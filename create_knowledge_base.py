import os
import math
#import numpy as np
from PyPDF2 import PdfReader
import docx

from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct, SparseVector

from tqdm import tqdm

# --- Configurazione ---
DOCUMENTS_DIR = "./documents"
MODEL_NAME = 'BAAI/bge-m3'
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "knowledge_base"

DENSE_VECTOR_NAME  = "dense"   
SPARSE_VECTOR_NAME = "sparse"  

# --- Chunking config ---
CHUNK_SIZE    = 500   # max characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks


# NOTE: SentenceTransformer does not support DirectML, so this script always runs on CPU.
# This is fine — you only need to run this once to build the knowledge base.
print("Running on CPU (SentenceTransformer does not support DirectML).")

# ── 1. Carica il modello di embedding ────────────────────────────────────────────────────────
print(f"Caricamento modello embedding: {MODEL_NAME}...")
bge_model = BGEM3FlagModel(MODEL_NAME, use_fp16=False, device="cpu") #True se usi gpu CUDA
dense_dim = 1024

print(f"Dimensione vettore denso: {dense_dim}")

# ── 2. Chunker ────────────────────────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "? ", "! ", ". ", ", "]
)
 


# ── 3. Carica i documenti ─────────────────────────────────────────────────────────────────
def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
 
        if filename.endswith(".txt"):
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
 
        elif filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            raw_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
 
        elif filename.endswith(".docx"):
            doc = docx.Document(filepath)
            raw_text = "\n\n".join(p.text for p in doc.paragraphs)
 
        else:
            continue  # formato non supportato
 
        chunks = text_splitter.split_text(raw_text)
        for i, chunk in enumerate(chunks):
            documents.append({
                'source':   filename,
                'chunk_id': i,
                'text':     chunk
            })
 
    return documents
 
print("Caricamento documenti...")
knowledge_base = load_documents_from_directory(DOCUMENTS_DIR)
print(f"Caricati {len(knowledge_base)} chunk di documenti.")

# ── 4. Crea collezione Qdrant ────────────────────────────────────────────────────────────────
client = QdrantClient(url=QDRANT_URL)

if client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"⚠️  Collection '{COLLECTION_NAME}' già esistente — la elimino e la ricreo.")
    client.delete_collection(collection_name=COLLECTION_NAME)

print("Creazione collection Qdrant (dense + sparse)...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    # Vettori densi (cosine similarity)
    vectors_config={
        DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE)
    },
    # Vettori sparsi (lexical/BM25-like da BGE-M3)
    sparse_vectors_config={
        SPARSE_VECTOR_NAME: SparseVectorParams(
            index=SparseIndexParams(on_disk=False)  # tieni in RAM per velocità
        )
    }
)
print("Collection creata.")

# ── 5. encoding (BGE-M3 dense + native learned sparse) ───────────────────────────────────────────────────────────────────────────────
def encode_batch(texts: list[str]) -> tuple[list[list[float]], list[SparseVector]]:
    """
    Encode a list of texts with BGE-M3.
    Returns:
        dense_vecs  — list of 1024-dim float lists
        sparse_vecs — list of Qdrant SparseVector objects
    """
    output = bge_model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
 
    dense_vecs  = output["dense_vecs"].tolist()          # shape (B, 1024)
    lex_weights = output["lexical_weights"]              # list of dicts {token_id: weight}
 
    sparse_vecs = [
        SparseVector(
            indices=list(lw.keys()),
            values=[float(v) for v in lw.values()],     # ensure plain Python floats
        )
        for lw in lex_weights
    ]
 
    return dense_vecs, sparse_vecs
 

# ── 6. Prepara i punti per Qdrant ────────────────────────────────────────────────────────────────
ENCODE_BATCH_SIZE = 16

points = []
texts = [item["text"] for item in knowledge_base]

print ("creazione embedding (dense + sparse)...")
for batch_start in tqdm(range(0, len(texts), ENCODE_BATCH_SIZE), desc="Encoding documents"):
    batch_texts = texts[batch_start:batch_start + ENCODE_BATCH_SIZE]
    batch_items = knowledge_base[batch_start:batch_start + ENCODE_BATCH_SIZE]
    dense_vecs, sparse_vecs = encode_batch(batch_texts)

    for local_i, (item, dv, sv) in enumerate(zip(batch_items, dense_vecs, sparse_vecs)):
        global_id = batch_start + local_i
        points.append(
            PointStruct(
                id=global_id,
                vector={
                    DENSE_VECTOR_NAME: dv,
                    SPARSE_VECTOR_NAME: sv
                },
                payload={
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "text": item["text"]
                }
            )
        )


# ── 7. Upsert in Qdrant ────────────────────────────────────────────────────────────────────────   
def upsert_in_batches(client, collection_name, points, batch_size=50):
    with tqdm(total=len(points), desc="Uploading to Qdrant") as pbar:
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(collection_name=collection_name, points=batch)
            pbar.update(len(batch))

upsert_in_batches(client, "knowledge_base", points) 
print(f"✅ Inseriti {len(points)} punti in Qdrant. Knowledge base pronta!")