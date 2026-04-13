import os
import math
from collections import Counter
#import numpy as np
from PyPDF2 import PdfReader
import docx


from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

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


# NOTE: SentenceTransformer does not support DirectML, so this script always runs on CPU.
# This is fine — you only need to run this once to build the knowledge base.
print("Running on CPU (SentenceTransformer does not support DirectML).")

# ── 1. Carica il modello di embedding ────────────────────────────────────────────────────────
print(f"Caricamento modello embedding: {MODEL_NAME}...")
st_model = SentenceTransformer(MODEL_NAME, device="cpu", model_kwargs={"use_safetensors": True})
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dense_dim = st_model.get_sentence_embedding_dimension()

SPECIAL_IDS = {
    tokenizer.pad_token_id,
    tokenizer.unk_token_id,
    tokenizer.cls_token_id,
    tokenizer.sep_token_id,
    tokenizer.bos_token_id,
    tokenizer.eos_token_id,
}
SPECIAL_IDS.discard(None)

print(f"Dimensione vettore denso: {dense_dim}")

# ── 2. Carica i documenti ─────────────────────────────────────────────────────────────────
def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if filename.endswith(".txt"):
            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = f.read().split('\n\n')

        elif filename.endswith(".pdf"):
            reader = PdfReader(filepath)
            full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            chunks = full_text.split('\n\n')

        elif filename.endswith(".docx"):
            doc = docx.Document(filepath)
            full_text = "\n\n".join(p.text for p in doc.paragraphs)
            chunks = full_text.split('\n\n')

        else:
            continue  # formato non supportato

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                documents.append({
                    'source':   filename,
                    'chunk_id': i,
                    'text':     chunk.strip()
                })
    return documents

print("Caricamento documenti...")
knowledge_base = load_documents_from_directory(DOCUMENTS_DIR)
print(f"Caricati {len(knowledge_base)} chunk di documenti.")

# ── 3. Crea collezione Qdrant ────────────────────────────────────────────────────────────────
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

# ── 4. encoding ───────────────────────────────────────────────────────────────────────────────
print("Creazione embedding (dense + sparse)...")

def get_dense_vector(text: str) -> list:
    """Restituisce il vettore denso normalizzato per il testo dato."""
    return st_model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()


def get_sparse_vector(text: str) -> SparseVector:
    """
    Tokenizza il testo e costruisce un vettore sparso basato sulla frequenza dei token.
    I token speciali (pad, unk, cls, sep, bos, eos) vengono ignorati.
    il peso di ogni token è calcolato come log(1 + frequenza), dove la frequenza è il numero di volte che il token appare nel testo.
    """
    token_ids = tokenizer(
        text,
        truncation=True,
        max_length=8192,
        add_special_tokens=False
    )["input_ids"]

    tf = Counter(token_ids)
    indices, values = [], []
    for token_id, count in tf.items():
        if token_id not in SPECIAL_IDS:
            indices.append(token_id)
            values.append(math.log1p(count))

    return SparseVector(indices=indices, values=values)


def encode_text(text: str):
    return get_dense_vector(text), get_sparse_vector(text)

points = []
for i, item in enumerate(tqdm(knowledge_base, desc="Encoding documents")):
    dense_vec, sparse_vec = encode_text(item['text'])
    points.append(
        PointStruct(
            id=i,
            vector={
                DENSE_VECTOR_NAME:  dense_vec,
                SPARSE_VECTOR_NAME: sparse_vec,
            },
            payload={
                "source":   item['source'],
                "chunk_id": item['chunk_id'],
                "text":     item['text']
            }
        )
    )

# ── 5. Upsert in Qdrant ────────────────────────────────────────────────────────────────────────   
def upsert_in_batches(client, collection_name, points, batch_size=50):
    with tqdm(total=len(points), desc="Uploading to Qdrant") as pbar:
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(collection_name=collection_name, points=batch)
            pbar.update(len(batch))

upsert_in_batches(client, "knowledge_base", points)
print(f"✅ Inseriti {len(points)} punti in Qdrant. Knowledge base pronta!")