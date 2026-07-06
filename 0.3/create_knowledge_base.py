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

# --- Chunking config ---
CHUNK_SIZE    = 500   # max characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks


# NOTE: SentenceTransformer does not support DirectML, so this script always runs on CPU.
# This is fine — you only need to run this once to build the knowledge base.
print("Running on CPU (SentenceTransformer does not support DirectML).")

# ── 1. Carica il modello di embedding ────────────────────────────────────────────────────────
print(f"Caricamento modello embedding: {MODEL_NAME}...")
st_model = SentenceTransformer(MODEL_NAME, device="cpu", model_kwargs={"use_safetensors": True})
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dense_dim = st_model.get_sentence_embedding_dimension()

SPECIAL_IDS = {             # Questi ID speciali non devono essere inclusi nel vettore sparso, altrimenti distorcono la rappresentazione.
    tokenizer.pad_token_id,       # ID del token di padding (usato per allineare le sequenze)
    tokenizer.unk_token_id,       # ID del token "unknown" (usato per parole fuori vocabolario)
    tokenizer.cls_token_id,       # ID del token "classificazione" (usato in alcuni modelli per indicare l'inizio della sequenza)
    tokenizer.sep_token_id,       # ID del token "separatore" (usato per separare le sequenze)
    tokenizer.bos_token_id,       # ID del token "inizio della sequenza" (usato in alcuni modelli, come GPT, per indicare l'inizio del testo)
    tokenizer.eos_token_id,       # ID del token "fine della sequenza" (usato in alcuni modelli, come GPT, per indicare la fine del testo)
}                           # questi id non portano informazioni semantiche utili, quindi è meglio escluderli dal vettore sparso.
SPECIAL_IDS.discard(None)

print(f"Dimensione vettore denso: {dense_dim}")

# ── 2. Chunker ────────────────────────────────────────────────────────────────────────────────
def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into chunks of at most `chunk_size` characters, with `overlap`
    characters of context carried over from the previous chunk.
 
    Strategy:
      1. Try to split on sentence boundaries ('. ', '? ', '! ') to avoid
         cutting mid-sentence whenever possible.
      2. Fall back to a hard character split only when no boundary is found.
    """
    chunks = []
    start = 0
    text_len = len(text)
 
    while start < text_len:
        end = min(start + chunk_size, text_len)
 
        # If we're not at the very end, try to find a sentence boundary
        # within the last 20 % of the window so we don't cut mid-sentence.
        if end < text_len:
            search_from = start + int(chunk_size * 0.8)
            best_boundary = -1
            for sep in ('. ', '? ', '! ', '\n'):
                pos = text.rfind(sep, search_from, end)
                if pos != -1 and pos > best_boundary:
                    best_boundary = pos + len(sep)   # include the separator in the chunk
 
            if best_boundary != -1:
                end = best_boundary
 
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
 
        # Move forward, stepping back by `overlap` to preserve context
        start = end - overlap if end - overlap > start else end
 
    return chunks
 


# ── 2. Carica i documenti ─────────────────────────────────────────────────────────────────
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
 
        chunks = split_into_chunks(raw_text)
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

    tf = Counter(token_ids)  #un dizionario che mappa ogni token alla sua frequenza nel testo. Esempio: {101: 3, 102: 1, 103: 2} 101 appare 3 volte, 102 appare 1 volta, 103 appare 2 volte.
    indices, values = [], []
    for token_id, count in tf.items():
        if token_id not in SPECIAL_IDS:
            indices.append(token_id)
            values.append(math.log1p(count))   #log1p = log(1+x) per punteggio assegnato ai token

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