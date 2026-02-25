import pickle  # Per salvare oggetti Python su disco
import os      # Per interagire con il sistema operativo (file, cartelle)
from sentence_transformers import SentenceTransformer  # Modello per creare embedding
import faiss   # Libreria per ricerca vettoriale veloce
import numpy as np  # Per operazioni matematiche con array
from PyPDF2 import PdfReader  # Legge file PDF
import docx     # Legge file Word

# --- Configurazione ---
DOCUMENTS_DIR = "./documents"  # Cartella dove mettere i tuoi file .txt
KNOWLEDGE_BASE_PKL = "knowledge_base.pkl"
FAISS_INDEX = "documents_index.faiss"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # Un buon modello per embedding

"""
Questo script crea una knowledge base a partire da documenti testuali, salvando i chunk di testo (paragrafi) in una matrice e donando loro informazioni di contesto (nome del file, id del chunk). 
il tutto viene salvato su disco per essere utilizzato successivamente.
"""

# 1. Carica i documenti (qui assumiamo siano file .txt, ma puoi estendere con PyPDF2, etc.)
def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Dividi in chunk (es. per paragrafi o frasi)
                chunks = content.split('\n\n')  # Dividi per doppi a capo
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Salta chunk vuoti
                        documents.append({
                            'source': filename,
                            'chunk_id': i,
                            'text': chunk.strip()
                        })
    return documents

print("Caricamento documenti...")
knowledge_base = load_documents_from_directory(DOCUMENTS_DIR)
print(f"Caricati {len(knowledge_base)} chunk di documenti.")

print(knowledge_base[:2])  # Stampa i primi 2 chunk per verifica

# 2. Carica il modello di embedding
print(f"Caricamento modello embedding: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# 3. Crea gli embedding per tutti i chunk
print("Creazione embedding...")
texts = [item['text'] for item in knowledge_base]
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# 4. Normalizza gli embedding e crea l'indice FAISS (per similarità coseno)
print("Creazione indice FAISS...")
dimension = embeddings.shape[1]
# Usiamo Inner Product, che diventa similarità coseno dopo la normalizzazione
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)  # Normalizza per usare cosine similarity
index.add(embeddings.astype('float32'))

# 5. Salva tutto su disco
print("Salvataggio knowledge base e indice...")
with open(KNOWLEDGE_BASE_PKL, 'wb') as f:
    pickle.dump(knowledge_base, f)
faiss.write_index(index, FAISS_INDEX)

print("Fatto! Knowledge base pronto.")