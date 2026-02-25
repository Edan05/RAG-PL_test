import pickle                                           # Per caricare/salvare oggetti Python su disco
import faiss                                            # Libreria per ricerca vettoriale efficiente (sviluppata da Meta/Facebook)
import numpy as np
from sentence_transformers import SentenceTransformer   # Per creare embedding di frasi

#1. Questa funzione carica dal disco tutti i componenti necessari per fare retrieval.

def load_retrieval_components(kb_pkl, faiss_index_path, model_name):
    with open(kb_pkl, 'rb') as f:
        knowledge_base = pickle.load(f)
    index = faiss.read_index(faiss_index_path)
    model = SentenceTransformer(model_name)
    return knowledge_base, index, model

#2. Questa funzione prende una query, la trasforma in embedding, e cerca i chunk più simili nella knowledge base usando FAISS.

def retrieve(query, model, index, knowledge_base, k=6):
    # 1. Crea l'embedding per la query
    query_embedding = model.encode([query], convert_to_numpy=True)
    # 2. Normalizza (come fatto per i documenti)
    faiss.normalize_L2(query_embedding)
    # 3. Cerca nell'indice
    distances, indices = index.search(query_embedding.astype('float32'), k)
    # 4. Recupera i chunk corrispondenti
    retrieved_chunks = [knowledge_base[i] for i in indices[0]]
    return retrieved_chunks

# Esempio di utilizzo (se eseguito direttamente)
if __name__ == "__main__":
    kb, idx, emb_model = load_retrieval_components("knowledge_base.pkl", "documents_index.faiss", 'sentence-transformers/all-MiniLM-L6-v2')
    query = input("Inserisci la tua domanda: ")
    results = retrieve(query, emb_model, idx, kb, k=6)
    print("\n--- Chunk Recuperati ---")
    for r in results:
        print(f"Fonte: {r['source']} (Chunk {r['chunk_id']})")
        print(f"Testo: {r['text'][:200]}...\n")

