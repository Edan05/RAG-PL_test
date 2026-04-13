from datasets import load_dataset
import os
import json

# 1. Dataset di articoli news(perfetto per RAG)
print("Scaricamento dataset news...")
ds = load_dataset("ag_news", split="train")  # Carica il dataset intero dal training set
print(f"Dataset caricato con {len(ds)} articoli.")
# Limita a 2000 elementi per il processing
ds = ds.select(range(min(2000, len(ds))))

# 2. Salva i documenti in una cartella (txt per testare)
DOCUMENTS_DIR = "./documents"
# Crea la cartella documents se non esiste
os.makedirs("./documents", exist_ok=True)
for i, paper in enumerate(ds):
    filename = f"./documents/article_{i}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Articolo: {paper['text']}\n")
    print(f"Salvato: {filename}")