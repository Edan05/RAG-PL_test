from datasets import load_dataset
import os

# CNN/DailyMail has full news articles — much better for RAG testing
print("Downloading full news articles...")
ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
ds = ds.select(range(200))
print(f"Loaded {len(ds)} articles.")

DOCUMENTS_DIR = "./documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

for i, article in enumerate(ds):
    filename = f"{DOCUMENTS_DIR}/article_{i}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        # Write headline-style marker + full article body
        f.write(f"URL: {article['id']}\n\n")
        f.write(article['article'])
    print(f"Saved: {filename}")

print(f"\nDone! {len(ds)} full articles saved to {DOCUMENTS_DIR}/")