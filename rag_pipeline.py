import math
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
import torch

# --- Configurazione ---
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
GENERATOR_MODEL_ID   = "Qwen/Qwen3-4B-Instruct-2507"
qdrant_url           = "http://localhost:6333"
collection_name      = "knowledge_base"
TOP_K                = 6

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"

# --- Device Detection ---
try:
    import torch_directml
    dml_device = torch_directml.device()
    use_dml = True
    print(f"✅ DirectML GPU detected: {torch_directml.device_name(0)}")
except Exception as e:
    dml_device = None
    use_dml = False
    print(f"⚠️  DirectML not available ({e}), falling back to CPU.")

# 1. ── Carica il modello di embedding ────────────────────────────────────────────────────────

print (f"Caricamento modello di embedding {EMBEDDING_MODEL_NAME}...")
st_model  = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", model_kwargs={"use_safetensors": True})
tokenizer_emb = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

SPECIAL_IDS = {
    tokenizer_emb.pad_token_id,
    tokenizer_emb.unk_token_id,
    tokenizer_emb.cls_token_id,
    tokenizer_emb.sep_token_id,
    tokenizer_emb.bos_token_id,
    tokenizer_emb.eos_token_id,
}
SPECIAL_IDS.discard(None)

# 2. ── Connessione a Qdrant ────────────────────────────────────────────────────────────────
print(f"connessione a Qdrant su {qdrant_url}...")
qdrant_client = QdrantClient(url=qdrant_url)

# 3. ── Funzione di encoding della query ────────────────────────────────────────────────────────
def encode_query(query: str):
    output = st_model.encode(
        query,
        return_dense=True,
        return_sparse=True,
        convert_to_numpy=True
    )
    dense = output['dense_vecs'].tolist()

    sparse_dict = output['lexical_weights'] # sparse_dict è un dizionario dove ogni chiave è un token e il valore è il peso associato. la mggior parte dei token non compare, quindi si chiama sparso.
    indices = [int(k)   for k in sparse_dict.keys()]    #quanti token compaiono
    values  = [float(v) for v in sparse_dict.values()]  #quanto pesano quei token
    return dense, indices, values

# ── 4.Funzione di retrieval ibrida (dense + sparse via RFF fusion) ───────────────────────────────────────────────────────────────
def get_dense_vector(text: str) -> list:
    return st_model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()


def get_sparse_vector(text: str) -> SparseVector:
    token_ids = tokenizer_emb(
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


def retrieve(query: str, k: int = TOP_K) -> list:
    """Hybrid retrieval: dense cosine + sparse lexical, fused with Reciprocal Rank Fusion."""
    dense_vec  = get_dense_vector(query)
    sparse_vec = get_sparse_vector(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=k * 2
            ),
            Prefetch(
                query=sparse_vec,
                using=SPARSE_VECTOR_NAME,
                limit=k * 2
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k
    )

    return [
        {
            "source":   r.payload["source"],
            "chunk_id": r.payload["chunk_id"],
            "text":     r.payload["text"],
            "score":    r.score
        }
        for r in results.points
    ]

# ── 5. Carica il modello generativo ────────────────────────────────────────────────────────────────
print(f"Caricamento modello generativo {GENERATOR_MODEL_ID}...")

# Detect AMD GPU via ROCm (exposed as 'cuda' in PyTorch)
model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_ID, dtype=torch.float16, use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_ID)

if use_dml:
    model = model.to(dml_device)  # Move model to AMD GPU via DirectML

# Importante: molti modelli non hanno un token di padding. Settiamo eos_token come pad_token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Pass the DirectML device object to the pipeline (or -1 for CPU)
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=dml_device if use_dml else -1  # -1 = CPU
)

# ── 6. Loop interattivo ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"RAG Pipeline pronta!")
print(f"  Generator : {GENERATOR_MODEL_ID}")
print(f"  Retrieval : {EMBEDDING_MODEL_NAME} (dense + sparse hybrid, RRF fusion)")
print("="*60)

# Conversation history to maintain context across queries
conversation_history = []
MAX_HISTORY_EXCHANGES = 5  # Keep last 5 exchanges to avoid overly long prompts

while True:
    query = input("\n🤔 fai una domanda (o 'history' per vedere la cronologia, 'clear' per cancellarla, 'quit' per uscire): ").strip()
    if query.lower() == 'quit':
        break
    elif query.lower() == 'history':
        if conversation_history:
            print("\n📝 Cronologia conversazione:")
            for i, (q, a) in enumerate(conversation_history, 1):
                print(f"\n[{i}] Q: {q}")
                print(f"    A: {a[:200]}..." if len(a) > 200 else f"    A: {a}")
        else:
            print("Nessuna conversazione precedente.")
        continue
    elif query.lower() == 'clear':
        conversation_history.clear()
        print("✨ Conversazione cancellata.")
        continue

    # --- Retrieval ---
    print("🔍 cercando documenti...")
    retrieved_chunks = retrieve(query, k=TOP_K)

    # --- Augmentation (Costruzione del prompt) ---
    context = "\n\n".join([f"From {r['source']}: {r['text']}" for r in retrieved_chunks])
    
    # Build conversation history context
    history_context = ""
    if conversation_history:
        history_context = "Previous conversation:\n"
        for prev_q, prev_a in conversation_history[-MAX_HISTORY_EXCHANGES:]:
            history_context += f"User question: {prev_q}\nAssistant answer: {prev_a}\n\n"
        history_context += "---\n\n"
    
    prompt = f"""Use the following document fragments to answer the question in the language it's written in.
If the question is not related to the provided documents, answer based on your general knowledge. Always try to use the provided documents if they are relevant.
If you don't know the answer, just say you don't know, without making anything up. Keep your answer below 500 tokens.
If you fall back to your general knowledge, please specify that in your answer.
If you loop or get stuck, just say you got stuck and end the answer.

{history_context}Context:
{context}

Question: {query}
Answer:"""

    # --- Generation ---
    print("✍️ Generazione risposta...")
    responses = generator(
        prompt,
        generation_config=GenerationConfig(
        max_new_tokens=500,
        max_length=None,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    )
    generated_text = responses[0]['generated_text']

    # Estrai solo la parte dopo "Answer:" (con fallback se non trovato)
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        # Se "Answer:" non è nel testo, prendi tutto dopo il prompt
        answer = generated_text.split(prompt)[-1].strip()
    
    # Store in conversation history
    conversation_history.append((query, answer))
    
    if answer:
        print(f"\n💬 {answer}")
    else:
        print(f"\n💬 [No response generated. Full text: {generated_text[:300]}...]")

    print("\n--- FONTI USATE ---")
    for r in retrieved_chunks:
        print(f"- {r['source']} (chunk {r['chunk_id']}): {r['text'][:100]}...")