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
ROUTER_MODEL_ID    = "Qwen/Qwen3-0.6B"

qdrant_url           = "http://localhost:6333"
collection_name      = "knowledge_base"
TOP_K                = 5

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"

# --- Device Detection ---
use_dml = False
dml_device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
else:
    try:
        import torch_directml
        dml_device = torch_directml.device()
        use_dml = True
        device = dml_device
        print(f"✅ DirectML GPU detected: {torch_directml.device_name(0)}")
    except Exception as e:
        device = torch.device("cpu")
        print(f"⚠️  DirectML not available ({e}), falling back to CPU.")

# 1. ── Carica il modello di embedding ────────────────────────────────────────────────────────

print("caricamento modello di embedding...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", model_kwargs={"use_safetensors": True})
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

# 3. ── Funzioni di encoding della query ────────────────────────────────────────────────────────

#vettore denso della query
def get_dense_vector(text: str) -> list:
    return embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()


#vettore sparso della query
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

# 4. ── Funzione di ricerca in Qdrant ───────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = TOP_K) -> list:
    """Hybrid retrieval: dense cosine + sparse lexical, fused with Reciprocal Rank Fusion."""
    dense = get_dense_vector(query)
    sparse = get_sparse_vector(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=dense,
                using=DENSE_VECTOR_NAME,
                limit=k * 2
            ),
            Prefetch(
                query=sparse,
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

# 5. ── Carica il modelli generativi (Router e LLM) ───────────────────────────────────────────────────────────────
print(f"Caricamento modello generativo {GENERATOR_MODEL_ID}...")

generator = pipeline(
    "text-generation",
    model=GENERATOR_MODEL_ID,
    dtype=torch.float16,
    device=device
)

if generator.tokenizer.pad_token is None:
    generator.tokenizer.pad_token = generator.tokenizer.eos_token

print(f"Modello generativo caricato su {device}.")

print(f"Caricamento modello router {ROUTER_MODEL_ID}...")
router = pipeline(
    "text-generation",
    model=ROUTER_MODEL_ID,
    dtype=torch.float16,
    device=device
)
if router.tokenizer.pad_token is None:
    router.tokenizer.pad_token = router.tokenizer.eos_token
print(f"Modello router caricato su {device}.")

# 6. ── Routing ────────────────────────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = """You are a query classifier. You must respond with EXACTLY one word, nothing else.

Classify the user query into one of these two categories:
- RETRIEVAL: questions about specific facts, documents, data, or knowledge that requires looking up information
- GENERATIVE: general conversation, creative tasks, or questions answerable from common knowledge

Rules:
- Output ONLY the single word: RETRIEVAL or GENERATIVE
- No punctuation, no explanation, no other words
- If unsure, output RETRIEVAL
- If asked about files or documents in a database output RETRIEVAL

Examples:
Query: "do you have files about politics in the database?" → RETRIEVAL
Query: "What does the contract say about termination?" → RETRIEVAL
Query: "What is the capital of France?" → GENERATIVE
Query: "Summarize the Q3 report" → RETRIEVAL
Query: "Write me a poem" → GENERATIVE
Query: "What are the product specifications?" → RETRIEVAL
Query: "How do I bake a cake?" → GENERATIVE
Query: "What can you tell me about the american economic policy?" → RETRIEVAL
Query: "What is 2+2?" → GENERATIVE
"""

def route_query(query: str, retries: int = 5) -> str:
    """Route query to retrieval or generative pipeline."""
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": f"/no_think\Query: {query}"}
    ]
    
    for attempt in range(retries):
        response = router(
            messages,
            max_new_tokens=20,       # one word needs at most 2-3 tokens, 5 is safe headroom
            do_sample=False,        # greedy decoding — you want determinism, not creativity
            temperature=None,       # must be None when do_sample=False
            top_p=None,             # same
            pad_token_id=router.tokenizer.pad_token_id
        )
        
        # extract only the new assistant text, not the prompt
        answer = response[0]['generated_text'][-1]['content'].strip().upper()

        print(f"Router attempt {attempt+1}: '{answer}'")
        
        if "RETRIEVAL" in answer:
            return "retrieval"
        elif "GENERATIVE" in answer:
            return "generative"
        # if neither matched, retry
    
    print(f"⚠️  Router non ha classificato la query dopo {retries} tentativi, usando retrieval come fallback.")
    return "retrieval"  # safe default: better to retrieve unnecessarily than to skip it

# 7. ── propts generatore ───────────────────────────────────────────────────────────────
GENERATOR_SYSTEM_PROMPT_RETRIEVAL = """/no_think
You are a helpful assistant that answers questions based on provided document fragments.
Rules:
- Always answer in the same language as the question, without explaining why
- Use the provided context if relevant
- If the context is not relevant, answer from general knowledge and explicitly say so
- If you don't know, say you don't know — never make anything up
- Keep answers under 500 tokens"""

GENERATOR_SYSTEM_PROMPT_GENERATIVE = """/no_think
You are a helpful assistant.
Rules:
- Always answer in the same language as the question, without explaining why
- Answer from your general knowledge
- If you don't know, say you don't know — never make anything up
- Keep answers under 500 tokens"""

def build_messages(query: str, conversation_history: list, context: str = None) -> list:
    # pick the right system prompt depending on whether we have context
    system_prompt = GENERATOR_SYSTEM_PROMPT_RETRIEVAL if context else GENERATOR_SYSTEM_PROMPT_GENERATIVE
    messages = [{"role": "system", "content": system_prompt}]

    # inject conversation history for both branches
    for prev_q, prev_a in conversation_history[-MAX_HISTORY_EXCHANGES:]:
        messages.append({"role": "user",      "content": prev_q})
        messages.append({"role": "assistant", "content": prev_a})

    # append current query, with context if retrieval branch
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    else:
        messages.append({"role": "user", "content": query})

    return messages

# 8. ── Loop interattivo ───────────────────────────────────────────────────────────────
conversation_history = []
MAX_HISTORY_EXCHANGES = 5

print("\n" + "="*60)
print(f"RAG Pipeline pronta!")
print(f"  Generator : {GENERATOR_MODEL_ID}")
print(f"  Router    : {ROUTER_MODEL_ID}")
print(f"  Retrieval : {EMBEDDING_MODEL_NAME} (dense + sparse hybrid, RRF fusion)")
print("="*60)

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

    # --- Routing ---
    route = route_query(query)
    print(f"📡 Router: {route.upper()}")

    # --- Retrieval (if needed) ---
    context = None
    if route == "retrieval":
        print("🔍 Ricerca documenti...")
        retrieved_chunks = retrieve(query, k=TOP_K)
        context = "\n\n".join([f"From {r['source']}: {r['text']}" for r in retrieved_chunks])

    # --- Build messages and generate ---
    print("✍️  Generazione risposta...")
    messages  = build_messages(query, conversation_history, context)
    responses = generator(
        messages,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        pad_token_id=generator.tokenizer.pad_token_id
    )
    answer = responses[0]['generated_text'][-1]['content'].strip()

    # --- Store and display ---
    conversation_history.append((query, answer))
    print(f"\n💬 {answer}")

    if route == "retrieval":
        print("\n--- FONTI USATE ---")
        for r in retrieved_chunks:
            print(f"- {r['source']} (chunk {r['chunk_id']}): {r['text'][:100]}...")


