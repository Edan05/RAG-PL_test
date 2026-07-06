import numpy as np
import onnxruntime_genai as og
 
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
 
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
import torch


# --- Configurazione ---
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
RERANKER_MODEL_NAME  = "BAAI/bge-reranker-v2-m3" #cross encoder reranker
GENERATOR_MODEL_ID   = "Qwen/Qwen3-4B-Instruct-2507"
ROUTER_MODEL_ID    = "Qwen/Qwen3-0.6B"

qdrant_url           = "http://localhost:6333"
collection_name      = "knowledge_base"
TOP_K                = 5 
FETCH_K              = 20  # quanti risultati da prefetchare da Qdrant prima di rerankare e ridurre a TOP_K

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"

# --- Device Detection ---
# Paths to the exported ONNX models (output of onnxruntime-genai builder)
if torch.cuda.is_available():
    GENERATOR_ONNX_PATH = "./qwen_gen_onnx_cuda"
    ROUTER_ONNX_PATH    = "./qwen_router_onnx_cuda"
    LLM_DEVICE = "CUDA"
    bge_fp16 = True
    print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")

else:
    GENERATOR_ONNX_PATH = "./qwen_gen_onnx_dml"
    ROUTER_ONNX_PATH    = "./qwen_router_onnx_dml"
    LLM_DEVICE = "DirectML"
    bge_fp16 = False
    print("ℹ️  No CUDA GPU detected. BGE-M3 and reranker will use CPU fp32.")


# 1. ── Carica il modello di embedding ────────────────────────────────────────────────────────

print("caricamento modello di embedding...")
bge_model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=bge_fp16, device="cpu")

# 2. ── Connessione a Qdrant ────────────────────────────────────────────────────────────────
print(f"connessione a Qdrant su {qdrant_url}...")
qdrant_client = QdrantClient(url=qdrant_url)

# 3. ── Funzioni di encoding della query ────────────────────────────────────────────────────────

def encode_query(text: str) -> list[float]:
    output = bge_model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    lw = output["lexical_weights"][0]
    dense = output["dense_vecs"][0].tolist()
    sparse = SparseVector(
        indices=list(lw.keys()),
        values=[float(v) for v in lw.values()]
    )
    return dense, sparse

# 4. ── Carica il reranker BGE ───────────────────────────────────────────────────────────────

print(f"Caricamento reranker {RERANKER_MODEL_NAME}...")
reranker = CrossEncoder(
    RERANKER_MODEL_NAME,
    device="cpu",
    automodel_args={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32},
)


# 5. ── Funzione di ricerca in Qdrant ───────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K, fetch_k: int = FETCH_K) -> list[dict]:
    """
    1. Hybrid retrieval from Qdrant (dense cosine + BGE-M3 sparse, fused with RRF).
       Fetches `fetch_k` candidates — more than we need so the reranker has
       enough material to work with.
    2. Reranks all candidates with BGE-reranker-v2-m3 (cross-encoder).
    3. Returns the best `top_k` chunks sorted by reranker score.
    """
    dense, sparse = encode_query(query)
 
    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=dense,  using=DENSE_VECTOR_NAME,  limit=fetch_k),
            Prefetch(query=sparse, using=SPARSE_VECTOR_NAME, limit=fetch_k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=fetch_k,
    )
 
    candidates = [
        {
            "source":   r.payload["source"],
            "chunk_id": r.payload["chunk_id"],
            "text":     r.payload["text"],
            "rrf_score": r.score,
        }
        for r in results.points
    ]
 
    if not candidates:
        return []
 
    # Cross-encoder reranking ─────────────────────────────────────────────────
    pairs         = [[query, c["text"]] for c in candidates]
    raw_scores     = reranker.predict(pairs, convert_to_numpy=True)
    rerank_scores = (1 / (1 + np.exp(-raw_scores))).tolist()  # sigmoid → [0, 1]
 
    for chunk, score in zip(candidates, rerank_scores):
        chunk["rerank_score"] = float(score)
 
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# 6. ── Carica il modelli generativi (Router e LLM) ───────────────────────────────────────────────────────────────
"""
og.Model loads the exported ONNX model and routes execution to DirectML.
"""

print(f"Caricamento modello generativo da {GENERATOR_ONNX_PATH}...")
gen_model     = og.Model(GENERATOR_ONNX_PATH)
gen_tokenizer = og.Tokenizer(gen_model)
print("Modello generativo caricato.")
 
print(f"Caricamento modello router da {ROUTER_ONNX_PATH}...")
router_model  = og.Model(ROUTER_ONNX_PATH)
router_tok    = og.Tokenizer(router_model)
print("Modello router caricato.")
 
# tokenizzo ogni prompt per gli llm
def _encode_messages(messages: list[dict], og_tokenizer) -> list[int]:
    prompt = ""
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]

        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return og_tokenizer.encode(prompt)



# 7. ── Routing ────────────────────────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = """You are a classifier. Answer with a single word only: YES or NO.

Is this query a question about specific files, documents, or records stored in a private database?

NO  = general knowledge, conversation, math, cooking, history, science, creative tasks, greetings
YES = explicitly asks about specific stored documents, contracts, reports, or database content

Output ONLY the word YES or NO. Nothing else.

Examples:
"ciao" → NO
"come faccio una torta?" → NO
"what is 2+2?" → NO
"what is the capital of France?" → NO
"write me a poem" → NO
"what does the contract say about termination?" → YES
"summarize the Q3 report" → YES
"do you have files about X in the database?" → YES
"what are the product specifications in the document?" → YES"""

def route_query(query: str, retries: int = 5) -> str:
    """Route query to retrieval or generative pipeline."""
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Query: {query}"}
    ]

    input_tokens = _encode_messages(messages, router_tok)

    for attempt in range(retries):
        # posso settare i parametri di generazione usando og.GeneratorParams e .set_search_options, ad esempio per limitare la lunghezza massima o forzare temperature basse per output più deterministici
        params = og.GeneratorParams(router_model)
        params.set_search_options(
            max_length=len(input_tokens) + 250,  # just a few tokens for the answer
            do_sample=True,   #not deterministic, to allow some variability in the output and gives room for retries if the format isn't correct
            temperature=0.1,  # almost deterministic output
        )

        gen_runner = og.Generator(router_model, params)
        gen_runner.append_tokens(input_tokens)   # ← feed tokens here

        answer_tokens = []
        # questo while loop genera un token alla volta e lo mette in answer tokens. (autoregressive decoding, default per i modelli di linguaggio)
        while not gen_runner.is_done():
            gen_runner.generate_next_token()
            new_token = gen_runner.get_next_tokens()[0]
            answer_tokens.append(new_token)

        answer_raw = router_tok.decode(answer_tokens).strip()

        #DEBUG: print raw router output for inspection
        print(f"Router attempt {attempt+1}: '{answer_raw}'")

        # Extract only what comes after </think> if present, to allow the model to "think" before answering
        if "</think>" in answer_raw:
            answer = answer_raw.split("</think>")[-1].strip().upper()
        else:
            continue # if the expected format isn't met, retry

        if "YES" in answer:
            return "retrieval"
        elif "NO" in answer:
            return "generative"

    print(f"⚠️  Router non ha classificato la query dopo {retries} tentativi, usando retrieval come fallback.")
    return "retrieval"  # meglio fare retrieval se il router fallisce



# 8. ── prompt generatore ───────────────────────────────────────────────────────────────
GENERATOR_SYSTEM_PROMPT_RETRIEVAL = """You are a helpful assistant that answers questions based on provided document fragments.
Rules:
- Always answer in the same language the user is using, without explaining why
- Use the provided context if relevant
- If the context is not relevant to the question, ignore it and answer from your general knowledge, explicitly stating that the provided documents were not relevant
- Only say you don't know if BOTH the context and your general knowledge cannot answer the question
- Keep answers under 500 tokens"""

GENERATOR_SYSTEM_PROMPT_GENERATIVE = """You are a helpful assistant.
Rules:
- Always answer in the same language the user is using, without explaining why
- Answer from your general knowledge
- If you don't know, say you don't know — never make anything up
- Keep answers under 500 tokens"""

MAX_HISTORY_EXCHANGES = 5

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

# 9. ── Loop interattivo ───────────────────────────────────────────────────────────────
conversation_history = []

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
        retrieved_chunks = retrieve(query, top_k=TOP_K, fetch_k=FETCH_K)
        context = "\n\n".join([f"From {r['source']}: {r['text']}" for r in retrieved_chunks])

    # --- Build messages and generate (streaming)---

    print("✍️  Generazione risposta...")
    messages     = build_messages(query, conversation_history, context)
    input_tokens = _encode_messages(messages, gen_tokenizer)


    #DEBUG: print prompt for inspection #################################################################################
    """
    prompt_check = ""
    for m in messages:
        prompt_check += f"\n[{m['role'].upper()}]: {m['content'][:20000]}"
    print(f"prompt_check", prompt_check[:5000], "..." if len(prompt_check) > 5000 else "")
    """
    params = og.GeneratorParams(gen_model)
    params.set_search_options(
        max_length=len(input_tokens) + 512,  # allow up to 512 tokens for the answer
        temperature=0.7,
        do_sample=True,
    )
 
    gen_runner    = og.Generator(gen_model, params)
    gen_runner.append_tokens(input_tokens)
    answer_tokens = []
    
    # end="" rimuove la newline dal print (print() ha /n di default) e flush=true forza la stampa immediata, senza passare per la memoria intermedia(evita ritardi nello streaming e blocchi di testo)
    print("\n💬 ", end="", flush=True)
    # la differenza tra questo while loop e quello del router è che decodo ogni token appena generato e lo printo immediatamente per attivare lo streaming.
    while not gen_runner.is_done():
        gen_runner.generate_next_token()
        new_token  = gen_runner.get_next_tokens()[0]
        token_text = gen_tokenizer.decode([new_token])
        print(token_text, end="", flush=True)
        answer_tokens.append(token_text)
 
    print()
    answer = "".join(answer_tokens).strip()
 
    # --- Store and display ---
    conversation_history.append((query, answer))
 
    if route == "retrieval":
        print("\n--- FONTI USATE ---")
        for r in retrieved_chunks:
            print(f"- {r['source']} (chunk {r['chunk_id']}): {r['text'][:100]}...")
