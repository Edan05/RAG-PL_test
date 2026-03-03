from retrieval import load_retrieval_components, retrieve
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configurazione ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
GENERATOR_MODEL_ID = "microsoft/phi-2"  # Un modello piccolo ma capace
#GENERATOR_MODEL_ID = "Qwen/Qwen3-0.6B" # modello troppo piccolo, troppi problemi logici, non riesce a rispondere a molte domande (anche semplici) e spesso si blocca o va in loop. Da evitare.
KNOWLEDGE_BASE_PKL = "knowledge_base.pkl"
FAISS_INDEX = "documents_index.faiss"
TOP_K = 6

# 1. Carica i componenti di retrieval
kb, index, emb_model = load_retrieval_components(KNOWLEDGE_BASE_PKL, FAISS_INDEX, EMBEDDING_MODEL_NAME)

# 2. Carica il modello generativo (CPU mode)
print(f"Caricamento modello generativo {GENERATOR_MODEL_ID}...")

model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_ID)
# Importante: molti modelli non hanno un token di padding. Settiamo eos_token come pad_token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Creiamo una pipeline di text-generation
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)  # -1 = CPU


# 3. Loop interattivo
print("\n" + "="*50)
print("RAG Pipeline ready!          Using Microsoft Phi-2 as generator and all-MiniLM-L6-v2 for retrieval.")
print("="*50)

# Conversation history to maintain context across queries
conversation_history = []
MAX_HISTORY_EXCHANGES = 5  # Keep last 5 exchanges to avoid overly long prompts

while True:
    query = input("\n🤔 Ask a question (or type 'quit' to exit, 'history' to see conversation, 'clear' to reset): ")
    if query.lower() == 'quit':
        break
    elif query.lower() == 'history':
        if conversation_history:
            print("\n📝 Conversation History:")
            for i, (q, a) in enumerate(conversation_history, 1):
                print(f"\n[{i}] Q: {q}")
                print(f"    A: {a[:200]}..." if len(a) > 200 else f"    A: {a}")
        else:
            print("No conversation history yet.")
        continue
    elif query.lower() == 'clear':
        conversation_history.clear()
        print("✨ Conversation history cleared!")
        continue

    # --- Retrieval ---
    print("🔍 Searching documents...")
    retrieved_chunks = retrieve(query, emb_model, index, kb, k=TOP_K)

    # --- Augmentation (Costruzione del prompt) ---
    context = "\n\n".join([f"From {r['source']}: {r['text']}" for r in retrieved_chunks])
    
    # Build conversation history context
    history_context = ""
    if conversation_history:
        history_context = "Previous conversation:\n"
        for prev_q, prev_a in conversation_history[-MAX_HISTORY_EXCHANGES:]:
            history_context += f"Q: {prev_q}\nyour Answer: {prev_a}\n\n"
        history_context += "---\n\n"
    
    prompt = f"""Use the following document fragments to answer the question.
If you don't know the answer, just say you don't know, without making anything up. Keep your answer below 500 tokens.

{history_context}Context:
{context}

Question: {query}
Answer:"""

    # --- Generation ---
    print("✍️ Generating answer...")
    responses = generator(
        prompt,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
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
        print(f"\n💬 Answer: {answer}")
    else:
        print(f"\n💬 Answer: [No response generated. Full text: {generated_text[:300]}...]")

    print("\n--- Sources Used ---")
    for r in retrieved_chunks:
        print(f"- {r['source']} (chunk {r['chunk_id']}): {r['text'][:100]}...")