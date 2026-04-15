# pip install groq sentence-transformers numpy
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq                              # ✅ Groq client

# ── 1. YOUR DOCUMENTS ──────────────────────────────────────────────
documents = [
    "Our return policy allows returns within 30 days of purchase with a receipt.",
    "Shipping takes 3-5 business days for standard, 1-2 days for express delivery.",
    "We offer a 1-year warranty on all electronics purchased from our store.",
    "Customer support is available Monday to Friday, 9am to 6pm EST.",
    "Payment methods accepted: Visa, Mastercard, PayPal, and Apple Pay.",
]

# ── 2. INDEXING — embed all documents once ─────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents)

# ── 3. RETRIEVER — find top-k most relevant chunks ─────────────────
def retrieve(query: str, k: int = 2) -> list[str]:
    query_vec = embedder.encode([query])
    scores = np.dot(doc_embeddings, query_vec.T).flatten()
    # print(scores)
    scores /= (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_vec))
    top_k = np.argsort(scores)[::-1][:k]
    return [documents[i] for i in top_k]

# ── 4. GENERATION — inject context into prompt ─────────────────────
def rag_answer(query: str) -> str:
    context_chunks = retrieve(query)
    context = "\n".join(f"- {c}" for c in context_chunks)
    # print(f"Context:\n{context}\n")
    prompt = f"""You are a helpful customer support assistant.
Answer ONLY based on the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}"""

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",                   # ✅ free, fast, no quota issues
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

# ── 5. TEST IT ──────────────────────────────────────────────────────
queries = [
    "Can I return something I bought 3 weeks ago?",
    "How long will my delivery take?",
    "Do you accept American Express?",
    "Hi what is my name?",
    "who is Neil Armstrong?",
]

for q in queries:
    print(f"\nQ: {q}")
    print(f"A: {rag_answer(q)}")