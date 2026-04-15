import os
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq

# ── 1. DOCUMENTS ──────────────────────────────────────────────
documents = [
    "Our return policy allows returns within 30 days of purchase with a receipt.",
    "Shipping takes 3-5 business days for standard, 1-2 days for express delivery.",
    "We offer a 1-year warranty on all electronics purchased from our store.",
    "Customer support is available Monday to Friday, 9am to 6pm EST.",
    "Payment methods accepted: Visa, Mastercard, PayPal, and Apple Pay.",
]

# ── 2. BUILD PIPELINE (LOADED ONCE) ───────────────────────────
def build_pipeline():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Dense embeddings
    doc_embeddings = embedder.encode(documents, normalize_embeddings=True)

    # BM25 (sparse search)
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    return {
        "embedder": embedder,
        "doc_embeddings": doc_embeddings,
        "bm25": bm25,
        "documents": documents
    }

# ── 3. HYBRID SEARCH (DENSE + BM25) ───────────────────────────
def hybrid_search(pipeline, query: str, top_k: int = 3, filters=None):
    embedder = pipeline["embedder"]
    doc_embeddings = pipeline["doc_embeddings"]
    bm25 = pipeline["bm25"]
    documents = pipeline["documents"]

    # Dense similarity
    query_vec = embedder.encode([query], normalize_embeddings=True)
    dense_scores = np.dot(doc_embeddings, query_vec.T).flatten()

    # BM25 scores
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Combine scores (hybrid)
    final_scores = 0.7 * dense_scores + 0.3 * bm25_scores

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [documents[i] for i in top_indices]

# ── 4. GENERATION (LLM) ───────────────────────────────────────
def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""You are a helpful customer support assistant.
Answer ONLY based on the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}"""

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    return response.choices[0].message.content