import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq

# ── 1. Documents WITH metadata ─────────────────────────────────────
docs = [
    {"text": "Acme Corp contract: 30-day termination clause requires written notice.", "client": "Acme", "year": 2023, "type": "contract"},
    {"text": "BetaCo invoice #4521 for consulting services, Q1 2023.", "client": "BetaCo", "year": 2023, "type": "invoice"},
    {"text": "Acme Corp NDA signed, confidentiality period 5 years.", "client": "Acme", "year": 2022, "type": "nda"},
    {"text": "Acme Corp renewal: contract extended with updated payment terms.", "client": "Acme", "year": 2023, "type": "contract"},
    {"text": "GammaCo termination clause: 60-day notice period, penalty fees apply.", "client": "GammaCo", "year": 2023, "type": "contract"},
]

# ── 2. Index into ChromaDB (vector search) ─────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.Client()
collection = db.create_collection("legal_docs", metadata={"hnsw:space": "cosine"})

collection.add(
    documents=[d["text"] for d in docs],
    embeddings=embedder.encode([d["text"] for d in docs]).tolist(),
    metadatas=[{"client": d["client"], "year": d["year"], "type": d["type"]} for d in docs],
    ids=[f"doc_{i}" for i in range(len(docs))]
)

# ── 3. BM25 index (keyword search) ────────────────────────────────
tokenized = [d["text"].lower().split() for d in docs]
bm25 = BM25Okapi(tokenized)

# ── 4. RRF fusion ─────────────────────────────────────────────────
def reciprocal_rank_fusion(bm25_ids, vector_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

# ── 5. Hybrid search with metadata pre-filter ─────────────────────
def hybrid_search(query: str, filters: dict = None, top_k: int = 3):
    # Build ChromaDB-compatible where clause
    if filters and len(filters) > 1:
        where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
    elif filters and len(filters) == 1:
        k, v = next(iter(filters.items()))
        where = {k: {"$eq": v}}
    else:
        where = None

    vec_results = collection.query(
        query_embeddings=embedder.encode([query]).tolist(),
        n_results=top_k,
        where=where,
        include=["documents"]
    )
    vector_ids = [r.split("_")[1] for r in vec_results["ids"][0]]

    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ids = [str(i) for i in sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)]

    if filters:
        bm25_ids = [i for i in bm25_ids if all(
            str(docs[int(i)].get(k)) == str(v) for k, v in filters.items()
        )]

    fused = reciprocal_rank_fusion(bm25_ids[:top_k], vector_ids)[:top_k]
    return [docs[int(i)]["text"] for i in fused]

# ── 6. RAG answer using Groq ───────────────────────────────────────
def rag_answer(query: str, filters: dict = None):
    chunks = hybrid_search(query, filters)
    context = "\n".join(f"- {c}" for c in chunks)
    prompt = f"""Answer using only this context. If not found, say so.

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

# ── 7. Test queries ────────────────────────────────────────────────
print("=== Test 1: Filtered query (Acme, 2023) ===")
print(rag_answer(
    query="What are the termination clause terms?",
    filters={"client": "Acme", "year": 2023}
))

print("\n=== Test 2: No filter (all docs) ===")
print(rag_answer(
    query="Which client has a 60-day notice period?"
))