from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import QueryRequest, QueryResponse
from app.rag import build_pipeline, hybrid_search, generate_answer

pipeline = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline["rag"] = build_pipeline()
    yield
    pipeline.clear()

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": "rag" in pipeline}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        chunks = hybrid_search(
            pipeline["rag"],
            query=req.question,
            top_k=req.top_k
        )

        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        answer = generate_answer(req.question, chunks)

        return QueryResponse(
            answer=answer,
            sources=chunks,
            model="llama-3.3-70b"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))