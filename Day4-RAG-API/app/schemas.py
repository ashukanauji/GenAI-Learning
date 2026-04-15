from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    filters: Optional[dict] = None
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str = "llama-3.3-70b"