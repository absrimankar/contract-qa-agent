from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    message: str
    page_count: int
    chunk_count: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question about the contract")


class QueryResponse(BaseModel):
    question: str
    answer: str
