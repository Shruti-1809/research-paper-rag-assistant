"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


# --- Paper Schemas ------------------------------------------------------------

class PaperUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    page_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    is_processed: bool
    chunk_count: int
    created_at: datetime


class PaperListResponse(BaseModel):
    papers: List[PaperUploadResponse]
    total: int


class PaperProcessResponse(BaseModel):
    paper_id: str
    chunk_count: int
    message: str


# --- Chunk / Citation Schemas -------------------------------------------------

class Citation(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: Optional[str] = None
    section: Optional[str] = None
    page_number: Optional[int] = None
    text_snippet: str
    relevance_score: float


# --- Chat Schemas -------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    paper_id: str
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    citations: List[Citation]
    confidence_score: Optional[float] = None
    retrieved_chunk_ids: List[str]


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    paper_id: Optional[str] = None
    title: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    message_count: int


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    citations: Optional[List[Citation]] = None
    confidence_score: Optional[float] = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    session: ChatSessionResponse
    messages: List[ChatMessageResponse]


# --- Summary Schemas ----------------------------------------------------------

class SummaryRequest(BaseModel):
    paper_id: str
    include_abstract: bool = True
    include_methodology: bool = True
    include_contributions: bool = True
    include_results: bool = True


class SummaryResponse(BaseModel):
    paper_id: str
    paper_title: Optional[str] = None
    summary: str
    sections: Dict[str, str]


# --- Comparison Schemas -------------------------------------------------------

class ComparisonRequest(BaseModel):
    paper_ids: List[str] = Field(..., min_length=2, max_length=5)
    aspect: str = Field(
        default="methodology",
        description="What to compare: methodology, results, contributions, dataset"
    )


class ComparisonResponse(BaseModel):
    paper_ids: List[str]
    aspect: str
    comparison: str
    per_paper: Dict[str, str]


# --- Health / Status ----------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    vector_store: str