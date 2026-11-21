"""Knowledge base API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    byte_size: int
    checksum: str
    status: Literal["pending", "processing", "completed", "failed"]
    chunks_created: int
    created_at: str


class DocumentListResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    byte_size: int
    checksum: str
    status: Literal["pending", "processing", "completed", "failed"]
    created_at: str
    updated_at: str
    completed_at: str | None
    error_message: str | None


class ChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    start_char: int
    end_char: int
    metadata: dict
    created_at: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResultResponse(BaseModel):
    chunk: ChunkResponse
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]
    query: str
    top_k: int