"""Evaluation dataset loading and validation.

The dataset uses stable *logical* chunk identifiers (e.g. ``C1``) for
relevance judgments. These are mapped to real stored chunk ids at runtime by
the runner, so the dataset itself is independent of any database-generated ids
and can be extended by adding documents/queries to the JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from retrieval_stack.evaluation import EvaluationDataset, EvaluationQuery


class LogicalChunk(BaseModel):
    """A chunk referenced by a stable logical id."""

    logical_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("logical_id", "content")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("value must be a non-empty string")
        return v


class EvalDocument(BaseModel):
    """A synthetic support document with its logical chunks."""

    id: str
    filename: str
    content_type: str
    content: str
    chunks: list[LogicalChunk] = Field(default_factory=list)

    @field_validator("id", "filename", "content_type", "content")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("value must be a non-empty string")
        return v


class RetrievalEvalDataset(BaseModel):
    """Synchronous offline retrieval evaluation dataset."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    tenant_id: str
    documents: list[EvalDocument] = Field(default_factory=list)
    queries: list[EvaluationQuery] = Field(default_factory=list)
    k_values: list[int] = Field(default_factory=lambda: [3, 5, 10])

    @field_validator("tenant_id", "name")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("value must be a non-empty string")
        return v

    @field_validator("k_values")
    @classmethod
    def valid_k_values(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("k_values must be non-empty")
        for k in v:
            if not isinstance(k, int) or k <= 0:
                raise ValueError("k values must be positive integers")
        return v

    @model_validator(mode="after")
    def unique_logical_ids(self) -> "RetrievalEvalDataset":
        seen: set[str] = set()
        for doc in self.documents:
            for chunk in doc.chunks:
                if chunk.logical_id in seen:
                    raise ValueError(f"duplicate logical chunk id: {chunk.logical_id}")
                seen.add(chunk.logical_id)
        return self

    @model_validator(mode="after")
    def unique_query_ids(self) -> "RetrievalEvalDataset":
        ids = [q.query_id for q in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate query ids")
        return self

    def to_retrieval_stack_dataset(self) -> EvaluationDataset:
        """Return the queries as a Retrieval Stack ``EvaluationDataset``."""
        return EvaluationDataset(queries=list(self.queries))

    def logical_chunk_ids(self) -> list[str]:
        return [c.logical_id for d in self.documents for c in d.chunks]


def load_dataset(path: str | Path) -> RetrievalEvalDataset:
    """Load and validate a retrieval evaluation dataset from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RetrievalEvalDataset.model_validate(data)
