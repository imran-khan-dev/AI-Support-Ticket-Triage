"""Offline retrieval evaluation for the KnowledgeService.

This package is a reproducible, offline evaluation tool. It does not touch
production API endpoints, does not add background jobs, and does not change
retrieval algorithms or tenant isolation. It reuses the Retrieval Stack
evaluation utilities (precision/recall/MRR/NDCG) and the Eval Harness metric
aggregation/reporting primitives.
"""

from __future__ import annotations

from .dataset import (
    RetrievalEvalDataset,
    EvalDocument,
    LogicalChunk,
    load_dataset,
)
from .runner import RetrievalEvalRunner, RetrievalEvalReport

__all__ = [
    "RetrievalEvalDataset",
    "EvalDocument",
    "LogicalChunk",
    "load_dataset",
    "RetrievalEvalRunner",
    "RetrievalEvalReport",
]
