"""Offline retrieval evaluation for the KnowledgeService.

This package is a reproducible, offline evaluation tool. It does not touch
production API endpoints, does not add background jobs, and does not change
retrieval algorithms or tenant isolation. It reuses the Retrieval Stack
evaluation utilities (precision/recall/MRR/NDCG) and the Eval Harness metric
aggregation/reporting primitives.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the sibling Retrieval-Stack and Eval Harness packages are importable
# when this tool is used outside of pytest (conftest.py adjusts ``sys.path``
# under pytest). Insert early so submodule imports below resolve.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("Retrieval-Stack", "eval_harness"):
    _path = _BACKEND_ROOT / _sub
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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
