"""Offline retrieval evaluation runner.

Runs every query of an evaluation dataset against each retrieval configuration
exposed by :class:`app.knowledge.service.KnowledgeService`, computes
Precision@K, Recall@K, MRR and NDCG@K with the Retrieval Stack evaluation
utilities, and assembles a comparable report across configurations.

The runner is intentionally offline: it never calls a production endpoint,
never starts a background job, and never changes retrieval algorithms or
tenant isolation. Embedding and reranking models are loaded lazily and only
when a configuration that needs them is actually run, so a BM25-only
evaluation never downloads a model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from app.knowledge.repository import create_chunks, create_document
from app.knowledge.service import KnowledgeService
from retrieval_stack.evaluation import (
    EvaluationReport,
    evaluate_search_fn,
)

from .dataset import RetrievalEvalDataset


class _EvalChunk:
    __slots__ = ("id", "content")

    def __init__(self, chunk_id: str, content: str):
        self.id = chunk_id
        self.content = content


class _EvalHit:
    """Search hit carrying a stable logical chunk id for metric computation."""

    __slots__ = ("chunk_id", "chunk")

    def __init__(self, logical_id: str, content: str):
        self.chunk_id = logical_id
        self.chunk = _EvalChunk(logical_id, content)


_METHODS: dict[str, dict[str, Any]] = {
    "bm25": {
        "label": "BM25",
        "needs_embedder": False,
        "needs_reranker": False,
    },
    "dense": {
        "label": "Dense retrieval",
        "needs_embedder": True,
        "needs_reranker": False,
    },
    "hybrid": {
        "label": "Hybrid (BM25 + Dense, RRF)",
        "needs_embedder": True,
        "needs_reranker": False,
    },
    "hybrid_rerank": {
        "label": "Hybrid + cross-encoder reranking",
        "needs_embedder": True,
        "needs_reranker": True,
    },
}

DEFAULT_METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class RetrievalEvalReport:
    """Result of an offline retrieval evaluation run."""

    report: EvaluationReport
    dataset: RetrievalEvalDataset
    unavailable: dict[str, str] = field(default_factory=dict)
    k_values: list[int] = field(default_factory=list)

    @property
    def methods_run(self) -> list[str]:
        return [a.method for a in self.report.aggregate]

    @property
    def num_queries(self) -> int:
        return self.dataset.queries and len(self.dataset.queries) or 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_meta": {
                "dataset": self.dataset.name,
                "version": self.dataset.version,
                "tenant_id": self.dataset.tenant_id,
                "num_queries": len(self.dataset.queries),
                "k_values": self.dataset.k_values,
                "methods_requested": list(_METHODS.keys()),
                "methods_run": self.methods_run,
                "methods_unavailable": list(self.unavailable.keys()),
            },
            "unavailable_methods": self.unavailable,
            "comparison_table": self.comparison_table(),
            "aggregate": [a.model_dump() for a in self.report.aggregate],
            "per_query": [p.model_dump() for p in self.report.per_query],
        }

    def comparison_table(self, k: int | None = None) -> str:
        from retrieval_stack.evaluation import comparison_table

        return comparison_table(self.report, k)


class RetrievalEvalRunner:
    def __init__(
        self,
        db,
        tenant_id: str,
        dataset: RetrievalEvalDataset,
        methods: list[str] | None = None,
        enable_reranking: bool = False,
    ):
        self._db = db
        self._tenant_id = tenant_id
        self.dataset = dataset
        self.methods = list(methods) if methods else list(DEFAULT_METHODS)
        self._enable_reranking = enable_reranking or "hybrid_rerank" in self.methods
        self._service: KnowledgeService | None = None
        self._logical_to_actual: dict[str, str] = {}
        self._actual_to_logical: dict[str, str] = {}
        self.unavailable: dict[str, str] = {}
        self._setup_done = False

    # -- setup ---------------------------------------------------------

    def _ensure_tenant(self) -> None:
        from app.database.repository import _canonical_uuid

        canonical = _canonical_uuid(self._tenant_id)
        if canonical is None:
            raise ValueError(f"tenant_id is not a valid UUID: {self._tenant_id}")
        with self._db.write_transaction() as conn:
            conn.execute(
                "INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE) "
                "ON CONFLICT (id) DO NOTHING",
                (canonical, f"eval-{self._tenant_id[:8]}"),
            )
            # Start from a clean slate for this eval tenant so repeated runs are
            # reproducible and never accumulate duplicate chunks.
            conn.execute(
                "DELETE FROM knowledge_documents WHERE tenant_id = %s",
                (canonical,),
            )

    def _ingest_documents(self) -> None:
        for doc in self.dataset.documents:
            document = create_document(
                self._db,
                tenant_id=self._tenant_id,
                filename=doc.filename,
                content_type=doc.content_type,
                byte_size=len(doc.content.encode("utf-8")),
                checksum=hashlib.sha256(doc.content.encode("utf-8")).hexdigest(),
            )
            chunk_dicts = [
                {
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "token_count": max(1, len(c.content.split())),
                    "start_char": 0,
                    "end_char": len(c.content),
                    "metadata": c.metadata,
                }
                for c in doc.chunks
            ]
            stored = create_chunks(
                self._db,
                tenant_id=self._tenant_id,
                document_id=document.id,
                chunks=chunk_dicts,
            )
            if len(stored) != len(doc.chunks):
                raise RuntimeError(
                    f"expected {len(doc.chunks)} chunks for {doc.filename}, stored {len(stored)}"
                )
            for logical_chunk, stored_chunk in zip(doc.chunks, stored):
                self._logical_to_actual[logical_chunk.logical_id] = stored_chunk.id
                self._actual_to_logical[stored_chunk.id] = logical_chunk.logical_id

    def setup(self) -> None:
        if self._setup_done:
            return
        self._ensure_tenant()
        self._ingest_documents()
        self._service = KnowledgeService(
            self._db, enable_reranking=self._enable_reranking
        )
        self._service.invalidate_index(self._tenant_id)
        self._setup_done = True

    @property
    def service(self) -> KnowledgeService:
        if self._service is None:
            self.setup()
        return self._service

    # -- search functions ----------------------------------------------

    def _embedder_required(self, method: str) -> bool:
        return _METHODS[method]["needs_embedder"]

    def _make_search_fn(self, method: str, top_k: int):
        service = self.service
        tenant_id = self._tenant_id
        actual_to_logical = self._actual_to_logical

        def bm25_fn(query: str, k: int):
            results = service.search(tenant_id, query, top_k=k, rerank=False)
            return self._wrap(results, actual_to_logical)

        def dense_fn(query: str, k: int):
            results = service.search_dense(tenant_id, query, top_k=k, rerank=False)
            return self._wrap(results, actual_to_logical)

        def hybrid_fn(query: str, k: int):
            results = service.search_hybrid(tenant_id, query, top_k=k, rerank=False)
            return self._wrap(results, actual_to_logical)

        def hybrid_rerank_fn(query: str, k: int):
            results = service.search_hybrid(tenant_id, query, top_k=k, rerank=True)
            return self._wrap(results, actual_to_logical)

        return {
            "bm25": bm25_fn,
            "dense": dense_fn,
            "hybrid": hybrid_fn,
            "hybrid_rerank": hybrid_rerank_fn,
        }[method]

    @staticmethod
    def _wrap(results, actual_to_logical: dict[str, str]):
        wrapped = []
        for result in results:
            chunk = getattr(result, "chunk", None)
            actual_id = getattr(chunk, "id", None) if chunk is not None else None
            logical_id = actual_to_logical.get(actual_id, actual_id) if actual_id else None
            if logical_id is None:
                continue
            content = chunk.content if chunk is not None else ""
            wrapped.append(_EvalHit(logical_id, content))
        return wrapped

    # -- run -----------------------------------------------------------

    def run(self) -> RetrievalEvalReport:
        self.setup()
        dataset_rs = self.dataset.to_retrieval_stack_dataset()
        k_values = list(self.dataset.k_values)

        search_fns = {}
        for method in self.methods:
            if self._embedder_required(method) and not _sentence_transformers_available():
                self.unavailable[method] = (
                    "sentence_transformers is not installed; the embedding model "
                    "cannot be loaded. Run BM25-only evaluation (--methods bm25) "
                    "without downloading embedding or reranking models."
                )
                continue
            search_fns[method] = self._make_search_fn(method, max(k_values))

        per_query = []
        aggregate = []
        for method, fn in search_fns.items():
            try:
                pq, ag = evaluate_search_fn(method, fn, dataset_rs, k_values)
                per_query.extend(pq)
                aggregate.append(ag)
            except Exception as exc:  # noqa: BLE001 - report per-method failure, keep going
                self.unavailable[method] = f"{type(exc).__name__}: {exc}"

        report = EvaluationReport(
            k_values=k_values,
            per_query=per_query,
            aggregate=aggregate,
        )
        return RetrievalEvalReport(
            report=report,
            dataset=self.dataset,
            unavailable=self.unavailable,
            k_values=k_values,
        )


def run_retrieval_evaluation(
    db,
    tenant_id: str,
    dataset: RetrievalEvalDataset,
    methods: list[str] | None = None,
    enable_reranking: bool = False,
) -> RetrievalEvalReport:
    """Convenience: run the full offline retrieval evaluation."""
    runner = RetrievalEvalRunner(
        db=db,
        tenant_id=tenant_id,
        dataset=dataset,
        methods=methods,
        enable_reranking=enable_reranking,
    )
    return runner.run()
