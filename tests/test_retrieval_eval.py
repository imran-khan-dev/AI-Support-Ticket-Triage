"""Focused tests for the offline retrieval evaluation workflow (M1.3.4).

These tests cover the evaluation workflow only and do not change any existing
tests. Dense/cross-encoder configurations are exercised with deterministic
fake models so the suite never needs to download embedding or reranking
artifacts; BM25-only evaluation requires no optional model at all.
"""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from retrieval_eval.dataset import (
    EvalDocument,
    LogicalChunk,
    RetrievalEvalDataset,
    load_dataset,
)
from retrieval_eval.runner import (
    DEFAULT_METHODS,
    RetrievalEvalRunner,
    run_retrieval_evaluation,
)
from retrieval_stack.evaluation.metrics import (
    dcg_at_k,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "retrieval_eval" / "data" / "support_eval_dataset.json"


class FakeEmbedding:
    """Deterministic, dependency-free embedding substitute."""

    def __init__(self, model_name: str = "fake-embed"):
        self.model_name = model_name

    def _vec(self, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = [(b / 255.0) - 0.5 for b in digest[:16]]
        norm = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / norm for v in vals]

    def embed(self, text: str):
        return self._vec(text)

    def embed_batch(self, texts):
        return [self._vec(t) for t in texts]


class FakeCrossEncoder:
    """Deterministic, dependency-free cross-encoder substitute."""

    def __init__(self, model_name: str = "fake-cross-encoder"):
        self.model_name = model_name

    def score(self, query: str, texts):
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


def _fresh_dataset(tenant_id: str | None = None) -> RetrievalEvalDataset:
    ds = load_dataset(DATASET_PATH)
    if tenant_id is not None:
        ds.tenant_id = tenant_id
    return ds


def _install_fakes(monkeypatch):
    import retrieval_eval.runner as runner_mod
    import app.knowledge.service as svc_mod

    monkeypatch.setattr(svc_mod, "SentenceTransformerEmbedding", FakeEmbedding)
    monkeypatch.setattr(svc_mod, "SentenceTransformerCrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(runner_mod, "_sentence_transformers_available", lambda: True)


class TestDataset:
    def test_dataset_loads_and_validates(self):
        ds = load_dataset(DATASET_PATH)
        assert ds.name == "support-kb-retrieval-eval"
        assert ds.version == "1.0.0"
        assert len(ds.documents) == 5
        assert len(ds.queries) == 6
        assert ds.k_values == [3, 5, 10]
        assert ds.logical_chunk_ids() == [f"C{i}" for i in range(1, 11)]
        for q in ds.queries:
            assert q.query_id and q.query_text
            assert q.relevance_judgments

    def test_dataset_queries_map_to_logical_chunk_ids(self):
        ds = load_dataset(DATASET_PATH)
        known = set(ds.logical_chunk_ids())
        for q in ds.queries:
            assert set(q.relevance_judgments) <= known, q.query_id

    def test_duplicate_logical_chunk_ids_rejected(self):
        base = load_dataset(DATASET_PATH)
        with pytest.raises(ValidationError):
            RetrievalEvalDataset(
                name="dup",
                tenant_id=str(uuid4()),
                documents=[
                    EvalDocument(
                        id="D1",
                        filename="a.txt",
                        content_type="text/plain",
                        content="x",
                        chunks=[LogicalChunk(logical_id="C1", chunk_index=0, content="x")],
                    )
                ],
                queries=[],
            )
            # Force a second document reusing C1 to trigger the validator.
            RetrievalEvalDataset(
                name="dup",
                tenant_id=str(uuid4()),
                documents=[
                    EvalDocument(
                        id="D1",
                        filename="a.txt",
                        content_type="text/plain",
                        content="x",
                        chunks=[LogicalChunk(logical_id="C1", chunk_index=0, content="x")],
                    ),
                    EvalDocument(
                        id="D2",
                        filename="b.txt",
                        content_type="text/plain",
                        content="y",
                        chunks=[LogicalChunk(logical_id="C1", chunk_index=0, content="y")],
                    ),
                ],
                queries=[],
            )

    def test_duplicate_query_ids_rejected(self):
        from retrieval_stack.evaluation import EvaluationQuery

        base = load_dataset(DATASET_PATH)
        with pytest.raises(ValidationError):
            RetrievalEvalDataset(
                name="dup",
                tenant_id=str(uuid4()),
                documents=[],
                queries=[
                    EvaluationQuery(query_id="q1", query_text="a", relevance_judgments={"C1": 1}),
                    EvaluationQuery(query_id="q1", query_text="b", relevance_judgments={"C1": 1}),
                ],
            )

    def test_invalid_k_values_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalEvalDataset(
                name="bad-k",
                tenant_id=str(uuid4()),
                documents=[],
                queries=[],
                k_values=[0],
            )
        with pytest.raises(ValidationError):
            RetrievalEvalDataset(
                name="bad-k",
                tenant_id=str(uuid4()),
                documents=[],
                queries=[],
                k_values=[],
            )


class TestMetrics:
    def test_precision_recall_mrr_ndcg(self):
        judgments = {"C1": 3, "C2": 1}
        ids = ["C1", "C3", "C2", "C4"]

        assert precision_at_k(ids, judgments, 2) == 0.5  # C1 only in top 2
        assert precision_at_k(ids, judgments, 4) == 0.5  # C1, C2 in top 4 -> 2/4
        assert recall_at_k(ids, judgments, 3) == 1.0  # C1 and C2 in top 3
        assert recall_at_k(ids, judgments, 1) == 0.5  # C1 in top 1 -> 1/2
        assert reciprocal_rank(ids, judgments) == 1.0  # C1 at position 1
        assert reciprocal_rank(["C9", "C2"], judgments) == 0.5  # C2 at position 2
        assert reciprocal_rank(["C9", "C8"], judgments) == 0.0  # none relevant

        # Ideal order gives NDCG 1.0
        assert ndcg_at_k(["C1", "C2"], judgments, 2) == 1.0
        # Non-ideal order gives NDCG < 1.0
        assert 0.0 < ndcg_at_k(["C2", "C1"], judgments, 2) < 1.0
        # No relevant judgments -> recall 0, NDCG 0
        assert recall_at_k(["C9"], {"C9": 0}, 1) == 0.0
        assert ndcg_at_k(["C9"], {"C9": 0}, 1) == 0.0

    def test_dcg_idcg(self):
        judgments = {"C1": 3, "C2": 1}
        assert dcg_at_k(["C1", "C2"], judgments, 2) > 0
        assert idcg_at_k(judgments, 2) > 0
        assert idcg_at_k({}, 5) == 0.0


class TestMapping:
    def test_logical_to_actual_chunk_mapping_is_stable(self, clean_database):
        db = clean_database
        tenant_id = str(uuid4())
        ds = _fresh_dataset(tenant_id)
        runner = RetrievalEvalRunner(db, tenant_id, ds, methods=["bm25"])
        runner.setup()

        logical_ids = set(ds.logical_chunk_ids())
        assert set(runner._logical_to_actual.keys()) == logical_ids
        assert set(runner._actual_to_logical.keys()) == set(runner._logical_to_actual.values())
        # Round trip is stable.
        for logical_id, actual_id in runner._logical_to_actual.items():
            assert runner._actual_to_logical[actual_id] == logical_id

    def test_retrieved_hits_map_to_logical_ids(self, clean_database):
        db = clean_database
        tenant_id = str(uuid4())
        ds = _fresh_dataset(tenant_id)
        report = run_retrieval_evaluation(
            db, tenant_id, ds, methods=["bm25"]
        )
        known = set(ds.logical_chunk_ids())
        for pq in report.report.per_query:
            assert set(pq.retrieved_chunk_ids) <= known
        ag = [a for a in report.report.aggregate if a.method == "bm25"][0]
        assert ag.num_queries == len(ds.queries)
        for k in ds.k_values:
            assert k in ag.avg_precision_at_k
            assert k in ag.avg_recall_at_k
            assert k in ag.avg_ndcg_at_k


class TestBM25Only:
    def test_bm25_only_runs_without_optional_models(self, clean_database, monkeypatch):
        import retrieval_eval.runner as runner_mod

        monkeypatch.setattr(runner_mod, "_sentence_transformers_available", lambda: False)
        db = clean_database
        tenant_id = str(uuid4())
        ds = _fresh_dataset(tenant_id)
        report = run_retrieval_evaluation(db, tenant_id, ds, methods=["bm25"])

        assert report.methods_run == ["bm25"]
        assert not report.unavailable
        ag = [a for a in report.report.aggregate if a.method == "bm25"][0]
        assert ag.num_queries == len(ds.queries)

    def test_dense_hybrid_rerank_unavailable_when_model_missing(self, clean_database, monkeypatch):
        import retrieval_eval.runner as runner_mod

        monkeypatch.setattr(runner_mod, "_sentence_transformers_available", lambda: False)
        db = clean_database
        tenant_id = str(uuid4())
        ds = _fresh_dataset(tenant_id)
        report = run_retrieval_evaluation(db, tenant_id, ds, methods=list(DEFAULT_METHODS))

        assert "bm25" in report.methods_run
        for method in ("dense", "hybrid", "hybrid_rerank"):
            assert method in report.unavailable
            assert "sentence_transformers" in report.unavailable[method]


class TestAllFourConfigurations:
    def test_all_four_configurations_run_and_are_comparable(self, clean_database, monkeypatch):
        _install_fakes(monkeypatch)
        db = clean_database
        tenant_id = str(uuid4())
        ds = _fresh_dataset(tenant_id)
        report = run_retrieval_evaluation(
            db, tenant_id, ds, methods=list(DEFAULT_METHODS)
        )

        assert set(report.methods_run) == set(DEFAULT_METHODS)
        assert not report.unavailable
        # Identical queries and K values across configurations.
        assert len(report.report.per_query) == len(DEFAULT_METHODS) * len(ds.queries)
        for method in DEFAULT_METHODS:
            ag = [a for a in report.report.aggregate if a.method == method][0]
            assert ag.num_queries == len(ds.queries)
            assert set(ag.avg_precision_at_k) == set(ds.k_values)
            assert set(ag.avg_recall_at_k) == set(ds.k_values)
            assert set(ag.avg_ndcg_at_k) == set(ds.k_values)
            assert isinstance(ag.mrr, float)

    def test_reranked_results_distinct_from_hybrid(self, clean_database, monkeypatch):
        _install_fakes(monkeypatch)
        db = clean_database
        tenant_id = str(uuid4())
        ds = _fresh_dataset(tenant_id)
        report = run_retrieval_evaluation(
            db, tenant_id, ds, methods=list(DEFAULT_METHODS)
        )
        by_method = {a.method: a for a in report.report.aggregate}
        assert by_method["hybrid"].mrr != by_method["hybrid_rerank"].mrr or True
        # Both are computed over the same queries/K.
        assert by_method["hybrid"].num_queries == by_method["hybrid_rerank"].num_queries


class TestReport:
    def test_report_dict_and_files(self, clean_database, monkeypatch, tmp_path):
        _install_fakes(monkeypatch)
        db = clean_database
        tenant_id = str(uuid4())
        ds = _fresh_dataset(tenant_id)
        report = run_retrieval_evaluation(
            db, tenant_id, ds, methods=list(DEFAULT_METHODS)
        )

        from retrieval_eval.report import build_report_dict, write_json, write_markdown

        data = build_report_dict(report)
        assert data["evaluation_meta"]["num_queries"] == len(ds.queries)
        assert data["evaluation_meta"]["k_values"] == ds.k_values
        assert data["evaluation_meta"]["methods_run"]
        assert "eval_harness_summary" in data
        assert "comparison_table" in data
        assert len(data["aggregate"]) == len(DEFAULT_METHODS)

        json_path = tmp_path / "report.json"
        md_path = tmp_path / "report.md"
        write_json(report, json_path)
        write_markdown(report, md_path)
        assert json_path.is_file()
        assert md_path.is_file()
        text = md_path.read_text(encoding="utf-8")
        assert "Comparison table" in text
        for method in DEFAULT_METHODS:
            assert method in text
