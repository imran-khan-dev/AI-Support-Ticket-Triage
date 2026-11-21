"""Report assembly for offline retrieval evaluation.

Produces a JSON report and a Markdown comparison table. Uses the Retrieval
Stack evaluation export utilities as the canonical retrieval report and the
Eval Harness metric aggregation (`MetricScore`, `aggregate_scores`) for an
additional cross-configuration summary, so both reusable components are
exercised without duplicating retrieval or metric logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from eval_harness import MetricScore, aggregate_scores
from eval_harness.metrics.models import MetricType
from retrieval_stack.evaluation import comparison_table, export_report_json

from .runner import RetrievalEvalReport


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonify(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonify(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _eval_harness_summary(report: RetrievalEvalReport) -> dict[str, Any]:
    """Aggregate per-method metrics using Eval Harness scoring primitives."""
    scores: list[MetricScore] = []
    for agg in report.report.aggregate:
        method = agg.method
        for k in report.k_values:
            scores.append(
                MetricScore(
                    metric_type=MetricType.PRECISION,
                    name=f"{method}:precision@{k}",
                    value=agg.avg_precision_at_k.get(k, 0.0),
                    details={"method": method, "k": k},
                )
            )
            scores.append(
                MetricScore(
                    metric_type=MetricType.RECALL,
                    name=f"{method}:recall@{k}",
                    value=agg.avg_recall_at_k.get(k, 0.0),
                    details={"method": method, "k": k},
                )
            )
            scores.append(
                MetricScore(
                    metric_type=MetricType.CUSTOM,
                    name=f"{method}:ndcg@{k}",
                    value=agg.avg_ndcg_at_k.get(k, 0.0),
                    details={"method": method, "k": k},
                )
            )
        scores.append(
            MetricScore(
                metric_type=MetricType.CUSTOM,
                name=f"{method}:mrr",
                value=agg.mrr,
                details={"method": method},
            )
        )
    return _jsonify(aggregate_scores(scores))


def build_report_dict(report: RetrievalEvalReport) -> dict[str, Any]:
    data = report.to_dict()
    try:
        data["eval_harness_summary"] = _eval_harness_summary(report)
    except Exception as exc:  # noqa: BLE001 - eval harness summary is additive
        data["eval_harness_summary"] = {"error": f"{type(exc).__name__}: {exc}"}
    return data


def write_json(report: RetrievalEvalReport, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(build_report_dict(report), indent=2), encoding="utf-8"
    )


def write_canonical_json(report: RetrievalEvalReport, path: str | Path) -> None:
    export_report_json(report.report, path)


def write_markdown(report: RetrievalEvalReport, path: str | Path, k: int | None = None) -> None:
    lines: list[str] = []
    lines.append(f"# Retrieval Evaluation Report: {report.dataset.name}")
    lines.append("")
    lines.append(f"- **Dataset version:** {report.dataset.version}")
    lines.append(f"- **Tenant:** {report.dataset.tenant_id}")
    lines.append(f"- **Queries:** {len(report.dataset.queries)}")
    lines.append(f"- **K values:** {report.k_values}")
    lines.append(f"- **Methods run:** {', '.join(report.methods_run) or 'none'}")
    if report.unavailable:
        lines.append(f"- **Methods unavailable:** {', '.join(report.unavailable)}")
        for method, reason in report.unavailable.items():
            lines.append(f"  - {method}: {reason}")
    lines.append("")
    lines.append("## Comparison table")
    lines.append("")
    lines.append(report.comparison_table(k))
    lines.append("")
    lines.append("## Per-method aggregates")
    lines.append("")
    for agg in report.report.aggregate:
        lines.append(f"### {agg.method} ({agg.num_queries} queries)")
        for k in report.k_values:
            lines.append(
                f"  - Precision@{k}: {agg.avg_precision_at_k.get(k, 0.0):.4f} | "
                f"Recall@{k}: {agg.avg_recall_at_k.get(k, 0.0):.4f} | "
                f"NDCG@{k}: {agg.avg_ndcg_at_k.get(k, 0.0):.4f}"
            )
        lines.append(f"  - MRR: {agg.mrr:.4f}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
