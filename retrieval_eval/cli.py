"""Command line interface for the offline retrieval evaluation runner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .dataset import load_dataset
from .report import write_canonical_json, write_json, write_markdown
from .runner import DEFAULT_METHODS, run_retrieval_evaluation


def _resolve_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url.strip()
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "DATABASE_URL":
                return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "DATABASE_URL is not set. Set the DATABASE_URL environment variable "
        "or pass --database-url."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retrieval_eval",
        description="Offline retrieval evaluation for KnowledgeService.",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "data" / "support_eval_dataset.json"),
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL connection string (defaults to DATABASE_URL env).",
    )
    parser.add_argument(
        "--tenant-id",
        default=None,
        help="Tenant id to evaluate against (defaults to the dataset's tenant_id).",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=f"Comma-separated methods (default: {','.join(DEFAULT_METHODS)}). "
        "Use 'bm25' to run without embedding/reranking models.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "eval_output"),
        help="Directory for the JSON and Markdown reports.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="K value for the Markdown comparison table (default: first k).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    database_url = args.database_url or _resolve_database_url()
    os.environ.setdefault("DATABASE_URL", database_url)

    from app.database import get_database

    db = get_database(database_url)
    db.initialize()

    dataset = load_dataset(args.dataset)
    tenant_id = args.tenant_id or dataset.tenant_id
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    report = run_retrieval_evaluation(
        db=db,
        tenant_id=tenant_id,
        dataset=dataset,
        methods=methods,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "retrieval_eval_report.json"
    canonical_path = output_dir / "retrieval_eval_canonical.json"
    md_path = output_dir / "retrieval_eval_report.md"

    write_json(report, json_path)
    write_canonical_json(report, canonical_path)
    write_markdown(report, md_path, k=args.k)

    print(f"Queries: {len(dataset.queries)} | K values: {dataset.k_values}")
    print(f"Methods run: {', '.join(report.methods_run) or 'none'}")
    if report.unavailable:
        print(f"Methods unavailable: {', '.join(report.unavailable)}")
        for method, reason in report.unavailable.items():
            print(f"  - {method}: {reason}")
    print(report.comparison_table(args.k))
    print(f"Wrote {json_path}")
    print(f"Wrote {canonical_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
