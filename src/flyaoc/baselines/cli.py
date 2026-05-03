"""Command-line entry point for FlyAOC baseline reruns."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flyaoc.baselines.data import load_gene_inputs
from flyaoc.baselines.normalize import normalize_result
from flyaoc.baselines.types import BaselineRunConfig, RawBaselineResult
from flyaoc.evaluation.io import read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FlyAOC paper baselines.")
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["memorization", "pipeline", "single_agent", "multi_agent"],
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "openai_compatible", "bedrock_proxy"],
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--paper-budget", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gene-id")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BaselineRunConfig(
        baseline=args.baseline,
        provider=args.provider,
        model=args.model,
        paper_budget=args.paper_budget,
        output_dir=args.output_dir,
        limit=args.limit,
        gene_id=args.gene_id,
        workers=args.workers,
        resume=args.resume,
        max_turns=args.max_turns,
        max_cost=args.max_cost,
        verbose=args.verbose,
    )
    run_baseline(config)


def run_baseline(config: BaselineRunConfig) -> dict[str, Any]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = config.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    genes = load_gene_inputs(gene_id=config.gene_id, limit=config.limit)
    if config.workers == 1:
        raw_results = [_run_or_resume(config, gene, raw_dir) for gene in genes]
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            raw_results = list(
                executor.map(lambda gene: _run_or_resume(config, gene, raw_dir), genes)
            )

    prediction_rows = [normalize_result(result) for result in raw_results]
    write_jsonl(config.output_dir / "predictions.jsonl", prediction_rows)

    summary = {
        "baseline": config.baseline,
        "provider": config.provider,
        "model": config.model,
        "paper_budget": config.paper_budget,
        "n_genes": len(raw_results),
        "run_status_counts": dict(Counter(row["run_status"] for row in prediction_rows)),
        "raw_dir": str(raw_dir),
        "predictions_file": str(config.output_dir / "predictions.jsonl"),
    }
    (config.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _run_or_resume(
    config: BaselineRunConfig,
    gene,
    raw_dir: Path,
) -> RawBaselineResult | dict[str, Any]:
    raw_path = raw_dir / f"{gene.gene_id}.json"
    if config.resume and raw_path.exists():
        return json.loads(raw_path.read_text())
    try:
        from flyaoc.baselines.runners import BaselineRunner
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Install rerun dependencies with `uv sync --extra baselines` before running baselines."
        ) from exc

    runner = BaselineRunner(config)
    result = runner.run_gene(gene)
    raw_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    if config.verbose:
        print(f"{gene.gene_symbol}: {result.run_status}")
    return result


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


if __name__ == "__main__":
    main()
