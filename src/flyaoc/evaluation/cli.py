"""Command-line entry point for evaluating normalized FlyAOC predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flyaoc.evaluation import evaluate_prediction_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate normalized FlyAOC prediction JSONL.")
    parser.add_argument("prediction_file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_prediction_file(args.prediction_file, benchmark_limit=args.benchmark_limit)
    payload = json.dumps(result["aggregate"], indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
