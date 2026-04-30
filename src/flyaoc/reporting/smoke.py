"""Smoke-test entry point."""

from __future__ import annotations

import json
from pathlib import Path

from flyaoc.evaluation import evaluate_prediction_file


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    prediction_file = root / "artifacts/predictions/smoke/smoke_predictions.jsonl"
    result = evaluate_prediction_file(prediction_file, benchmark_limit=5)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
