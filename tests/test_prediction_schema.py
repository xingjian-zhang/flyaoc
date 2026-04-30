import json
from pathlib import Path

from flyaoc.evaluation.io import read_jsonl


def test_smoke_prediction_schema() -> None:
    rows = read_jsonl("artifacts/predictions/smoke/smoke_predictions.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["gene_id"].startswith("FBgn")
    assert row["task1_function_predictions"]
    assert row["task2_expression_predictions"]
    assert row["task3_synonym_predictions"]["symbol_synonyms"]


def test_manifest_prediction_files_are_normalized() -> None:
    manifest = json.loads(Path("artifacts/predictions/manifest.json").read_text())
    required_keys = {
        "gene_id",
        "gene_symbol",
        "baseline",
        "provider",
        "model",
        "paper_budget",
        "run_status",
        "task1_function_predictions",
        "task2_expression_predictions",
        "task3_synonym_predictions",
    }
    forbidden_keys = {"raw_response", "trace", "usage", "error", "evidence", "cost_usd"}

    for artifact in manifest["prediction_files"]:
        rows = read_jsonl(Path("artifacts/predictions") / artifact["path"])
        assert len(rows) == artifact["n_genes"]
        assert {row["run_status"] for row in rows} <= {"ok", "failed", "empty_output"}
        for row in rows:
            assert required_keys <= set(row)
            assert forbidden_keys.isdisjoint(row)
            for item in row["task1_function_predictions"]:
                assert "evidence" not in item
            for item in row["task2_expression_predictions"]:
                assert "evidence" not in item
