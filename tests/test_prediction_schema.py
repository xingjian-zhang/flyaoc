from flyaoc.evaluation.io import read_jsonl


def test_smoke_prediction_schema() -> None:
    rows = read_jsonl("artifacts/predictions/smoke/smoke_predictions.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["gene_id"].startswith("FBgn")
    assert row["task1_function_predictions"]
    assert row["task2_expression_predictions"]
    assert row["task3_synonym_predictions"]["symbol_synonyms"]
