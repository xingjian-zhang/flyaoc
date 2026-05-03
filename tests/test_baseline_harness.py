import json
import sys
import types
from pathlib import Path

import pytest

from flyaoc.baselines import cli
from flyaoc.baselines.normalize import normalize_result
from flyaoc.baselines.types import BaselineRunConfig, GeneInput, RawBaselineResult
from flyaoc.evaluation.io import read_jsonl


def test_baseline_config_validates_supported_combinations(tmp_path: Path) -> None:
    BaselineRunConfig(
        baseline="memorization",
        provider="openai",
        model="gpt-5-mini",
        paper_budget=0,
        output_dir=tmp_path,
    ).validate()
    BaselineRunConfig(
        baseline="multi_agent",
        provider="bedrock_proxy",
        model="claude-sonnet-4",
        paper_budget=16,
        output_dir=tmp_path,
    ).validate()

    with pytest.raises(ValueError):
        BaselineRunConfig(
            baseline="memorization",
            provider="openai",
            model="gpt-5-mini",
            paper_budget=1,
            output_dir=tmp_path,
        ).validate()


def test_cli_help_parses_without_baseline_optional_imports(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        [
            "--baseline",
            "single_agent",
            "--provider",
            "openai_compatible",
            "--model",
            "model-name",
            "--paper-budget",
            "8",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert args.baseline == "single_agent"
    assert args.provider == "openai_compatible"


def test_raw_to_normalized_prediction_strips_private_fields() -> None:
    raw = RawBaselineResult(
        gene=GeneInput("FBgn0000014", "abd-A"),
        baseline="single_agent",
        provider="openai",
        model="gpt-5-mini",
        paper_budget=8,
        output={
            "task1_function": [{"go_id": "GO:0003700", "evidence": {"text": "private"}}],
            "task2_expression": [{"anatomy_id": "FBbt:00001919", "evidence": {"text": "private"}}],
            "task3_synonyms": {"fullname_synonyms": ["abdominal A"], "symbol_synonyms": ["abd-A"]},
        },
        usage={"raw_text": "private"},
    )

    row = normalize_result(raw)
    assert row["run_status"] == "ok"
    assert row["task1_function_predictions"] == [{"go_id": "GO:0003700"}]
    assert row["task2_expression_predictions"] == [{"anatomy_id": "FBbt:00001919"}]
    assert row["task3_synonym_predictions"]["symbol_synonyms"] == ["abd-A"]
    assert "usage" not in row
    assert "evidence" not in json.dumps(row)


def test_failed_run_normalizes_to_empty_predictions() -> None:
    raw = RawBaselineResult(
        gene=GeneInput("FBgn0000014", "abd-A"),
        baseline="pipeline",
        provider="openai",
        model="gpt-5-mini",
        paper_budget=4,
        error="API failed",
    )

    row = normalize_result(raw)
    assert row["run_status"] == "failed"
    assert row["task1_function_predictions"] == []
    assert row["task2_expression_predictions"] == []
    assert row["task3_synonym_predictions"] == {"fullname_synonyms": [], "symbol_synonyms": []}


def test_mocked_baseline_run_writes_predictions_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeRunner:
        def __init__(self, config: BaselineRunConfig):
            self.config = config

        def run_gene(self, gene: GeneInput) -> RawBaselineResult:
            return RawBaselineResult(
                gene=gene,
                baseline=self.config.baseline,
                provider=self.config.provider,
                model=self.config.model,
                paper_budget=self.config.paper_budget,
                output={
                    "task1_function": [{"go_id": "GO:0003700"}],
                    "task2_expression": [{"anatomy_id": "FBbt:00001919"}],
                    "task3_synonyms": {"fullname_synonyms": [], "symbol_synonyms": ["abd-A"]},
                },
            )

    fake_module = types.SimpleNamespace(BaselineRunner=FakeRunner)
    monkeypatch.setitem(sys.modules, "flyaoc.baselines.runners", fake_module)
    monkeypatch.setattr(
        cli,
        "load_gene_inputs",
        lambda **_: [GeneInput("FBgn0000014", "abd-A", "summary", ["PMC1"])],
    )

    summary = cli.run_baseline(
        BaselineRunConfig(
            baseline="memorization",
            provider="openai",
            model="gpt-5-mini",
            paper_budget=0,
            output_dir=tmp_path,
        )
    )

    rows = read_jsonl(tmp_path / "predictions.jsonl")
    assert summary["run_status_counts"] == {"ok": 1}
    assert rows[0]["baseline"] == "memorization"
    assert rows[0]["task1_function_predictions"] == [{"go_id": "GO:0003700"}]
    assert (tmp_path / "raw" / "FBgn0000014.json").exists()
    assert json.loads((tmp_path / "run_summary.json").read_text())["n_genes"] == 1


def test_paper_cache_uses_released_dataset_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("whoosh")
    from agent.core import papers

    class FakeDataset:
        def __iter__(self):
            yield {
                "pmcid": "PMC1",
                "title": "paper",
                "abstract": "abstract",
                "sections": {"RESULTS": ["body"]},
            }

    monkeypatch.setattr(papers, "load_dataset", lambda *_, **__: FakeDataset())
    monkeypatch.setattr(papers, "_paper_cache", None)
    monkeypatch.setattr(papers, "CACHE_DIR", tmp_path)

    result = papers.get_paper_text_core("PMC1")
    assert result["pmcid"] == "PMC1"
    assert result["sections"]["RESULTS"] == ["body"]
