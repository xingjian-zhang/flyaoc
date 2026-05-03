"""Shared types for runnable FlyAOC baseline harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

BaselineName = Literal["memorization", "pipeline", "single_agent", "multi_agent"]
ProviderName = Literal["openai", "openai_compatible", "bedrock_proxy"]
RunStatus = Literal["ok", "failed", "empty_output"]


@dataclass(frozen=True)
class GeneInput:
    gene_id: str
    gene_symbol: str
    summary: str = ""
    pmcids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BaselineRunConfig:
    baseline: BaselineName
    provider: ProviderName
    model: str
    paper_budget: int
    output_dir: Path
    limit: int | None = None
    gene_id: str | None = None
    workers: int = 1
    resume: bool = True
    max_turns: int = 50
    max_cost: float = 1.0
    verbose: bool = False

    def validate(self) -> None:
        if self.baseline == "memorization" and self.paper_budget != 0:
            raise ValueError("The memorization baseline must use --paper-budget 0.")
        if self.baseline != "memorization" and self.paper_budget <= 0:
            raise ValueError("Retrieval baselines require --paper-budget > 0.")
        if self.workers < 1:
            raise ValueError("--workers must be at least 1.")
        if self.max_turns < 1:
            raise ValueError("--max-turns must be at least 1.")
        if self.max_cost <= 0:
            raise ValueError("--max-cost must be positive.")


@dataclass
class RawBaselineResult:
    gene: GeneInput
    baseline: BaselineName
    provider: ProviderName
    model: str
    paper_budget: int
    output: dict[str, Any] | None = None
    retrieved_pmcids: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def run_status(self) -> RunStatus:
        if self.error:
            return "failed"
        if not self.output or _is_empty_output(self.output):
            return "empty_output"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gene_id": self.gene.gene_id,
            "gene_symbol": self.gene.gene_symbol,
            "baseline": self.baseline,
            "provider": self.provider,
            "model": self.model,
            "paper_budget": self.paper_budget,
            "run_status": self.run_status,
            "retrieved_pmcids": self.retrieved_pmcids,
            "output": self.output,
            "usage": self.usage,
            "error": self.error,
        }


def _is_empty_output(output: dict[str, Any]) -> bool:
    task1 = output.get("task1_function") or output.get("task1_function_predictions") or []
    task2 = output.get("task2_expression") or output.get("task2_expression_predictions") or []
    task3 = output.get("task3_synonyms") or output.get("task3_synonym_predictions") or {}
    return not task1 and not task2 and not task3.get("fullname_synonyms") and not task3.get(
        "symbol_synonyms"
    )
