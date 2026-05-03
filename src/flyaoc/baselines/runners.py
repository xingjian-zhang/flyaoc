"""Dispatcher for FlyAOC baseline reruns."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from flyaoc.baselines.providers import validate_provider_environment
from flyaoc.baselines.types import BaselineName, BaselineRunConfig, GeneInput, RawBaselineResult

_ADAPTER_MODULES: dict[BaselineName, str] = {
    "memorization": "flyaoc.baselines.adapters.memorization",
    "pipeline": "flyaoc.baselines.adapters.pipeline",
    "single_agent": "flyaoc.baselines.adapters.single_agent",
    "multi_agent": "flyaoc.baselines.adapters.multi_agent",
}


class BaselineRunner:
    """Run a configured paper baseline through its explicit adapter."""

    def __init__(self, config: BaselineRunConfig):
        self.config = config

    def run_gene(self, gene: GeneInput) -> RawBaselineResult:
        try:
            validate_provider_environment(self.config.provider)
            adapter = load_adapter(self.config.baseline)
            return adapter.run_gene(self.config, gene)
        except Exception as exc:
            return RawBaselineResult(
                gene=gene,
                baseline=self.config.baseline,
                provider=self.config.provider,
                model=self.config.model,
                paper_budget=self.config.paper_budget,
                error=str(exc),
            )


def load_adapter(baseline: BaselineName) -> ModuleType:
    """Load the adapter for a public baseline name."""
    return import_module(_ADAPTER_MODULES[baseline])
