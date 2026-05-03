"""Reviewer-facing wrappers around the original FlyAOC baseline harness."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from agent.agentic.budget import BudgetConfig
from agent.agentic.runner import run_agent_mcp
from agent.config import ExecutionConfig, FeatureFlags
from agent.pipeline.agent import run_agent as run_pipeline_agent
from flyaoc.baselines.types import BaselineRunConfig, GeneInput, RawBaselineResult


class BaselineRunner:
    """Run one of the paper baselines using the original implementation."""

    def __init__(self, config: BaselineRunConfig):
        self.config = config

    def run_gene(self, gene: GeneInput) -> RawBaselineResult:
        try:
            _check_provider_environment(self.config.provider)
            if self.config.baseline == "pipeline":
                result = asyncio.run(self._run_pipeline(gene))
            else:
                result = asyncio.run(self._run_agentic(gene))
            return RawBaselineResult(
                gene=gene,
                baseline=self.config.baseline,
                provider=self.config.provider,
                model=self.config.model,
                paper_budget=self.config.paper_budget,
                output=result.get("output"),
                usage=result.get("usage", {}),
                error=result.get("error"),
            )
        except Exception as exc:
            return RawBaselineResult(
                gene=gene,
                baseline=self.config.baseline,
                provider=self.config.provider,
                model=self.config.model,
                paper_budget=self.config.paper_budget,
                error=str(exc),
            )

    async def _run_agentic(self, gene: GeneInput) -> dict[str, Any]:
        baseline = self.config.baseline
        config = ExecutionConfig(
            budget=BudgetConfig(
                max_turns=self.config.max_turns,
                max_papers=self.config.paper_budget,
                max_cost_usd=self.config.max_cost,
            ),
            model=self.config.model,
            verbose=self.config.verbose,
            features=FeatureFlags(
                multi_agent=baseline == "multi_agent",
                no_literature=baseline == "memorization",
            ),
        )
        result = await run_agent_mcp(
            gene.gene_id,
            gene.gene_symbol,
            summary=gene.summary,
            config=config,
        )
        return result.to_dict()

    async def _run_pipeline(self, gene: GeneInput) -> dict[str, Any]:
        return await run_pipeline_agent(
            gene.gene_id,
            gene.gene_symbol,
            summary=gene.summary,
            model=self.config.model,
            verbose=self.config.verbose,
            max_papers=self.config.paper_budget,
        )


def _check_provider_environment(provider: str) -> None:
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --provider openai.")
        return

    if provider in {"openai_compatible", "bedrock_proxy"}:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI-compatible provider routes."
            )
        if not os.environ.get("OPENAI_BASE_URL"):
            raise RuntimeError(
                "OPENAI_BASE_URL is required for --provider openai_compatible or bedrock_proxy."
            )
        return

    raise ValueError(f"Unsupported provider: {provider}")
