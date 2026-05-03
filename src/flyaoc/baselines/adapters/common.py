"""Shared adapter helpers for baseline reruns."""

from __future__ import annotations

from typing import Any

from flyaoc.baselines.types import BaselineRunConfig, GeneInput, RawBaselineResult


def raw_result_from_mapping(
    config: BaselineRunConfig,
    gene: GeneInput,
    result: dict[str, Any],
) -> RawBaselineResult:
    """Convert an original-harness result mapping to the public raw result."""
    return RawBaselineResult(
        gene=gene,
        baseline=config.baseline,
        provider=config.provider,
        model=config.model,
        paper_budget=config.paper_budget,
        output=result.get("output"),
        usage=result.get("usage", {}),
        error=result.get("error"),
    )


def agentic_execution_config(
    config: BaselineRunConfig,
    *,
    multi_agent: bool,
    no_literature: bool,
):
    """Build the original agentic ExecutionConfig for an adapter.

    Imports stay local so default no-API table reproduction does not need the
    optional baseline dependency stack.
    """
    from agent.agentic.budget import BudgetConfig
    from agent.config import ExecutionConfig, FeatureFlags

    return ExecutionConfig(
        budget=BudgetConfig(
            max_turns=config.max_turns,
            max_papers=config.paper_budget,
            max_cost_usd=config.max_cost,
        ),
        model=config.model,
        verbose=config.verbose,
        features=FeatureFlags(
            multi_agent=multi_agent,
            no_literature=no_literature,
        ),
    )
