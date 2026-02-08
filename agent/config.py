"""Unified configuration for agent execution.

This module consolidates the various configuration parameters that were
previously scattered across function signatures (8-9 parameters) into
structured config objects.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.agentic.budget import BudgetConfig

# =============================================================================
# Shared Experiment Defaults
# =============================================================================
# These defaults are used by both LangGraph and MCP agents to ensure
# controlled comparison in experiments.

DEFAULT_MODEL = "gpt-5-mini"
"""Default LLM model for both agents."""

DEFAULT_TEMPERATURE = 1.0
"""Default temperature (recommended for gpt-5-mini)."""

DEFAULT_MAX_PAPERS = 8
"""Default paper budget for experiments."""


@dataclass
class FeatureFlags:
    """Feature flags for agent execution.

    These control optional behaviors that can be enabled for specific
    experiments or use cases.
    """

    hide_go_terms: bool = False
    """Enable specificity gap benchmark (hide GO terms from ontology search)."""

    multi_agent: bool = False
    """Enable Multi-Agent mode (delegate paper reading to paper reader agents)."""

    no_literature: bool = False
    """Memorization baseline: no literature access (tests LLM prior knowledge)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hide_go_terms": self.hide_go_terms,
            "multi_agent": self.multi_agent,
            "no_literature": self.no_literature,
        }


@dataclass
class ExecutionConfig:
    """Unified configuration for agent execution.

    This replaces the 8-9 individual parameters that were previously
    passed to run_agent_mcp() and similar functions:
    - budget: BudgetConfig
    - model: str
    - verbose: bool
    - trace_dir: Path | None
    - hide_terms: bool (now in features.hide_go_terms)
    - multi_agent: bool (now in features.multi_agent)
    """

    budget: BudgetConfig = field(
        default_factory=lambda: BudgetConfig(max_papers=DEFAULT_MAX_PAPERS)
    )
    """Budget limits for the agent run."""

    model: str = DEFAULT_MODEL
    """Model to use for the agent."""

    temperature: float = DEFAULT_TEMPERATURE
    """Temperature for LLM sampling."""

    verbose: bool = False
    """Enable verbose output."""

    trace_dir: Path | None = None
    """Directory to save trace files."""

    features: FeatureFlags = field(default_factory=FeatureFlags)
    """Optional feature flags."""

    @classmethod
    def from_args(cls, args: Namespace) -> ExecutionConfig:
        """Build config from argparse namespace.

        This provides a convenient way to construct config from CLI arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            ExecutionConfig instance
        """
        return cls(
            budget=BudgetConfig(
                max_turns=getattr(args, "max_turns", 50),
                max_papers=getattr(args, "max_papers", DEFAULT_MAX_PAPERS),
                max_cost_usd=getattr(args, "max_cost", 1.0),
            ),
            model=getattr(args, "model", DEFAULT_MODEL),
            temperature=getattr(args, "temperature", DEFAULT_TEMPERATURE),
            verbose=getattr(args, "verbose", False),
            trace_dir=Path(args.trace_dir) if getattr(args, "trace_dir", None) else None,
            features=FeatureFlags(
                hide_go_terms=getattr(args, "hide_terms", False),
                multi_agent=getattr(args, "multi_agent", False),
                no_literature=getattr(args, "no_literature", False),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "budget": self.budget.to_dict(),
            "model": self.model,
            "temperature": self.temperature,
            "verbose": self.verbose,
            "trace_dir": str(self.trace_dir) if self.trace_dir else None,
            "features": self.features.to_dict(),
        }

    def with_budget(self, **kwargs: Any) -> ExecutionConfig:
        """Create a copy with modified budget settings.

        Args:
            **kwargs: Budget parameters to override (max_turns, max_papers, max_cost_usd)

        Returns:
            New ExecutionConfig with updated budget
        """
        new_budget = BudgetConfig(
            max_turns=kwargs.get("max_turns", self.budget.max_turns),
            max_papers=kwargs.get("max_papers", self.budget.max_papers),
            max_cost_usd=kwargs.get("max_cost_usd", self.budget.max_cost_usd),
        )
        return ExecutionConfig(
            budget=new_budget,
            model=self.model,
            temperature=self.temperature,
            verbose=self.verbose,
            trace_dir=self.trace_dir,
            features=self.features,
        )
