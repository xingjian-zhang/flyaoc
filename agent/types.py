"""Typed result objects for agent execution.

This module provides strongly typed dataclasses for agent run results,
replacing the untyped dict[str, Any] returns throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentOutput:
    """Parsed agent output with task annotations.

    This represents the structured output from a successful agent run,
    containing annotations for all three tasks.
    """

    task1_function: list[dict[str, Any]] = field(default_factory=list)
    task2_expression: list[dict[str, Any]] = field(default_factory=list)
    task3_synonyms: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentOutput | None:
        """Create AgentOutput from a dictionary.

        Args:
            data: Dictionary with task1_function, task2_expression, task3_synonyms

        Returns:
            AgentOutput instance or None if data is None
        """
        if data is None:
            return None
        return cls(
            task1_function=data.get("task1_function", []),
            task2_expression=data.get("task2_expression", []),
            task3_synonyms=data.get("task3_synonyms", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task1_function": self.task1_function,
            "task2_expression": self.task2_expression,
            "task3_synonyms": self.task3_synonyms,
        }


@dataclass
class UsageStats:
    """Token and cost usage statistics from an agent run.

    This mirrors the BudgetState.to_dict() output for serialization
    compatibility while providing type safety.
    """

    turns_used: int = 0
    papers_read: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def uncached_tokens(self) -> int:
        """Uncached input tokens (full price)."""
        return self.input_tokens - self.cached_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of input tokens that were cached."""
        if self.input_tokens == 0:
            return 0.0
        return self.cached_tokens / self.input_tokens

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageStats:
        """Create UsageStats from a dictionary.

        Args:
            data: Dictionary with usage statistics

        Returns:
            UsageStats instance
        """
        return cls(
            turns_used=data.get("turns_used", 0),
            papers_read=data.get("papers_read", 0),
            cost_usd=data.get("cost_usd", 0.0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turns_used": self.turns_used,
            "papers_read": self.papers_read,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "uncached_tokens": self.uncached_tokens,
            "cache_hit_rate": self.cache_hit_rate,
            "total_tokens": self.total_tokens,
        }


@dataclass
class AgentRunResult:
    """Complete result from a single agent run.

    This is the primary return type for run_agent_mcp() and similar functions,
    replacing the previous dict[str, Any] return type.
    """

    output: AgentOutput | None
    usage: UsageStats
    raw_response: str | None = None
    error: str | None = None
    trace: dict[str, Any] | None = None

    @property
    def success(self) -> bool:
        """Whether the run completed successfully."""
        return self.error is None and self.output is not None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRunResult:
        """Create AgentRunResult from a dictionary.

        Args:
            data: Dictionary with output, usage, error, trace, etc.

        Returns:
            AgentRunResult instance
        """
        output_data = data.get("output")
        return cls(
            output=AgentOutput.from_dict(output_data),
            usage=UsageStats.from_dict(data.get("usage", {})),
            raw_response=data.get("raw_response"),
            error=data.get("error"),
            trace=data.get("trace"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        This produces the same structure as the previous dict returns
        for backwards compatibility.
        """
        result: dict[str, Any] = {
            "output": self.output.to_dict() if self.output else None,
            "usage": self.usage.to_dict(),
        }
        if self.raw_response is not None:
            result["raw_response"] = self.raw_response
        if self.error is not None:
            result["error"] = self.error
        if self.trace is not None:
            result["trace"] = self.trace
        return result


@dataclass
class GeneRunResult:
    """Result for a single gene in a benchmark run.

    Combines gene identification with the agent run result.
    """

    gene_id: str
    gene_symbol: str
    result: AgentRunResult
    output_path: Path | None = None

    @property
    def success(self) -> bool:
        """Whether the gene run was successful."""
        return self.result.success

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "gene_id": self.gene_id,
            "gene_symbol": self.gene_symbol,
            **self.result.to_dict(),
        }
        if self.output_path:
            data["output_path"] = str(self.output_path)
        return data


@dataclass
class BenchmarkSummary:
    """Summary of a benchmark run across multiple genes.

    This replaces the untyped run_summary dict structure.
    """

    total: int
    successful: int
    failed: int
    total_cost: float
    total_tokens: int
    model: str = ""
    strategy: str = ""
    results: list[GeneRunResult] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of genes that completed successfully."""
        return self.successful / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "total_cost": round(self.total_cost, 4),
            "total_tokens": self.total_tokens,
            "model": self.model,
            "strategy": self.strategy,
            "errors": self.errors,
        }
