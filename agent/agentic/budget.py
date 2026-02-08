"""Budget configuration and tracking for the MCP backbone agent."""

from dataclasses import dataclass
from typing import Any


@dataclass
class BudgetConfig:
    """Budget limits for agent execution.

    Attributes:
        max_turns: Maximum number of LLM round-trips
        max_papers: Maximum number of papers to read
        max_cost_usd: Maximum cost in USD
    """

    max_turns: int = 50
    max_papers: int = 10
    max_cost_usd: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_turns": self.max_turns,
            "max_papers": self.max_papers,
            "max_cost_usd": self.max_cost_usd,
        }


@dataclass
class BudgetState:
    """Mutable tracking of budget consumption.

    Attributes:
        turns_used: Number of LLM round-trips used
        papers_read: Number of papers read
        cost_usd: Total cost in USD
        input_tokens: Total input tokens used
        output_tokens: Total output tokens used
        cached_tokens: Total cached input tokens (subset of input_tokens)
    """

    turns_used: int = 0
    papers_read: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def check(self, config: BudgetConfig) -> tuple[bool, str]:
        """Check if any budget limit is exceeded.

        Args:
            config: Budget configuration to check against

        Returns:
            Tuple of (exceeded: bool, reason: str)
        """
        if self.turns_used > config.max_turns:
            return True, f"Turn limit reached ({self.turns_used}/{config.max_turns})"
        if self.papers_read > config.max_papers:
            return True, f"Paper limit reached ({self.papers_read}/{config.max_papers})"
        if self.cost_usd > config.max_cost_usd:
            return True, f"Cost limit reached (${self.cost_usd:.4f}/${config.max_cost_usd})"
        return False, ""

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
