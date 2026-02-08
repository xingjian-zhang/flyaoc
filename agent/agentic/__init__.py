"""Single-Agent and Multi-Agent implementations.

Single-Agent: One agent with sequential tool use and growing context
Multi-Agent: Hierarchical delegation to subagents with bounded context

This package provides:
- BudgetConfig/BudgetState: Budget tracking and limits
- BudgetControlHooks: RunHooks for enforcing budget constraints
- AgentTrace: Structured logging and tracing
- run_agent_mcp: Main entry point for running the agentic methods
- Parallel execution utilities for batch processing
"""

from .budget import BudgetConfig, BudgetState
from .hooks import BudgetControlHooks, BudgetExhaustedError
from .parallel import GeneResult, GeneTask, get_remaining_genes, run_genes_parallel
from .runner import run_agent_mcp
from .tracing import AgentTrace, TraceEvent, setup_logging

__all__ = [
    "BudgetConfig",
    "BudgetState",
    "BudgetControlHooks",
    "BudgetExhaustedError",
    "run_agent_mcp",
    "AgentTrace",
    "TraceEvent",
    "setup_logging",
    # Parallel execution
    "GeneTask",
    "GeneResult",
    "run_genes_parallel",
    "get_remaining_genes",
]
