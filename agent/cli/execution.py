"""Execution functions for running the Single-Agent/Multi-Agent on individual genes.

This module provides the core execution functions used by both single-gene
and batch modes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..agentic import BudgetConfig, run_agent_mcp


def run_single_gene_quiet(
    gene_id: str,
    gene_symbol: str,
    summary: str = "",
    budget: BudgetConfig | None = None,
    model: str = "gpt-5-mini",
    verbose: bool = False,
    trace_dir: Path | None = None,
    hide_terms: bool = False,
    multi_agent: bool = False,
    no_literature: bool = False,
) -> dict[str, Any]:
    """Run the MCP agent on a single gene without printing (for batch mode).

    Returns:
        Result dictionary (converted from AgentRunResult for compatibility)
    """
    result = asyncio.run(
        run_agent_mcp(
            gene_id=gene_id,
            gene_symbol=gene_symbol,
            summary=summary,
            budget=budget,
            model=model,
            verbose=verbose,
            trace_dir=trace_dir,
            hide_terms=hide_terms,
            multi_agent=multi_agent,
            no_literature=no_literature,
        )
    )
    # Convert AgentRunResult to dict for backwards compatibility
    return result.to_dict()


def run_single_gene(
    gene_id: str,
    gene_symbol: str,
    summary: str = "",
    budget: BudgetConfig | None = None,
    model: str = "gpt-5-mini",
    verbose: bool = False,
    trace_dir: Path | None = None,
    hide_terms: bool = False,
    multi_agent: bool = False,
    no_literature: bool = False,
) -> dict[str, Any]:
    """Run the MCP agent on a single gene and return results.

    Args:
        gene_id: FlyBase gene ID
        gene_symbol: Gene symbol
        summary: Optional gene summary
        budget: Budget configuration
        model: OpenAI model to use
        verbose: If True, show detailed logging
        trace_dir: Directory to save trace files
        hide_terms: If True, enable GO term hiding for specificity gap benchmark
        multi_agent: If True, use paper reader subagent for context isolation
        no_literature: If True, memorization baseline (no literature access)

    Returns:
        Result dictionary with output, usage, and optional error
    """
    print(f"Processing gene: {gene_symbol} ({gene_id})")
    if hide_terms:
        print("  [GO term hiding enabled]")
    if multi_agent:
        print("  [Multi-Agent mode enabled]")
    if no_literature:
        print("  [Memorization mode enabled - no literature access]")
    result = run_single_gene_quiet(
        gene_id,
        gene_symbol,
        summary,
        budget,
        model,
        verbose,
        trace_dir,
        hide_terms,
        multi_agent,
        no_literature,
    )

    usage = result.get("usage", {})
    print(f"  Turns: {usage.get('turns_used', 0)}")
    print(f"  Papers read: {usage.get('papers_read', 0)}")
    print(f"  Cost: ${usage.get('cost_usd', 0):.4f} ({usage.get('total_tokens', 0)} tokens)")

    if result.get("error"):
        print(f"  Error: {result['error']}")

    return result
