"""Parallel gene processing for Single-Agent/Multi-Agent.

Provides utilities for running multiple genes concurrently with:
- Semaphore-based concurrency control
- Retry logic for rate limits
- Immediate result saving (checkpointing)
- Resume support (skip completed genes)
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .budget import BudgetConfig
from .runner import run_agent_mcp


@dataclass
class GeneTask:
    """A gene to process."""

    gene_id: str
    gene_symbol: str
    summary: str = ""


@dataclass
class GeneResult:
    """Result of processing a gene."""

    gene_id: str
    gene_symbol: str
    result: dict[str, Any] | None = None
    error: str | None = None


def is_rate_limit_error(e: BaseException) -> bool:
    """Check if exception is a rate limit error."""
    error_str = str(e).lower()
    return "rate limit" in error_str or "429" in error_str or "too many requests" in error_str


async def run_gene_with_retry(
    gene: GeneTask,
    budget: BudgetConfig,
    model: str = "gpt-5-mini",
    verbose: bool = False,
    trace_dir: Path | None = None,
    hide_terms: bool = False,
    multi_agent: bool = False,
    no_literature: bool = False,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Run agent on a gene with retry logic for rate limits.

    Args:
        gene: Gene task to process
        budget: Budget configuration
        model: OpenAI model to use
        verbose: Show detailed logging
        trace_dir: Directory for trace files
        hide_terms: Enable GO term hiding
        multi_agent: Use paper reader subagent for context isolation
        no_literature: Memorization baseline (no literature access)
        max_retries: Maximum retry attempts

    Returns:
        Result dict from run_agent_mcp

    Raises:
        Exception: If all retries exhausted
    """

    @retry(
        retry=retry_if_exception(is_rate_limit_error),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential_jitter(initial=5, max=60, jitter=5),
        reraise=True,
    )
    async def _run_with_retry() -> dict[str, Any]:
        result = await run_agent_mcp(
            gene_id=gene.gene_id,
            gene_symbol=gene.gene_symbol,
            summary=gene.summary,
            budget=budget,
            model=model,
            verbose=verbose,
            trace_dir=trace_dir,
            hide_terms=hide_terms,
            multi_agent=multi_agent,
            no_literature=no_literature,
        )
        # Convert AgentRunResult to dict for backwards compatibility
        return result.to_dict()

    return await _run_with_retry()


async def run_gene_with_semaphore(
    semaphore: asyncio.Semaphore,
    gene: GeneTask,
    budget: BudgetConfig,
    output_dir: Path | None = None,
    model: str = "gpt-5-mini",
    verbose: bool = False,
    trace_dir: Path | None = None,
    hide_terms: bool = False,
    multi_agent: bool = False,
    no_literature: bool = False,
    max_retries: int = 5,
    on_complete: Callable[[GeneResult], None] | None = None,
) -> GeneResult:
    """Run a gene with semaphore-controlled concurrency.

    Args:
        semaphore: Asyncio semaphore for concurrency control
        gene: Gene task to process
        budget: Budget configuration
        output_dir: Directory to save result (for checkpointing)
        model: OpenAI model to use
        verbose: Show detailed logging
        trace_dir: Directory for trace files
        hide_terms: Enable GO term hiding
        multi_agent: Use paper reader subagent for context isolation
        no_literature: Memorization baseline (no literature access)
        max_retries: Maximum retry attempts for rate limits
        on_complete: Callback when gene completes (for progress updates)

    Returns:
        GeneResult with result or error
    """
    async with semaphore:
        try:
            result = await run_gene_with_retry(
                gene=gene,
                budget=budget,
                model=model,
                verbose=verbose,
                trace_dir=trace_dir,
                hide_terms=hide_terms,
                multi_agent=multi_agent,
                no_literature=no_literature,
                max_retries=max_retries,
            )

            # Save immediately for checkpointing
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{gene.gene_id}.json"
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)

            gene_result = GeneResult(
                gene_id=gene.gene_id,
                gene_symbol=gene.gene_symbol,
                result=result,
                error=result.get("error"),
            )

        except Exception as e:
            gene_result = GeneResult(
                gene_id=gene.gene_id,
                gene_symbol=gene.gene_symbol,
                error=str(e),
            )

        # Notify completion
        if on_complete:
            on_complete(gene_result)

        return gene_result


async def run_genes_parallel(
    genes: list[GeneTask],
    budget: BudgetConfig,
    workers: int = 5,
    output_dir: Path | None = None,
    model: str = "gpt-5-mini",
    verbose: bool = False,
    trace_dir: Path | None = None,
    hide_terms: bool = False,
    multi_agent: bool = False,
    no_literature: bool = False,
    max_retries: int = 5,
    on_complete: Callable[[GeneResult], None] | None = None,
) -> list[GeneResult]:
    """Run multiple genes with semaphore-controlled concurrency.

    Args:
        genes: List of gene tasks to process
        budget: Budget configuration (shared across all genes)
        workers: Number of concurrent workers
        output_dir: Directory to save results (enables checkpointing)
        model: OpenAI model to use
        verbose: Show detailed logging
        trace_dir: Directory for trace files
        hide_terms: Enable GO term hiding
        multi_agent: Use paper reader subagent for context isolation
        no_literature: Memorization baseline (no literature access)
        max_retries: Maximum retry attempts for rate limits
        on_complete: Callback when each gene completes

    Returns:
        List of GeneResults (in completion order may differ from input order)
    """
    semaphore = asyncio.Semaphore(workers)

    tasks = [
        run_gene_with_semaphore(
            semaphore=semaphore,
            gene=gene,
            budget=budget,
            output_dir=output_dir,
            model=model,
            verbose=verbose,
            trace_dir=trace_dir,
            hide_terms=hide_terms,
            multi_agent=multi_agent,
            no_literature=no_literature,
            max_retries=max_retries,
            on_complete=on_complete,
        )
        for gene in genes
    ]

    # gather with return_exceptions=True ensures one failure doesn't crash all
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert any exceptions to GeneResult
    final_results: list[GeneResult] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            final_results.append(
                GeneResult(
                    gene_id=genes[i].gene_id,
                    gene_symbol=genes[i].gene_symbol,
                    error=str(result),
                )
            )
        else:
            final_results.append(result)

    return final_results


def get_remaining_genes(genes: list[GeneTask], output_dir: Path) -> list[GeneTask]:
    """Filter out genes that already have completed results.

    Enables resume functionality - rerunning the same command will
    skip genes that were already processed.

    Args:
        genes: Full list of genes to process
        output_dir: Directory where results are saved

    Returns:
        List of genes without existing results
    """
    remaining = []
    for gene in genes:
        result_path = output_dir / f"{gene.gene_id}.json"
        if result_path.exists():
            # Could optionally verify the file is valid JSON
            continue
        remaining.append(gene)
    return remaining
