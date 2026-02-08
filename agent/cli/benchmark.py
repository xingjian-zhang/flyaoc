"""Benchmark orchestration for running the Single-Agent/Multi-Agent on multiple genes.

This module provides functions for running the agent on benchmark gene sets,
with support for parallel execution and JSON input mode.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from ..agentic import (
    BudgetConfig,
    GeneResult,
    GeneTask,
    get_remaining_genes,
    run_genes_parallel,
)
from ..core.metadata import capture_metadata, format_version_string
from .execution import run_single_gene, run_single_gene_quiet

if TYPE_CHECKING:
    pass


def run_from_json(
    input_path: Path,
    budget: BudgetConfig,
    model: str,
    verbose: bool = False,
    trace_dir: Path | None = None,
    hide_terms: bool = False,
    multi_agent: bool = False,
) -> dict:
    """Run the agent from a JSON input file.

    Args:
        input_path: Path to JSON input file
        budget: Budget configuration
        model: OpenAI model to use
        verbose: If True, show detailed logging
        trace_dir: Directory to save trace files
        hide_terms: If True, enable GO term hiding
        multi_agent: If True, use paper reader subagent

    Returns:
        Result dictionary
    """
    with open(input_path) as f:
        data = json.load(f)

    return run_single_gene(
        gene_id=data["gene_id"],
        gene_symbol=data["gene_symbol"],
        summary=data.get("summary", ""),
        budget=budget,
        model=model,
        verbose=verbose,
        trace_dir=trace_dir,
        hide_terms=hide_terms,
        multi_agent=multi_agent,
    )


async def run_benchmark_parallel(
    genes: list[GeneTask],
    output_dir: Path,
    budget: BudgetConfig,
    model: str,
    workers: int = 5,
    verbose: bool = False,
    trace_dir: Path | None = None,
    hide_terms: bool = False,
    multi_agent: bool = False,
    no_literature: bool = False,
) -> dict:
    """Run benchmark with parallel gene processing.

    Args:
        genes: List of GeneTask objects
        output_dir: Directory to write output JSON files
        budget: Budget configuration for each gene
        model: OpenAI model to use
        workers: Number of parallel workers
        verbose: If True, show detailed logging
        trace_dir: Directory to save trace files
        hide_terms: If True, enable GO term hiding
        multi_agent: If True, use paper reader subagent

    Returns:
        Results summary dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for already completed genes (resume support)
    remaining_genes = get_remaining_genes(genes, output_dir)
    completed_count = len(genes) - len(remaining_genes)

    results_summary: dict = {
        "strategy": "agentic",
        "model": model,
        "budget": budget.to_dict(),
        "hide_terms": hide_terms,
        "multi_agent": multi_agent,
        "no_literature": no_literature,
        "workers": workers,
        "total": len(genes),
        "successful": 0,
        "failed": 0,
        "total_cost": 0.0,
        "total_tokens": 0,
        "errors": [],
    }

    # Load results for already completed genes
    for gene in genes:
        result_path = output_dir / f"{gene.gene_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
            usage = result.get("usage", {})
            results_summary["total_cost"] += usage.get("cost_usd", 0)
            results_summary["total_tokens"] += usage.get("total_tokens", 0)
            if result.get("error"):
                results_summary["failed"] += 1
                results_summary["errors"].append(
                    {"gene_id": gene.gene_id, "error": result["error"]}
                )
            else:
                results_summary["successful"] += 1

    if completed_count > 0:
        print(
            f"Resuming: {completed_count} genes already completed, {len(remaining_genes)} remaining"
        )

    if not remaining_genes:
        print("All genes already completed")
        return results_summary

    # Progress bar
    desc = "Processing genes (hidden terms)" if hide_terms else "Processing genes"
    pbar = tqdm(total=len(remaining_genes), desc=desc, unit="gene")

    def on_complete(result: GeneResult) -> None:
        """Callback when a gene completes."""
        pbar.update(1)
        pbar.set_postfix_str(f"{result.gene_symbol}")

    # Run remaining genes in parallel
    results = await run_genes_parallel(
        genes=remaining_genes,
        budget=budget,
        workers=workers,
        output_dir=output_dir,
        model=model,
        verbose=verbose,
        trace_dir=trace_dir,
        hide_terms=hide_terms,
        multi_agent=multi_agent,
        no_literature=no_literature,
        on_complete=on_complete,
    )

    pbar.close()

    # Process results
    for gene_result in results:
        usage = gene_result.result.get("usage", {}) if gene_result.result else {}
        results_summary["total_cost"] += usage.get("cost_usd", 0)
        results_summary["total_tokens"] += usage.get("total_tokens", 0)

        if gene_result.error:
            results_summary["failed"] += 1
            results_summary["errors"].append(
                {"gene_id": gene_result.gene_id, "error": gene_result.error}
            )
        else:
            results_summary["successful"] += 1

    return results_summary


def run_benchmark(
    csv_path: Path,
    output_dir: Path,
    budget: BudgetConfig,
    model: str,
    limit: int | None = None,
    verbose: bool = False,
    trace_dir: Path | None = None,
    hide_terms: bool = False,
    multi_agent: bool = False,
    workers: int = 1,
    no_literature: bool = False,
) -> None:
    """Run the agent on benchmark genes from CSV.

    Args:
        csv_path: Path to genes CSV (columns: FBgn_ID, Gene_Symbol, Summary)
        output_dir: Directory to write output JSON files
        budget: Budget configuration for each gene
        model: OpenAI model to use
        limit: Maximum number of genes to process (for testing)
        verbose: If True, show detailed logging
        trace_dir: Directory to save trace files (one per gene)
        hide_terms: If True, enable GO term hiding for specificity gap benchmark
        multi_agent: If True, use paper reader subagent for context isolation
        workers: Number of parallel workers (1 = sequential)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        genes_raw = list(reader)

    if limit:
        genes_raw = genes_raw[:limit]

    # Capture experiment metadata for reproducibility
    metadata = capture_metadata()
    print(f"Code version: {format_version_string(metadata)}")
    print(f"Workers: {workers}")

    if workers > 1:
        # Parallel execution
        gene_tasks = [
            GeneTask(
                gene_id=g["FBgn_ID"],
                gene_symbol=g["Gene_Symbol"],
                summary=g.get("Summary", ""),
            )
            for g in genes_raw
        ]

        results_summary = asyncio.run(
            run_benchmark_parallel(
                genes=gene_tasks,
                output_dir=output_dir,
                budget=budget,
                model=model,
                workers=workers,
                verbose=verbose,
                trace_dir=trace_dir,
                hide_terms=hide_terms,
                multi_agent=multi_agent,
                no_literature=no_literature,
            )
        )
        results_summary["metadata"] = metadata
    else:
        # Sequential execution (original behavior)
        results_summary: dict = {
            "metadata": metadata,
            "strategy": "agentic",
            "model": model,
            "budget": budget.to_dict(),
            "hide_terms": hide_terms,
            "multi_agent": multi_agent,
            "no_literature": no_literature,
            "workers": workers,
            "total": len(genes_raw),
            "successful": 0,
            "failed": 0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "errors": [],
        }

        # Progress bar
        desc = "Processing genes (hidden terms)" if hide_terms else "Processing genes"
        pbar = tqdm(genes_raw, desc=desc, unit="gene")
        for gene in pbar:
            gene_id = gene["FBgn_ID"]
            gene_symbol = gene["Gene_Symbol"]
            gene_summary = gene.get("Summary", "")

            pbar.set_postfix_str(f"{gene_symbol}")

            # Check if already completed (resume support)
            output_path = output_dir / f"{gene_id}.json"
            if output_path.exists():
                with open(output_path) as f:
                    result = json.load(f)
            else:
                try:
                    # Suppress individual gene output when using progress bar
                    result = run_single_gene_quiet(
                        gene_id,
                        gene_symbol,
                        gene_summary,
                        budget,
                        model,
                        verbose,
                        trace_dir,
                        hide_terms,
                        multi_agent,
                        no_literature,
                    )

                    # Save output
                    with open(output_path, "w") as f:
                        json.dump(result, f, indent=2)

                except Exception as e:
                    results_summary["failed"] += 1
                    results_summary["errors"].append({"gene_id": gene_id, "error": str(e)})
                    continue

            # Track costs
            usage = result.get("usage", {})
            results_summary["total_cost"] += usage.get("cost_usd", 0)
            results_summary["total_tokens"] += usage.get("total_tokens", 0)

            if result.get("error"):
                results_summary["failed"] += 1
                results_summary["errors"].append({"gene_id": gene_id, "error": result["error"]})
            else:
                results_summary["successful"] += 1

        pbar.close()

    # Save summary
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nCompleted: {results_summary['successful']}/{results_summary['total']} successful")
    print(f"Total cost: ${results_summary['total_cost']:.4f}")
    print(f"Results saved to: {output_dir}")
