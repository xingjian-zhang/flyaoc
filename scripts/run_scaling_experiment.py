#!/usr/bin/env python3
"""Run scaling experiment to map cost vs performance Pareto frontier.

This script runs agents with varying max_papers settings to measure how
performance scales with paper budget. Supports all three agent methods.

Output structure (default):
    outputs/scaling/{method}-{commit_short}/
        experiment_summary.json
        papers_1/
            {gene_id}.json
            run_summary.json
        papers_3/
            ...

Usage:
    # Run with default method (single-agent)
    uv run python -m scripts.run_scaling_experiment

    # Run specific method
    uv run python -m scripts.run_scaling_experiment --method single
    uv run python -m scripts.run_scaling_experiment --method multi
    uv run python -m scripts.run_scaling_experiment --method pipeline

    # Run with parallel workers (faster)
    uv run python -m scripts.run_scaling_experiment --workers 4

    # Dry run (show what would be run)
    uv run python -m scripts.run_scaling_experiment --dry-run

    # Run with specific genes and paper configs
    uv run python -m scripts.run_scaling_experiment --papers 1 2 4 8 --max-genes 10

    # Custom output directory
    uv run python -m scripts.run_scaling_experiment -o outputs/scaling/my_experiment
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv(override=True)

from agent.agentic import (  # noqa: E402
    BudgetConfig,
    GeneResult,
    GeneTask,
    get_remaining_genes,
    run_agent_mcp,
    run_genes_parallel,
)
from agent.core.metadata import capture_metadata, format_version_string  # noqa: E402
from agent.pipeline.agent import run_agent as run_pipeline_agent  # noqa: E402

# Agent method types
METHOD_SINGLE = "single"
METHOD_MULTI = "multi"
METHOD_PIPELINE = "pipeline"
METHOD_MEMORIZATION = "memorization"
VALID_METHODS = [METHOD_SINGLE, METHOD_MULTI, METHOD_PIPELINE, METHOD_MEMORIZATION]

# Paper configurations to test
PAPER_CONFIGS = [1, 2, 3, 5, 10, 15, 20]

# Test genes selected for variety in corpus size (25 genes)
TEST_GENES = [
    # Large corpus (200+ papers)
    ("FBgn0284084", "wg", 627),
    ("FBgn0000229", "bsk", 515),
    ("FBgn0003514", "sqh", 371),
    ("FBgn0011648", "Mad", 316),
    ("FBgn0000546", "EcR", 314),
    ("FBgn0259246", "brp", 306),
    ("FBgn0003716", "tkv", 280),
    ("FBgn0000606", "eve", 277),
    # Medium corpus (100-200 papers)
    ("FBgn0000014", "abd-A", 170),
    ("FBgn0031424", "VGlut1", 168),
    ("FBgn0000320", "eya", 160),
    ("FBgn0003345", "sd", 145),
    ("FBgn0283521", "lola", 144),
    ("FBgn0250816", "AGO3", 141),
    ("FBgn0040477", "cid", 128),
    ("FBgn0011202", "dia", 126),
    ("FBgn0024234", "gbb", 122),
    ("FBgn0024836", "stan", 122),
    # Small corpus (50-100 papers)
    ("FBgn0001316", "klar", 62),
    ("FBgn0026086", "Adar", 62),
    ("FBgn0030941", "wgn", 61),
    ("FBgn0038901", "Burs", 60),
    ("FBgn0250823", "gish", 59),
    ("FBgn0004449", "Ten-m", 57),
    ("FBgn0023213", "eIF4G1", 54),
]


def load_gene_summaries(csv_path: Path) -> dict[str, str]:
    """Load gene summaries from CSV file.

    Args:
        csv_path: Path to genes CSV with FBgn_ID, Gene_Symbol, Summary columns

    Returns:
        Dict mapping gene_id to summary
    """
    summaries = {}
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                summaries[row["FBgn_ID"]] = row.get("Summary", "")
    return summaries


def load_all_genes_from_csv(csv_path: Path) -> list[tuple[str, str, int]]:
    """Load all genes from CSV file.

    Args:
        csv_path: Path to genes CSV

    Returns:
        List of (gene_id, symbol, corpus_size) tuples.
        Corpus size is set to 0 since CSV doesn't have this info.
    """
    genes = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                genes.append((row["FBgn_ID"], row["Gene_Symbol"], 0))
    return genes


def run_pipeline_single_gene(
    gene_id: str,
    gene_symbol: str,
    summary: str,
    max_papers: int,
    output_dir: Path,
    model: str = "gpt-5-mini",
    verbose: bool = False,
    hide_terms: bool = False,
) -> dict:
    """Run Pipeline agent on single gene and save results.

    Args:
        gene_id: FlyBase gene ID
        gene_symbol: Gene symbol
        summary: Gene summary text
        max_papers: Maximum papers to read
        output_dir: Directory to save output
        model: Model to use
        verbose: Show detailed logging
        hide_terms: Enable specificity gap benchmark (hide GO terms)

    Returns:
        Result dict with output, usage, error
    """
    result = asyncio.run(
        run_pipeline_agent(
            gene_id=gene_id,
            gene_symbol=gene_symbol,
            summary=summary,
            model=model,
            verbose=verbose,
            max_papers=max_papers,
            hide_terms=hide_terms,
        )
    )

    # Normalize result format to match agentic agent output
    normalized_result = {
        "output": result.get("output"),
        "usage": {
            "cost_usd": result.get("usage", {}).get("total_cost", 0),
            "total_tokens": result.get("usage", {}).get("total_tokens", 0),
            "papers_read": max_papers,  # Pipeline always reads max_papers
        },
        "error": result.get("error"),
    }

    # Save result
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{gene_id}.json"
    with open(output_path, "w") as f:
        json.dump(normalized_result, f, indent=2)

    return normalized_result


def run_single_gene(
    gene_id: str,
    gene_symbol: str,
    summary: str,
    max_papers: int,
    output_dir: Path,
    model: str = "gpt-5-mini",
    verbose: bool = False,
    multi_agent: bool = False,
    no_literature: bool = False,
    hide_terms: bool = False,
) -> dict:
    """Run agent on single gene and save results (sequential mode).

    Args:
        gene_id: FlyBase gene ID
        gene_symbol: Gene symbol
        summary: Gene summary text
        max_papers: Maximum papers to read
        output_dir: Directory to save output
        model: Model to use
        verbose: Show detailed logging
        multi_agent: Use paper reader subagent for context isolation
        no_literature: Memorization baseline (no literature access)
        hide_terms: Enable specificity gap benchmark (hide GO terms)

    Returns:
        Result dict with output, usage, error
    """
    # Import config for feature flags
    from agent.config import ExecutionConfig, FeatureFlags

    budget = BudgetConfig(
        max_turns=50,
        max_papers=max_papers,
        max_cost_usd=10.0,  # Higher limit for expensive models (gpt-4o, gpt-5)
    )

    trace_dir = output_dir / "traces"

    # Build config with feature flags
    config = ExecutionConfig(
        budget=budget,
        model=model,
        verbose=verbose,
        trace_dir=trace_dir,
        features=FeatureFlags(
            multi_agent=multi_agent,
            no_literature=no_literature,
            hide_go_terms=hide_terms,
        ),
    )

    agent_result = asyncio.run(
        run_agent_mcp(
            gene_id=gene_id,
            gene_symbol=gene_symbol,
            summary=summary,
            config=config,
        )
    )

    # Convert AgentRunResult to dict for serialization
    result = agent_result.to_dict()

    # Save result
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{gene_id}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


async def run_config_parallel(
    genes: list[GeneTask],
    max_papers: int,
    output_dir: Path,
    summaries: dict[str, str],
    model: str = "gpt-5-mini",
    verbose: bool = False,
    workers: int = 5,
    multi_agent: bool = False,
    no_literature: bool = False,
    hide_terms: bool = False,
) -> dict:
    """Run a single paper config with parallel gene processing.

    Args:
        genes: List of GeneTask objects
        max_papers: Maximum papers to read
        output_dir: Directory to save output
        summaries: Dict mapping gene_id to summary
        model: Model to use
        verbose: Show detailed logging
        workers: Number of parallel workers
        multi_agent: Use paper reader subagent for context isolation
        no_literature: Memorization baseline (no literature access)
        hide_terms: Enable specificity gap benchmark (hide GO terms)

    Returns:
        Config results dict
    """
    budget = BudgetConfig(
        max_turns=50,
        max_papers=max_papers,
        max_cost_usd=10.0,  # Higher limit for expensive models
    )

    trace_dir = output_dir / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for already completed genes (resume support)
    remaining_genes = get_remaining_genes(genes, output_dir)
    completed_count = len(genes) - len(remaining_genes)

    if completed_count > 0:
        print(
            f"  Resuming: {completed_count} genes already completed, {len(remaining_genes)} remaining"
        )

    config_results: dict = {
        "max_papers": max_papers,
        "genes": [],
        "total_cost": 0.0,
        "successful": 0,
        "failed": 0,
    }

    # Load results for already completed genes
    for gene in genes:
        result_path = output_dir / f"{gene.gene_id}.json"
        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
            usage = result.get("usage", {})
            config_results["genes"].append(
                {
                    "gene_id": gene.gene_id,
                    "gene_symbol": gene.gene_symbol,
                    "papers_read": usage.get("papers_read", 0),
                    "cost_usd": usage.get("cost_usd", 0),
                    "error": result.get("error"),
                }
            )
            config_results["total_cost"] += usage.get("cost_usd", 0)
            if result.get("error"):
                config_results["failed"] += 1
            else:
                config_results["successful"] += 1

    if not remaining_genes:
        print("  All genes already completed")
        return config_results

    # Progress bar for parallel execution
    pbar = tqdm(total=len(remaining_genes), desc=f"papers={max_papers}", unit="gene")

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
        multi_agent=multi_agent,
        no_literature=no_literature,
        hide_terms=hide_terms,
        on_complete=on_complete,
    )

    pbar.close()

    # Process results
    for gene_result in results:
        usage = gene_result.result.get("usage", {}) if gene_result.result else {}
        config_results["genes"].append(
            {
                "gene_id": gene_result.gene_id,
                "gene_symbol": gene_result.gene_symbol,
                "papers_read": usage.get("papers_read", 0),
                "cost_usd": usage.get("cost_usd", 0),
                "error": gene_result.error,
            }
        )
        config_results["total_cost"] += usage.get("cost_usd", 0)
        if gene_result.error:
            config_results["failed"] += 1
        else:
            config_results["successful"] += 1

    return config_results


def run_scaling_experiment(
    paper_configs: list[int],
    genes: list[tuple[str, str, int]] | None = None,
    max_genes: int | None = None,
    model: str = "gpt-5-mini",
    verbose: bool = False,
    dry_run: bool = False,
    output_dir: Path | None = None,
    workers: int = 1,
    method: str = METHOD_SINGLE,
    hide_terms: bool = False,
) -> dict:
    """Run the full scaling experiment.

    Args:
        paper_configs: List of max_papers values to test
        genes: List of (gene_id, symbol, corpus_size) tuples, or None for defaults
        max_genes: Limit to first N genes (applied after genes selection)
        model: Model to use
        verbose: Show detailed logging
        dry_run: If True, just print what would be run
        output_dir: Base output directory (default: outputs/scaling/{method}-{commit_short})
        workers: Number of parallel workers (1 = sequential)
        method: Agent method to use (single, multi, pipeline)
        hide_terms: Enable specificity gap benchmark (hide GO terms)

    Returns:
        Summary dict with costs and results per configuration
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method: {method}. Must be one of {VALID_METHODS}")

    genes = genes or TEST_GENES
    if max_genes:
        genes = genes[:max_genes]

    # Load gene summaries
    csv_path = Path("data/genes_top100.csv")
    summaries = load_gene_summaries(csv_path)

    # Capture experiment metadata for reproducibility
    metadata = capture_metadata()
    commit_short = metadata["git"].get("commit_short", "unknown")

    # Default output dir includes method and commit hash for traceability
    if output_dir is None:
        output_dir = Path(f"outputs/scaling/{method}-{commit_short}")

    # Memorization method ignores paper configs - run once per gene
    if method == METHOD_MEMORIZATION:
        paper_configs = [0]  # Single run with 0 papers

    if dry_run:
        print("DRY RUN - Would execute:")
        print(f"  Code version: {format_version_string(metadata)}")
        print(f"  Method: {method}")
        print(f"  Output dir: {output_dir}")
        total_runs = len(paper_configs) * len(genes)
        print(f"  Paper configs: {paper_configs}")
        print(f"  Genes: {[g[1] for g in genes]}")
        print(f"  Total runs: {total_runs}")
        print(f"  Workers: {workers}")
        print(f"  Hide terms: {hide_terms}")
        # Cost estimates vary by method
        cost_per_paper = {"single": 0.10, "multi": 0.15, "pipeline": 0.03, "memorization": 0.01}
        est_cost = sum(
            max(max_papers, 1) * cost_per_paper.get(method, 0.10) for max_papers in paper_configs
        ) * len(genes)
        print(f"  Estimated cost: ${est_cost:.2f}")
        return {}

    print(f"Code version: {format_version_string(metadata)}")
    print(f"Method: {method}")
    print(f"Output dir: {output_dir}")
    print(f"Workers: {workers}")
    if hide_terms:
        print("Hide terms: ENABLED (specificity gap benchmark)")

    # Convert tuples to GeneTask objects
    gene_tasks = [
        GeneTask(gene_id=g[0], gene_symbol=g[1], summary=summaries.get(g[0], "")) for g in genes
    ]

    # Results tracking
    experiment_summary: dict = {
        "metadata": metadata,
        "model": model,
        "method": method,
        "paper_configs": paper_configs,
        "workers": workers,
        "hide_terms": hide_terms,
        "genes": [{"id": g[0], "symbol": g[1], "corpus_size": g[2]} for g in genes],
        "results_by_config": {},
    }

    # Determine flags for different methods
    multi_agent = method == METHOD_MULTI
    no_literature = method == METHOD_MEMORIZATION

    total_cost = 0.0

    for max_papers in paper_configs:
        config_output_dir = output_dir / f"papers_{max_papers}"

        print(f"\n{'=' * 60}")
        print(f"Running config: max_papers={max_papers}")
        print(f"Output dir: {config_output_dir}")
        print(f"{'=' * 60}")

        if workers > 1 and method != METHOD_PIPELINE:
            # Parallel execution (agentic methods - Pipeline handles parallelism internally)
            config_results = asyncio.run(
                run_config_parallel(
                    genes=gene_tasks,
                    max_papers=max_papers,
                    output_dir=config_output_dir,
                    summaries=summaries,
                    model=model,
                    verbose=verbose,
                    workers=workers,
                    multi_agent=multi_agent,
                    no_literature=no_literature,
                    hide_terms=hide_terms,
                )
            )
        else:
            # Sequential execution (or Pipeline which handles parallelism internally)
            config_results = {
                "max_papers": max_papers,
                "genes": [],
                "total_cost": 0.0,
                "successful": 0,
                "failed": 0,
            }

            pbar = tqdm(genes, desc=f"papers={max_papers}", unit="gene")
            for gene_id, gene_symbol, _corpus_size in pbar:
                pbar.set_postfix_str(f"{gene_symbol}")

                summary = summaries.get(gene_id, "")

                # Check if already completed (resume support)
                result_path = config_output_dir / f"{gene_id}.json"
                if result_path.exists():
                    with open(result_path) as f:
                        result = json.load(f)
                else:
                    try:
                        if method == METHOD_PIPELINE:
                            result = run_pipeline_single_gene(
                                gene_id=gene_id,
                                gene_symbol=gene_symbol,
                                summary=summary,
                                max_papers=max_papers,
                                output_dir=config_output_dir,
                                model=model,
                                verbose=verbose,
                                hide_terms=hide_terms,
                            )
                        else:
                            result = run_single_gene(
                                gene_id=gene_id,
                                gene_symbol=gene_symbol,
                                summary=summary,
                                max_papers=max_papers,
                                output_dir=config_output_dir,
                                model=model,
                                verbose=verbose,
                                multi_agent=multi_agent,
                                no_literature=no_literature,
                                hide_terms=hide_terms,
                            )
                    except Exception as e:
                        config_results["genes"].append(
                            {
                                "gene_id": gene_id,
                                "gene_symbol": gene_symbol,
                                "error": str(e),
                            }
                        )
                        config_results["failed"] += 1
                        continue

                usage = result.get("usage", {})
                cost = usage.get("cost_usd", 0)
                papers_read = usage.get("papers_read", 0)

                config_results["genes"].append(
                    {
                        "gene_id": gene_id,
                        "gene_symbol": gene_symbol,
                        "papers_read": papers_read,
                        "cost_usd": cost,
                        "error": result.get("error"),
                    }
                )

                config_results["total_cost"] += cost

                if result.get("error"):
                    config_results["failed"] += 1
                else:
                    config_results["successful"] += 1

            pbar.close()

        total_cost += config_results["total_cost"]

        # Save config summary
        config_output_dir.mkdir(parents=True, exist_ok=True)
        config_summary_path = config_output_dir / "run_summary.json"
        with open(config_summary_path, "w") as f:
            json.dump(config_results, f, indent=2)

        experiment_summary["results_by_config"][max_papers] = {
            "successful": config_results["successful"],
            "failed": config_results["failed"],
            "total_cost": config_results["total_cost"],
            "avg_papers_read": sum(g.get("papers_read", 0) for g in config_results["genes"])
            / len(genes),
        }

        print(f"Config complete: {config_results['successful']}/{len(genes)} successful")
        print(f"Config cost: ${config_results['total_cost']:.4f}")

    # Save experiment summary
    experiment_summary["total_cost"] = total_cost
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_summary_path = output_dir / "experiment_summary.json"
    with open(experiment_summary_path, "w") as f:
        json.dump(experiment_summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Summary saved to: {experiment_summary_path}")

    return experiment_summary


def main():
    parser = argparse.ArgumentParser(description="Run MCP agent scaling experiment")
    parser.add_argument(
        "--papers",
        type=int,
        nargs="+",
        help="Specific paper configs to run (default: all)",
    )
    parser.add_argument(
        "--genes",
        type=str,
        nargs="+",
        help="Specific gene IDs to test (default: 25 test genes)",
    )
    parser.add_argument(
        "--all-genes",
        action="store_true",
        help="Use all 100 genes from data/genes_top100.csv instead of 25 test genes",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        help="Limit to first N genes (from TEST_GENES or --all-genes list)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Base output directory (default: outputs/scaling/{commit_short})",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=VALID_METHODS,
        default=METHOD_SINGLE,
        help=f"Agent method to use: {', '.join(VALID_METHODS)} (default: single)",
    )
    parser.add_argument(
        "--hide-terms",
        action="store_true",
        help="Enable specificity gap benchmark (hide GO terms)",
    )
    # Keep --multi-agent for backwards compatibility
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="(Deprecated) Use --method multi instead",
    )

    args = parser.parse_args()

    # Handle backwards compatibility
    if args.multi_agent:
        print("Warning: --multi-agent is deprecated, use --method multi instead")
        args.method = METHOD_MULTI

    paper_configs = args.papers or PAPER_CONFIGS

    # Determine which genes to use
    genes = None
    if args.all_genes:
        csv_path = Path("data/genes_top100.csv")
        if not csv_path.exists():
            print(f"Error: Gene file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        genes = load_all_genes_from_csv(csv_path)
        print(f"Using all {len(genes)} genes from {csv_path}")
    elif args.genes:
        genes = [(g[0], g[1], g[2]) for g in TEST_GENES if g[0] in args.genes]
        if not genes:
            print(f"Error: No matching genes found for {args.genes}", file=sys.stderr)
            print(f"Available: {[g[0] for g in TEST_GENES]}", file=sys.stderr)
            sys.exit(1)

    # Apply --max-genes limit if specified
    if args.max_genes and genes:
        genes = genes[: args.max_genes]
        print(f"Limited to first {len(genes)} genes")

    run_scaling_experiment(
        paper_configs=paper_configs,
        genes=genes,
        max_genes=args.max_genes,
        model=args.model,
        verbose=args.verbose,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        workers=args.workers,
        method=args.method,
        hide_terms=args.hide_terms,
    )


if __name__ == "__main__":
    main()
