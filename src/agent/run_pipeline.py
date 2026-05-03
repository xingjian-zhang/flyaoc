#!/usr/bin/env python3
"""Entry point for running the Pipeline agent (fixed parallel DAG).

Usage:
    # Run on a single gene
    python -m agent.run_pipeline --gene-id FBgn0000014 --gene-symbol abd-A

    # Run benchmark on first N genes (with progress bar)
    python -m agent.run_pipeline -n 5

    # Run on benchmark genes from CSV
    python -m agent.run_pipeline --benchmark data/genes_top100.csv

    # Run on a JSON input file
    python -m agent.run_pipeline --input schemas/examples/example_input_abd-A.json
"""

import argparse
import asyncio
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv(override=True)

from .config import DEFAULT_MAX_PAPERS, DEFAULT_MODEL  # noqa: E402
from .core.metadata import capture_metadata, format_version_string  # noqa: E402
from .pipeline import run_agent  # noqa: E402


async def run_single_gene(
    gene_id: str,
    gene_symbol: str,
    summary: str = "",
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    hide_terms: bool = False,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> dict:
    """Run the agent on a single gene and return results."""
    if not verbose:
        print(f"Processing gene: {gene_symbol} ({gene_id})")
        if hide_terms:
            print("  (hidden terms mode enabled)")
    result = await run_agent(
        gene_id,
        gene_symbol,
        summary,
        model=model,
        verbose=verbose,
        hide_terms=hide_terms,
        max_papers=max_papers,
    )
    usage = result.get("usage", {})
    if not verbose:
        print(f"  Cost: ${usage.get('total_cost', 0):.4f} ({usage.get('total_tokens', 0)} tokens)")
    return result


async def run_from_json(input_path: Path) -> dict:
    """Run the agent from a JSON input file."""
    with open(input_path) as f:
        data = json.load(f)

    return await run_single_gene(
        gene_id=data["gene_id"],
        gene_symbol=data["gene_symbol"],
        summary=data.get("summary", ""),
    )


async def run_benchmark(
    csv_path: Path,
    output_dir: Path,
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    hide_terms: bool = False,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> None:
    """Run the agent on benchmark genes from CSV.

    Args:
        csv_path: Path to genes CSV (columns: FBgn_ID, Gene_Symbol, Summary)
        output_dir: Directory to write output JSON files
        limit: Maximum number of genes to process (for testing)
        model: Model to use for inference
        hide_terms: Enable specificity gap benchmark (hide GO terms)
        max_papers: Maximum papers to process per gene
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        genes = list(reader)

    if limit:
        genes = genes[:limit]

    # Capture experiment metadata for reproducibility
    metadata = capture_metadata()
    print(f"Code version: {format_version_string(metadata)}")

    results_summary: dict = {
        "metadata": metadata,
        "total": len(genes),
        "successful": 0,
        "failed": 0,
        "total_cost": 0.0,
        "total_tokens": 0,
        "errors": [],
    }

    # Check for already completed genes (resume support)
    completed_ids = set()
    for gene in genes:
        output_path = output_dir / f"{gene['FBgn_ID']}.json"
        if output_path.exists():
            completed_ids.add(gene["FBgn_ID"])
            # Load existing result for summary stats
            with open(output_path) as f:
                result = json.load(f)
            usage = result.get("usage", {})
            results_summary["total_cost"] += usage.get("total_cost", 0)
            results_summary["total_tokens"] += usage.get("total_tokens", 0)
            if "error" in result:
                results_summary["failed"] += 1
            else:
                results_summary["successful"] += 1

    if completed_ids:
        print(
            f"Resuming: {len(completed_ids)} genes already completed, {len(genes) - len(completed_ids)} remaining"
        )

    # Progress bar
    pbar = tqdm(genes, desc="Processing genes", unit="gene")
    for gene in pbar:
        gene_id = gene["FBgn_ID"]
        gene_symbol = gene["Gene_Symbol"]
        gene_summary = gene.get("Summary", "")

        pbar.set_postfix_str(f"{gene_symbol}")

        # Skip already completed genes
        if gene_id in completed_ids:
            continue

        try:
            result = await run_single_gene(
                gene_id,
                gene_symbol,
                gene_summary,
                model=model,
                hide_terms=hide_terms,
                max_papers=max_papers,
            )

            # Track costs
            usage = result.get("usage", {})
            results_summary["total_cost"] += usage.get("total_cost", 0)
            results_summary["total_tokens"] += usage.get("total_tokens", 0)

            # Save output
            output_path = output_dir / f"{gene_id}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            if "error" in result:
                results_summary["failed"] += 1
                results_summary["errors"].append({"gene_id": gene_id, "error": result["error"]})
            else:
                results_summary["successful"] += 1

        except Exception as e:
            results_summary["failed"] += 1
            results_summary["errors"].append({"gene_id": gene_id, "error": str(e)})

    pbar.close()

    # Save summary
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nCompleted: {results_summary['successful']}/{results_summary['total']} successful")
    print(f"Total cost: ${results_summary['total_cost']:.4f}")
    print(f"Results saved to: {output_dir}")


async def async_main():
    """Async main entry point."""
    parser = argparse.ArgumentParser(description="Run the Drosophila gene annotation agent")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--gene-id",
        help="FlyBase gene ID (e.g., FBgn0000014)",
    )
    input_group.add_argument(
        "-n",
        "--num-genes",
        type=int,
        metavar="N",
        help="Run benchmark on first N genes from data/genes_top100.csv",
    )
    input_group.add_argument(
        "--input",
        type=Path,
        help="JSON input file path",
    )
    input_group.add_argument(
        "--benchmark",
        type=Path,
        help="CSV file with benchmark genes (use -n for simpler interface)",
    )

    # Additional options
    parser.add_argument(
        "--gene-symbol",
        help="Gene symbol (required with --gene-id)",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Gene summary text",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=DEFAULT_MAX_PAPERS,
        help=f"Maximum papers to process per gene (default: {DEFAULT_MAX_PAPERS})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file/directory path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of genes to process (for --benchmark mode)",
    )
    parser.add_argument(
        "--hide-terms",
        action="store_true",
        help="Enable specificity gap benchmark (hide GO terms from ontology)",
    )

    args = parser.parse_args()

    # Default output directory
    default_output_dir = Path("outputs/pipeline")

    # Handle single gene mode
    if args.gene_id:
        if not args.gene_symbol:
            parser.error("--gene-symbol is required with --gene-id")

        result = await run_single_gene(
            args.gene_id,
            args.gene_symbol,
            args.summary,
            model=args.model,
            verbose=args.verbose,
            hide_terms=args.hide_terms,
            max_papers=args.max_papers,
        )

        default_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output or default_output_dir / f"{args.gene_id}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Output saved to: {output_path}")
        if args.verbose:
            print(json.dumps(result, indent=2))

    # Handle quick benchmark mode (-n N)
    elif args.num_genes:
        csv_path = Path("data/genes_top100.csv")
        if not csv_path.exists():
            parser.error(f"Benchmark file not found: {csv_path}")
        output_dir = args.output or default_output_dir
        await run_benchmark(
            csv_path,
            output_dir,
            limit=args.num_genes,
            model=args.model,
            hide_terms=args.hide_terms,
            max_papers=args.max_papers,
        )

    # Handle JSON input mode
    elif args.input:
        result = await run_from_json(args.input)

        default_output_dir.mkdir(parents=True, exist_ok=True)
        gene_id = result.get("output", result).get("gene_id", "unknown")
        output_path = args.output or default_output_dir / f"{gene_id}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Output saved to: {output_path}")
        if args.verbose:
            print(json.dumps(result, indent=2))

    # Handle benchmark mode with custom CSV
    elif args.benchmark:
        output_dir = args.output or default_output_dir
        await run_benchmark(
            args.benchmark,
            output_dir,
            args.limit,
            model=args.model,
            hide_terms=args.hide_terms,
            max_papers=args.max_papers,
        )


def main():
    """Main entry point - runs the async main function."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
