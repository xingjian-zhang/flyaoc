"""Main entry point for the Single-Agent and Multi-Agent CLI.

This module provides the main function that orchestrates the CLI behavior
based on parsed arguments.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from ..agentic import BudgetConfig  # noqa: E402
from .args import create_parser  # noqa: E402
from .benchmark import run_benchmark, run_from_json  # noqa: E402
from .execution import run_single_gene  # noqa: E402


def main() -> None:
    """Main entry point for the Single-Agent and Multi-Agent CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Create budget config
    budget = BudgetConfig(
        max_turns=args.max_turns,
        max_papers=args.max_papers,
        max_cost_usd=args.max_cost,
    )

    # Default directories based on mode
    multi_agent = getattr(args, "multi_agent", False)
    default_output_dir = Path("outputs/multi_agent" if multi_agent else "outputs/single_agent")
    trace_dir = args.trace_dir  # None by default; trace is embedded in output JSON

    # Handle single gene mode
    if args.gene_id:
        if not args.gene_symbol:
            parser.error("--gene-symbol is required with --gene-id")

        result = run_single_gene(
            args.gene_id,
            args.gene_symbol,
            args.summary,
            budget,
            args.model,
            args.verbose,
            trace_dir,
            args.hide_terms,
            args.multi_agent,
            getattr(args, "no_literature", False),
        )

        default_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output or default_output_dir / f"{args.gene_id}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Output saved to: {output_path}")
        if args.verbose:
            print("\nOutput:")
            print(json.dumps(result.get("output"), indent=2))

    # Handle quick benchmark mode (-n N)
    elif args.num_genes:
        csv_path = Path("data/genes_top100.csv")
        if not csv_path.exists():
            parser.error(f"Benchmark file not found: {csv_path}")
        output_dir = args.output or default_output_dir
        run_benchmark(
            csv_path,
            output_dir,
            budget,
            args.model,
            args.num_genes,
            args.verbose,
            trace_dir,
            args.hide_terms,
            args.multi_agent,
            args.workers,
            getattr(args, "no_literature", False),
        )

    # Handle JSON input mode
    elif args.input:
        result = run_from_json(
            args.input,
            budget,
            args.model,
            args.verbose,
            trace_dir,
            args.hide_terms,
            args.multi_agent,
        )

        default_output_dir.mkdir(parents=True, exist_ok=True)
        gene_id = result.get("output", result).get("gene_id", "unknown")
        output_path = args.output or default_output_dir / f"{gene_id}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Output saved to: {output_path}")
        if args.verbose:
            print("\nOutput:")
            print(json.dumps(result.get("output"), indent=2))

    # Handle benchmark mode with custom CSV
    elif args.benchmark:
        output_dir = args.output or default_output_dir
        run_benchmark(
            args.benchmark,
            output_dir,
            budget,
            args.model,
            args.limit,
            args.verbose,
            trace_dir,
            args.hide_terms,
            args.multi_agent,
            args.workers,
        )


if __name__ == "__main__":
    main()
