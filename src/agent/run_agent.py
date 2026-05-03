#!/usr/bin/env python3
"""Entry point for the Single-Agent and Multi-Agent methods.

Single-Agent: One agent with growing context (default)
Multi-Agent: Hierarchical delegation with bounded context (--multi-agent flag)

This module is a thin wrapper that imports from the agent.cli package.
The actual implementation has been split into focused modules:

- args.py: Argument parser setup
- execution.py: run_single_gene, run_single_gene_quiet
- benchmark.py: run_benchmark, run_benchmark_parallel, run_from_json
- main.py: main() entry point

Usage:
    # Single-Agent mode (default)
    python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A

    # Multi-Agent mode (hierarchical delegation)
    python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A --multi-agent

    # Run benchmark on first N genes (with progress bar)
    python -m agent.run_agent -n 5

    # Run benchmark with parallel workers
    python -m agent.run_agent -n 10 --workers 5

    # Run on a single gene with budget constraints
    python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A \
        --max-turns 30 --max-papers 5 --max-cost 0.50

    # Run on benchmark genes from CSV
    python -m agent.run_agent --benchmark data/genes_top100.csv --limit 5

    # Run with a JSON input file
    python -m agent.run_agent --input schemas/examples/example_input_abd-A.json
"""

# Re-export everything for backwards compatibility
from agent.cli import (
    create_parser,
    main,
    run_benchmark,
    run_benchmark_parallel,
    run_from_json,
    run_single_gene,
    run_single_gene_quiet,
)

__all__ = [
    # Entry point
    "main",
    # Argument parsing
    "create_parser",
    # Execution functions
    "run_single_gene",
    "run_single_gene_quiet",
    # Benchmark functions
    "run_benchmark",
    "run_benchmark_parallel",
    "run_from_json",
]

if __name__ == "__main__":
    main()
