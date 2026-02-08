"""CLI package for the MCP backbone gene annotation agent.

This package provides the command-line interface for running the MCP agent
on single genes or benchmark sets.

Usage:
    # As a module
    uv run python -m agent.cli --gene-id FBgn0000014 --gene-symbol abd-A

    # Import functions
    from agent.cli import main, run_single_gene, run_benchmark
"""

from .args import create_parser
from .benchmark import run_benchmark, run_benchmark_parallel, run_from_json
from .execution import run_single_gene, run_single_gene_quiet
from .main import main

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
