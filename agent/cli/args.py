"""Argument parser for the Single-Agent and Multi-Agent CLI.

This module provides the argument parser used by the main CLI entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the MCP agent CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Run Drosophila gene annotation agent (Single-Agent or Multi-Agent mode)"
    )

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

    # Budget controls
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum LLM turns (default: 50)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum papers to read (default: 10)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=1.0,
        help="Maximum cost in USD (default: 1.0)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-5-mini)",
    )

    # Parallel execution
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers for benchmark mode (default: 1 = sequential)",
    )

    # Output
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

    # Logging and tracing
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed logging output",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Directory to save trace files for debugging",
    )

    # Specificity gap benchmark
    parser.add_argument(
        "--hide-terms",
        action="store_true",
        help="Enable GO term hiding for specificity gap benchmark (116 terms hidden)",
    )

    # Multi-agent mode for context isolation
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Enable Multi-Agent mode with hierarchical delegation (bounded context)",
    )

    # Memorization baseline (no literature)
    parser.add_argument(
        "--no-literature",
        action="store_true",
        help="Memorization baseline: no literature access (tests LLM prior knowledge)",
    )

    return parser
