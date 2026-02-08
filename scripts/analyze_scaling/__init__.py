"""Scaling analysis package for evaluating agent performance across paper budgets.

This package provides tools for analyzing scaling experiment results,
including metrics collection, reporting, plotting, and I/O utilities.

Usage:
    # Run as module
    uv run python -m scripts.analyze_scaling --base-dir outputs/scaling --plot

    # Import functions directly
    from scripts.analyze_scaling import collect_all_metrics, generate_plots
"""

from .cli import PAPER_CONFIGS, main
from .collection import collect_all_metrics
from .config_metrics import ConfigMetrics
from .extraction import extract_metrics
from .io import save_results
from .plotting import generate_plots
from .reporting import find_knee_point, print_detailed_metrics, print_summary_table

__all__ = [
    # CLI
    "main",
    "PAPER_CONFIGS",
    # Data types
    "ConfigMetrics",
    # Collection
    "collect_all_metrics",
    "extract_metrics",
    # Reporting
    "print_summary_table",
    "print_detailed_metrics",
    "find_knee_point",
    # I/O
    "save_results",
    # Plotting
    "generate_plots",
]
