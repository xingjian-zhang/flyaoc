#!/usr/bin/env python3
"""Analyze scaling experiment results and generate Pareto frontier plots.

This module is a thin wrapper that imports from the scripts.analyze_scaling package.
The actual implementation has been split into focused modules:

- config_metrics.py: ConfigMetrics dataclass
- extraction.py: extract_metrics function
- collection.py: collect_all_metrics function
- reporting.py: print_summary_table, print_detailed_metrics, find_knee_point
- plotting.py: generate_plots function
- io.py: save_results function
- cli.py: main() entry point

Usage:
    uv run python -m scripts.analyze_scaling --base-dir outputs/scaling --plot
"""

# Re-export everything for backwards compatibility
# Data loading - re-export for backwards compatibility
from eval.data_loader import load_eval_results, load_run_summary
from scripts.analyze_scaling.cli import PAPER_CONFIGS, main
from scripts.analyze_scaling.collection import collect_all_metrics
from scripts.analyze_scaling.config_metrics import ConfigMetrics
from scripts.analyze_scaling.extraction import extract_metrics
from scripts.analyze_scaling.io import save_results
from scripts.analyze_scaling.plotting import generate_plots
from scripts.analyze_scaling.reporting import (
    find_knee_point,
    print_detailed_metrics,
    print_summary_table,
)

__all__ = [
    # Constants
    "PAPER_CONFIGS",
    # Data class
    "ConfigMetrics",
    # Functions
    "extract_metrics",
    "collect_all_metrics",
    "print_summary_table",
    "print_detailed_metrics",
    "find_knee_point",
    "generate_plots",
    "save_results",
    # Data loading
    "load_eval_results",
    "load_run_summary",
    # Entry point
    "main",
]

if __name__ == "__main__":
    main()
