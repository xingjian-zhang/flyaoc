#!/usr/bin/env python3
"""In-depth analysis of scaling experiment results for academic publication.

This module is a thin wrapper that imports from the scripts.deep_analysis package.
The actual implementation has been split into focused modules:

- dataclasses.py: GeneTrajectory, FalsePositive, TruePositive, ToolCall, PaperRead
- trajectories.py: compute_gene_trajectories, analyze_scaling_categories
- go_analysis.py: analyze_go_aspects, extract/categorize false positives
- tool_analysis.py: extract_detailed_tool_usage, analyze_paper_selection
- statistics.py: perform_statistical_tests, deep_task_comparison
- case_studies.py: generate_detailed_case_study
- reporting.py: generate_academic_report
- plotting.py: generate_publication_plots
- cli.py: main() entry point

Usage:
    uv run python -m scripts.deep_analysis --base-dir outputs/scaling/d37b178 --plot -v
"""

# Re-export everything for backwards compatibility
from eval.data_loader import (
    load_agent_output_scaling as load_agent_output,
)

# Data loading - re-export aliases for backwards compatibility
from eval.data_loader import (
    load_all_scaling_eval_results as load_all_eval_results,
)
from eval.data_loader import (
    load_go_ontology,
)
from eval.data_loader import (
    load_ground_truth as _load_ground_truth_base,
)
from eval.data_loader import (
    load_scaling_eval_results as load_eval_results,
)
from eval.data_loader import (
    load_scaling_run_summary as load_run_summary,
)
from eval.data_loader import (
    load_scaling_trace as load_trace,
)
from scripts.deep_analysis.case_studies import generate_detailed_case_study
from scripts.deep_analysis.cli import main
from scripts.deep_analysis.dataclasses import (
    FalsePositive,
    GeneTrajectory,
    PaperRead,
    ToolCall,
    TruePositive,
)
from scripts.deep_analysis.go_analysis import (
    analyze_go_aspects,
    analyze_per_paper_yield,
    categorize_false_positives_detailed,
    extract_false_positives_detailed,
)
from scripts.deep_analysis.plotting import generate_publication_plots
from scripts.deep_analysis.reporting import (
    PAPER_CONFIGS,
    generate_academic_report,
)
from scripts.deep_analysis.statistics import (
    deep_task_comparison,
    perform_statistical_tests,
)
from scripts.deep_analysis.tool_analysis import (
    analyze_paper_selection_effectiveness,
    extract_detailed_tool_usage,
)
from scripts.deep_analysis.trajectories import (
    analyze_scaling_categories,
    compute_gene_trajectories,
)


def load_ground_truth(path: str = "data/ground_truth_top100.jsonl") -> dict[str, dict]:
    """Load ground truth indexed by gene_id (wrapper for backwards compatibility)."""
    return _load_ground_truth_base(ground_truth_path=path, as_dict=True)  # type: ignore[return-value]


__all__ = [
    # Data classes
    "GeneTrajectory",
    "FalsePositive",
    "TruePositive",
    "ToolCall",
    "PaperRead",
    # Constants
    "PAPER_CONFIGS",
    # Analysis functions
    "compute_gene_trajectories",
    "analyze_scaling_categories",
    "analyze_per_paper_yield",
    "analyze_go_aspects",
    "extract_false_positives_detailed",
    "categorize_false_positives_detailed",
    "extract_detailed_tool_usage",
    "analyze_paper_selection_effectiveness",
    "perform_statistical_tests",
    "deep_task_comparison",
    "generate_detailed_case_study",
    "generate_academic_report",
    "generate_publication_plots",
    # Data loading
    "load_eval_results",
    "load_run_summary",
    "load_trace",
    "load_agent_output",
    "load_ground_truth",
    "load_all_eval_results",
    "load_go_ontology",
    # Entry point
    "main",
]

if __name__ == "__main__":
    main()
