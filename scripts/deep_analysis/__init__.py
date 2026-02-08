"""Deep analysis subpackage for scaling experiments.

This package provides comprehensive behavioral analysis of scaling experiments.

Usage:
    # As a module
    uv run python -m scripts.deep_analysis --base-dir outputs/scaling/d37b178 --plot -v

    # Import functions
    from scripts.deep_analysis import compute_gene_trajectories, GeneTrajectory
"""

from .case_studies import generate_detailed_case_study
from .cli import main
from .dataclasses import (
    FalsePositive,
    GeneTrajectory,
    PaperRead,
    ToolCall,
    TruePositive,
)
from .go_analysis import (
    analyze_go_aspects,
    analyze_per_paper_yield,
    categorize_false_positives_detailed,
    extract_false_positives_detailed,
)
from .plotting import generate_publication_plots
from .reporting import PAPER_CONFIGS, generate_academic_report
from .statistics import deep_task_comparison, perform_statistical_tests
from .tool_analysis import (
    analyze_paper_selection_effectiveness,
    extract_detailed_tool_usage,
)
from .trajectories import analyze_scaling_categories, compute_gene_trajectories

__all__ = [
    # Entry point
    "main",
    # Constants
    "PAPER_CONFIGS",
    # Data classes
    "GeneTrajectory",
    "FalsePositive",
    "TruePositive",
    "ToolCall",
    "PaperRead",
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
]
