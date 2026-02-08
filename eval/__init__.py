"""Evaluation framework for agent annotation quality assessment.

Simplified structure (5 files):
- evaluator.py: Main CLI + metrics + orchestration + NL resolution
- task1_go.py: GO evaluation + Wang similarity
- task2_expression.py: Expression evaluation + anatomy similarity
- task3_synonyms.py: Synonym evaluation
- data_loader.py: File I/O utilities
"""

from .data_loader import (
    DataLoadError,
    get_all_gene_ids,
    load_agent_output,
    load_ground_truth,
)
from .evaluator import (
    BatchResult,
    EvaluationResult,
    ReferenceCoverageResult,
    evaluate_batch,
    evaluate_gene,
    evaluate_reference_coverage,
)
from .task1_go import Task1Result, evaluate_go
from .task2_expression import Task2Result, evaluate_expression
from .task3_synonyms import Task3Result, evaluate_synonyms

__all__ = [
    # Main API
    "evaluate_gene",
    "evaluate_batch",
    "EvaluationResult",
    "BatchResult",
    # Task evaluation
    "evaluate_go",
    "evaluate_expression",
    "evaluate_synonyms",
    "Task1Result",
    "Task2Result",
    "Task3Result",
    # Data loading
    "load_ground_truth",
    "load_agent_output",
    "get_all_gene_ids",
    "DataLoadError",
    # Reference coverage
    "evaluate_reference_coverage",
    "ReferenceCoverageResult",
]
