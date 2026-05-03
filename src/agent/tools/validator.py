"""Validation tool for agent output annotations.

This module provides a LangChain tool wrapper around the core validation functionality.
"""

from langchain_core.tools import tool

from ..core.validation import ValidationError, submit_annotations_core, validate_output

# Re-export for backwards compatibility
__all__ = ["ValidationError", "validate_output", "submit_annotations", "ValidationTool"]


@tool
def submit_annotations(annotations: dict) -> dict:
    """Submit final gene annotations for validation.

    This tool validates your annotations against the benchmark schema.
    Call this when you have completed extracting all annotations for a gene.

    Args:
        annotations: Complete output dictionary containing:
            - gene_id: FlyBase gene ID (e.g., "FBgn0000014")
            - gene_symbol: Gene symbol (e.g., "abd-A")
            - task1_function: List of GO annotations
            - task2_expression: List of expression records
            - task3_synonyms: Dict with fullname_synonyms and symbol_synonyms

    Returns:
        Dictionary with:
        - valid: Boolean indicating if output passes validation
        - errors: List of validation error messages (empty if valid)
        - annotation_count: Summary of annotations submitted
    """
    return submit_annotations_core(annotations)


# Alias for LangChain integration
ValidationTool = submit_annotations
