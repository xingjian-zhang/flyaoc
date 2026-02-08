#!/usr/bin/env python3
"""MCP server for output validation.

This server provides one tool:
- submit_annotations: Validate and submit gene annotations

Run as:
    python -m agent.mcp_servers.validation_server
"""

import json

from mcp.server.fastmcp import FastMCP

from ..core.validation import submit_annotations_core

# Create MCP server
mcp = FastMCP("Drosophila Validation Server")


@mcp.tool()
def submit_annotations(annotations: dict) -> str:
    """Submit final gene annotations for validation.

    This tool validates your annotations against the benchmark schema.
    Call this when you have completed extracting all annotations for a gene.

    Args:
        annotations: Complete output dictionary containing:
            - gene_id: FlyBase gene ID (e.g., "FBgn0000014")
            - gene_symbol: Gene symbol (e.g., "abd-A")
            - task1_function: List of GO annotations, each with:
                - go_id: GO term ID (e.g., "GO:0003700")
                - qualifier: Relationship type (e.g., "enables", "involved_in")
                - aspect: "P" (Process), "F" (Function), or "C" (Component)
                - is_negated: Boolean (optional, default false)
                - evidence: Dict with pmcid and text (optional)
            - task2_expression: List of expression records, each with:
                - expression_type: "polypeptide" or "transcript"
                - anatomy_id: FBbt term ID (optional)
                - stage_id: FBdv term ID (optional)
                - evidence: Dict with pmcid and text (optional)
            - task3_synonyms: Dict with:
                - fullname_synonyms: List of full name synonyms
                - symbol_synonyms: List of symbol synonyms

    Returns:
        JSON string with validation result containing:
        - valid: Boolean indicating if output passes validation
        - errors: List of validation error messages (empty if valid)
        - annotation_count: Summary of annotations submitted
    """
    result = submit_annotations_core(annotations)
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
