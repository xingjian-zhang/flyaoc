"""LangGraph workflow definition for the gene annotation agent."""

import os
from typing import Any, Literal

from langchain_community.callbacks import get_openai_callback
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..config import DEFAULT_MAX_PAPERS, DEFAULT_MODEL
from .nodes import (
    compile_output,
    extract_all,
    read_papers,
    resolve_ontology,
    search_literature,
)
from .state import AgentState
from .tracing import logger, setup_logging


def should_continue_after_search(state: AgentState) -> Literal["read_papers", "end"]:
    """Determine if we should continue after search.

    If no papers found, go to end. Otherwise, continue to read papers.
    """
    if not state.get("relevant_papers"):
        return "end"
    return "read_papers"


def should_continue_after_read(state: AgentState) -> Literal["extract_all", "end"]:
    """Determine if we should continue after reading papers.

    If no paper texts retrieved, go to end. Otherwise, continue to extraction.
    """
    if not state.get("paper_texts"):
        return "end"
    return "extract_all"


def create_agent_graph() -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    """Create the LangGraph workflow for gene annotation.

    Flow:
        search_lit → read_papers → extract_all → resolve → compile

    Sequential flow with unified 1-pass extraction and deferred ontology resolution.
    Early exit to END if no papers found or no texts retrieved.
    """
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_lit", search_literature)
    workflow.add_node("read_papers", read_papers)
    workflow.add_node("extract_all", extract_all)
    workflow.add_node("resolve", resolve_ontology)
    workflow.add_node("compile", compile_output)

    # Set entry point
    workflow.set_entry_point("search_lit")

    # Add edges from search
    workflow.add_conditional_edges(
        "search_lit",
        should_continue_after_search,
        {
            "read_papers": "read_papers",
            "end": END,
        },
    )

    # Add edges from read_papers
    workflow.add_conditional_edges(
        "read_papers",
        should_continue_after_read,
        {
            "extract_all": "extract_all",
            "end": END,
        },
    )

    # Sequential edges: extract_all → resolve → compile → END
    workflow.add_edge("extract_all", "resolve")
    workflow.add_edge("resolve", "compile")
    workflow.add_edge("compile", END)

    return workflow.compile()


async def run_agent(
    gene_id: str,
    gene_symbol: str,
    summary: str = "",
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    hide_terms: bool = False,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> dict:
    """Run the agent on a single gene.

    Args:
        gene_id: FlyBase gene ID (e.g., "FBgn0000014")
        gene_symbol: Gene symbol (e.g., "abd-A")
        summary: Optional gene summary
        model: OpenAI model to use (default: gpt-5-mini)
        verbose: Enable verbose logging
        hide_terms: Enable specificity gap benchmark (hide GO terms)
        max_papers: Maximum papers to process (default: 8)

    Returns:
        Dict with 'output' (or 'error') and 'usage' containing cost info
    """
    # Set up logging
    setup_logging(verbose=verbose)
    logger.info(f"[START] Processing {gene_symbol} ({gene_id}) with model={model}")

    # Set environment variable for hidden terms (checked by ontology core)
    if hide_terms:
        os.environ["HIDE_GO_TERMS"] = "1"
        logger.info("[START] Hidden terms mode enabled (specificity gap benchmark)")
    elif "HIDE_GO_TERMS" in os.environ:
        del os.environ["HIDE_GO_TERMS"]

    graph = create_agent_graph()

    initial_state = {
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "summary": summary,
        "config": {
            "model": model,
            "verbose": verbose,
            "hide_terms": hide_terms,
            "max_papers": max_papers,
        },
        "relevant_papers": [],
        "paper_texts": {},
        # Text-based annotations (before resolution)
        "text_function_annotations": [],
        "text_expression_annotations": [],
        "synonyms_found": [],
        # Resolved annotations (after ontology lookup)
        "go_annotations": [],
        "expression_records": [],
        "messages": [],
        "final_output": None,
        "error": None,
    }

    # Run the graph with cost tracking (async invocation for parallel nodes)
    with get_openai_callback() as cb:
        final_state = await graph.ainvoke(initial_state)  # type: ignore[arg-type]

    usage = {
        "total_cost": cb.total_cost,
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
    }

    logger.info(f"[DONE] Cost: ${cb.total_cost:.4f} ({cb.total_tokens} tokens)")

    if final_state.get("error"):
        logger.warning(f"[DONE] Completed with errors: {final_state['error']}")
        return {
            "error": final_state["error"],
            "partial_output": final_state.get("final_output"),
            "usage": usage,
        }

    output = final_state.get("final_output", {})
    t1 = len(output.get("task1_function", [])) if output else 0
    t2 = len(output.get("task2_expression", [])) if output else 0
    logger.info(f"[DONE] Success: {t1} GO, {t2} expression annotations")

    return {
        "output": output,
        "usage": usage,
    }
