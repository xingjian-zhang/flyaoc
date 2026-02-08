"""Graph node implementations for the gene annotation workflow."""

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from ..config import DEFAULT_MAX_PAPERS, DEFAULT_MODEL, DEFAULT_TEMPERATURE
from ..core.prompt_templates import HIDDEN_TERMS_ADDENDUM, SYSTEM_PROMPT
from ..tools import (
    get_paper_text,
    search_anatomy_terms,
    search_corpus,
    search_go_terms,
    search_stage_terms,
    submit_annotations,
)
from .prompts import (
    EXPRESSION_EXTRACTION_PROMPT,
    FUNCTION_EXTRACTION_PROMPT,
    SYNONYM_EXTRACTION_PROMPT,
    UNIFIED_EXTRACTION_PROMPT,
)
from .state import AgentState, TextExpressionAnnotation, TextFunctionAnnotation
from .tracing import logger


def get_llm(model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> ChatOpenAI:
    """Get the LLM instance."""
    return ChatOpenAI(model=model, temperature=temperature)


def get_llm_with_tools(
    model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE
) -> Runnable:
    """Get the LLM instance with tools bound."""
    llm = ChatOpenAI(model=model, temperature=temperature)
    tools = [
        search_corpus,
        get_paper_text,
        search_go_terms,
        search_anatomy_terms,
        search_stage_terms,
        submit_annotations,
    ]
    return llm.bind_tools(tools)


FINAL_ANSWER_PROMPT = """Based on the information gathered from the ontology searches above,
please provide your final answer as a JSON array. Do not make any more tool calls.
Return ONLY the JSON array with your findings, or an empty array [] if no relevant data was found."""


def _get_response_text(response: AIMessage) -> str:
    """Extract text content from an AIMessage response."""
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        # Content can be a list of content blocks
        text_parts = []
        for block in response.content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
        return " ".join(text_parts)
    return ""


def search_literature(state: AgentState) -> dict[str, Any]:
    """Search the corpus for papers mentioning the gene.

    This node uses the search_corpus tool to find relevant papers.
    """
    gene_symbol = state["gene_symbol"]
    gene_id = state["gene_id"]
    config = state.get("config", {})
    max_papers = config.get("max_papers", DEFAULT_MAX_PAPERS)

    logger.info(f"[SEARCH] Searching corpus for {gene_symbol} ({gene_id})")

    # Search with gene symbol
    results = search_corpus.invoke({"query": gene_symbol, "limit": 20})  # type: ignore[attr-defined]

    # Also search with gene ID for broader coverage
    id_results = search_corpus.invoke({"query": gene_id, "limit": 10})  # type: ignore[attr-defined]

    # Combine and deduplicate
    seen = set()
    relevant_papers = []
    for paper in results + id_results:
        pmcid = paper["pmcid"]
        if pmcid not in seen:
            seen.add(pmcid)
            relevant_papers.append(pmcid)

    # Limit to max_papers
    relevant_papers = relevant_papers[:max_papers]

    logger.info(f"[SEARCH] Found {len(relevant_papers)} papers (max: {max_papers})")

    return {
        "relevant_papers": relevant_papers,
        "messages": [
            HumanMessage(content=f"Found {len(relevant_papers)} papers for gene {gene_symbol}")
        ],
    }


def read_papers(state: AgentState) -> dict[str, Any]:
    """Read the full text of relevant papers.

    This node fetches full text for each paper found in the search step.
    """
    relevant_papers = state.get("relevant_papers", [])
    paper_texts = state.get("paper_texts", {})

    logger.info(f"[READ] Reading {len(relevant_papers)} papers")

    for pmcid in relevant_papers:
        if pmcid not in paper_texts:
            result = get_paper_text.invoke({"pmcid": pmcid})  # type: ignore[attr-defined]
            if "error" not in result:
                paper_texts[pmcid] = result
                logger.debug(f"[READ] Retrieved {pmcid}")

    logger.info(f"[READ] Retrieved {len(paper_texts)} paper texts")

    return {
        "paper_texts": paper_texts,
        "messages": [HumanMessage(content=f"Retrieved {len(paper_texts)} paper texts")],
    }


def _extract_json_from_response(text: str) -> list[dict]:
    """Extract JSON objects from LLM response text."""
    results = []

    # First try to parse the entire text as JSON
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try to find JSON code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_blocks = re.findall(code_block_pattern, text)
    for block in code_blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, list):
                results.extend([item for item in parsed if isinstance(item, dict)])
            elif isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            pass

    if results:
        return results

    # Find JSON by bracket matching (handles nested objects)
    def find_json_objects(s: str) -> list[str]:
        objects = []
        i = 0
        while i < len(s):
            if s[i] == "{":
                depth = 1
                start = i
                i += 1
                while i < len(s) and depth > 0:
                    if s[i] == "{":
                        depth += 1
                    elif s[i] == "}":
                        depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(s[start:i])
            elif s[i] == "[":
                depth = 1
                start = i
                i += 1
                while i < len(s) and depth > 0:
                    if s[i] == "[":
                        depth += 1
                    elif s[i] == "]":
                        depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(s[start:i])
            else:
                i += 1
        return objects

    for candidate in find_json_objects(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                results.extend([item for item in parsed if isinstance(item, dict)])
            elif isinstance(parsed, dict) and parsed not in results:
                results.append(parsed)
        except json.JSONDecodeError:
            pass

    return results


def _get_paper_text_for_extraction(paper: dict, max_chars: int = 8000) -> str:
    """Get relevant text from a paper for extraction, limited to max_chars."""
    parts = []

    if paper.get("abstract"):
        parts.append(f"ABSTRACT:\n{paper['abstract']}")

    sections = paper.get("sections", {})
    for section_name in ["RESULTS", "INTRO", "DISCUSS"]:
        if section_name in sections and sections[section_name]:
            section_text = sections[section_name]
            if isinstance(section_text, list):
                section_text = "\n".join(section_text)
            parts.append(f"{section_name}:\n{section_text}")

    full_text = "\n\n".join(parts)

    # Truncate if too long
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n... [truncated]"

    return full_text


async def extract_functions(state: AgentState) -> dict[str, Any]:
    """Extract GO function annotations from papers IN PARALLEL.

    This node uses the LLM to extract GO terms from paper text,
    using the ontology search tools to find correct IDs.
    Papers are processed concurrently using asyncio.gather().
    """
    gene_symbol = state["gene_symbol"]
    gene_id = state["gene_id"]
    paper_texts = state.get("paper_texts", {})
    config = state.get("config", {})
    model = config.get("model", DEFAULT_MODEL)

    logger.info(
        f"[EXTRACT_FN] Extracting GO annotations from {len(paper_texts)} papers (model={model})"
    )

    async def process_paper(pmcid: str, paper: dict) -> list[dict]:
        """Process a single paper asynchronously."""
        text = _get_paper_text_for_extraction(paper)
        if not text:
            return []

        llm = get_llm_with_tools(model=model, temperature=1.0)

        prompt = FUNCTION_EXTRACTION_PROMPT.format(
            gene_symbol=gene_symbol,
            gene_id=gene_id,
            pmcid=pmcid,
            text=text,
        )

        messages: list[Any] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # Allow multiple tool calls for ontology lookup
        response: AIMessage | None = None
        for _ in range(5):  # Max 5 rounds of tool calls
            response = await llm.ainvoke(messages)  # type: ignore[assignment]
            messages.append(response)

            if response.tool_calls:  # type: ignore[union-attr]
                # Execute tool calls
                for tool_call in response.tool_calls:  # type: ignore[union-attr]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "search_go_terms":
                        result = search_go_terms.invoke(tool_args)  # type: ignore[attr-defined]
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    messages.append(
                        ToolMessage(
                            content=json.dumps(result),
                            tool_call_id=tool_call["id"],
                        )
                    )
            else:
                # No more tool calls, extract annotations from response
                break

        # If still no text response, force a final answer without tools
        response_text = _get_response_text(response) if response else ""
        if not response_text and response and response.tool_calls:  # type: ignore[union-attr]
            messages.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
            llm_no_tools = get_llm(model=model, temperature=1.0)
            final_response = await llm_no_tools.ainvoke(messages)  # type: ignore[assignment]
            response_text = _get_response_text(final_response)

        annotations = []
        if response_text:
            extracted = _extract_json_from_response(response_text)
            for ann in extracted:
                # Validate required fields
                if all(k in ann for k in ["go_id", "qualifier", "aspect"]):
                    # Ensure evidence has pmcid
                    if "evidence" not in ann:
                        ann["evidence"] = {"pmcid": pmcid}
                    elif "pmcid" not in ann["evidence"]:
                        ann["evidence"]["pmcid"] = pmcid
                    annotations.append(ann)
        return annotations

    # Process all papers in parallel
    tasks = [process_paper(pmcid, paper) for pmcid, paper in paper_texts.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results, handling any exceptions gracefully
    all_annotations: list[dict] = []
    pmcid_list = list(paper_texts.keys())
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning(f"[EXTRACT_FN] Error processing {pmcid_list[i]}: {result}")
        else:
            all_annotations.extend(result)

    logger.info(f"[EXTRACT_FN] Extracted {len(all_annotations)} GO annotations")

    return {
        "go_annotations": all_annotations,
        "messages": [HumanMessage(content=f"Extracted {len(all_annotations)} GO annotations")],
    }


async def extract_expression(state: AgentState) -> dict[str, Any]:
    """Extract expression annotations from papers IN PARALLEL.

    This node uses the LLM to extract anatomy and stage terms,
    using the ontology search tools to find correct FBbt/FBdv IDs.
    Papers are processed concurrently using asyncio.gather().
    """
    gene_symbol = state["gene_symbol"]
    gene_id = state["gene_id"]
    paper_texts = state.get("paper_texts", {})
    config = state.get("config", {})
    model = config.get("model", DEFAULT_MODEL)

    logger.info(f"[EXTRACT_EXPR] Extracting expression from {len(paper_texts)} papers")

    async def process_paper(pmcid: str, paper: dict) -> list[dict]:
        """Process a single paper asynchronously."""
        text = _get_paper_text_for_extraction(paper)
        if not text:
            return []

        llm = get_llm_with_tools(model=model, temperature=1.0)

        prompt = EXPRESSION_EXTRACTION_PROMPT.format(
            gene_symbol=gene_symbol,
            gene_id=gene_id,
            pmcid=pmcid,
            text=text,
        )

        messages: list[Any] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # Allow multiple tool calls for ontology lookup
        response: AIMessage | None = None
        for _ in range(5):  # Increased to 5 rounds
            response = await llm.ainvoke(messages)  # type: ignore[assignment]
            messages.append(response)

            if response.tool_calls:  # type: ignore[union-attr]
                for tool_call in response.tool_calls:  # type: ignore[union-attr]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name == "search_anatomy_terms":
                        result = search_anatomy_terms.invoke(tool_args)  # type: ignore[attr-defined]
                    elif tool_name == "search_stage_terms":
                        result = search_stage_terms.invoke(tool_args)  # type: ignore[attr-defined]
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    messages.append(
                        ToolMessage(
                            content=json.dumps(result),
                            tool_call_id=tool_call["id"],
                        )
                    )
            else:
                break

        # If still no text response, force a final answer without tools
        response_text = _get_response_text(response) if response else ""
        if not response_text and response and response.tool_calls:  # type: ignore[union-attr]
            messages.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
            llm_no_tools = get_llm(model=model, temperature=1.0)
            final_response = await llm_no_tools.ainvoke(messages)  # type: ignore[assignment]
            response_text = _get_response_text(final_response)

        records = []
        if response_text:
            extracted = _extract_json_from_response(response_text)
            for rec in extracted:
                if "expression_type" in rec:
                    # Ensure evidence has pmcid
                    if "evidence" not in rec:
                        rec["evidence"] = {"pmcid": pmcid}
                    elif "pmcid" not in rec["evidence"]:
                        rec["evidence"]["pmcid"] = pmcid
                    records.append(rec)
        return records

    # Process all papers in parallel
    tasks = [process_paper(pmcid, paper) for pmcid, paper in paper_texts.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results, handling any exceptions gracefully
    all_records: list[dict] = []
    pmcid_list = list(paper_texts.keys())
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning(f"[EXTRACT_EXPR] Error processing {pmcid_list[i]}: {result}")
        else:
            all_records.extend(result)

    logger.info(f"[EXTRACT_EXPR] Extracted {len(all_records)} expression records")

    return {
        "expression_records": all_records,
        "messages": [HumanMessage(content=f"Extracted {len(all_records)} expression records")],
    }


async def extract_synonyms(state: AgentState) -> dict[str, Any]:
    """Extract gene synonyms from papers.

    This node uses the LLM to identify alternative gene names.
    Made async for compatibility with parallel graph execution.
    """
    gene_symbol = state["gene_symbol"]
    gene_id = state["gene_id"]
    paper_texts = state.get("paper_texts", {})
    config = state.get("config", {})
    model = config.get("model", DEFAULT_MODEL)

    logger.info("[EXTRACT_SYN] Extracting synonyms from up to 5 papers")

    llm = get_llm(model=model, temperature=1.0)

    all_synonyms: set[str] = set()

    # Combine text from all papers for synonym extraction
    combined_text = ""
    for pmcid, paper in list(paper_texts.items())[:5]:  # Limit to 5 papers
        text = _get_paper_text_for_extraction(paper, max_chars=3000)
        combined_text += f"\n\n--- Paper {pmcid} ---\n{text}"

    if combined_text:
        prompt = SYNONYM_EXTRACTION_PROMPT.format(
            gene_symbol=gene_symbol,
            gene_id=gene_id,
            text=combined_text[:10000],  # Limit total text
        )

        messages: list[Any] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response: AIMessage = await llm.ainvoke(messages)  # type: ignore[assignment]

        response_text = _get_response_text(response)
        if response_text:
            extracted = _extract_json_from_response(response_text)
            for item in extracted:
                if "fullname_synonyms" in item:
                    all_synonyms.update(item["fullname_synonyms"])
                if "symbol_synonyms" in item:
                    all_synonyms.update(item["symbol_synonyms"])

    logger.info(f"[EXTRACT_SYN] Found {len(all_synonyms)} synonyms")

    return {
        "synonyms_found": list(all_synonyms),
        "messages": [HumanMessage(content=f"Found {len(all_synonyms)} synonyms")],
    }


# =============================================================================
# NEW: Unified 1-Pass Extraction (text-based, no ontology IDs)
# =============================================================================


async def extract_all(state: AgentState) -> dict[str, Any]:
    """Extract all annotations from papers in a single pass (text-based).

    This node extracts function, expression, and synonym annotations using
    natural language descriptions. Ontology resolution happens in the next node.
    """
    gene_symbol = state["gene_symbol"]
    gene_id = state["gene_id"]
    paper_texts = state.get("paper_texts", {})
    config = state.get("config", {})
    model = config.get("model", DEFAULT_MODEL)
    hide_terms = config.get("hide_terms", False)

    # Build system prompt with optional hidden terms addendum
    system_prompt = SYSTEM_PROMPT
    if hide_terms:
        system_prompt = system_prompt + HIDDEN_TERMS_ADDENDUM
        logger.info("[EXTRACT_ALL] Hidden terms mode enabled")

    logger.info(f"[EXTRACT_ALL] Extracting from {len(paper_texts)} papers (1-pass)")

    async def process_paper(pmcid: str, paper: dict) -> dict[str, Any]:
        """Process a single paper and extract text-based annotations."""
        text = _get_paper_text_for_extraction(paper)
        if not text:
            return {"function": [], "expression": [], "synonyms": []}

        llm = get_llm(model=model)
        prompt = UNIFIED_EXTRACTION_PROMPT.format(
            gene_symbol=gene_symbol,
            gene_id=gene_id,
            pmcid=pmcid,
            text=text,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        response_text = _get_response_text(response)

        # Parse JSON response
        result = {"function": [], "expression": [], "synonyms": []}
        if response_text:
            extracted = _extract_json_from_response(response_text)
            if extracted and len(extracted) > 0:
                data = extracted[0] if isinstance(extracted, list) else extracted

                # Extract function annotations
                for ann in data.get("function_annotations", []):
                    result["function"].append(
                        TextFunctionAnnotation(
                            function_description=ann.get("function_description", ""),
                            qualifier=ann.get("qualifier", "involved_in"),
                            aspect=ann.get("aspect", "P"),
                            is_negated=ann.get("is_negated", False),
                            evidence_text=ann.get("evidence_text", ""),
                            pmcid=pmcid,
                        )
                    )

                # Extract expression annotations
                for ann in data.get("expression_annotations", []):
                    result["expression"].append(
                        TextExpressionAnnotation(
                            expression_type=ann.get("expression_type", "polypeptide"),
                            anatomy_description=ann.get("anatomy_description", ""),
                            stage_description=ann.get("stage_description", ""),
                            evidence_text=ann.get("evidence_text", ""),
                            pmcid=pmcid,
                        )
                    )

                # Extract synonyms
                result["synonyms"] = data.get("synonyms", [])

        return result

    # Process all papers in parallel
    tasks = [process_paper(pmcid, paper) for pmcid, paper in paper_texts.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate results
    all_function: list[TextFunctionAnnotation] = []
    all_expression: list[TextExpressionAnnotation] = []
    all_synonyms: list[str] = []

    pmcid_list = list(paper_texts.keys())
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning(f"[EXTRACT_ALL] Error processing {pmcid_list[i]}: {result}")
        else:
            all_function.extend(result["function"])
            all_expression.extend(result["expression"])
            all_synonyms.extend(result["synonyms"])

    logger.info(
        f"[EXTRACT_ALL] Extracted {len(all_function)} functions, "
        f"{len(all_expression)} expressions, {len(all_synonyms)} synonyms"
    )

    return {
        "text_function_annotations": all_function,
        "text_expression_annotations": all_expression,
        "synonyms_found": all_synonyms,
        "messages": [
            HumanMessage(
                content=f"Extracted {len(all_function)} functions, {len(all_expression)} expressions"
            )
        ],
    }


async def resolve_ontology(state: AgentState) -> dict[str, Any]:
    """Resolve text descriptions to ontology IDs.

    This node converts text-based annotations to ontology IDs using search tools.
    """
    text_functions = state.get("text_function_annotations", [])
    text_expressions = state.get("text_expression_annotations", [])

    logger.info(
        f"[RESOLVE] Resolving {len(text_functions)} functions, {len(text_expressions)} expressions"
    )

    # Resolve GO terms
    go_annotations = []
    search_hits = 0
    for ann in text_functions:
        desc = ann.get("function_description", "")
        if not desc:
            continue

        # Search for GO term (no aspect filter - let search find best match)
        results = search_go_terms.invoke({"query": desc, "limit": 3})  # type: ignore[attr-defined]

        if results and len(results) > 0:
            search_hits += 1
            # Take the top result
            go_term = results[0]
            # Infer aspect from GO term namespace
            namespace = go_term.get("namespace", "")
            if "biological_process" in namespace:
                inferred_aspect = "P"
            elif "molecular_function" in namespace:
                inferred_aspect = "F"
            elif "cellular_component" in namespace:
                inferred_aspect = "C"
            else:
                inferred_aspect = ann.get("aspect", "P")  # fallback to LLM's guess
            go_annotations.append(
                {
                    "go_id": go_term.get("go_id"),
                    "qualifier": ann.get("qualifier", "involved_in"),
                    "aspect": inferred_aspect,
                    "is_negated": ann.get("is_negated", False),
                    "evidence": {
                        "pmcid": ann.get("pmcid"),
                        "text": ann.get("evidence_text", "")[:500],
                    },
                }
            )

    logger.info(f"[RESOLVE] GO search: {search_hits}/{len(text_functions)} resolved")

    # Resolve anatomy and stage terms
    expression_records = []
    for ann in text_expressions:
        anatomy_desc = ann.get("anatomy_description", "")
        stage_desc = ann.get("stage_description", "")

        anatomy_id = None
        stage_id = None

        # Search anatomy
        if anatomy_desc:
            results = search_anatomy_terms.invoke({"query": anatomy_desc, "limit": 3})  # type: ignore[attr-defined]
            if results and len(results) > 0:
                anatomy_id = results[0].get("fbbt_id")

        # Search stage
        if stage_desc:
            results = search_stage_terms.invoke({"query": stage_desc, "limit": 3})  # type: ignore[attr-defined]
            if results and len(results) > 0:
                stage_id = results[0].get("fbdv_id")

        # Only add if we resolved at least one term
        if anatomy_id or stage_id:
            expression_records.append(
                {
                    "expression_type": ann.get("expression_type", "polypeptide"),
                    "anatomy_id": anatomy_id,
                    "stage_id": stage_id,
                    "evidence": {
                        "pmcid": ann.get("pmcid"),
                        "text": ann.get("evidence_text", "")[:500],
                    },
                }
            )

    logger.info(
        f"[RESOLVE] Resolved {len(go_annotations)} GO, {len(expression_records)} expression"
    )

    return {
        "go_annotations": go_annotations,
        "expression_records": expression_records,
        "messages": [
            HumanMessage(
                content=f"Resolved {len(go_annotations)} GO, {len(expression_records)} expression"
            )
        ],
    }


# =============================================================================
# Original compile_output (unchanged)
# =============================================================================


def compile_output(state: AgentState) -> dict[str, Any]:
    """Compile and validate final output.

    This node merges all extracted annotations, deduplicates,
    and validates against the output schema.
    """
    gene_id = state["gene_id"]
    gene_symbol = state["gene_symbol"]

    logger.info("[COMPILE] Compiling and validating output")

    # Deduplicate GO annotations
    seen_go = set()
    unique_go = []
    for ann in state.get("go_annotations", []):
        key = (ann.get("go_id"), ann.get("qualifier"), ann.get("aspect"))
        if key not in seen_go:
            seen_go.add(key)
            unique_go.append(ann)

    # Deduplicate expression records
    seen_expr = set()
    unique_expr = []
    for rec in state.get("expression_records", []):
        key = (
            rec.get("expression_type"),
            rec.get("anatomy_id"),
            rec.get("stage_id"),
        )
        if key not in seen_expr:
            seen_expr.add(key)
            unique_expr.append(rec)

    # Organize synonyms
    synonyms_raw = state.get("synonyms_found", [])
    fullname_synonyms = []
    symbol_synonyms = []

    for syn in synonyms_raw:
        if not syn or syn.lower() == gene_symbol.lower():
            continue
        # Heuristic: short names (<=6 chars) or names with capitals are symbols
        if len(syn) <= 6 or (syn[0].isupper() and any(c.islower() for c in syn)):
            if syn not in symbol_synonyms:
                symbol_synonyms.append(syn)
        else:
            if syn not in fullname_synonyms:
                fullname_synonyms.append(syn)

    # Build final output
    output = {
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "task1_function": unique_go,
        "task2_expression": unique_expr,
        "task3_synonyms": {
            "fullname_synonyms": fullname_synonyms,
            "symbol_synonyms": symbol_synonyms,
        },
    }

    # Validate
    validation_result = submit_annotations.invoke({"annotations": output})  # type: ignore[attr-defined]

    if validation_result["valid"]:
        return {
            "final_output": output,
            "messages": [
                HumanMessage(
                    content=f"Output compiled successfully: "
                    f"{len(unique_go)} GO terms, "
                    f"{len(unique_expr)} expression records, "
                    f"{len(fullname_synonyms) + len(symbol_synonyms)} synonyms"
                )
            ],
        }
    else:
        return {
            "final_output": output,  # Still return output even if invalid
            "error": f"Validation errors: {validation_result['errors']}",
            "messages": [
                HumanMessage(content=f"Output has validation errors: {validation_result['errors']}")
            ],
        }
