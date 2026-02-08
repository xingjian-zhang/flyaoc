"""Main runner for the Single-Agent and Multi-Agent methods.

This module provides the run_agent_mcp function that orchestrates the
OpenAI Agents SDK with MCP servers for gene annotation.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents import Agent, MaxTurnsExceeded, ModelSettings, Runner
from agents.mcp import MCPServerStdio

from agent.types import AgentOutput, AgentRunResult, UsageStats

from .budget import BudgetConfig

if TYPE_CHECKING:
    from agent.config import ExecutionConfig

from .hooks import BudgetControlHooks, BudgetExhaustedError
from .prompts import (
    HIDDEN_TERMS_ADDENDUM,
    MEMORIZATION_SYSTEM_PROMPT,
    MULTI_AGENT_MODE_ADDENDUM,
    SYSTEM_PROMPT_V2,
)
from .tracing import AgentTrace, setup_logging


def get_mcp_server_params(
    hide_terms: bool = False,
    max_papers: int = 0,
    multi_agent: bool = False,
    no_literature: bool = False,
) -> list[dict[str, Any]]:
    """Get MCP server parameters for stdio transport.

    Args:
        hide_terms: If True, pass HIDE_GO_TERMS env var to ontology server
        max_papers: Max papers to read (0 = no limit). Passed to literature server.
        multi_agent: If True, hide get_paper_text tool (subagent handles paper reading)
        no_literature: If True, exclude literature server (memorization baseline)

    Returns:
        List of server parameter dicts for MCPServerStdio
    """
    python_path = sys.executable
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent

    # Environment for ontology server (may include HIDE_GO_TERMS)
    ontology_env = dict(os.environ)
    if hide_terms:
        ontology_env["HIDE_GO_TERMS"] = "1"

    servers = []

    # Literature server (excluded for memorization baseline)
    if not no_literature:
        literature_env = dict(os.environ)
        if max_papers > 0:
            literature_env["MAX_PAPERS"] = str(max_papers)
        if multi_agent:
            # Hide get_paper_text - subagent's analyze_papers_batch handles paper reading
            literature_env["HIDE_GET_PAPER_TEXT"] = "1"

        servers.append(
            {
                "name": "literature",
                "params": {
                    "command": python_path,
                    "args": ["-m", "agent.mcp_servers.literature_server"],
                    "cwd": str(project_root),
                    "env": literature_env,
                },
            }
        )

    # Ontology server (always included)
    servers.append(
        {
            "name": "ontology",
            "params": {
                "command": python_path,
                "args": ["-m", "agent.mcp_servers.ontology_server"],
                "cwd": str(project_root),
                "env": ontology_env,
            },
        }
    )

    # Validation server (always included)
    servers.append(
        {
            "name": "validation",
            "params": {
                "command": python_path,
                "args": ["-m", "agent.mcp_servers.validation_server"],
                "cwd": str(project_root),
            },
        }
    )

    return servers


def create_task_prompt(
    gene_id: str,
    gene_symbol: str,
    summary: str,
    budget: BudgetConfig,
) -> str:
    """Create the task prompt for the agent.

    Args:
        gene_id: FlyBase gene ID
        gene_symbol: Gene symbol
        summary: Optional gene summary
        budget: Budget configuration

    Returns:
        Formatted task prompt
    """
    return f"""Annotate gene {gene_symbol} ({gene_id}).

{f"Gene Summary: {summary}" if summary else ""}

## Resources
- You MUST read exactly {budget.max_papers} papers before submitting annotations
- You have {budget.max_turns} turns to complete the task

## CRITICAL REQUIREMENT
**You MUST read at least {budget.max_papers} papers using `get_paper_text` before calling `submit_annotations`.**
If fewer than {budget.max_papers} relevant papers exist for this gene, read all available papers.
Do NOT submit early - extract as much information as possible from each paper.

## Tools Available
- `search_corpus`: Find papers mentioning the gene
- `get_paper_text`: Read a paper's full text
- `search_go_terms`: Look up GO term IDs by name/keyword
- `search_anatomy_terms`: Look up FBbt anatomy term IDs
- `search_stage_terms`: Look up FBdv developmental stage IDs
- `submit_annotations`: Validate your final output (only after reading {budget.max_papers} papers!)

## Strategy Tips
- Papers with the gene name in the title are often most relevant
- Look for experimental results: mutant phenotypes, biochemical assays, localization studies
- One well-studied paper can yield multiple annotations - be thorough
- Always use ontology search tools to find correct term IDs

CRITICAL: Operate fully autonomously - never ask for user guidance or clarification.

## Output
After reading {budget.max_papers} papers, provide your annotations as JSON:
```json
{{
    "gene_id": "{gene_id}",
    "gene_symbol": "{gene_symbol}",
    "task1_function": [
        {{"go_id": "GO:XXXXXXX", "qualifier": "involved_in|enables|located_in", "aspect": "P|F|C", "is_negated": false, "evidence": {{"pmcid": "PMCXXXXXXX", "text": "quote from paper"}}}}
    ],
    "task2_expression": [
        {{"expression_type": "polypeptide|transcript", "anatomy_id": "FBbt:XXXXXXXX", "stage_id": "FBdv:XXXXXXXX", "evidence": {{"pmcid": "PMCXXXXXXX", "text": "quote"}}}}
    ],
    "task3_synonyms": {{
        "fullname_synonyms": ["..."],
        "symbol_synonyms": ["..."]
    }}
}}
```"""


def create_memorization_task_prompt(
    gene_id: str,
    gene_symbol: str,
    summary: str,
    max_turns: int,
) -> str:
    """Create the task prompt for memorization baseline (no literature).

    Args:
        gene_id: FlyBase gene ID
        gene_symbol: Gene symbol
        summary: Optional gene summary
        max_turns: Maximum turns allowed

    Returns:
        Formatted task prompt for memorization baseline
    """
    return f"""Annotate gene {gene_symbol} ({gene_id}).

{f"Gene Summary: {summary}" if summary else ""}

Use your prior knowledge and the ontology tools to annotate this gene.
You have {max_turns} turns to complete the task.

Remember:
- Only include annotations you are confident about from your training knowledge
- Use ontology search tools to find correct term IDs
- For evidence, use: {{"pmcid": null, "text": "Based on prior knowledge: [explanation]"}}
"""


def parse_agent_output(response: str) -> dict[str, Any] | None:
    """Parse the agent's response to extract JSON output.

    Args:
        response: Raw response from the agent

    Returns:
        Parsed output dictionary or None if parsing fails
    """
    if not response:
        return None

    # Try to find JSON in the response
    # First, try direct JSON parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    matches = re.findall(json_pattern, response)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try to find raw JSON object
    brace_pattern = r"\{[\s\S]*\}"
    matches = re.findall(brace_pattern, response)
    for match in matches:
        try:
            parsed = json.loads(match)
            # Verify it looks like our expected output
            if "gene_id" in parsed or "task1_function" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def extract_validated_annotations_from_trace(trace: AgentTrace) -> dict[str, Any] | None:
    """Extract annotations from the last successful submit_annotations call.

    Looks for submit_annotations tool calls where the subsequent tool_result
    shows valid=true. Returns annotations from the last such call.

    Args:
        trace: AgentTrace instance with recorded events

    Returns:
        Validated annotations dict or None if not found
    """
    last_valid_annotations = None
    pending_annotations = None

    for event in trace.events:
        if event.event_type == "tool_call" and event.data.get("tool") == "submit_annotations":
            args = event.data.get("args") or {}
            pending_annotations = args.get("annotations")

        elif event.event_type == "tool_result" and event.data.get("tool") == "submit_annotations":
            result = event.data.get("result")
            # Handle various result formats:
            # - {"text": {"valid": true}} (nested dict)
            # - {"valid": true} (flat dict)
            # - string (error message - skip)
            if isinstance(result, dict):
                text_field = result.get("text")
                # text_field could be a dict or a string
                if isinstance(text_field, dict) and text_field.get("valid") is True:
                    if pending_annotations:
                        last_valid_annotations = pending_annotations
                elif result.get("valid") is True:
                    if pending_annotations:
                        last_valid_annotations = pending_annotations
            pending_annotations = None  # Reset for next call

    return last_valid_annotations


async def run_agent_mcp(
    gene_id: str,
    gene_symbol: str,
    summary: str = "",
    config: ExecutionConfig | None = None,
    # Legacy parameters (deprecated, use config instead)
    budget: BudgetConfig | None = None,
    model: str | None = None,
    verbose: bool | None = None,
    trace_dir: Path | None = None,
    hide_terms: bool | None = None,
    multi_agent: bool | None = None,
    no_literature: bool | None = None,
) -> AgentRunResult:
    """Run the MCP-based agent on a single gene.

    Args:
        gene_id: FlyBase gene ID (e.g., "FBgn0000014")
        gene_symbol: Gene symbol (e.g., "abd-A")
        summary: Optional gene summary
        config: Unified execution configuration (preferred)

        Legacy parameters (deprecated, use config instead):
        budget: Budget limits
        model: OpenAI model to use
        verbose: If True, show detailed logging
        trace_dir: Directory to save trace files
        hide_terms: Enable GO term hiding for specificity gap benchmark
        multi_agent: Use paper reader subagent for context isolation
        no_literature: Memorization baseline (no literature access)

    Returns:
        AgentRunResult with output, usage, error (if any), and trace

    Example:
        # New style (preferred)
        config = ExecutionConfig(budget=BudgetConfig(max_papers=5), verbose=True)
        result = await run_agent_mcp("FBgn0000014", "abd-A", config=config)

        # Legacy style (still works but deprecated)
        result = await run_agent_mcp("FBgn0000014", "abd-A", budget=budget, verbose=True)
    """
    # Import ExecutionConfig at runtime to avoid circular import
    from agent.config import ExecutionConfig as _ExecutionConfig

    # Handle backwards compatibility: merge legacy params with config
    effective_config: _ExecutionConfig
    if config is None:
        effective_config = _ExecutionConfig(
            budget=budget or BudgetConfig(),
            model=model or "gpt-5-mini",
            verbose=verbose if verbose is not None else False,
            trace_dir=trace_dir,
        )
        if hide_terms is not None:
            effective_config.features.hide_go_terms = hide_terms
        if multi_agent is not None:
            effective_config.features.multi_agent = multi_agent
        if no_literature is not None:
            effective_config.features.no_literature = no_literature
    else:
        # Config provided - use it, but allow individual params to override
        effective_config = config
        if budget is not None:
            effective_config = effective_config.with_budget(
                max_turns=budget.max_turns,
                max_papers=budget.max_papers,
                max_cost_usd=budget.max_cost_usd,
            )
        if model is not None:
            effective_config.model = model
        if verbose is not None:
            effective_config.verbose = verbose
        if trace_dir is not None:
            effective_config.trace_dir = trace_dir
        if hide_terms is not None:
            effective_config.features.hide_go_terms = hide_terms
        if multi_agent is not None:
            effective_config.features.multi_agent = multi_agent
        if no_literature is not None:
            effective_config.features.no_literature = no_literature

    # Extract config values for use in function
    budget_config = effective_config.budget
    model_name = effective_config.model
    temperature = effective_config.temperature
    verbose_mode = effective_config.verbose
    hide_go_terms = effective_config.features.hide_go_terms
    multi_agent_mode = effective_config.features.multi_agent
    no_literature_mode = effective_config.features.no_literature

    # Set up logging
    setup_logging(verbose=verbose_mode)

    # Set environment variable for subagent to read (subagent runs in same process)
    if hide_go_terms:
        os.environ["HIDE_GO_TERMS"] = "1"
    else:
        os.environ.pop("HIDE_GO_TERMS", None)  # Clear if previously set

    # Create trace
    trace = AgentTrace(gene_id=gene_id, gene_symbol=gene_symbol)
    trace.log_info(f"Starting annotation for {gene_symbol} ({gene_id})")
    trace.log_info(
        f"Budget: {budget_config.max_turns} turns, {budget_config.max_papers} papers, ${budget_config.max_cost_usd}"
    )
    if hide_go_terms:
        trace.log_info("GO term hiding enabled (specificity gap benchmark)")
    if multi_agent_mode:
        trace.log_info("Multi-Agent mode enabled (context isolation)")
    if no_literature_mode:
        trace.log_info("Memorization mode enabled (no literature access)")

    hooks = BudgetControlHooks(budget_config, model=model_name, trace=trace)

    # Use different prompts for memorization mode vs normal mode
    if no_literature_mode:
        task_prompt = create_memorization_task_prompt(
            gene_id, gene_symbol, summary, budget_config.max_turns
        )
    else:
        task_prompt = create_task_prompt(gene_id, gene_symbol, summary, budget_config)

    server_params = get_mcp_server_params(
        hide_terms=hide_go_terms,
        max_papers=budget_config.max_papers,
        multi_agent=multi_agent_mode,
        no_literature=no_literature_mode,
    )

    mcp_servers = []
    try:
        # Create MCP server connections
        # Increase timeout to 60s to allow for corpus/ontology loading on first run
        for sp in server_params:
            server = MCPServerStdio(
                name=sp["name"],
                params=sp["params"],
                client_session_timeout_seconds=60.0,
            )
            mcp_servers.append(server)

        # Connect to all servers
        for server in mcp_servers:
            await server.connect()

        # Create the agent with conditional prompts
        if no_literature_mode:
            # Memorization baseline uses completely different prompt
            system_prompt = MEMORIZATION_SYSTEM_PROMPT
        else:
            system_prompt = SYSTEM_PROMPT_V2
            if hide_go_terms:
                system_prompt = system_prompt + HIDDEN_TERMS_ADDENDUM
            if multi_agent_mode:
                system_prompt = system_prompt + MULTI_AGENT_MODE_ADDENDUM

        # Build tools list (function tools in addition to MCP tools)
        tools = []
        if multi_agent_mode:
            from .subagents import analyze_papers_batch

            tools.append(analyze_papers_batch)

        agent = Agent(
            name="GeneAnnotationAgent",
            instructions=system_prompt,
            model=model_name,
            model_settings=ModelSettings(temperature=temperature),
            mcp_servers=mcp_servers,
            tools=tools,
        )

        # Run the agent
        result = await Runner.run(
            starting_agent=agent,
            input=task_prompt,
            hooks=hooks,
            max_turns=budget_config.max_turns,
        )

        # Extract output from validated submit_annotations call (authoritative source)
        output = extract_validated_annotations_from_trace(trace)

        # Fallback to parsing final response if no validated submission found
        if output is None:
            output = parse_agent_output(result.final_output)
            if output is not None:
                trace.log_info("Output parsed from final response (no validated submission)")

        trace.log_info(f"Completed annotation for {gene_symbol}")

        return AgentRunResult(
            output=AgentOutput.from_dict(output),
            usage=UsageStats.from_dict(hooks.state.to_dict()),
            raw_response=result.final_output,
            trace=trace.to_dict(),
        )

    except BudgetExhaustedError as e:
        trace.log_error(f"Budget exhausted: {e.reason}")
        return AgentRunResult(
            output=None,
            usage=UsageStats.from_dict(hooks.state.to_dict()),
            error=f"Budget exhausted: {e.reason}",
            trace=trace.to_dict(),
        )

    except MaxTurnsExceeded as e:
        trace.log_error(f"Max turns exceeded: {e}")
        return AgentRunResult(
            output=None,
            usage=UsageStats.from_dict(hooks.state.to_dict()),
            error=f"Max turns exceeded ({budget_config.max_turns})",
            trace=trace.to_dict(),
        )

    except Exception as e:
        trace.log_error(str(e), context="run_agent_mcp")
        return AgentRunResult(
            output=None,
            usage=UsageStats.from_dict(hooks.state.to_dict()),
            error=str(e),
            trace=trace.to_dict(),
        )

    finally:
        # Clean up server connections
        # Note: MCP library has known issues with asyncio cleanup, suppress all errors
        # Cannot use contextlib.suppress() with async cleanup, so we use try/except
        for server in mcp_servers:
            try:  # noqa: SIM105
                await server.cleanup()
            except BaseException:
                # Ignore all cleanup errors including CancelledError
                pass
