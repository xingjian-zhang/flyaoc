"""Budget control hooks for the Single-Agent and Multi-Agent methods.

This module provides RunHooks that enforce budget limits during agent execution.
"""

import json
from typing import Any

from agents import Agent, RunContextWrapper, RunHooks
from agents.items import ModelResponse

from .budget import BudgetConfig, BudgetState
from .tracing import AgentTrace

# Pricing per 1K tokens (updated Feb 2026)
# Source: https://platform.openai.com/docs/pricing
# "input" = uncached input price, "cached" = cached input price (typically 50% of input)
MODEL_PRICING = {
    "gpt-5": {"input": 0.00125, "cached": 0.000625, "output": 0.01},
    "gpt-5-mini": {"input": 0.00025, "cached": 0.000125, "output": 0.002},
    "gpt-4o": {"input": 0.0025, "cached": 0.00125, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "cached": 0.000075, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "cached": 0.005, "output": 0.03},
    "gpt-4": {"input": 0.03, "cached": 0.015, "output": 0.06},
    # Default fallback
    "default": {"input": 0.00025, "cached": 0.000125, "output": 0.002},
}


class BudgetExhaustedError(Exception):
    """Raised when budget limits are exceeded."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class BudgetControlHooks(RunHooks):
    """Hooks for enforcing budget limits during agent execution.

    This hooks implementation tracks:
    - Number of LLM turns (via on_agent_start)
    - Number of papers read (via on_tool_start)
    - Token usage and cost (via on_llm_response)
    """

    def __init__(
        self,
        config: BudgetConfig,
        model: str = "gpt-4o",
        trace: AgentTrace | None = None,
    ):
        """Initialize budget control hooks.

        Args:
            config: Budget configuration with limits
            model: Model name for cost calculation
            trace: Optional trace object for logging events
        """
        self.config = config
        self.state = BudgetState()
        self.model = model
        self.trace = trace
        self._pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

    async def on_agent_start(self, context: RunContextWrapper[Any], agent: Agent[Any]) -> None:
        """Called when an agent starts a turn.

        Increments turn counter and checks budget.
        """
        self.state.turns_used += 1

        # Log budget state
        if self.trace:
            exceeded, reason = self.state.check(self.config)
            self.trace.log_budget_update(
                turns=self.state.turns_used,
                papers=self.state.papers_read,
                cost=self.state.cost_usd,
                exceeded=exceeded,
                reason=reason,
            )
            if exceeded:
                raise BudgetExhaustedError(reason)
        else:
            exceeded, reason = self.state.check(self.config)
            if exceeded:
                raise BudgetExhaustedError(reason)

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        """Called before each LLM call.

        Logs the input being sent to the LLM for tracing.
        """
        if self.trace:
            # Convert input items to serializable dicts
            messages = []
            for item in input_items:
                item_dict = self._serialize_input_item(item)
                if item_dict:
                    messages.append(item_dict)

            self.trace.log_llm_input(system_prompt, messages)

    def _serialize_input_item(self, item: Any) -> dict[str, Any] | None:
        """Convert an input item to a serializable dict."""
        item_type = getattr(item, "type", None)

        # Handle different item types
        if item_type == "message":
            content = getattr(item, "content", None)
            role = getattr(item, "role", "unknown")

            # Content can be a string or a list of content items
            if isinstance(content, str):
                content_serialized = content
            elif isinstance(content, list):
                content_serialized = []
                for c in content:
                    c_type = getattr(c, "type", None)
                    if c_type == "input_text" or c_type == "output_text":
                        content_serialized.append(
                            {
                                "type": "text",
                                "text": getattr(c, "text", ""),
                            }
                        )
                    else:
                        content_serialized.append({"type": c_type or "unknown"})
            else:
                content_serialized = str(content) if content else ""

            return {
                "type": "message",
                "role": role,
                "content": content_serialized,
            }

        elif item_type == "mcp_call":
            return {
                "type": "mcp_call",
                "name": getattr(item, "name", "unknown"),
                "arguments": getattr(item, "arguments", "{}"),
                "output": getattr(item, "output", None),
                "status": getattr(item, "status", None),
            }

        elif item_type == "function_call":
            return {
                "type": "function_call",
                "name": getattr(item, "name", "unknown"),
                "arguments": getattr(item, "arguments", "{}"),
            }

        elif item_type == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": getattr(item, "call_id", "unknown"),
                "output": getattr(item, "output", ""),
            }

        elif hasattr(item, "role") and hasattr(item, "content"):
            # EasyInputMessageParam or similar dict-like message
            return {
                "type": "message",
                "role": getattr(item, "role", "unknown"),
                "content": getattr(item, "content", ""),
            }

        else:
            # Fallback: try to capture what we can
            return {
                "type": item_type or "unknown",
                "raw": str(item)[:500],
            }

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent[Any], tool: Any
    ) -> None:
        """Called before each tool call.

        Tracks paper reads for monitoring. Paper limit is enforced by the MCP server
        which returns an error message, allowing the agent to submit annotations.
        Note: Tool arguments are not available here - they are logged in on_llm_end
        via McpCall items in the response.
        """
        # Get tool name (args not available in this hook - only in ModelResponse)
        tool_name = getattr(tool, "name", str(tool))

        # Track paper reads (enforcement is done by MCP server)
        if tool_name == "get_paper_text":
            self.state.papers_read += 1
            # Log if approaching or exceeding limit (but don't kill agent)
            if self.state.papers_read > self.config.max_papers and self.trace:
                self.trace.log_info(
                    f"Paper limit reached ({self.state.papers_read}/{self.config.max_papers}) - "
                    "MCP server will return error message"
                )

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Any,
        result: str,
    ) -> None:
        """Called after each tool call completes.

        Logs the tool result for tracing, parsing JSON if possible.
        Also captures subagent usage from analyze_papers_batch results.
        """
        tool_name = getattr(tool, "name", str(tool))

        # Try to parse result as JSON
        result_parsed = None
        try:
            result_parsed = json.loads(result)
            # If it's an MCP response with nested JSON in "text" field, parse that too
            if isinstance(result_parsed, dict) and "text" in result_parsed:
                try:
                    result_parsed["text"] = json.loads(result_parsed["text"])
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep text as-is if not JSON
        except json.JSONDecodeError:
            pass  # Keep as string

        # Capture subagent usage from analyze_papers_batch result
        if tool_name == "analyze_papers_batch" and isinstance(result_parsed, dict):
            subagent_usage = result_parsed.get("usage", {})
            input_tokens = subagent_usage.get("input_tokens", 0)
            output_tokens = subagent_usage.get("output_tokens", 0)
            cached_tokens = subagent_usage.get("cached_tokens", 0)

            if input_tokens or output_tokens:
                uncached_tokens = input_tokens - cached_tokens

                self.state.input_tokens += input_tokens
                self.state.output_tokens += output_tokens
                self.state.cached_tokens += cached_tokens

                # Calculate incremental cost with cache discount
                uncached_cost = (uncached_tokens / 1000) * self._pricing["input"]
                cached_cost = (cached_tokens / 1000) * self._pricing.get(
                    "cached", self._pricing["input"] * 0.5
                )
                output_cost = (output_tokens / 1000) * self._pricing["output"]
                self.state.cost_usd += uncached_cost + cached_cost + output_cost

                if self.trace:
                    cache_pct = (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0
                    self.trace.log_info(
                        f"Subagent usage: +{input_tokens} in ({cached_tokens} cached, {cache_pct:.0f}%), +{output_tokens} out "
                        f"(total: {self.state.input_tokens}/{self.state.output_tokens}, ${self.state.cost_usd:.4f})"
                    )

        if self.trace:
            self.trace.log_tool_result(tool_name, result, result_parsed=result_parsed)

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        """Called after each LLM call returns.

        Updates token usage and cost tracking, including cached token breakdown.
        """
        # Extract usage from response
        usage = getattr(response, "usage", None)
        if usage:
            # Get token counts
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0

            # Get cached token breakdown from input_tokens_details
            input_details = getattr(usage, "input_tokens_details", None)
            cached_tokens = 0
            if input_details:
                cached_tokens = getattr(input_details, "cached_tokens", 0) or 0

            uncached_tokens = input_tokens - cached_tokens

            # Accumulate tokens
            self.state.input_tokens += input_tokens
            self.state.output_tokens += output_tokens
            self.state.cached_tokens += cached_tokens

            # Calculate cost with cache discount
            uncached_cost = (uncached_tokens / 1000) * self._pricing["input"]
            cached_cost = (cached_tokens / 1000) * self._pricing.get(
                "cached", self._pricing["input"] * 0.5
            )
            output_cost = (output_tokens / 1000) * self._pricing["output"]
            self.state.cost_usd += uncached_cost + cached_cost + output_cost

            if self.trace:
                cache_pct = (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0
                self.trace.log_info(
                    f"Tokens: +{input_tokens} in ({cached_tokens} cached, {cache_pct:.0f}%), +{output_tokens} out "
                    f"(total: {self.state.input_tokens}/{self.state.output_tokens}, ${self.state.cost_usd:.4f})"
                )

            # Check cost budget after updating
            exceeded, reason = self.state.check(self.config)
            if exceeded and "Cost" in reason:
                if self.trace:
                    self.trace.log_budget_update(
                        self.state.turns_used,
                        self.state.papers_read,
                        self.state.cost_usd,
                        exceeded=True,
                        reason=reason,
                    )
                raise BudgetExhaustedError(reason)

        # Log LLM response with proper extraction from OpenAI Agents SDK types
        if self.trace:
            content = ""
            tool_calls = []

            if hasattr(response, "output"):
                for item in response.output:
                    item_type = getattr(item, "type", None)

                    # Extract MCP tool calls with arguments
                    if item_type == "mcp_call":
                        tool_name = getattr(item, "name", "unknown")
                        tool_calls.append(tool_name)
                        # Parse arguments JSON string
                        args_str = getattr(item, "arguments", "{}")
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args = {"_raw": args_str}
                        self.trace.log_tool_call(tool_name, args)

                    # Extract function tool calls
                    elif item_type == "function_call":
                        tool_name = getattr(item, "name", "unknown")
                        tool_calls.append(tool_name)
                        args_str = getattr(item, "arguments", "{}")
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args = {"_raw": args_str}
                        self.trace.log_tool_call(tool_name, args)

                    # Extract text content from assistant messages
                    elif item_type == "message":
                        msg_content = getattr(item, "content", [])
                        for content_item in msg_content:
                            content_type = getattr(content_item, "type", None)
                            if content_type == "output_text":
                                text = getattr(content_item, "text", "")
                                if text and not content:
                                    content = text[:500]  # Capture more content

            self.trace.log_llm_response(
                content_preview=content,
                tool_calls=tool_calls if tool_calls else None,
                input_tokens=self.state.input_tokens,
                output_tokens=self.state.output_tokens,
            )

    def get_remaining_budget_prompt(self) -> str:
        """Generate a prompt snippet showing remaining budget.

        Useful for injecting into the agent's context to help it plan.
        """
        remaining_turns = self.config.max_turns - self.state.turns_used
        remaining_papers = self.config.max_papers - self.state.papers_read
        remaining_cost = self.config.max_cost_usd - self.state.cost_usd

        return f"""
=== REMAINING BUDGET ===
- Turns: {remaining_turns}
- Papers you can still read: {remaining_papers}
- Cost: ${remaining_cost:.4f}

Plan your actions carefully within these limits.
"""
