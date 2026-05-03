"""Tracing and logging for the MCP backbone agent.

Provides structured logging of agent execution for debugging and analysis.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure module logger
logger = logging.getLogger("agent.mcp")


@dataclass
class TraceEvent:
    """A single event in the agent execution trace."""

    timestamp: str
    event_type: str  # "tool_call", "tool_result", "llm_request", "llm_response", "budget", "error"
    data: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            **self.data,
        }


@dataclass
class AgentTrace:
    """Collects trace events during agent execution."""

    gene_id: str
    gene_symbol: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    events: list[TraceEvent] = field(default_factory=list)

    def _now(self) -> str:
        return datetime.now().isoformat()

    def log_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        """Log a tool being called."""
        event = TraceEvent(
            timestamp=self._now(),
            event_type="tool_call",
            data={"tool": tool_name, "args": args},  # Store full args in trace
        )
        self.events.append(event)
        logger.info(f"[TOOL CALL] {tool_name}({_truncate_args(args, max_value_len=100)})")

    def log_tool_result(
        self,
        tool_name: str,
        result: Any,
        result_parsed: Any = None,
        error: str | None = None,
    ) -> None:
        """Log a tool result.

        Args:
            tool_name: Name of the tool
            result: Raw result string
            result_parsed: Parsed result (dict/list) if JSON parsing succeeded
            error: Error message if the tool failed
        """
        event = TraceEvent(
            timestamp=self._now(),
            event_type="tool_result",
            data={
                "tool": tool_name,
                "result_raw": str(result),
                "result": result_parsed if result_parsed is not None else str(result),
                "error": error,
            },
        )
        self.events.append(event)
        if error:
            logger.warning(f"[TOOL ERROR] {tool_name}: {error}")
        else:
            logger.info(f"[TOOL RESULT] {tool_name}: {_truncate_str(str(result), 200)}")

    def log_llm_input(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]],
    ) -> None:
        """Log the input being sent to the LLM.

        Args:
            system_prompt: The system prompt (if any)
            messages: List of input messages/items as dicts
        """
        event = TraceEvent(
            timestamp=self._now(),
            event_type="llm_input",
            data={
                "system_prompt": system_prompt,
                "messages": messages,
                "message_count": len(messages),
            },
        )
        self.events.append(event)
        logger.debug(f"[LLM INPUT] system_prompt={bool(system_prompt)}, {len(messages)} messages")

    def log_llm_response(
        self,
        content_preview: str,
        tool_calls: list[str] | None,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Log an LLM response."""
        event = TraceEvent(
            timestamp=self._now(),
            event_type="llm_response",
            data={
                "content": content_preview,  # Store full content passed in
                "content_preview": _truncate_str(content_preview, 500),
                "tool_calls": tool_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
        self.events.append(event)

        if tool_calls:
            logger.info(f"[LLM] Calling tools: {', '.join(tool_calls)}")
        else:
            logger.info(f"[LLM] Response: {_truncate_str(content_preview, 200)}")
        logger.debug(f"[TOKENS] in={input_tokens}, out={output_tokens}")

    def log_budget_update(
        self, turns: int, papers: int, cost: float, exceeded: bool = False, reason: str = ""
    ) -> None:
        """Log budget state change."""
        event = TraceEvent(
            timestamp=self._now(),
            event_type="budget",
            data={
                "turns_used": turns,
                "papers_read": papers,
                "cost_usd": cost,
                "exceeded": exceeded,
                "reason": reason,
            },
        )
        self.events.append(event)

        if exceeded:
            logger.warning(f"[BUDGET EXCEEDED] {reason}")
        else:
            logger.debug(f"[BUDGET] turns={turns}, papers={papers}, cost=${cost:.4f}")

    def log_error(self, error: str, context: str = "") -> None:
        """Log an error."""
        event = TraceEvent(
            timestamp=self._now(),
            event_type="error",
            data={"error": error, "context": context},
        )
        self.events.append(event)
        logger.error(f"[ERROR] {context}: {error}" if context else f"[ERROR] {error}")

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        event = TraceEvent(
            timestamp=self._now(),
            event_type="info",
            data={"message": message},
        )
        self.events.append(event)
        logger.info(f"[INFO] {message}")

    def to_dict(self) -> dict:
        """Convert trace to dictionary for serialization."""
        return {
            "gene_id": self.gene_id,
            "gene_symbol": self.gene_symbol,
            "start_time": self.start_time,
            "end_time": self._now(),
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

    def save(self, path: Path) -> None:
        """Save trace to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"[TRACE] Saved to {path}")


def _truncate_str(s: str, max_len: int) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _truncate_args(args: dict, max_value_len: int = 100) -> str:
    """Format args for logging, truncating long values."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > max_value_len:
            v_str = v_str[: max_value_len - 3] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure logging for the MCP agent.

    Args:
        verbose: If True, show DEBUG level logs
        log_file: Optional file to write logs to
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure format
    fmt = "%(asctime)s %(levelname)s %(message)s"
    datefmt = "%H:%M:%S"

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt))

    # Configure logger
    logger.setLevel(level)
    logger.handlers = [console_handler]

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(file_handler)

    logger.propagate = False
