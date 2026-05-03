"""Provider environment validation for optional baseline reruns."""

from __future__ import annotations

import os


def validate_provider_environment(provider: str) -> None:
    """Validate environment variables needed by a configured provider route."""
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --provider openai.")
        return

    if provider in {"openai_compatible", "bedrock_proxy"}:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI-compatible provider routes."
            )
        if not os.environ.get("OPENAI_BASE_URL"):
            raise RuntimeError(
                "OPENAI_BASE_URL is required for --provider openai_compatible or bedrock_proxy."
            )
        return

    raise ValueError(f"Unsupported provider: {provider}")
