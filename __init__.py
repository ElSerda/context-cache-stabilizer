"""context-cache-stabilizer: Stabilizes LLM request prefix between turns to preserve KV/prompt cache.

Registers llm_request middleware that rewrites provider kwargs before execution:
  - Sorts tool definitions alphabetically (prevents shuffle-induced cache breaks)
  - Freezes dynamic timestamps in messages (prevents timestamp drift)
  - Optionally injects Anthropic-style cache breakpoints

Works with any LM Studio-compatible backend. No core Hermes changes needed.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _setting(key: str, default):
    """Read a plugin setting from config.yaml."""
    try:
        return ctx.get_config(key, default=default) if ctx is not None else default
    except Exception:  # noqa: BLE001
        logger.debug(f"Failed to read setting '{key}', using default {default}")
        return default


# Global context reference (set by register())
ctx = None

# Fixed timestamp value for cache preservation (same-day sessions)
_FROZEN_TIMESTAMP = "2026-11-22 00:00"

# Timestamp pattern matches ISO-like timestamps with optional seconds
_TIMESTAMP_RE = re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?')


def _stabilize_tools(request: dict[str, Any]) -> dict[str, Any]:
    """Sort tool definitions alphabetically to prevent KV-cache breaks from shuffle."""
    if "tools" not in request or not isinstance(request["tools"], list):
        return request

    tools_copy = []
    for t in request["tools"]:
        # Deep copy only the function name (the sort key) and preserve rest by reference
        tool_dict = {**t}
        if "function" in tool_dict:
            tool_dict["function"] = {**tool_dict["function"]}
        tools_copy.append(tool_dict)

    # Sort by function name (stable sort preserves relative order of equal keys)
    tools_copy.sort(key=lambda t: t.get("function", {}).get("name", ""))
    request["tools"] = tools_copy
    logger.debug(f"Stabilized {len(tools_copy)} tool definitions")
    return request


def _freeze_timestamps(request: dict[str, Any]) -> dict[str, Any]:
    """Replace dynamic timestamps with a fixed value to preserve cache."""
    if "messages" not in request or not isinstance(request["messages"], list):
        return request

    frozen_count = 0
    stable_messages = []
    for msg in request["messages"]:
        content = msg.get("content", "")
        if isinstance(content, str):
            # Replace all timestamps with a fixed value
            new_content, count = _TIMESTAMP_RE.subn(_FROZEN_TIMESTAMP, content)
            frozen_count += count
            stable_messages.append({**msg, "content": new_content})
        else:
            stable_messages.append(msg)

    request["messages"] = stable_messages
    if frozen_count > 0:
        logger.debug(f"Froze {frozen_count} timestamps")
    return request


def _inject_cache_breakpoints(request: dict[str, Any]) -> dict[str, Any]:
    """Inject Anthropic-style cache breakpoints if configured."""
    breakpoint_tokens = _setting("cache_breakpoint_tokens", "")
    if not breakpoint_tokens or "messages" not in request:
        return request

    # Split on spaces and inject as cache markers
    tokens = [t for t in breakpoint_tokens.split() if t]
    if not tokens:
        return request

    # Find the first system message and prepend cache markers
    stable_messages = []
    for msg in request["messages"]:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system" and isinstance(content, str):
            # Prepend cache breakpoint tokens (Anthropic format)
            cache_prefix = "".join(
                f"<|cache_control_{token}|>" for token in tokens
            )
            new_content = cache_prefix + content
            stable_messages.append({**msg, "content": new_content})
        else:
            stable_messages.append(msg)

    request["messages"] = stable_messages
    logger.debug(f"Injected {len(tokens)} cache breakpoint tokens")
    return request


def on_llm_request(**kwargs) -> dict[str, Any] | None:
    """Middleware hook: rewrite LLM request to stabilize the prefix.

    Receives:
      - request: full provider kwargs (messages, tools, model settings, etc.)
      - original_request: unmodified request before any middleware ran

    Returns:
      {"request": rewritten_request} or None to skip this middleware

    Note: Wrapped in try/except so a crash here doesn't break the turn.
    """
    try:
        request = dict(kwargs.get("request", {}))

        # Apply stabilizations in order
        if _setting("stabilize_tools_order", True):
            request = _stabilize_tools(request)

        if _setting("freeze_timestamps", True):
            request = _freeze_timestamps(request)

        # Optional: inject cache breakpoints
        request = _inject_cache_breakpoints(request)

        return {"request": request}
    except Exception as e:  # noqa: BLE001 - middleware must never break the turn
        logger.error(f"llm_request middleware failed: {e}")
        return None


def register(ctx_instance) -> None:
    """Register this plugin with Hermes.

    Args:
        ctx_instance: PluginContext from Hermes
    """
    global ctx
    ctx = ctx_instance

    # Register llm_request middleware to stabilize the prompt prefix
    ctx.register_middleware("llm_request", on_llm_request)
