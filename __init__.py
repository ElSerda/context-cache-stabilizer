"""context-cache-stabilizer: measures what Hermes sends to the model on every request.

Read-only. Nothing in the request or the reply is changed:
  - llm_execution middleware: sees the request exactly as it goes to the provider (after every
    llm_request middleware, such as webgraft removing its footers) and hands it on unchanged;
  - post_api_request hook: the token counts the provider reported for that request.

Each request adds one line to <hermes home>/plugin-data/context-cache-stabilizer/requests.jsonl:
sizes (tools, system prompt sections, messages by role), how much of the request is identical to the
previous one of the same conversation (what a prefix cache can reuse), then the real token counts.
No message text is stored, only sizes, tool names and system prompt headings.

`hermes context-cache report` prints them.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PLUGIN = "context-cache-stabilizer"
LOG_FILE = "requests.jsonl"
MAX_LOG_BYTES = 20 * 1024 * 1024  # then the file is renamed to requests.1.jsonl and a new one starts
MAX_SESSIONS = 200
TOP_TOOLS = 12

_ctx = None
_lock = threading.Lock()
# Per conversation: the previous request's serialized form and message fingerprints.
_previous: "OrderedDict[str, dict]" = OrderedDict()


def _setting(key: str, default):
    try:
        value = _ctx.get_config(key, default=default) if _ctx is not None else default
    except Exception:  # noqa: BLE001 - a settings problem must never break a turn
        return default
    return default if value is None else value


def _log_path() -> Optional[Path]:
    try:
        from plugins.plugin_storage import plugin_data_dir  # Hermes plugin public API
        return plugin_data_dir(PLUGIN) / LOG_FILE
    except Exception:  # noqa: BLE001 - outside Hermes
        return None


def _write(record: dict) -> None:
    path = _log_path()
    if path is None:
        return
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _lock:
        try:
            if path.is_file() and path.stat().st_size > MAX_LOG_BYTES:
                os.replace(path, path.with_name("requests.1.jsonl"))
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            logger.debug("%s: could not write %s: %s", PLUGIN, path, exc)


def _text(content: Any) -> str:
    """The text of a message content: a string, or the text parts of a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return "" if content is None else json.dumps(content, ensure_ascii=False, default=str)


def _size(value: Any) -> int:
    """Characters of a value as it travels in the request body (JSON)."""
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _sections(system: str) -> list[list]:
    """[heading, chars] for each top-level part of the system prompt: Markdown "# " / "## " headings
    and XML-style blocks opened on their own line (<available_skills>)."""
    parts: list[list] = []
    title, start = "(start)", 0
    pos = 0
    for line in system.splitlines(keepends=True):
        s = line.strip()
        is_heading = s.startswith(("# ", "## ")) or (
            s.startswith("<") and s.endswith(">") and not s.startswith("</") and " " not in s and len(s) < 60)
        if is_heading and pos > start:
            parts.append([title, pos - start])
            title, start = s[:80], pos
        elif is_heading:
            title = s[:80]
        pos += len(line)
    if pos > start:
        parts.append([title, pos - start])
    return parts


def _messages(request: dict) -> list:
    items = request.get("messages")
    if not isinstance(items, list):
        items = request.get("input") if isinstance(request.get("input"), list) else []
    return [m for m in items if isinstance(m, dict)]


def _system_text(request: dict, messages: list) -> str:
    """The system prompt: Anthropic `system`, Responses `instructions`, or system/developer messages."""
    if request.get("system") is not None:
        return _text(request["system"])
    if isinstance(request.get("instructions"), str):
        return request["instructions"]
    return "".join(_text(m.get("content")) for m in messages if m.get("role") in ("system", "developer"))


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return "?"
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(fn.get("name") or "?")


def measure(request: dict, session_id: str = "", **context: Any) -> dict:
    """The record for one request (pure: also used by the tests)."""
    messages = _messages(request)
    tools = request.get("tools") if isinstance(request.get("tools"), list) else []
    system = _system_text(request, messages)
    tool_sizes = sorted(((_tool_name(t), _size(t)) for t in tools), key=lambda x: -x[1])
    by_role: dict[str, dict] = {}
    for m in messages:
        role = str(m.get("role") or m.get("type") or "?")
        slot = by_role.setdefault(role, {"count": 0, "chars": 0})
        slot["count"] += 1
        slot["chars"] += _size(m)
    last_user = next((len(_text(m.get("content"))) for m in reversed(messages) if m.get("role") == "user"), 0)
    # What the provider receives, in order: tools, system, messages (Anthropic and OpenAI both
    # read their cached prefix in this order). Serialized once to compare with the previous request.
    serial = json.dumps({"tools": tools, "system": request.get("system"), "instructions": request.get("instructions"),
                         "messages": messages}, ensure_ascii=False, default=str)
    fingerprints = [hash(json.dumps(m, ensure_ascii=False, default=str, sort_keys=True)) for m in messages]
    record = {
        "type": "request", "ts": round(time.time(), 3), "session_id": session_id,
        "api_request_id": context.get("api_request_id"), "api_call_count": context.get("api_call_count"),
        "model": context.get("model") or request.get("model"), "provider": context.get("provider"),
        "api_mode": context.get("api_mode"),
        "total_chars": len(serial), "est_tokens": len(serial) // 4,
        "tools": {"count": len(tools), "chars": sum(c for _, c in tool_sizes), "top": tool_sizes[:TOP_TOOLS]},
        "system": {"chars": len(system), "sections": _sections(system)},
        "messages": {"count": len(messages), "by_role": by_role, "last_user_chars": last_user},
    }
    with _lock:
        prev = _previous.get(session_id) if session_id else None
        if session_id:
            _previous[session_id] = {"serial": serial, "fingerprints": fingerprints}
            _previous.move_to_end(session_id)
            while len(_previous) > MAX_SESSIONS:
                _previous.popitem(last=False)
    if prev:
        common = len(os.path.commonprefix([prev["serial"], serial]))
        leading = 0
        for a, b in zip(prev["fingerprints"], fingerprints):
            if a != b:
                break
            leading += 1
        record["prefix"] = {"common_chars": common, "common_pct": round(100 * common / max(1, len(serial)), 1),
                            "same_leading_messages": leading, "previous_messages": len(prev["fingerprints"])}
    return record


def _on_llm_execution(**kwargs: Any) -> Any:
    """llm_execution middleware: measure, then hand the same request to the next layer."""
    request = kwargs.get("request")
    if _setting("enabled", True) and isinstance(request, dict):
        try:
            _write(measure(request, str(kwargs.get("session_id") or ""),
                           **{k: kwargs.get(k) for k in ("api_request_id", "api_call_count", "model",
                                                         "provider", "api_mode")}))
        except Exception as exc:  # noqa: BLE001 - measuring must never break a request
            logger.debug("%s: measure failed: %s", PLUGIN, exc)
    return kwargs["next_call"](request)


def _on_post_api_request(**kwargs: Any) -> None:
    """The provider's token counts for the request measured above (same api_request_id)."""
    if not _setting("enabled", True):
        return
    usage = kwargs.get("usage") or {}
    started, first = kwargs.get("started_at"), kwargs.get("first_chunk_at")
    _write({
        "type": "usage", "ts": round(time.time(), 3), "session_id": kwargs.get("session_id") or "",
        "api_request_id": kwargs.get("api_request_id"), "model": kwargs.get("model"),
        **{k: usage.get(k) for k in ("prompt_tokens", "input_tokens", "cache_read_tokens",
                                      "cache_write_tokens", "output_tokens", "reasoning_tokens")},
        "duration_s": round(kwargs["api_duration"], 3) if isinstance(kwargs.get("api_duration"), (int, float)) else None,
        # Time to first streamed chunk: on a local server that reports no cached tokens (LM Studio),
        # a short one is the sign the prompt was mostly read from the KV cache.
        "ttfb_s": round(first - started, 3) if isinstance(first, (int, float)) and isinstance(started, (int, float)) else None,
    })


def register(ctx) -> None:
    global _ctx
    _ctx = ctx
    ctx.register_middleware("llm_execution", _on_llm_execution)
    ctx.register_hook("post_api_request", _on_post_api_request)
    if hasattr(ctx, "register_cli_command"):
        from . import report
        report.register(ctx)
