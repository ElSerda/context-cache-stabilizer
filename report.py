"""`hermes context-cache report`: the measured requests of a conversation, as tables."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def _records(path: Optional[Path]) -> list[dict]:
    if path is None:
        return []
    rows: list[dict] = []
    for p in (path.with_name("requests.1.jsonl"), path):
        try:
            lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
        except OSError:
            continue
        for line in lines:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def _join(rows: list[dict]) -> list[dict]:
    """Request records with their usage record (same api_request_id) under "usage"."""
    usage = {r.get("api_request_id"): r for r in rows if r.get("type") == "usage" and r.get("api_request_id")}
    return [dict(r, usage=usage.get(r.get("api_request_id"), {})) for r in rows if r.get("type") == "request"]


def _k(n: Any) -> str:
    if not isinstance(n, (int, float)):
        return "-"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(int(n))


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = lambda r: "  ".join(str(x).rjust(w) if i else str(x).ljust(w) for i, (x, w) in enumerate(zip(r, widths)))
    return "\n".join([fmt(header), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows])


def render(rows: list[dict], session: str = "", last: int = 30) -> str:
    requests = _join(rows)
    if not requests:
        return "No request measured yet."
    if not session:
        session = requests[-1].get("session_id", "")
    picked = [r for r in requests if r.get("session_id", "").startswith(session)]
    if not picked:
        return f"No request measured for session {session}."
    out = [f"Session {picked[-1].get('session_id')}  ({len(picked)} requests, model {picked[-1].get('model')})", ""]
    table = []
    for prev, r in zip([None] + picked[:-1], picked):
        prefix = r.get("prefix") or {}
        if prev and "common_chars" in prefix and "previous_kept_pct" not in prefix:  # 0.2.0 records
            r["prefix"] = dict(prefix, previous_kept_pct=round(100 * prefix["common_chars"] / max(1, prev["total_chars"]), 1))
    for i, r in enumerate(picked[-last:], max(1, len(picked) - last + 1)):
        u, roles, prefix = r["usage"], r["messages"]["by_role"], r.get("prefix") or {}
        table.append([
            str(i), time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0))),
            _k(r.get("est_tokens")), _k(u.get("prompt_tokens")), _k(u.get("cache_read_tokens")),
            _k(r["tools"]["chars"]), _k(r["system"]["chars"]),
            _k(roles.get("user", {}).get("chars")), _k(roles.get("assistant", {}).get("chars")),
            _k(roles.get("tool", {}).get("chars")),
            f"{prefix['previous_kept_pct']}%" if "previous_kept_pct" in prefix else "-",
            f"{100 - prefix['common_pct']:.1f}%" if "common_pct" in prefix else "-",
            f"{u['ttfb_s']}s" if isinstance(u.get("ttfb_s"), (int, float)) else "-",
        ])
    out.append(_table(["#", "time", "est tok", "prompt tok", "cached", "tools ch", "system ch",
                       "user ch", "assist ch", "tool ch", "prev kept", "new", "ttfb"], table))
    latest = picked[-1]
    out += ["", "System prompt of the latest request, largest parts first:"]
    sections = sorted(latest["system"]["sections"], key=lambda s: -s[1])
    out.append(_table(["part", "chars"], [[t, str(c)] for t, c in sections]))
    out += ["", f"Tools of the latest request ({latest['tools']['count']}), largest first:"]
    out.append(_table(["tool", "chars"], [[n, str(c)] for n, c in latest["tools"]["top"]]))
    out += ["", "est tok = request characters / 4. prompt tok / cached = what the provider reported "
                "(LM Studio reports no cached tokens: a short ttfb is the sign). prev kept = share of the "
                "previous request found unchanged at the start of this one (below 100%: a prefix cache "
                "recomputes from the first difference). new = share of this request not in the previous one."]
    return "\n".join(out)


def _setup(parser) -> None:
    sub = parser.add_subparsers(dest="context_cache_cmd")
    rep = sub.add_parser("report", help="Show the measured requests of a conversation")
    rep.add_argument("--session", default="", help="session id or its beginning (default: the latest)")
    rep.add_argument("--last", type=int, default=30, help="how many requests to show (default 30)")
    sub.add_parser("path", help="Print where the measurements are written")


def _handle(args) -> int:
    from . import _log_path
    path = _log_path()
    if getattr(args, "context_cache_cmd", None) == "path":
        print(path)
        return 0
    print(render(_records(path), getattr(args, "session", ""), getattr(args, "last", 30)))
    return 0


def register(ctx) -> None:
    ctx.register_cli_command(name="context-cache", help="Measure what Hermes sends to the model",
                             setup_fn=_setup, handler_fn=_handle)
