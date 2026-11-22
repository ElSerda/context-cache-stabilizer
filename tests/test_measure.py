"""Measurement tests with a fake Hermes context (no Hermes needed)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    name = "context_cache_stabilizer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ccs = load()
report = importlib.import_module("context_cache_stabilizer.report")

SYSTEM = "You are Hermes.\n# Finishing the job\nDo it.\n<available_skills>\n- a\n- b\n</available_skills>\n## Skills\nmore\n"
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "run", "parameters": {}}},
         {"type": "function", "function": {"name": "read_file", "description": "read a file " * 20, "parameters": {}}}]


def request(*turns):
    msgs = [{"role": "system", "content": SYSTEM}]
    for role, text in turns:
        msgs.append({"role": role, "content": text})
    return {"model": "m", "messages": msgs, "tools": TOOLS}


class FakeCtx:
    def __init__(self, settings=None):
        self.settings, self.hooks, self.middleware, self.commands = settings or {}, {}, {}, {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, name, fn, **_):
        self.hooks[name] = fn

    def register_middleware(self, kind, fn):
        self.middleware[kind] = fn

    def register_cli_command(self, name, **kw):
        self.commands[name] = kw


@pytest.fixture
def log(tmp_path, monkeypatch):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(ccs, "_log_path", lambda: path)
    ccs._previous.clear()
    return path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sizes_by_block():
    ccs._previous.clear()
    r = ccs.measure(request(("user", "hello"), ("assistant", "hi"), ("tool", "x" * 500)), "s")
    assert r["tools"]["count"] == 2 and r["tools"]["top"][0][0] == "read_file"
    assert r["tools"]["chars"] == sum(c for _, c in r["tools"]["top"])
    assert r["system"]["chars"] == len(SYSTEM)
    assert [t for t, _ in r["system"]["sections"]] == ["(start)", "# Finishing the job", "<available_skills>", "## Skills"]
    assert sum(c for _, c in r["system"]["sections"]) == len(SYSTEM)
    assert r["messages"]["by_role"]["tool"]["chars"] > 500 and r["messages"]["last_user_chars"] == 5
    assert "prefix" not in r  # first request of the conversation


def test_prefix_reuse_against_the_previous_request():
    ccs._previous.clear()
    ccs.measure(request(("user", "q1")), "s")
    r = ccs.measure(request(("user", "q1"), ("assistant", "a1"), ("user", "q2")), "s")
    assert r["prefix"]["same_leading_messages"] == 2 and r["prefix"]["previous_messages"] == 2
    assert 50 < r["prefix"]["common_pct"] < 100
    changed = request(("user", "q1 edited"), ("assistant", "a1"), ("user", "q2"))
    r = ccs.measure(changed, "s")
    assert r["prefix"]["same_leading_messages"] == 1  # only the system message is unchanged
    other = ccs.measure(request(("user", "q1")), "other")
    assert "prefix" not in other  # conversations are kept apart


def test_anthropic_and_responses_shapes():
    ccs._previous.clear()
    r = ccs.measure({"system": [{"type": "text", "text": SYSTEM}], "messages": [{"role": "user", "content": "q"}],
                     "tools": [{"name": "terminal", "input_schema": {}}]}, "s")
    assert r["system"]["chars"] == len(SYSTEM) and r["tools"]["top"][0][0] == "terminal"
    r = ccs.measure({"instructions": SYSTEM, "input": [{"role": "user", "content": [{"type": "input_text", "text": "q"}]}]}, "t")
    assert r["system"]["chars"] == len(SYSTEM) and r["messages"]["count"] == 1


def test_middleware_hands_the_request_on_unchanged_and_logs(log):
    ctx = FakeCtx()
    ccs.register(ctx)
    req = request(("user", "q"))
    sent = []
    result = ctx.middleware["llm_execution"](request=req, original_request=req, next_call=lambda r: sent.append(r) or "reply",
                                             session_id="s", api_request_id="r1", model="m")
    assert result == "reply" and sent == [req]
    ctx.hooks["post_api_request"](session_id="s", api_request_id="r1", model="m", api_duration=1.5,
                                  started_at=100.0, first_chunk_at=100.4,
                                  usage={"prompt_tokens": 900, "cache_read_tokens": 800, "input_tokens": 100})
    req_row, usage_row = rows(log)
    assert req_row["type"] == "request" and req_row["api_request_id"] == "r1"
    assert usage_row["prompt_tokens"] == 900 and usage_row["ttfb_s"] == 0.4
    assert "q" not in json.dumps(req_row["messages"])  # sizes only, no text


def test_a_measuring_error_never_blocks_the_request(log, monkeypatch):
    ctx = FakeCtx()
    ccs.register(ctx)
    monkeypatch.setattr(ccs, "measure", lambda *a, **k: 1 / 0)
    assert ctx.middleware["llm_execution"](request=request(), next_call=lambda r: "ok") == "ok"


def test_disabled_writes_nothing(log):
    ctx = FakeCtx({"enabled": False})
    ccs.register(ctx)
    assert ctx.middleware["llm_execution"](request=request(), next_call=lambda r: "ok") == "ok"
    ctx.hooks["post_api_request"](session_id="s", api_request_id="r1", usage={})
    assert not log.exists()


def test_report_joins_usage_and_shows_the_latest_session(log):
    ctx = FakeCtx()
    ccs.register(ctx)
    for i, turns in enumerate([[("user", "q1")], [("user", "q1"), ("assistant", "a1"), ("user", "q2")]]):
        ctx.middleware["llm_execution"](request=request(*turns), next_call=lambda r: None, session_id="sess-1",
                                        api_request_id=f"r{i}", model="m")
        ctx.hooks["post_api_request"](session_id="sess-1", api_request_id=f"r{i}", usage={"prompt_tokens": 1000 + i})
    text = report.render(report._records(log))
    assert "Session sess-1  (2 requests" in text and "1.0k" in text
    assert "<available_skills>" in text and "read_file" in text
    assert report.render([], "") == "No request measured yet."
    assert "No request measured for session zzz" in report.render(report._records(log), "zzz")
