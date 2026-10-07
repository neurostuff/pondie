"""`CodexCaller` sends the call it was given, and reads back what codex says it cost.

The fake below is an executable on disk, not a patched `subprocess.run`, so what is pinned
is what actually crosses the process boundary: argv, stdin, environment and the files the
call wrote for codex to read.
"""

from __future__ import annotations

import json
import stat
import sys

import pytest

from pondie.extraction import llm
from pondie.extraction.llm import CodexCaller, MalformedReply
from pondie.extraction.models import ModelCall

FAKE = """#!{python}
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]
if argv[:2] == ["features", "list"]:
    # Every feature `CodexCaller` disables but one, which this codex has never heard of.
    for name in {features!r}:
        print(f"{{name:40}} stable             true")
    sys.exit(0)
config = dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv) if a == "-c")
instructions = json.loads(config.get("model_instructions_file", '""'))
schema = argv[argv.index("--output-schema") + 1] if "--output-schema" in argv else None
seen = Path({log!r})
calls = json.loads(seen.read_text()) if seen.exists() else []
calls.append({{
    "argv": argv,
    "stdin": sys.stdin.read(),
    "env": {{k: os.environ.get(k) for k in ("OPENAI_API_KEY", "CODEX_API_KEY", "KEEP")}},
    "instructions": Path(instructions).read_text() if instructions else None,
    "schema": json.loads(Path(schema).read_text()) if schema else None,
}})
seen.write_text(json.dumps(calls))
replies = json.loads(Path({replies!r}).read_text())
for line in replies[min(len(calls), len(replies)) - 1]:
    print(json.dumps(line))
"""


def ok(text, **usage):
    return [
        {"type": "thread.started", "thread_id": "th-1"},
        {"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": text}},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 30,
                "reasoning_output_tokens": 20,
                **usage,
            },
        },
    ]


def failed(status):
    message = json.dumps({"type": "error", "status": status, "error": {"message": "no"}})
    return [{"type": "turn.failed", "error": {"message": message}}]


@pytest.fixture
def codex(tmp_path, monkeypatch):
    """A fake `codex` that answers with each reply list in turn; returns (caller, calls)."""
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    log, replies = tmp_path / "calls.json", tmp_path / "replies.json"
    binary = tmp_path / "codex"
    features = [name for name in llm._CODEX_TOOLS if name != "sleep_tool"]
    binary.write_text(
        FAKE.format(python=sys.executable, log=str(log), replies=str(replies), features=features)
    )
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)

    def make(*answers):
        replies.write_text(json.dumps(answers))
        return CodexCaller(binary=str(binary)), lambda: json.loads(log.read_text())

    return make


def call(**kw):
    return ModelCall(
        **{
            "model": "@gw-provider/gpt-x",
            "system": "be terse",
            "prompt": "PAPER",
            "effort": "medium",
            "service_tier": "flex",
            **kw,
        }
    )


def test_the_call_reaches_codex_as_it_would_reach_the_gateway(codex, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "gateway-key")
    monkeypatch.setenv("KEEP", "yes")
    caller, calls = codex(ok('{"a": 1}'))
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    caller(call(json_schema=schema), paper="S1", stage="single")
    [sent] = calls()
    argv = sent["argv"]
    assert argv[argv.index("-m") + 1] == "gpt-x", "the gateway's provider prefix is dropped"
    assert 'model_reasoning_effort="medium"' in argv
    assert sent["instructions"] == "be terse", "the system prompt replaces codex's own"
    assert sent["stdin"] == "PAPER"
    assert sent["schema"] == schema
    assert not any("service_tier" in a for a in argv), "the account refuses flex"
    assert sent["env"]["OPENAI_API_KEY"] is None, "a gateway key must not redirect billing"
    assert sent["env"]["KEEP"] == "yes"
    assert "shell_tool" in argv and argv[argv.index("shell_tool") - 1] == "--disable"


def test_a_feature_this_codex_does_not_know_is_not_disabled(codex):
    """`--disable` of an unknown name is a hard error in codex, so it would fail every call."""
    caller, calls = codex(ok('{"a": 1}'))
    caller(call(), paper="S1", stage="demands")
    argv = calls()[0]["argv"]
    assert "sleep_tool" not in argv
    assert "view_image" in argv


def test_a_missing_binary_fails_before_the_first_call(tmp_path):
    """Not at the first call, where `_transient` reads `FileNotFoundError` as the wire."""
    with pytest.raises(RuntimeError, match="cannot run"):
        CodexCaller(binary=str(tmp_path / "no-codex"))


def test_no_schema_is_sent_for_plain_json_mode(codex):
    caller, calls = codex(ok('{"a": 1}'))
    caller(call(), paper="S1", stage="demands")
    assert calls()[0]["schema"] is None


def test_the_reply_carries_codex_usage_and_thread(codex):
    caller, _ = codex(ok('{"a": 1, "b": null}'))
    reply = caller(call(json_schema={"type": "object"}), paper="S1", stage="single")
    assert reply.payload == {"a": 1}, "strict mode's nulls are dropped as on the gateway"
    assert reply.trace_id == "th-1"
    cost = reply.cost
    assert (
        cost.input_tokens,
        cost.cached_tokens,
        cost.output_tokens,
        cost.reasoning_tokens,
        cost.calls,
    ) == (100, 40, 30, 20, 1)


def test_a_rate_limit_is_retried_without_spending_an_attempt(codex):
    caller, calls = codex(failed(429), ok('{"a": 1}'))
    reply = caller(call(attempts=1), paper="S1", stage="single")
    assert reply.payload == {"a": 1} and len(calls()) == 2


def test_a_refused_request_spends_its_attempts_and_says_why(codex):
    caller, calls = codex(failed(400))
    with pytest.raises(RuntimeError, match="CodexError.*400"):
        caller(call(attempts=2), paper="S1", stage="single")
    assert len(calls()) == 2


def test_no_system_prompt_is_refused_rather_than_replaced_by_codexs(codex):
    caller, _ = codex(ok('{"a": 1}'))
    with pytest.raises(RuntimeError, match="needs a system prompt"):
        caller(call(system="", attempts=1), paper="S1", stage="single")


def test_an_unparseable_body_is_a_failed_attempt(codex):
    caller, _ = codex(ok('{"broken": '), ok('{"ok": 1}'))
    assert caller(call(attempts=2), paper="S1", stage="single").payload == {"ok": 1}
    caller, _ = codex(ok('{"broken": '))
    with pytest.raises(MalformedReply) as raised:
        caller(call(attempts=1), paper="S1", stage="single")
    assert raised.value.cost.input_tokens == 100
