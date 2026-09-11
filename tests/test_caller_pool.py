"""One client for the whole run, and the per-call metadata on the request.

`GatewayCaller` used to build an `OpenAI()` per call, because it named the paper and the
stage in `default_headers` and those change every call. Every client brings its own httpx
connection pool, so that was a TLS handshake per call and no connection reused -- invisible
at eight workers, and at ninety-six a burst of handshakes against a 1024 file-descriptor
soft limit.

Both halves are pinned, because reverting either restores the cost silently: a client
rebuilt per call still returns the right answer, and metadata left on the client still
reaches Portkey's analytics. Only the file-descriptor count would say, and nothing reads it.
"""

from __future__ import annotations

import json
import threading

import pytest

from pondie.extraction.llm import GatewayCaller
from pondie.extraction.models import ModelCall


class Reply:
    def __init__(self, body):
        self.choices = [
            type(
                "C", (), {"message": type("M", (), {"content": body})(), "finish_reason": "stop"}
            )()
        ]
        self.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()


class Raw:
    headers: dict = {}

    def parse(self):
        return Reply('{"ok": 1}')


class FakeClient:
    """Records what every request was given, and how many clients were built."""

    built: list["FakeClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.requests: list[dict] = []
        FakeClient.built.append(self)
        self.chat = type(
            "Chat", (), {"completions": type("Cmp", (), {"with_raw_response": self})()}
        )()

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return Raw()


@pytest.fixture
def openai(monkeypatch):
    FakeClient.built = []
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_GATEWAY", "https://gateway.invalid")
    import openai as sdk

    monkeypatch.setattr(sdk, "OpenAI", FakeClient)
    return FakeClient


def call(n=1):
    return ModelCall(model="m", prompt="p", max_output_tokens=10, attempts=n)


def test_one_client_serves_every_call(openai):
    caller = GatewayCaller()
    for i in range(5):
        caller(call(), paper=f"S{i}", stage="demands")
    assert len(openai.built) == 1, "a client per call is a connection pool per call"
    assert len(openai.built[0].requests) == 5


def test_the_first_calls_racing_still_build_one_client(openai):
    """The scheduler starts every worker at once, so the first calls arrive together --
    the one moment an unlocked lazy build makes N clients and discards N-1."""
    caller = GatewayCaller()
    start = threading.Barrier(8)

    def one(i):
        start.wait()
        caller(call(), paper=f"S{i}", stage="demands")

    threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(openai.built) == 1


def test_the_paper_and_stage_travel_on_the_request(openai):
    caller = GatewayCaller()
    caller(call(), paper="S7", stage="satisfy")
    sent = json.loads(openai.built[0].requests[0]["extra_headers"]["x-portkey-metadata"])
    assert sent["paper"] == "S7"
    assert sent["stage"] == "satisfy"
    assert sent["pipeline"] == "pondie"
    assert sent["run_id"]


def test_each_request_carries_its_own_metadata(openai):
    """The failure a shared client invites: one header set on the client, so every call
    after the first is attributed to the first call's paper."""
    caller = GatewayCaller()
    caller(call(), paper="S1", stage="demands")
    caller(call(), paper="S2", stage="fill")
    papers = [
        json.loads(r["extra_headers"]["x-portkey-metadata"])["paper"]
        for r in openai.built[0].requests
    ]
    assert papers == ["S1", "S2"]


def test_nothing_that_varies_per_call_is_on_the_client(openai):
    caller = GatewayCaller()
    caller(call(), paper="S1", stage="demands")
    assert (
        "default_headers" not in openai.built[0].kwargs
    ), "a per-call value in default_headers is what made the client unshareable"
