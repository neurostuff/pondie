"""The one place a prompt becomes a network call.

Everything above this module deals in `ModelCall` and `ModelReply`, so a stage never touches
a client, a header or a usage object, and a test substitutes a `Caller` rather than patching
the SDK.

Raw response first, parsed second: the trace id is a header, and the SDK discards headers
once it has built the response object. Cost is returned rather than logged, because a stage
that has to scrape its own spend out of its own logging cannot be summed.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

from pondie.extraction.models import Cost, ModelCall, ModelReply

#: One id for a whole run, so calls from several stages are attributable to it.
RUN_ID = os.environ.get("PONDIE_RUN_ID") or uuid.uuid4().hex[:12]


@runtime_checkable
class Caller(Protocol):
    """What a stage needs from a model. Implemented by `GatewayCaller`, `CodexCaller` and
    by fakes."""

    def __call__(self, call: ModelCall, *, paper: str, stage: str) -> ModelReply: ...


def load_env(path: Path) -> list[str]:
    """Read a shell-style env file into the process. Values are never returned or printed."""
    names = []
    for raw in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))
        names.append(name.strip())
    return names


#: Statuses another attempt may clear. Flex is capacity-scheduled and answers 429 when
#: there is none; the 5xx family and the timeouts are the provider or the wire. A 400 or a
#: 422 is the request itself, and retrying it only spends the budget -- worse here, since the
#: post-condition loop lengthens the prompt on each retry, so an over-length context would
#: retry its way further from success.
RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: How many times a call may fail to land before it counts against `attempts`.
UNREACHABLE_TRIES = 4
#: The same for flex's "not sufficient resources", which lasts minutes to hours rather than
#: seconds: four tries within two minutes lost 7 papers of one night's runs to it. Spaced
#: 30 s doubling to 10 min, about 45 minutes in all.
CAPACITY_TRIES = 8


def _no_capacity(error: BaseException) -> bool:
    """Flex's 429 for having no capacity, as opposed to a rate limit."""
    return "resource_unavailable" in str(error) or "sufficient resources" in str(error)


def _transient(error: BaseException) -> bool:
    """Whether this failure is worth another call rather than another answer."""
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        return int(status) in RETRYABLE
    # No status at all is a connection reset, a DNS failure or a read timeout -- the wire,
    # not the request.
    return isinstance(error, (ConnectionError, TimeoutError, OSError)) or any(
        name in type(error).__name__ for name in ("Connection", "Timeout", "APIError")
    )


class MalformedReply(ValueError):
    """The call succeeded and the body it returned is not JSON.

    Carries what it cost, because the tokens were spent whatever the body says, and the
    stage that catches this adds them to the paper's total rather than losing them.
    """

    def __init__(self, message: str, *, body: str, cost: Cost):
        super().__init__(message)
        self.body, self.cost = body, cost


@dataclass(frozen=True)
class _Sent:
    """One request that reached the model: its body as text, unparsed, and what it cost."""

    body: str
    finish: str
    cost: Cost
    trace_id: str = ""
    cache_status: str = ""


class _RetryingCaller:
    """The attempt budget, the transport retries and the parse, around one `_send`.

    A subclass says how one request reaches a model; this says what counts as a failure and
    how often to try again, so two routes to the same model fail and retry identically and
    a comparison between them measures the route, not the retry policy.
    """

    def _send(self, call: ModelCall, *, constrain: bool, paper: str, stage: str) -> _Sent:
        """One request. Raises on any failure to get a body back."""
        raise NotImplementedError

    def __call__(self, call: ModelCall, *, paper: str, stage: str) -> ModelReply:
        last: Exception | None = None
        # Dropped for the rest of this call if the provider says it does not know the
        # parameter, so a gateway without JSON mode degrades to the old behaviour instead
        # of failing every attempt on an argument error.
        constrain = call.json_object
        # Two budgets, because they answer different questions. `call.attempts` is for
        # faults the model owns -- an unparseable body, a post-condition it failed. Reaching
        # the provider at all is not one of those, and a call that never landed spent no
        # tokens, so it must not consume an attempt. Four papers in one batch died on
        # `1 attempt(s) failed` while `settings.attempts` was 3, because a transport error
        # raised straight past the loop that absorbs model faults; re-running all four later
        # succeeded, which is what transient means.
        attempt = unreachable = 0
        while attempt < call.attempts:
            try:
                sent = self._send(call, constrain=constrain, paper=paper, stage=stage)
            except Exception as error:  # noqa: BLE001 -- retried, then surfaced
                last = error
                if constrain and "response_format" in str(error):
                    print(
                        f"  {stage}: gateway rejected response_format; "
                        f"retrying without JSON mode",
                        file=sys.stderr,
                    )
                    constrain = False
                    continue
                capacity = _no_capacity(error)
                if _transient(error) and unreachable < (
                    CAPACITY_TRIES if capacity else UNREACHABLE_TRIES
                ):
                    unreachable += 1
                    # The SDK has already backed off twice inside this one call, so this
                    # spaces whole calls. Jittered: eight workers that hit the same limit
                    # would otherwise return in lockstep and hit it again.
                    wait = min(15.0 * 2.0**unreachable, 600.0) if capacity else min(
                        2.0**unreachable, 30.0
                    )
                    time.sleep(wait * (0.5 + random.random() / 2))
                    continue
                attempt += 1
                continue
            body, finish, spent = sent.body, sent.finish, sent.cost
            # Parse inside the loop. A reply that is not JSON is a failed attempt like any
            # other: it used to be parsed in the `return` below, where it escaped both this
            # loop and the post-condition loop in `_ModelPass`, so the one fault the retry
            # machinery exists to absorb was the one it could not see. It also escaped the
            # accounting, and a paper that spent 40,000 tokens logged `calls: 0`.
            try:
                payload = _as_json(body)
            except json.JSONDecodeError as error:
                # The finish reason is the difference between two faults that look
                # identical from the parse error alone: `length` is the model being cut
                # off mid-object and wants a bigger `max_output_tokens`, anything else is
                # a body that ended where it meant to and came out malformed anyway, which
                # wants a retry or JSON mode. It was recorded on `ModelReply` and read by
                # nothing, so an investigation into 25 unparseable papers had to rule
                # truncation out by measuring instead of by looking.
                last = MalformedReply(
                    f"reply was not valid JSON (finish_reason={finish!r}): {error}",
                    body=body,
                    cost=spent,
                )
                # A body that will not parse is the model's fault and spent real tokens, so
                # it costs an attempt -- unlike a call that never landed.
                attempt += 1
                continue
            if constrain and call.json_schema is not None:
                payload = _without_nulls(payload)
            return ModelReply(
                payload=payload,
                stop_reason=finish,
                trace_id=sent.trace_id,
                cache_status=sent.cache_status,
                cost=spent,
            )
        if isinstance(last, MalformedReply):
            raise last
        # `from last` chains the cause, and the CLI prints `str(error)`, so the chain was
        # invisible: a gateway 400 -- a model name the account cannot reach, a parameter it
        # rejects -- surfaced as `1 attempt(s) failed` and took two further runs at debug to
        # read off the wire. The reason belongs in the message that gets printed.
        raise RuntimeError(
            f"{stage} for {paper}: {call.attempts} attempt(s) failed"
            + (f" after {unreachable} that never reached the provider" if unreachable else "")
            + (f": {type(last).__name__}: {str(last)[:400]}" if last else "")
        ) from last


class GatewayCaller(_RetryingCaller):
    """An OpenAI-compatible gateway, with every request tagged for the analytics API."""

    def __init__(
        self, api_key_env: str = "OPENAI_API_KEY", base_url_env: str = "OPENAI_API_GATEWAY"
    ):
        self._key_env, self._base_env = api_key_env, base_url_env
        self._shared = None
        self._lock = threading.Lock()

    def _client(self):
        """One client for the whole run, built on first use.

        The connection pool is the reason it is shared. Every `OpenAI()` builds its own
        httpx pool, so a client per call reuses no connection and pays a TLS handshake per
        call -- invisible at eight workers, and at ninety-six a burst of handshakes against
        a 1024 file-descriptor soft limit. What forced a client per call was the Portkey
        metadata, which named the paper and the stage in `default_headers`; that is
        per-request information and is passed per request below, so nothing is left that
        varies between calls.

        Locked and double-checked because the scheduler runs items on a thread pool and the
        first calls arrive together: without it the first N workers each build a client and
        all but one is discarded, which is the per-call cost this removes, on the one batch
        where it is largest. The client itself is thread-safe and is meant to be shared.
        """
        if self._shared is None:
            with self._lock:
                if self._shared is None:
                    from openai import OpenAI

                    self._shared = OpenAI(
                        api_key=os.environ[self._key_env],
                        base_url=os.environ.get(self._base_env),
                    )
        return self._shared

    def _send(self, call: ModelCall, *, constrain: bool, paper: str, stage: str) -> _Sent:
        # Per request, not per client: this is what a call is about, and putting it on the
        # client is what used to make the client unshareable.
        metadata = {
            "x-portkey-metadata": json.dumps(
                {"paper": paper, "stage": stage, "run_id": RUN_ID, "pipeline": "pondie"}
            )
        }
        started = time.time()
        raw = self._client().chat.completions.with_raw_response.create(
            model=call.model,
            messages=([{"role": "system", "content": call.system}] if call.system else [])
            + [{"role": "user", "content": call.prompt}],
            max_completion_tokens=call.max_output_tokens,
            reasoning_effort=call.effort,
            **({"response_format": _format(call)} if constrain else {}),
            **({"service_tier": call.service_tier} if call.service_tier else {}),
            extra_headers=metadata,
        )
        response = raw.parse()
        usage = response.usage
        out = getattr(usage, "completion_tokens_details", None)
        inp = getattr(usage, "prompt_tokens_details", None)
        return _Sent(
            body=response.choices[0].message.content or "",
            finish=response.choices[0].finish_reason or "",
            cost=Cost(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                reasoning_tokens=getattr(out, "reasoning_tokens", 0) or 0,
                cached_tokens=getattr(inp, "cached_tokens", 0) or 0,
                cache_write_tokens=getattr(inp, "cache_write_tokens", 0) or 0,
                seconds=round(time.time() - started, 2),
                calls=1,
            ),
            # Read off the RAW response: both are headers, and the SDK drops them once it
            # has turned the reply into a model object.
            trace_id=raw.headers.get("x-portkey-trace-id") or "",
            cache_status=raw.headers.get("x-portkey-cache-status") or "",
        )


class CodexError(RuntimeError):
    """`codex exec` ended without a completed turn. `status_code` is the provider's, if it
    gave one, so `_transient` sorts a 429 from a 400 exactly as it does for the gateway."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CodexUsageLimit(CodexError):
    """The `codex login` account's allowance is spent. `retry_at` is when codex says it
    resets, or None when the message names no time."""

    def __init__(self, message: str, retry_at: datetime | None):
        super().__init__(message)
        self.retry_at = retry_at


#: "You've hit your usage limit. Try again at 11:36 PM." -- a local clock time.
_RETRY_AT = re.compile(r"try again at (\d{1,2}):(\d{2})\s*([AP]M)", re.I)

#: The longest one wait for a reset may last, and the wait when no time is given.
_USAGE_WAIT_CAP = 24 * 3600.0
_USAGE_WAIT_UNKNOWN = 1800.0


def _retry_at(message: str, now: datetime | None = None) -> datetime | None:
    """The next local time matching the "Try again at" clause, or None."""
    match = _RETRY_AT.search(message)
    if not match:
        return None
    hour, minute, half = int(match[1]) % 12, int(match[2]), match[3].upper()
    now = now or datetime.now()
    at = now.replace(
        hour=hour + (12 if half == "PM" else 0), minute=minute, second=0, microsecond=0
    )
    return at if at > now else at + timedelta(days=1)


#: Codex features that put a tool, or instructions about one, in front of the model. A
#: pondie call is one prompt and one JSON answer; every tool is prompt the API route does
#: not send, and a turn the model could spend on something other than answering. Measured
#: on codex-cli 0.161 for a one-line prompt: 13,944 input tokens as installed, 4,569 with
#: these off and pondie's system prompt in place of codex's own. The remainder is codex's
#: collaboration-mode and multi-agent preamble, which no setting removes.
_CODEX_TOOLS = (
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "goals",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "plugins",
    "shell_tool",
    "skill_search",
    "sleep_tool",
    "tool_suggest",
    "unified_exec",
    "view_image",
)

#: One turn, no session saved, no user config or rules, and no write access.
_CODEX_EXEC_FLAGS = (
    "--json",
    "--ephemeral",
    "--skip-git-repo-check",
    "--sandbox",
    "read-only",
    "--ignore-user-config",
    "--ignore-rules",
)

#: Credentials that would make codex bill an API key instead of the `codex login` account.
#: `--env` loads the gateway's key into this process, and a child inherits it.
_API_KEY_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL")


class CodexCaller(_RetryingCaller):
    """The same calls, sent through `codex exec` on the `codex login` account.

    For comparing a route, not a model: the model and the reasoning effort are the call's,
    the system prompt replaces codex's own instructions, the schema goes in as
    `--output-schema`, and the attempt and retry policy is `_RetryingCaller`'s. What differs
    is what the route cannot carry:

    - `service_tier` is not sent. The account's backend refuses `flex` with a 400, and the
      tier is billing and scheduling, not what the model is asked.
    - `max_output_tokens` is not sent; codex has no per-request output cap.
    - `stop_reason` is empty; codex does not report the provider's finish reason.
    - Codex's preamble (see `_CODEX_TOOLS`) adds a few thousand input tokens per call.

    A gateway model name, `@provider/model`, is sent as `model`.
    """

    def __init__(self, binary: str = "codex", timeout: float = 3600.0):
        self._binary, self._timeout = binary, timeout
        # Asked of the installed codex, not assumed: an unknown name to `--disable` is a
        # hard error, so one feature renamed in an upgrade would fail every call. Asking
        # here also finds a missing binary before the first paper, where `_transient` would
        # otherwise retry its `FileNotFoundError` as a network fault.
        try:
            listed = subprocess.run(
                [binary, "features", "list"], capture_output=True, text=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"cannot run `{binary} features list`: {error}") from error
        known = {line.split()[0] for line in listed.splitlines() if line.strip()}
        self._disabled = tuple(name for name in _CODEX_TOOLS if name in known)

    def _send(self, call: ModelCall, *, constrain: bool, paper: str, stage: str) -> _Sent:
        if not call.system.strip():
            # Codex refuses an empty instructions file, and leaving it unset sends codex's
            # own agent prompt -- a different request from the gateway's, silently.
            raise ValueError("CodexCaller needs a system prompt; codex has no empty one")
        model = call.model.split("/", 1)[1] if call.model.startswith("@") else call.model
        env = {k: v for k, v in os.environ.items() if k not in _API_KEY_ENV}
        # An empty directory per call: nothing for codex to read as project context, and
        # nowhere a tool could write if one were left on.
        with tempfile.TemporaryDirectory(prefix="pondie-codex-") as work:
            instructions = Path(work) / "instructions.md"
            instructions.write_text(call.system, encoding="utf-8")
            config = {
                "model_reasoning_effort": json.dumps(call.effort),
                "model_instructions_file": json.dumps(str(instructions)),
                "include_environment_context": "false",
                "include_permissions_instructions": "false",
                "include_apps_instructions": "false",
                "web_search": '"disabled"',
            }
            args = [
                self._binary,
                "exec",
                *_CODEX_EXEC_FLAGS,
                *("-m", model),
                *(arg for key, value in config.items() for arg in ("-c", f"{key}={value}")),
                *(arg for name in self._disabled for arg in ("--disable", name)),
            ]
            if constrain and call.json_schema is not None:
                schema = Path(work) / "schema.json"
                schema.write_text(json.dumps(call.json_schema), encoding="utf-8")
                args += ["--output-schema", str(schema)]
            # A spent allowance is waited out here, below the retry loop: it is neither the
            # model's fault nor a blip, and failing on it fails every paper a long run
            # reaches until the reset -- each spends its attempts in seconds.
            while True:
                started = time.time()
                try:
                    done = subprocess.run(
                        [*args, "-"],
                        input=call.prompt,
                        capture_output=True,
                        text=True,
                        cwd=work,
                        env=env,
                        timeout=self._timeout,
                    )
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError(
                        f"codex exec gave no answer in {self._timeout:.0f}s"
                    ) from error
                try:
                    return _codex_reply(done, seconds=round(time.time() - started, 2))
                except CodexUsageLimit as limit:
                    now = datetime.now()
                    wait = (
                        (limit.retry_at - now).total_seconds() + 60
                        if limit.retry_at
                        else _USAGE_WAIT_UNKNOWN
                    )
                    wait = min(max(wait, 60.0), _USAGE_WAIT_CAP)
                    print(
                        f"  {stage} for {paper}: codex usage limit; waiting until "
                        f"{(now + timedelta(seconds=wait)):%H:%M}",
                        file=sys.stderr,
                    )
                    time.sleep(wait)


def _codex_reply(done: subprocess.CompletedProcess, *, seconds: float) -> _Sent:
    """The answer and the usage out of `codex exec --json`'s event stream."""
    thread = body = failure = ""
    usage = None
    for line in done.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "thread.started":
            thread = event.get("thread_id", "")
        elif kind == "item.completed" and event.get("item", {}).get("type") == "agent_message":
            body = event["item"].get("text", "")
        elif kind == "turn.completed":
            usage = event.get("usage") or {}
        elif kind == "turn.failed":
            failure = event.get("error", {}).get("message", "")
        elif kind == "error":
            failure = failure or event.get("message", "")
    if usage is None:
        detail = failure or done.stderr.strip()[-400:] or f"exit status {done.returncode}"
        if "usage limit" in detail.lower():
            raise CodexUsageLimit(f"codex exec: {detail}", _retry_at(detail))
        status = re.search(r'"status"\s*:\s*(\d{3})', detail)
        raise CodexError(f"codex exec: {detail}", int(status.group(1)) if status else None)
    return _Sent(
        body=body,
        finish="",
        cost=Cost(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=usage.get("reasoning_output_tokens", 0),
            cached_tokens=usage.get("cached_input_tokens", 0),
            cache_write_tokens=usage.get("cache_write_input_tokens", 0),
            seconds=seconds,
            calls=1,
        ),
        trace_id=thread,
    )


def _without_nulls(node):
    """Remove null-valued keys, everywhere: strict mode's spelling of an absent slot."""
    if isinstance(node, dict):
        return {k: _without_nulls(v) for k, v in node.items() if v is not None}
    if isinstance(node, list):
        return [_without_nulls(v) for v in node]
    return node


def _format(call: ModelCall) -> dict:
    """The `response_format` a call asks for: its schema, strictly, or plain JSON mode."""
    if call.json_schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": "reply", "strict": True, "schema": call.json_schema},
    }


def _as_json(body: str) -> dict:
    """Model output as a payload. A fenced block is unwrapped; anything else is an error."""
    text = body.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(text)
