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
import sys
import time
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from pondie.extraction.models import Cost, ModelCall, ModelReply

#: One id for a whole run, so calls from several stages are attributable to it.
RUN_ID = os.environ.get("PONDIE_RUN_ID") or uuid.uuid4().hex[:12]


@runtime_checkable
class Caller(Protocol):
    """What a stage needs from a model. Implemented by `GatewayCaller` and by fakes."""

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
        name in type(error).__name__
        for name in ("Connection", "Timeout", "APIError")
    )


class MalformedReply(ValueError):
    """The call succeeded and the body it returned is not JSON.

    Carries what it cost, because the tokens were spent whatever the body says, and the
    stage that catches this adds them to the paper's total rather than losing them.
    """

    def __init__(self, message: str, *, body: str, cost: Cost):
        super().__init__(message)
        self.body, self.cost = body, cost


class GatewayCaller:
    """An OpenAI-compatible gateway, with every request tagged for the analytics API."""

    def __init__(
        self, api_key_env: str = "OPENAI_API_KEY", base_url_env: str = "OPENAI_API_GATEWAY"
    ):
        self._key_env, self._base_env = api_key_env, base_url_env

    def _client(self, paper: str, stage: str):
        from openai import OpenAI

        return OpenAI(
            api_key=os.environ[self._key_env],
            base_url=os.environ.get(self._base_env),
            default_headers={
                "x-portkey-metadata": json.dumps(
                    {"paper": paper, "stage": stage, "run_id": RUN_ID, "pipeline": "pondie"}
                )
            },
        )

    def __call__(self, call: ModelCall, *, paper: str, stage: str) -> ModelReply:
        client = self._client(paper, stage)
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
            started = time.time()
            try:
                raw = client.chat.completions.with_raw_response.create(
                    model=call.model,
                    messages=(
                        [{"role": "system", "content": call.system}] if call.system else []
                    )
                    + [{"role": "user", "content": call.prompt}],
                    max_completion_tokens=call.max_output_tokens,
                    reasoning_effort=call.effort,
                    **({"response_format": {"type": "json_object"}} if constrain else {}),
                    **({"service_tier": call.service_tier} if call.service_tier else {}),
                )
                response = raw.parse()
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
                if _transient(error) and unreachable < UNREACHABLE_TRIES:
                    unreachable += 1
                    # The SDK has already backed off twice inside this one call, so this
                    # spaces whole calls. Jittered: eight workers that hit the same limit
                    # would otherwise return in lockstep and hit it again.
                    time.sleep(min(2.0 ** unreachable, 30.0) * (0.5 + random.random() / 2))
                    continue
                attempt += 1
                continue
            usage = response.usage
            out = getattr(usage, "completion_tokens_details", None)
            inp = getattr(usage, "prompt_tokens_details", None)
            body = response.choices[0].message.content or ""
            spent = Cost(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                reasoning_tokens=getattr(out, "reasoning_tokens", 0) or 0,
                cached_tokens=getattr(inp, "cached_tokens", 0) or 0,
                cache_write_tokens=getattr(inp, "cache_write_tokens", 0) or 0,
                seconds=round(time.time() - started, 2),
                calls=1,
            )
            # Parse inside the loop. A reply that is not JSON is a failed attempt like any
            # other: it used to be parsed in the `return` below, where it escaped both this
            # loop and the post-condition loop in `_ModelPass`, so the one fault the retry
            # machinery exists to absorb was the one it could not see. It also escaped the
            # accounting, and a paper that spent 40,000 tokens logged `calls: 0`.
            finish = response.choices[0].finish_reason or ""
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
            return ModelReply(
                payload=payload,
                stop_reason=finish,
                # Read off the RAW response: both are headers, and the SDK drops them once
                # it has turned the reply into a model object.
                trace_id=raw.headers.get("x-portkey-trace-id") or "",
                cache_status=raw.headers.get("x-portkey-cache-status") or "",
                cost=spent,
            )
        if isinstance(last, MalformedReply):
            raise last
        raise RuntimeError(
            f"{stage} for {paper}: {call.attempts} attempt(s) failed"
            + (f" after {unreachable} that never reached the provider" if unreachable else "")
        ) from last


def _as_json(body: str) -> dict:
    """Model output as a payload. A fenced block is unwrapped; anything else is an error."""
    text = body.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(text)
