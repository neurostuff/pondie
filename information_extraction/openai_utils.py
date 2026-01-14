from __future__ import annotations

import json
import random
import time
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)


def _response_format(schema: dict[str, Any], name: str, use_schema: bool) -> dict[str, Any]:
    if use_schema:
        return {"format": {"type": "json_schema", "name": name, "schema": schema}}
    return {"format": {"type": "json_object"}}


def _backoff_sleep(attempt: int, base: float) -> None:
    delay = base * (2**attempt)
    jitter = random.uniform(0, base)
    time.sleep(delay + jitter)


def _extract_output_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    output = getattr(response, "output", None) or []
    for item in output:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) in {"output_text", "text"}:
                return getattr(content, "text", "")
    return ""


def _incomplete_reason(response) -> str | None:
    details = getattr(response, "incomplete_details", None)
    if details is None:
        return None
    if isinstance(details, dict):
        return details.get("reason")
    return getattr(details, "reason", None)


def _request_with_retries(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str,
    service_tier: str | None,
    max_output_tokens: int,
    max_retries: int,
    timeout_s: float,
    schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    use_schema = True
    last_error: Exception | None = None
    max_tokens = max_output_tokens
    for attempt in range(max_retries):
        response_format = _response_format(schema, schema_name, use_schema)
        try:
            response = client.responses.create(
                model=model,
                input=messages,
                text=response_format,
                service_tier=service_tier,
                max_output_tokens=max_tokens,
                reasoning={"effort": "low"},
                timeout=timeout_s,
            )
            if getattr(response, "status", None) not in {None, "completed"}:
                if _incomplete_reason(response) == "max_output_tokens":
                    max_tokens = min(max_tokens * 2, max_output_tokens * 4)
                    last_error = RuntimeError("Model ran out of output tokens.")
                    continue
            text = _extract_output_text(response)
            if not text:
                raise RuntimeError("Empty model response.")
            data = json.loads(text)
            return data
        except BadRequestError as exc:
            message = str(exc).lower()
            if "json_schema" in message or "response_format" in message or "format" in message:
                use_schema = False
                last_error = exc
                continue
            if service_tier and "service_tier" in message:
                service_tier = None
                last_error = exc
                continue
            raise
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as exc:
            last_error = exc
            _backoff_sleep(attempt, base=1.5)
            continue
        except json.JSONDecodeError as exc:
            last_error = exc
            _backoff_sleep(attempt, base=0.5)
            use_schema = False
            continue
        except OpenAIError as exc:
            last_error = exc
            _backoff_sleep(attempt, base=1.0)
            continue
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")
