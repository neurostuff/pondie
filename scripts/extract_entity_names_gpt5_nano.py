#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

DEFAULT_EXAMPLE_PATH = Path(
    "outputs/text/20260112_221156/xTs9gzGeybhU/elsevier/15488423-10.1016_j.neuroimage.2004.06.041.md"
)

SYSTEM_PROMPT = """You extract only entity NAMES and explicit edges from scientific text.

Definitions (for guidance only; do not include in output):
- Group: participant cohort(s) or comparison groups (e.g., patients vs controls).
- Task: behavioral or experimental task/condition participants perform.
- Modality: data acquisition method (e.g., fMRI, PET, EEG, BehaviorOnly).
- Analysis: contrast-specific analysis (method/model + explicit contrast, e.g., "GLM: control > patient").

Rules:
- Return JSON only. No prose or markdown.
- Extract names for: Groups, Tasks, Modalities, Analyses.
- Use short, human-readable names as written in the text.
- Do not add IDs, counts, or attributes. Names only.
- Groups must be participant cohorts in the experiment (not stimuli, task conditions, or brain regions).
- Only create edges when explicitly stated.
- If something is unclear, leave it out.
- Include entities from all studies/experiments within the document; do not stop after the first study.
- Always list every group/task/modality/analysis mentioned, even if no edges can be formed for them.
- If groups are written as comparisons (e.g., "A vs B", "A versus B"), list each group separately and create one edge per group.
- Do not use combined comparison strings (e.g., "A vs B") as a group name.
- Analyses must be contrast-specific: include the contrast direction in the analysis name.
- If a study reports both directions (e.g., "control > patient" and "control < patient"), create separate analysis names for each.
- If a study reports group-specific associations with different directions (e.g., positive vs negative), create separate analyses that include the group name and direction (e.g., "association: positive in Group X").
- Read the full document; group/task details may appear near the end (e.g., Results or Discussion).
- Edges must reference only entities listed in the top-level lists; if an edge would introduce a new entity, add that entity to the list (only if explicitly mentioned) or omit the edge.
- Always include all keys with arrays (empty if none). When nothing is found, return empty arrays.
"""

FEW_SHOT_TEXT = (
    "Patients with Ardent syndrome and matched controls completed a visual oddball task "
    "during PET scanning. GLM voxel-based analysis reported Ardent syndrome > controls "
    "and controls > Ardent syndrome contrasts. In Study 2, associations between recent stress "
    "and cortical connectivity were positive in Ardent syndrome patients and negative in controls."
)

FEW_SHOT_OUTPUT = """{
  "groups": ["Ardent syndrome patients", "matched controls"],
  "tasks": ["visual oddball task"],
  "modalities": ["PET"],
  "analyses": [
    "GLM voxel-based analysis: Ardent syndrome > controls",
    "GLM voxel-based analysis: controls > Ardent syndrome",
    "association: positive in Ardent syndrome patients",
    "association: negative in matched controls"
  ],
  "edges": {
    "group_task": [
      {"group": "Ardent syndrome patients", "task": "visual oddball task"},
      {"group": "matched controls", "task": "visual oddball task"}
    ],
    "task_modality": [{"task": "visual oddball task", "modality": "PET"}],
    "analysis_task": [
      {"analysis": "GLM voxel-based analysis: Ardent syndrome > controls", "task": "visual oddball task"},
      {"analysis": "GLM voxel-based analysis: controls > Ardent syndrome", "task": "visual oddball task"}
    ],
    "analysis_group": [
      {"analysis": "GLM voxel-based analysis: Ardent syndrome > controls", "group": "Ardent syndrome patients"},
      {"analysis": "GLM voxel-based analysis: Ardent syndrome > controls", "group": "matched controls"},
      {"analysis": "GLM voxel-based analysis: controls > Ardent syndrome", "group": "Ardent syndrome patients"},
      {"analysis": "GLM voxel-based analysis: controls > Ardent syndrome", "group": "matched controls"},
      {"analysis": "association: positive in Ardent syndrome patients", "group": "Ardent syndrome patients"},
      {"analysis": "association: negative in matched controls", "group": "matched controls"}
    ],
    "group_modality": [
      {"group": "Ardent syndrome patients", "modality": "PET"},
      {"group": "matched controls", "modality": "PET"}
    ]
  }
}"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "groups": {"type": "array", "items": {"type": "string"}},
        "tasks": {"type": "array", "items": {"type": "string"}},
        "modalities": {"type": "array", "items": {"type": "string"}},
        "analyses": {"type": "array", "items": {"type": "string"}},
        "edges": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "group_task": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "group": {"type": "string"},
                            "task": {"type": "string"},
                        },
                        "required": ["group", "task"],
                    },
                },
                "task_modality": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "task": {"type": "string"},
                            "modality": {"type": "string"},
                        },
                        "required": ["task", "modality"],
                    },
                },
                "analysis_task": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "analysis": {"type": "string"},
                            "task": {"type": "string"},
                        },
                        "required": ["analysis", "task"],
                    },
                },
                "analysis_group": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "analysis": {"type": "string"},
                            "group": {"type": "string"},
                        },
                        "required": ["analysis", "group"],
                    },
                },
                "group_modality": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "group": {"type": "string"},
                            "modality": {"type": "string"},
                        },
                        "required": ["group", "modality"],
                    },
                },
            },
            "required": [
                "group_task",
                "task_modality",
                "analysis_task",
                "analysis_group",
                "group_modality",
            ],
        },
    },
    "required": ["groups", "tasks", "modalities", "analyses", "edges"],
}


def _build_messages(text: str) -> list[dict[str, str]]:
    example_user = (
        "Example.\n\n"
        "Text:\n"
        f"{FEW_SHOT_TEXT}\n\n"
        "Return JSON only."
    )
    real_user = (
        "Now extract from this document.\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": FEW_SHOT_OUTPUT},
        {"role": "user", "content": real_user},
    ]


def _response_format(use_schema: bool) -> dict[str, Any]:
    if use_schema:
        return {"format": {"type": "json_schema", "name": "entity_edges", "schema": SCHEMA}}
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


def _normalize_name(value: str) -> str:
    return " ".join(value.split()).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _normalize_output(data: dict[str, Any]) -> dict[str, Any]:
    groups = _dedupe([_normalize_name(v) for v in data.get("groups", [])])
    tasks = _dedupe([_normalize_name(v) for v in data.get("tasks", [])])
    modalities = _dedupe([_normalize_name(v) for v in data.get("modalities", [])])
    analyses = _dedupe([_normalize_name(v) for v in data.get("analyses", [])])
    edges = data.get("edges", {}) if isinstance(data.get("edges"), dict) else {}

    def norm_edge_list(items: list[dict[str, str]]) -> list[dict[str, str]]:
        cleaned = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized = {k: _normalize_name(str(v)) for k, v in item.items()}
            key = tuple(normalized.get(k, "").lower() for k in sorted(normalized))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned

    normalized_edges = {
        "group_task": norm_edge_list(edges.get("group_task", [])),
        "task_modality": norm_edge_list(edges.get("task_modality", [])),
        "analysis_task": norm_edge_list(edges.get("analysis_task", [])),
        "analysis_group": norm_edge_list(edges.get("analysis_group", [])),
        "group_modality": norm_edge_list(edges.get("group_modality", [])),
    }

    return {
        "groups": groups,
        "tasks": tasks,
        "modalities": modalities,
        "analyses": analyses,
        "edges": normalized_edges,
    }


def _request_with_retries(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str,
    service_tier: str | None,
    max_output_tokens: int,
    max_retries: int,
    timeout_s: float,
) -> dict[str, Any]:
    use_schema = True
    last_error: Exception | None = None
    max_tokens = max_output_tokens
    for attempt in range(max_retries):
        response_format = _response_format(use_schema)
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
            return _normalize_output(data)
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


def _read_text(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract group/task/modality/analysis names and edges using GPT-5 nano."
    )
    parser.add_argument(
        "--input-md",
        type=Path,
        default=DEFAULT_EXAMPLE_PATH,
        help="Path to markdown text to extract.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write extracted JSON (default: alongside input).",
    )
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--service-tier", default="flex")
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-chars", type=int, default=200_000)
    args = parser.parse_args()

    load_dotenv(".env")
    if not args.input_md.exists():
        print(f"Input not found: {args.input_md}", file=sys.stderr)
        return 2

    text = _read_text(args.input_md, args.max_chars)
    messages = _build_messages(text)
    client = OpenAI()
    service_tier = args.service_tier or None
    if service_tier and service_tier.lower() == "none":
        service_tier = None

    data = _request_with_retries(
        client=client,
        messages=messages,
        model=args.model,
        service_tier=service_tier,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        timeout_s=args.timeout,
    )

    output_path = args.output_json
    if output_path is None:
        output_path = args.input_md.with_suffix(".entities.gpt5-nano.json")
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
