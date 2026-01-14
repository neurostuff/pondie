#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from dotenv import load_dotenv
import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from information_extraction import schema as schema_mod


DEFAULT_EXAMPLE_PATH = Path(
    "outputs/text/20260112_221156/xTs9gzGeybhU/elsevier/15488423-10.1016_j.neuroimage.2004.06.041.md"
)

PROMPT_ROOT = Path("prompts")


@dataclass(frozen=True)
class EntitySpec:
    key: str
    prompt_path: Path
    model: type[Any]
    max_items: int | None = None


@dataclass(frozen=True)
class FieldMeta:
    prompt: str | None
    scope_hint: str | None
    allowed_values: list[str] | None
    description: str | None


@dataclass(frozen=True)
class EvidenceConfig:
    embedding_model: str
    nli_model: str
    top_k: int
    entailment_threshold: float
    embedding_batch_size: int
    nli_batch_size: int
    cache_dir: Path
    device: int
    show_progress: bool = False


@dataclass(frozen=True)
class SentenceRecord:
    text: str
    start: int
    end: int
    section_title: str
    source_type: str
    alignment_status: str


class AbbreviationExpander:
    _ID_KEYS = {
        "id",
        "group_id",
        "task_id",
        "modality_id",
        "analysis_id",
        "condition_id",
    }

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        ordered = [(short, long) for short, long in pairs if short and long]
        ordered.sort(key=lambda item: len(item[0]), reverse=True)
        self.patterns = [
            (re.compile(rf"\b{re.escape(short)}\b"), long) for short, long in ordered
        ]
        self.regex = None
        if ordered:
            escaped = [re.escape(short) for short, _ in ordered]
            self.regex = re.compile(r"\b(" + "|".join(escaped) + r")\b")
        self.short_to_long = {short: long for short, long in ordered}

    def expand_text(self, text: str) -> str:
        if not self.patterns:
            return text
        expanded = text
        for pattern, long in self.patterns:
            expanded = pattern.sub(long, expanded)
        return expanded

    def expand_with_map(self, text: str) -> tuple[str, list[int]]:
        if not self.regex:
            return text, list(range(len(text)))
        output: list[str] = []
        mapping: list[int] = []
        cursor = 0
        for match in self.regex.finditer(text):
            start, end = match.span()
            if start > cursor:
                chunk = text[cursor:start]
                output.append(chunk)
                mapping.extend(range(cursor, start))
            short = match.group(0)
            long = self.short_to_long.get(short, short)
            output.append(long)
            orig_len = max(end - start, 1)
            for i in range(len(long)):
                mapped = start + min(orig_len - 1, int(i * orig_len / max(len(long), 1)))
                mapping.append(mapped)
            cursor = end
        if cursor < len(text):
            output.append(text[cursor:])
            mapping.extend(range(cursor, len(text)))
        return "".join(output), mapping

    @classmethod
    def from_path(cls, path: Path) -> AbbreviationExpander | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        pairs: list[tuple[str, str]] = []
        for item in payload.get("abbreviations", []) or []:
            short = str(item.get("short", "")).strip()
            long = str(item.get("long", "")).strip()
            if not short or not long:
                continue
            if short.lower() == long.lower():
                continue
            if "|" in long or "\n" in long:
                continue
            if short.islower() and len(short) <= 2:
                continue
            if len(short) <= 1:
                continue
            pairs.append((short, long))
        return cls(pairs) if pairs else None

    def expand_record(self, value: Any) -> Any:
        if isinstance(value, dict):
            expanded: dict[str, Any] = {}
            for key, item in value.items():
                if key in self._ID_KEYS:
                    expanded[key] = item
                    continue
                if key == "value" and isinstance(item, str):
                    expanded[key] = self.expand_text(item)
                    continue
                expanded[key] = self.expand_record(item)
            return expanded
        if isinstance(value, list):
            return [self.expand_record(item) for item in value]
        if isinstance(value, str):
            return self.expand_text(value)
        return value


class EvidenceProgress:
    def __init__(self, total: int, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled and tqdm is not None
        self.bar = None
        if self.enabled:
            self.bar = tqdm(total=total, desc="Evidence", unit="field")

    def update(self, n: int = 1) -> None:
        if self.bar is not None:
            self.bar.update(n)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _find_literal_values(annotation: Any) -> list[str] | None:
    if annotation is None:
        return None
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is None:
        return None
    if origin is list:
        args = get_args(annotation)
        return _find_literal_values(args[0]) if args else None
    if origin is schema_mod.ExtractedValue:
        args = get_args(annotation)
        return _find_literal_values(args[0]) if args else None
    if origin is dict:
        return None
    if origin is Union:
        values: list[str] = []
        for arg in get_args(annotation):
            found = _find_literal_values(arg)
            if found:
                values.extend(found)
        deduped = list(dict.fromkeys(values))
        return deduped or None
    if origin is Literal:
        return [str(value) for value in get_args(annotation)]
    return None


def _iter_model_fields(
    model: type[Any],
) -> list[tuple[str, Any, str | None, dict[str, Any]]]:
    fields = getattr(model, "model_fields", None)
    if fields:
        return [
            (
                name,
                getattr(field, "annotation", None),
                getattr(field, "description", None),
                getattr(field, "json_schema_extra", None) or {},
            )
            for name, field in fields.items()
        ]
    legacy_fields = getattr(model, "__fields__", None) or {}
    out = []
    for name, field in legacy_fields.items():
        info = field.field_info
        out.append(
            (
                name,
                getattr(field, "outer_type_", None) or getattr(field, "type_", None),
                getattr(info, "description", None),
                getattr(info, "extra", None) or {},
            )
        )
    return out


def _field_meta_map(model: type[Any], fields: list[str]) -> dict[str, FieldMeta]:
    meta: dict[str, FieldMeta] = {}
    wanted = set(fields)
    for name, annotation, description, extra in _iter_model_fields(model):
        if name not in wanted:
            continue
        allowed_values = _find_literal_values(annotation)
        meta[name] = FieldMeta(
            prompt=extra.get("extraction_prompt"),
            scope_hint=extra.get("scope_hint"),
            allowed_values=allowed_values,
            description=description,
        )
    for field in fields:
        if field not in meta:
            raise KeyError(f"Field '{field}' not found on {model.__name__}")
    return meta


def _relax_id(schema: dict[str, Any]) -> None:
    if isinstance(schema, dict):
        if "properties" in schema and "id" in schema["properties"]:
            id_schema = schema["properties"]["id"]
            if isinstance(id_schema, dict):
                existing = id_schema.get("type")
                if isinstance(existing, list):
                    if "null" not in existing:
                        existing.append("null")
                elif isinstance(existing, str):
                    id_schema["type"] = [existing, "null"]
                else:
                    id_schema["type"] = ["string", "null"]
            if "required" in schema and "id" in schema["required"]:
                schema["required"] = [name for name in schema["required"] if name != "id"]
        for value in schema.values():
            if isinstance(value, dict):
                _relax_id(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _relax_id(item)


def _unwrap_extracted_value(schema: dict[str, Any]) -> dict[str, Any]:
    if "properties" in schema and "value" in schema["properties"]:
        title = str(schema.get("title", ""))
        props = schema.get("properties", {})
        if "evidence" in props or "ExtractedValue" in title:
            value_schema = props.get("value", {})
            if isinstance(value_schema, dict):
                existing = value_schema.get("type")
                if isinstance(existing, list):
                    if "null" not in existing:
                        existing.append("null")
                elif isinstance(existing, str):
                    value_schema["type"] = [existing, "null"]
            return _unwrap_extracted_value(value_schema)
    for key, value in list(schema.items()):
        if isinstance(value, dict):
            schema[key] = _unwrap_extracted_value(value)
        elif isinstance(value, list):
            schema[key] = [
                _unwrap_extracted_value(item) if isinstance(item, dict) else item
                for item in value
            ]
    return schema


def _prune_model_schema(
    model: type[Any],
    fields: list[str],
    *,
    include_id: bool = False,
) -> dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        schema = model.model_json_schema()
    else:
        schema = model.schema()
    properties = schema.get("properties", {})
    keep = set(fields)
    if include_id and "id" in properties:
        keep.add("id")
    schema["properties"] = {name: properties[name] for name in keep if name in properties}
    schema["required"] = [name for name in schema.get("required", []) if name in schema["properties"]]
    schema["additionalProperties"] = False
    _relax_id(schema)
    schema = _unwrap_extracted_value(schema)
    return schema


def _wrap_items_schema(schema: dict[str, Any], max_items: int | None) -> dict[str, Any]:
    items_schema: dict[str, Any] = {"type": "array", "items": schema}
    if max_items is not None:
        items_schema["maxItems"] = max_items
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"items": items_schema},
        "required": ["items"],
    }


PROMPT_SPEC_CACHE: dict[Path, dict[str, Any]] = {}


def _load_prompt_spec_cached(path: Path) -> dict[str, Any]:
    cached = PROMPT_SPEC_CACHE.get(path)
    if cached is not None:
        return cached
    data = _load_prompt_spec(path)
    PROMPT_SPEC_CACHE[path] = data
    return data


ENTITY_SPECS = [
    EntitySpec(
        key="study",
        prompt_path=PROMPT_ROOT / "study.json",
        model=schema_mod.StudyMetadataModel,
        max_items=1,
    ),
    EntitySpec(
        key="groups",
        prompt_path=PROMPT_ROOT / "group.json",
        model=schema_mod.GroupBase,
    ),
    EntitySpec(
        key="tasks",
        prompt_path=PROMPT_ROOT / "task.json",
        model=schema_mod.TaskBase,
    ),
    EntitySpec(
        key="modalities",
        prompt_path=PROMPT_ROOT / "modality.json",
        model=schema_mod.ModalityBase,
    ),
    EntitySpec(
        key="analyses",
        prompt_path=PROMPT_ROOT / "analysis.json",
        model=schema_mod.AnalysisBase,
    ),
]


def _load_prompt_spec(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt spec must be a JSON object: {path}")
    return data


def _build_prompt_description(
    prompt_spec: dict[str, Any],
    field_meta: dict[str, FieldMeta],
) -> str:
    overrides = prompt_spec.get("field_prompt_overrides") or {}
    fields = prompt_spec.get("fields") or []
    lines = [prompt_spec.get("prompt_description", "").strip(), "", "Fields:"]
    for field in fields:
        override_prompt = overrides.get(field)
        prompt_entry = field_meta.get(field)
        prompt_text = override_prompt or (prompt_entry.prompt if prompt_entry else None)
        scope_hint = prompt_entry.scope_hint if prompt_entry else None
        allowed_values = prompt_entry.allowed_values if prompt_entry else None
        details: list[str] = []
        if scope_hint:
            details.append(f"Scope: {scope_hint}")
        if allowed_values:
            details.append(f"Allowed values: {', '.join(allowed_values)}")
        if prompt_text:
            if details:
                lines.append(f"- {field}: {prompt_text} ({'; '.join(details)})")
            else:
                lines.append(f"- {field}: {prompt_text}")
        elif details:
            lines.append(f"- {field}: ({'; '.join(details)})")
        else:
            lines.append(f"- {field}")
    return "\n".join(line for line in lines if line is not None).strip()


def _build_example_output(
    fields: list[str],
    attributes: dict[str, Any],
) -> dict[str, Any]:
    item = {field: None for field in fields}
    for key, value in (attributes or {}).items():
        if key not in item:
            continue
        item[key] = value
    return {"items": [item]}


def _build_entity_prompt(
    prompt_spec: dict[str, Any],
    field_meta: dict[str, FieldMeta],
) -> str:
    description = _build_prompt_description(prompt_spec, field_meta)
    fields = prompt_spec.get("fields") or []

    template = {"items": [{field: None for field in fields}]}
    lines = [
        description,
        "",
        "Rules:",
        "- Use only information explicitly stated in the document.",
        "- Include every field listed; use null when a value is not stated.",
        "- Return JSON only (no prose, no markdown).",
        "- Match field types based on the schema (arrays, objects, numbers, booleans).",
        "",
        "Output format (items may contain multiple objects):",
        json.dumps(template, indent=2),
    ]

    examples = prompt_spec.get("examples") or []
    if examples:
        lines.append("")
        lines.append("Examples:")
        for example in examples:
            text = example.get("text", "").strip()
            extractions = example.get("extractions") or []
            items = []
            for extraction in extractions:
                attrs = extraction.get("attributes") or {}
                example_obj = _build_example_output(fields, attrs)
                items.extend(example_obj.get("items", []))
            example_output = {"items": items}
            lines.append("Text:")
            lines.append(text)
            lines.append("Output:")
            lines.append(json.dumps(example_output, indent=2))
    return "\n".join(lines).strip()


def _build_messages(text: str, entity_prompt: str) -> list[dict[str, str]]:
    system = (
        "You extract structured JSON from scientific text. "
        "Follow the instructions and return JSON only."
    )
    # Put document text in a dedicated, stable message to maximize prompt caching.
    doc_message = f"Document text:\n{text}"
    instructions = f"{entity_prompt}\n\nUse the document text above."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": doc_message},
        {"role": "user", "content": instructions},
    ]


def _build_entity_descriptions(entity_name: str, field_meta: dict[str, FieldMeta]) -> str:
    lines = [f"{entity_name} fields:"]
    for field_name, meta in field_meta.items():
        details: list[str] = []
        if meta.description:
            details.append(meta.description)
        if meta.allowed_values:
            details.append(f"Allowed values: {', '.join(meta.allowed_values)}")
        if details:
            lines.append(f"- {field_name}: {'; '.join(details)}")
        else:
            lines.append(f"- {field_name}")
    return "\n".join(lines)


def _build_verification_prompt(
    text: str,
    extraction: dict[str, Any],
    field_descriptions: dict[str, str],
) -> list[dict[str, str]]:
    system = (
        "You verify extracted entities against the document. "
        "Only keep entities and values explicitly supported by the text. "
        "Return JSON only."
    )
    doc_message = f"Document text:\n{text}"
    schema_description = "\n\n".join(
        [
            "Schema field descriptions:",
            field_descriptions["study"],
            field_descriptions["demographics"],
            field_descriptions["tasks"],
            field_descriptions["modalities"],
            field_descriptions["analyses"],
            field_descriptions["links"],
        ]
    )
    instructions = (
        f"{schema_description}\n\n"
        "Current extraction JSON:\n"
        f"{json.dumps(extraction, indent=2)}\n\n"
        "Verify the entities and fields using the document text above. "
        "Remove entities not supported by the text. Add missing entities if explicitly stated. "
        "For each field, set it to null if not explicitly stated; fill it if explicitly stated. "
        "Do not infer. Do not create or change IDs; leave IDs as-is and leave new IDs null "
        "(IDs will be assigned programmatically). "
        "Return JSON with a corrected extraction and a list of changes.\n\n"
        "Output format:\n"
        "{\n"
        '  "corrected": { ... },\n'
        '  "changes": [\n'
        '    {"action": "remove_entity", "target": "G2", "note": "Not supported by text"}\n'
        "  ]\n"
        "}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": doc_message},
        {"role": "user", "content": instructions},
    ]


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


def _read_text(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    last_was_space = False
    for idx, char in enumerate(text):
        if char.isspace():
            if last_was_space:
                continue
            norm_chars.append(" ")
            norm_to_orig.append(idx)
            last_was_space = True
            continue
        norm_chars.append(char)
        norm_to_orig.append(idx)
        last_was_space = False
    return "".join(norm_chars), norm_to_orig




def _is_references_title(title: str) -> bool:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", title.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return False
    return normalized.startswith(("references", "bibliography", "works cited", "literature cited"))


def _section_boundaries(text: str) -> list[tuple[int, str]]:
    boundaries = [(0, "Document")]
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip() or "Untitled section"
            boundaries.append((offset, title))
        offset += len(line)
    return boundaries


def _split_sentences_with_spans(
    normalized_text: str,
    norm_to_orig: list[int],
    *,
    section_bounds: list[tuple[int, str]],
) -> list[SentenceRecord]:
    sentences: list[SentenceRecord] = []
    pattern = re.compile(r"[^.!?]+[.!?]+(?=\s|$)|[^.!?]+$")
    for match in pattern.finditer(normalized_text):
        span_text = match.group(0)
        leading = len(span_text) - len(span_text.lstrip())
        trailing = len(span_text) - len(span_text.rstrip())
        start_norm = match.start() + leading
        end_norm = match.end() - trailing
        if start_norm >= end_norm:
            continue
        sentence = normalized_text[start_norm:end_norm].strip()
        if not sentence:
            continue
        if sentence.lstrip().startswith("#"):
            continue
        if start_norm >= len(norm_to_orig):
            continue
        end_index = max(start_norm, end_norm - 1)
        if end_index >= len(norm_to_orig):
            end_index = len(norm_to_orig) - 1
        start_orig = norm_to_orig[start_norm]
        end_orig = norm_to_orig[end_index] + 1
        section_title = "Document"
        for boundary_start, title in reversed(section_bounds):
            if start_orig >= boundary_start:
                section_title = title
                break
        if _is_references_title(section_title):
            continue
        sentences.append(
            SentenceRecord(
                text=sentence,
                start=start_orig,
                end=end_orig,
                section_title=section_title,
                source_type="sentence",
                alignment_status="match_exact",
            )
        )
    return sentences


def _safe_cache_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_cached_embeddings(
    cache_dir: Path, model_name: str, doc_hash: str, sentences_hash: str
) -> np.ndarray | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_cache_name(model_name)
    meta_path = cache_dir / f"{safe_name}-{doc_hash}.json"
    data_path = cache_dir / f"{safe_name}-{doc_hash}.npz"
    if not (meta_path.exists() and data_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if (
        meta.get("model") != model_name
        or meta.get("doc_hash") != doc_hash
        or meta.get("sentences_hash") != sentences_hash
    ):
        return None
    try:
        data = np.load(data_path)
        embeddings = data.get("embeddings")
    except Exception:
        return None
    if embeddings is None:
        return None
    return embeddings


def _write_cached_embeddings(
    cache_dir: Path,
    model_name: str,
    doc_hash: str,
    sentences_hash: str,
    embeddings: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_cache_name(model_name)
    meta_path = cache_dir / f"{safe_name}-{doc_hash}.json"
    data_path = cache_dir / f"{safe_name}-{doc_hash}.npz"
    meta = {
        "model": model_name,
        "doc_hash": doc_hash,
        "sentences_hash": sentences_hash,
        "count": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(data_path, embeddings=embeddings)


def _canonical_label(label: str, nli) -> str:
    normalized = label.strip().lower()
    if normalized in {"entailment", "neutral", "contradiction"}:
        return normalized
    if normalized.startswith("label_"):
        suffix = normalized.replace("label_", "")
        if suffix.isdigit() and getattr(nli.model, "config", None) is not None:
            mapped = nli.model.config.id2label.get(int(suffix))
            if mapped:
                return str(mapped).strip().lower()
    return normalized


def _scores_from_item(item: Any, nli) -> dict[str, float]:
    scores: list[dict[str, Any]] = []
    if isinstance(item, list):
        scores = item
    elif isinstance(item, dict) and "labels" in item and "scores" in item:
        scores = [
            {"label": label, "score": score}
            for label, score in zip(item["labels"], item["scores"])
        ]
    elif isinstance(item, dict) and "label" in item and "score" in item:
        scores = [item]

    out = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    for score_item in scores:
        label = _canonical_label(str(score_item.get("label", "")), nli)
        if label in out:
            out[label] = float(score_item.get("score", 0.0))
    return out


def _batch_nli_scores(
    nli, premises: list[str], hypothesis: str, batch_size: int
) -> list[dict[str, float]]:
    if not premises:
        return []
    inputs = [{"text": premise, "text_pair": hypothesis} for premise in premises]
    result = nli(
        inputs,
        truncation=True,
        max_length=512,
        top_k=None,
        batch_size=batch_size,
    )
    items: list[Any] = []
    if isinstance(result, list):
        if len(inputs) == 1 and result and isinstance(result[0], dict) and "label" in result[0]:
            items = [result]
        else:
            items = result
    else:
        items = [result]
    scores_list: list[dict[str, float]] = []
    for item in items[: len(inputs)]:
        scores_list.append(_scores_from_item(item, nli))
    while len(scores_list) < len(inputs):
        scores_list.append({"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0})
    return scores_list


class EvidenceIndex:
    def __init__(
        self,
        text: str,
        document_id: str | None,
        config: EvidenceConfig,
        progress: EvidenceProgress | None = None,
        llm_client: OpenAI | None = None,
        llm_service_tier: str | None = None,
        expander: AbbreviationExpander | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Evidence extraction requires the 'semantic' extras. "
                "Install with: uv pip install -e \".[semantic]\""
            ) from exc

        self.document_id = document_id
        self.raw_text = text
        self.config = config
        self.progress = progress
        self.llm_client = llm_client
        self.llm_service_tier = llm_service_tier
        self.llm_cache: dict[tuple[str, str], bool] = {}
        self.expander = expander
        self.normalized_text, self.norm_to_orig = _normalize_with_map(text)
        section_bounds = _section_boundaries(text)
        self.sentences = _split_sentences_with_spans(
            self.normalized_text,
            self.norm_to_orig,
            section_bounds=section_bounds,
        )
        if self.expander:
            self.sentence_texts = [
                self.expander.expand_text(sentence.text) for sentence in self.sentences
            ]
        else:
            self.sentence_texts = [sentence.text for sentence in self.sentences]

        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.nli = pipeline(
            "text-classification",
            model=config.nli_model,
            tokenizer=config.nli_model,
            device=config.device,
        )

        self.sentence_embeddings = self._load_or_compute_embeddings()
        self.query_cache: dict[str, np.ndarray] = {}


    def _load_or_compute_embeddings(self) -> np.ndarray:
        doc_hash = _hash_text(self.normalized_text)
        sentences_hash = _hash_text("\n".join(self.sentence_texts))
        cached = _load_cached_embeddings(
            self.config.cache_dir,
            self.config.embedding_model,
            doc_hash,
            sentences_hash,
        )
        if cached is not None:
            return cached
        if not self.sentence_texts:
            return np.zeros((0, 0), dtype=float)
        embeddings = self.embedding_model.encode(
            self.sentence_texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        _write_cached_embeddings(
            self.config.cache_dir,
            self.config.embedding_model,
            doc_hash,
            sentences_hash,
            embeddings,
        )
        return embeddings

    def _query_embedding(self, query: str) -> np.ndarray:
        cached = self.query_cache.get(query)
        if cached is not None:
            return cached
        embedding = self.embedding_model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        embedding = embedding[0]
        self.query_cache[query] = embedding
        return embedding

    def _value_text(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return f"{value:g}"
        return str(value).strip()

    def _match_value_span(
        self,
        normalized_sentence: str,
        normalized_value: str,
    ) -> tuple[int, int, str] | None:
        if not normalized_value:
            return None
        idx = normalized_sentence.find(normalized_value)
        if idx != -1:
            return idx, idx + len(normalized_value), "match_exact"
        if re.fullmatch(r"[-+]?\\d+(\\.\\d+)?", normalized_value):
            base_raw = normalized_value.rstrip("0").rstrip(".")
            base = re.escape(base_raw or normalized_value)
            if "." in normalized_value:
                pattern = rf"\\b{base}0*\\b"
            else:
                pattern = rf"\\b{base}(?:\\.0+)?\\b"
            match = re.search(pattern, normalized_sentence)
            if match:
                return match.start(), match.end(), "match_exact"
        matcher = difflib.SequenceMatcher(None, normalized_sentence, normalized_value)
        match = matcher.find_longest_match(
            0, len(normalized_sentence), 0, len(normalized_value)
        )
        if match.size and match.size / max(len(normalized_value), 1) >= 0.6:
            return match.a, match.a + match.size, "match_fuzzy"
        return None

    def _split_value_sentences(self, value_text: str) -> list[str]:
        normalized = _normalize_text(value_text)
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        return [part.strip() for part in parts if part.strip()]

    def _llm_supports(self, hypothesis: str, sentence: str) -> bool:
        if self.llm_client is None:
            return False
        key = (hypothesis, sentence)
        cached = self.llm_cache.get(key)
        if cached is not None:
            return cached
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"supports": {"type": "boolean"}},
            "required": ["supports"],
        }
        messages = [
            {
                "role": "system",
                "content": "Determine if the sentence supports the hypothesis. Return JSON only.",
            },
            {
                "role": "user",
                "content": f"Hypothesis: {hypothesis}\\nSentence: {sentence}",
            },
        ]
        try:
            result = _request_with_retries(
                self.llm_client,
                messages,
                model="gpt-5-nano",
                service_tier=self.llm_service_tier,
                max_output_tokens=50,
                max_retries=3,
                timeout_s=30.0,
                schema=schema,
                schema_name="SupportCheck",
            )
        except Exception:
            self.llm_cache[key] = False
            return False
        supports = bool(result.get("supports"))
        self.llm_cache[key] = supports
        return supports

    def find_evidence(
        self,
        value: Any,
        field_description: str | None,
        *,
        field_name: str | None = None,
        context_label: str | None = None,
    ) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            description = _evidence_description(field_name, field_description)
            value_text = self._value_text(value)
            value_sentences = self._split_value_sentences(value_text)
            if not value_sentences:
                return None
            if self.sentence_embeddings is None or len(self.sentence_embeddings) == 0:
                return None
            evidence: list[dict[str, Any]] = []
            seen_spans: set[tuple[int, int, str]] = set()
            context_prefix = ""
            if context_label:
                context_prefix = f"For {context_label}, "
            for value_sentence in value_sentences:
                hypothesis = f"{context_prefix}{value_sentence} is {description}"
                query_embedding = self._query_embedding(hypothesis)
                similarities = self.sentence_embeddings @ query_embedding
                top_k = min(self.config.top_k, len(similarities))
                if top_k <= 0:
                    continue
                top_indices = list(similarities.argsort()[-top_k:][::-1])
                candidate_texts = [self.sentence_texts[idx] for idx in top_indices]
                scores_list = _batch_nli_scores(
                    self.nli,
                    candidate_texts,
                    hypothesis,
                    self.config.nli_batch_size,
                )
                normalized_value = _normalize_text(value_sentence).lower()
                for idx, scores in zip(top_indices, scores_list):
                    record = self.sentences[idx]
                    candidate_text = self.sentence_texts[idx]
                    entailment = scores["entailment"]
                    neutral = scores["neutral"]
                    contradiction = scores["contradiction"]
                    supports = False
                    if (
                        entailment >= self.config.entailment_threshold
                        and entailment > neutral
                        and entailment > contradiction
                    ):
                        supports = True
                    elif entailment < 0.5 and contradiction < 0.1:
                        supports = self._llm_supports(hypothesis, candidate_text)
                    if not supports:
                        continue
                    raw_segment = self.raw_text[record.start:record.end]
                    if self.expander:
                        expanded_segment, expanded_to_raw = self.expander.expand_with_map(
                            raw_segment
                        )
                        normalized_segment, norm_to_expanded = _normalize_with_map(
                            expanded_segment
                        )
                        if not norm_to_expanded:
                            continue
                        norm_to_orig = [expanded_to_raw[i] for i in norm_to_expanded]
                    else:
                        normalized_segment, norm_to_orig = _normalize_with_map(raw_segment)
                    if not norm_to_orig:
                        continue
                    match = self._match_value_span(
                        normalized_segment.lower(), normalized_value
                    )
                    if match is None:
                        start_pos = record.start
                        end_pos = record.end
                        alignment_status = "match_fuzzy"
                        extraction_text = _normalize_text(self.raw_text[start_pos:end_pos])
                    else:
                        match_start, match_end, alignment_status = match
                        if match_start >= len(norm_to_orig):
                            start_pos = record.start
                            end_pos = record.end
                            alignment_status = "match_fuzzy"
                            extraction_text = _normalize_text(self.raw_text[start_pos:end_pos])
                        else:
                            end_index = max(match_start, match_end - 1)
                            if end_index >= len(norm_to_orig):
                                end_index = len(norm_to_orig) - 1
                            start_pos = record.start + norm_to_orig[match_start]
                            end_pos = record.start + norm_to_orig[end_index] + 1
                            extraction_text = _normalize_text(self.raw_text[start_pos:end_pos])
                    span_key = (start_pos, end_pos, record.section_title)
                    if span_key in seen_spans:
                        continue
                    seen_spans.add(span_key)
                    evidence.append(
                        {
                            "source": record.source_type,
                            "section": record.section_title,
                            "extraction_text": extraction_text,
                            "char_interval": {
                                "start_pos": start_pos,
                                "end_pos": end_pos,
                            },
                            "alignment_status": alignment_status,
                            "document_id": self.document_id,
                        }
                    )
            return evidence or None
        finally:
            if self.progress is not None:
                self.progress.update(1)


def _ensure_ids(items: list[Any], prefix: str, *, overwrite: bool = False) -> list[Any]:
    if overwrite:
        return [
            dict(item, id=f"{prefix}{idx}") if isinstance(item, dict) else item
            for idx, item in enumerate(items, start=1)
        ]
    used: set[str] = {
        item.get("id", "").strip()
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id").strip()
    }
    counter = 1
    updated: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            updated.append(item)
            continue
        item_copy = dict(item)
        value = item_copy.get("id")
        if not isinstance(value, str) or not value.strip():
            while f"{prefix}{counter}" in used:
                counter += 1
            item_copy["id"] = f"{prefix}{counter}"
            used.add(item_copy["id"])
            counter += 1
        updated.append(item_copy)
    return updated


def _assign_missing_condition_ids(tasks: list[Any]) -> list[Any]:
    used: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        conditions = task.get("conditions")
        if not isinstance(conditions, list):
            continue
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            value = condition.get("id")
            if isinstance(value, str) and value.strip():
                used.add(value.strip())

    counter = 1
    updated_tasks: list[Any] = []
    for task in tasks:
        if not isinstance(task, dict):
            updated_tasks.append(task)
            continue
        task_copy = dict(task)
        conditions = task_copy.get("conditions")
        if not isinstance(conditions, list):
            updated_tasks.append(task_copy)
            continue
        updated_conditions: list[Any] = []
        for condition in conditions:
            if not isinstance(condition, dict):
                updated_conditions.append(condition)
                continue
            condition_copy = dict(condition)
            value = condition_copy.get("id")
            if not isinstance(value, str) or not value.strip():
                while f"C{counter}" in used:
                    counter += 1
                condition_copy["id"] = f"C{counter}"
                used.add(condition_copy["id"])
                counter += 1
            updated_conditions.append(condition_copy)
        task_copy["conditions"] = updated_conditions
        updated_tasks.append(task_copy)
    return updated_tasks


def _normalize_task_conditions(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    condition_counter = 1
    normalized = []
    for task in tasks:
        task = dict(task)
        conditions = task.get("conditions")
        if conditions is None:
            normalized.append(task)
            continue
        if not isinstance(conditions, list):
            task["conditions"] = None
            normalized.append(task)
            continue
        condition_objs = []
        for item in conditions:
            if isinstance(item, dict):
                condition = dict(item)
                label = condition.get("condition_label")
                if label is None or (isinstance(label, str) and not label.strip()):
                    continue
                if not condition.get("id"):
                    condition["id"] = f"C{condition_counter}"
                condition_objs.append(condition)
                condition_counter += 1
                continue
            if isinstance(item, str):
                label = item.strip()
                if not label:
                    continue
                condition_objs.append({"id": f"C{condition_counter}", "condition_label": label})
                condition_counter += 1
        task["conditions"] = condition_objs or None
        normalized.append(task)
    return normalized


def _assign_missing_ids_to_record(record: dict[str, Any]) -> dict[str, Any]:
    demographics = record.get("demographics")
    if isinstance(demographics, dict):
        groups = demographics.get("groups")
        if isinstance(groups, list):
            demographics["groups"] = _ensure_ids(groups, "G")
        record["demographics"] = demographics
    tasks = record.get("tasks")
    if isinstance(tasks, list):
        tasks = _ensure_ids(tasks, "T")
        tasks = _assign_missing_condition_ids(tasks)
        record["tasks"] = tasks
    modalities = record.get("modalities")
    if isinstance(modalities, list):
        record["modalities"] = _ensure_ids(modalities, "M")
    analyses = record.get("analyses")
    if isinstance(analyses, list):
        record["analyses"] = _ensure_ids(analyses, "A")
    return record


def _is_extracted_value_type(annotation: Any) -> bool:
    annotation = _unwrap_optional(annotation)
    if annotation is None:
        return False
    origin = get_origin(annotation)
    if origin is schema_mod.ExtractedValue:
        return True
    if isinstance(annotation, type) and issubclass(annotation, schema_mod.ExtractedValue):
        return True
    return False


def _is_model_type(annotation: Any) -> bool:
    annotation = _unwrap_optional(annotation)
    if annotation is None:
        return False
    if annotation in {schema_mod.EvidenceSpan, schema_mod.CharInterval}:
        return False
    return isinstance(annotation, type) and issubclass(annotation, schema_mod.BaseModel)


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def _evidence_description(field_name: str | None, description: str | None) -> str:
    if field_name == "contrast_formula":
        return "statistical contrast formula"
    return description or field_name or "field"


def _group_context_label(group: dict[str, Any]) -> str | None:
    label = _unwrap_value(group.get("cohort_label"))
    if isinstance(label, str) and label.strip():
        return f"cohort {label.strip()}"
    label = _unwrap_value(group.get("medical_condition"))
    if isinstance(label, str) and label.strip():
        return f"cohort {label.strip()}"
    group_id = group.get("id")
    if isinstance(group_id, str) and group_id.strip():
        return f"group {group_id.strip()}"
    return None


def _task_context_label(task: dict[str, Any]) -> str | None:
    label = _unwrap_value(task.get("task_name"))
    if isinstance(label, str) and label.strip():
        return f"task {label.strip()}"
    label = _unwrap_value(task.get("task_description"))
    if isinstance(label, str) and label.strip():
        return f"task {label.strip()}"
    task_id = task.get("id")
    if isinstance(task_id, str) and task_id.strip():
        return f"task {task_id.strip()}"
    return None


def _analysis_context_label(analysis: dict[str, Any]) -> str | None:
    label = _unwrap_value(analysis.get("analysis_label"))
    if isinstance(label, str) and label.strip():
        return f"analysis {label.strip()}"
    label = _unwrap_value(analysis.get("contrast_formula"))
    if isinstance(label, str) and label.strip():
        return f"analysis {label.strip()}"
    analysis_id = analysis.get("id")
    if isinstance(analysis_id, str) and analysis_id.strip():
        return f"analysis {analysis_id.strip()}"
    return None


def _modality_context_label(modality: dict[str, Any]) -> str | None:
    modality_type = modality.get("modality_type")
    if isinstance(modality_type, dict):
        family = _unwrap_value(modality_type.get("family"))
        subtype = _unwrap_value(modality_type.get("subtype"))
        label_parts = []
        if isinstance(family, str) and family.strip():
            label_parts.append(family.strip())
        if isinstance(subtype, str) and subtype.strip():
            label_parts.append(subtype.strip())
        if label_parts:
            return f"modality {' '.join(label_parts)}"
    for key in ("sequence_name", "voxel_size", "manufacturer"):
        label = _unwrap_value(modality.get(key))
        if isinstance(label, str) and label.strip():
            return f"modality {label.strip()}"
    modality_id = modality.get("id")
    if isinstance(modality_id, str) and modality_id.strip():
        return f"modality {modality_id.strip()}"
    return None


def _wrap_model_with_evidence(
    obj: dict[str, Any],
    model: type[Any],
    evidence_index: EvidenceIndex,
    *,
    context_label: str | None = None,
    context_exclude: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return obj
    updated = dict(obj)
    for field_name, annotation, description, _extra in _iter_model_fields(model):
        if field_name not in obj:
            continue
        value = obj.get(field_name)
        effective_context = context_label
        if context_exclude and field_name in context_exclude:
            effective_context = None
        updated[field_name] = _wrap_value_with_evidence(
            value,
            annotation,
            description,
            evidence_index,
            field_name=field_name,
            context_label=effective_context,
            context_exclude=context_exclude,
        )
    return updated


def _wrap_value_with_evidence(
    value: Any,
    annotation: Any,
    description: str | None,
    evidence_index: EvidenceIndex,
    *,
    field_name: str | None = None,
    context_label: str | None = None,
    context_exclude: set[str] | None = None,
) -> Any:
    if value is None:
        return None

    def _wrap_extracted(item: Any) -> dict[str, Any]:
        if isinstance(item, dict) and "value" in item:
            updated = dict(item)
            raw = updated.get("value")
            if raw is None:
                updated["evidence"] = None
                return updated
            updated["evidence"] = evidence_index.find_evidence(
                raw,
                description,
                field_name=field_name,
                context_label=context_label,
            )
            return updated
        return {
            "value": item,
            "evidence": evidence_index.find_evidence(
                item,
                description,
                field_name=field_name,
                context_label=context_label,
            ),
        }
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        item_type = args[0] if args else None
        if _is_extracted_value_type(item_type):
            if isinstance(value, list):
                return [_wrap_extracted(item) for item in value]
            return value
        if _is_model_type(item_type):
            if isinstance(value, list):
                return [
                    _wrap_model_with_evidence(
                        item,
                        item_type,
                        evidence_index,
                        context_label=context_label,
                        context_exclude=context_exclude,
                    )
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            return value
        return value
    if _is_extracted_value_type(annotation):
        return _wrap_extracted(value)
    if _is_model_type(annotation):
        if isinstance(value, dict):
            return _wrap_model_with_evidence(
                value,
                annotation,
                evidence_index,
                context_label=context_label,
                context_exclude=context_exclude,
            )
        return value
    return value


def _has_evidence_value(value: Any) -> bool:
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _count_evidence_fields(value: Any, annotation: Any) -> int:
    if value is None:
        return 0
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        item_type = args[0] if args else None
        if _is_extracted_value_type(item_type):
            if isinstance(value, list):
                return sum(1 for item in value if _has_evidence_value(item))
            return 0
        if _is_model_type(item_type):
            if isinstance(value, list):
                return sum(
                    _count_model_evidence_fields(item, item_type)
                    for item in value
                    if isinstance(item, dict)
                )
            return 0
        return 0
    if _is_extracted_value_type(annotation):
        return 1 if _has_evidence_value(value) else 0
    if _is_model_type(annotation):
        if isinstance(value, dict):
            return _count_model_evidence_fields(value, annotation)
        return 0
    return 0


def _count_model_evidence_fields(obj: dict[str, Any], model: type[Any]) -> int:
    total = 0
    for field_name, annotation, _description, _extra in _iter_model_fields(model):
        if field_name not in obj:
            continue
        total += _count_evidence_fields(obj.get(field_name), annotation)
    return total


def _attach_evidence_to_record(
    record: dict[str, Any],
    *,
    text: str,
    document_id: str | None,
    config: EvidenceConfig,
    llm_client: OpenAI | None,
    llm_service_tier: str | None,
    expander: AbbreviationExpander | None,
) -> dict[str, Any]:
    total_fields = _count_model_evidence_fields(record, schema_mod.StudyRecord)
    if config.show_progress and total_fields <= 0:
        config = EvidenceConfig(
            embedding_model=config.embedding_model,
            nli_model=config.nli_model,
            top_k=config.top_k,
            entailment_threshold=config.entailment_threshold,
            embedding_batch_size=config.embedding_batch_size,
            nli_batch_size=config.nli_batch_size,
            cache_dir=config.cache_dir,
            device=config.device,
            show_progress=False,
        )
    progress = None
    if config.show_progress:
        if tqdm is None:
            print("tqdm is not installed; progress bar disabled.", file=sys.stderr)
        else:
            progress = EvidenceProgress(total_fields, enabled=True)
    evidence_index = EvidenceIndex(
        text,
        document_id,
        config,
        progress=progress,
        llm_client=llm_client,
        llm_service_tier=llm_service_tier,
        expander=expander,
    )
    updated = dict(record)

    study = record.get("study")
    if isinstance(study, dict):
        updated["study"] = _wrap_model_with_evidence(
            study, schema_mod.StudyMetadataModel, evidence_index
        )

    demographics = record.get("demographics")
    if isinstance(demographics, dict):
        groups = demographics.get("groups")
        if isinstance(groups, list):
            demographics = dict(demographics)
            demographics["groups"] = [
                _wrap_model_with_evidence(
                    group,
                    schema_mod.GroupBase,
                    evidence_index,
                    context_label=_group_context_label(group) if isinstance(group, dict) else None,
                )
                if isinstance(group, dict)
                else group
                for group in groups
            ]
            updated["demographics"] = demographics

    tasks = record.get("tasks")
    if isinstance(tasks, list):
        updated["tasks"] = [
            _wrap_model_with_evidence(
                task,
                schema_mod.TaskBase,
                evidence_index,
                context_label=_task_context_label(task) if isinstance(task, dict) else None,
                context_exclude={"task_name"},
            )
            if isinstance(task, dict)
            else task
            for task in tasks
        ]

    modalities = record.get("modalities")
    if isinstance(modalities, list):
        updated["modalities"] = [
            _wrap_model_with_evidence(
                modality,
                schema_mod.ModalityBase,
                evidence_index,
                context_label=_modality_context_label(modality)
                if isinstance(modality, dict)
                else None,
            )
            if isinstance(modality, dict)
            else modality
            for modality in modalities
        ]

    analyses = record.get("analyses")
    if isinstance(analyses, list):
        updated["analyses"] = [
            _wrap_model_with_evidence(
                analysis,
                schema_mod.AnalysisBase,
                evidence_index,
                context_label=_analysis_context_label(analysis)
                if isinstance(analysis, dict)
                else None,
                context_exclude={"analysis_label"},
            )
            if isinstance(analysis, dict)
            else analysis
            for analysis in analyses
        ]

    links = record.get("links")
    if isinstance(links, dict):
        updated["links"] = _wrap_model_with_evidence(
            links, schema_mod.StudyLinks, evidence_index
        )

    if progress is not None:
        progress.close()

    return updated


def _build_link_prompt(
    text: str,
    groups: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    modalities: list[dict[str, Any]],
    analyses: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You extract explicit links between entities. "
        "Only create links when explicitly stated. "
        "Return JSON only."
    )
    doc_message = f"Document text:\n{text}"

    def label_for(item: dict[str, Any], keys: list[str]) -> str:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return item.get("id", "")

    def modality_label(item: dict[str, Any]) -> str:
        modality_type = item.get("modality_type")
        if isinstance(modality_type, dict):
            for key in ["family", "subtype"]:
                value = modality_type.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return label_for(item, ["sequence_name", "voxel_size", "manufacturer"])

    def summarize(items: list[dict[str, Any]], label_keys: list[str]) -> list[dict[str, Any]]:
        summarized = []
        for item in items:
            entry = {"id": item.get("id", ""), "label": label_for(item, label_keys)}
            if "conditions" in item:
                entry["conditions"] = item.get("conditions")
            summarized.append(entry)
        return summarized

    entity_block = {
        "groups": summarize(groups, ["cohort_label", "medical_condition"]),
        "tasks": summarize(tasks, ["task_name", "task_description"]),
        "modalities": [
            {"id": item.get("id", ""), "label": modality_label(item)}
            for item in modalities
        ],
        "analyses": summarize(analyses, ["analysis_label", "contrast_formula", "analysis_method"]),
    }

    instructions = (
        "Use the entities listed below (IDs are required). "
        "Create edges only when explicitly stated in the document text. "
        "Do not infer. Always include all edge lists; use [] if none. "
        "For analysis_condition, only link when a named task condition is explicitly referenced.\n\n"
        f"Entities:\n{json.dumps(entity_block, indent=2)}\n\n"
        "Output JSON format:\n"
        "{\n"
        '  "group_task": [{"group_id": "G1", "task_id": "T1"}],\n'
        '  "task_modality": [{"task_id": "T1", "modality_id": "M1"}],\n'
        '  "analysis_task": [{"analysis_id": "A1", "task_id": "T1"}],\n'
        '  "analysis_group": [{"analysis_id": "A1", "group_id": "G1"}],\n'
        '  "analysis_condition": [{"analysis_id": "A1", "condition_id": "C1"}],\n'
        '  "group_modality": [{"group_id": "G1", "modality_id": "M1", "n_scanned": 20}]\n'
        "}\n"
        "\nUse the document text above."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": doc_message},
        {"role": "user", "content": instructions},
    ]


def _link_schema() -> dict[str, Any]:
    def edge_schema(fields: list[str]) -> dict[str, Any]:
        props = {field: {"type": "string"} for field in fields}
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": props,
            "required": fields,
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "group_task": {
                "type": "array",
                "items": edge_schema(["group_id", "task_id"]),
            },
            "task_modality": {
                "type": "array",
                "items": edge_schema(["task_id", "modality_id"]),
            },
            "analysis_task": {
                "type": "array",
                "items": edge_schema(["analysis_id", "task_id"]),
            },
            "analysis_group": {
                "type": "array",
                "items": edge_schema(["analysis_id", "group_id"]),
            },
            "analysis_condition": {
                "type": "array",
                "items": edge_schema(["analysis_id", "condition_id"]),
            },
            "group_modality": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "group_id": {"type": "string"},
                        "modality_id": {"type": "string"},
                        "n_scanned": {"type": ["number", "null"]},
                    },
                    "required": ["group_id", "modality_id"],
                },
            },
        },
        "required": [
            "group_task",
            "task_modality",
            "analysis_task",
            "analysis_group",
            "analysis_condition",
            "group_modality",
        ],
    }


def _build_record_schema(specs: list[EntitySpec], *, include_ids: bool) -> dict[str, Any]:
    def entity_items_schema(spec_key: str, *, include_id: bool) -> dict[str, Any]:
        spec = next(spec for spec in specs if spec.key == spec_key)
        prompt_spec = _load_prompt_spec_cached(spec.prompt_path)
        fields = prompt_spec.get("fields") or []
        item_schema = _prune_model_schema(spec.model, fields, include_id=include_id)
        return {"type": "array", "items": item_schema}

    study_spec = next(spec for spec in specs if spec.key == "study")
    study_fields = _load_prompt_spec_cached(study_spec.prompt_path).get("fields") or []
    study_schema = _prune_model_schema(study_spec.model, study_fields, include_id=False)
    study_schema["type"] = ["object", "null"]

    demographics_schema = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {"groups": entity_items_schema("groups", include_id=include_ids)},
        "required": ["groups"],
    }

    links_schema = _link_schema()
    links_schema["type"] = ["object", "null"]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "study": study_schema,
            "demographics": demographics_schema,
            "tasks": entity_items_schema("tasks", include_id=include_ids),
            "modalities": entity_items_schema("modalities", include_id=include_ids),
            "analyses": entity_items_schema("analyses", include_id=include_ids),
            "links": links_schema,
        },
        "required": ["study", "demographics", "tasks", "modalities", "analyses", "links"],
    }


def _verification_schema(entity_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "corrected": entity_schema,
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        "field": {"type": ["string", "null"]},
                        "note": {"type": "string"},
                    },
                    "required": ["action", "target", "note"],
                },
            },
        },
        "required": ["corrected", "changes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract entities with GPT-5 nano (no LangExtract) and link them."
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
        help="Where to write grounded JSON (default: alongside input, .grounded suffix).",
    )
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--service-tier", default="flex")
    parser.add_argument("--max-output-tokens", type=int, default=2500)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-chars", type=int, default=400_000)
    parser.add_argument("--skip-links", action="store_true", help="Skip the linking pass.")
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Re-run extraction even if a .entities.gpt5-nano.json file exists.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for evidence retrieval.",
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="NLI model used to validate evidence sentences.",
    )
    parser.add_argument("--evidence-top-k", type=int, default=10)
    parser.add_argument("--evidence-entailment-threshold", type=float, default=0.6)
    parser.add_argument("--evidence-embedding-batch-size", type=int, default=32)
    parser.add_argument("--evidence-nli-batch-size", type=int, default=32)
    parser.add_argument(
        "--evidence-cache-dir",
        type=Path,
        default=Path(".cache/evidence_embeddings"),
    )
    parser.add_argument(
        "--evidence-device",
        type=int,
        default=-1,
        help="Device for evidence models (-1 for CPU, 0+ for CUDA).",
    )
    parser.add_argument(
        "--evidence-progress",
        action="store_true",
        help="Show a progress bar while attaching evidence spans.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Set CPU thread counts for torch and BLAS env vars.",
    )
    args = parser.parse_args()

    if args.cpu_threads is not None:
        cpu_threads = max(1, args.cpu_threads)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(cpu_threads)
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None:
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(cpu_threads)

    load_dotenv(".env")
    if not args.input_md.exists():
        print(f"Input not found: {args.input_md}", file=sys.stderr)
        return 2

    text = _read_text(args.input_md, args.max_chars)
    expander = AbbreviationExpander.from_path(args.input_md.with_suffix(".abbreviations.json"))
    if expander is not None:
        expanded_text = expander.expand_text(text)
        if expanded_text != text:
            expanded_text_path = args.input_md.with_suffix(".expanded.md")
            expanded_text_path.write_text(expanded_text, encoding="utf-8")
    client = OpenAI()
    service_tier = args.service_tier or None
    if service_tier and service_tier.lower() == "none":
        service_tier = None

    raw_output_path = args.input_md.with_suffix(".entities.gpt5-nano.json")
    grounded_output_path = args.output_json
    if grounded_output_path is None:
        grounded_output_path = args.input_md.with_suffix(".entities.gpt5-nano.grounded.json")

    record: dict[str, Any] | None = None
    if raw_output_path.exists() and not args.force_reextract:
        try:
            existing = json.loads(raw_output_path.read_text(encoding="utf-8"))
            if hasattr(schema_mod.StudyRecord, "model_validate"):
                schema_mod.StudyRecord.model_validate(existing)
            else:
                schema_mod.StudyRecord.parse_obj(existing)
            record = existing
        except Exception:
            record = None

    meta_by_key: dict[str, dict[str, FieldMeta]] = {}
    if record is None:
        results: dict[str, Any] = {
            "study": None,
            "groups": [],
            "tasks": [],
            "modalities": [],
            "analyses": [],
            "links": None,
        }
        for spec in ENTITY_SPECS:
            prompt_spec = _load_prompt_spec_cached(spec.prompt_path)
            fields = prompt_spec.get("fields") or []
            field_meta = _field_meta_map(spec.model, fields)
            meta_by_key[spec.key] = field_meta
            entity_prompt = _build_entity_prompt(prompt_spec, field_meta)
            item_schema = _prune_model_schema(spec.model, fields, include_id=False)
            schema = _wrap_items_schema(item_schema, spec.max_items)
            messages = _build_messages(text, entity_prompt)
            data = _request_with_retries(
                client=client,
                messages=messages,
                model=args.model,
                service_tier=service_tier,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                timeout_s=args.timeout,
                schema=schema,
                schema_name=f"{spec.key}_schema",
            )
            items = data.get("items", []) if isinstance(data, dict) else []
            if spec.key == "study":
                results["study"] = items[0] if items else None
            else:
                results[spec.key] = items

        # Assign IDs and normalize conditions for linking.
        results["groups"] = _ensure_ids(results["groups"], "G", overwrite=True)
        results["tasks"] = _ensure_ids(results["tasks"], "T", overwrite=True)
        results["tasks"] = _normalize_task_conditions(results["tasks"])
        results["modalities"] = _ensure_ids(results["modalities"], "M", overwrite=True)
        results["analyses"] = _ensure_ids(results["analyses"], "A", overwrite=True)

        if not args.skip_links:
            link_messages = _build_link_prompt(
                text=text,
                groups=results["groups"],
                tasks=results["tasks"],
                modalities=results["modalities"],
                analyses=results["analyses"],
            )
            link_schema = _link_schema()
            links = _request_with_retries(
                client=client,
                messages=link_messages,
                model=args.model,
                service_tier=service_tier,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                timeout_s=args.timeout,
                schema=link_schema,
                schema_name="study_links",
            )
            results["links"] = links

        record = {
            "study": results["study"],
            "demographics": {"groups": results["groups"]},
            "tasks": results["tasks"],
            "modalities": results["modalities"],
            "analyses": results["analyses"],
            "links": results["links"],
        }

        # Verification pass: check entities and fields against the full text.
        field_descriptions = {
            "study": _build_entity_descriptions(
                "StudyMetadataModel",
                meta_by_key.get("study", {}),
            ),
            "demographics": (
                "Demographics (groups only):\n"
                + _build_entity_descriptions(
                    "GroupBase",
                    meta_by_key.get("groups", {}),
                )
            ),
            "tasks": _build_entity_descriptions(
                "TaskBase",
                meta_by_key.get("tasks", {}),
            ),
            "modalities": _build_entity_descriptions(
                "ModalityBase",
                meta_by_key.get("modalities", {}),
            ),
            "analyses": _build_entity_descriptions(
                "AnalysisBase",
                meta_by_key.get("analyses", {}),
            ),
            "links": (
                "StudyLinks fields:\n"
                "- group_task: link groups to tasks they performed.\n"
                "- task_modality: link tasks to modalities used.\n"
                "- analysis_task: link analyses to tasks they test.\n"
                "- analysis_group: link analyses to groups included.\n"
                "- analysis_condition: link analyses to task conditions.\n"
                "- group_modality: link groups to modalities they underwent (n_scanned if stated).\n"
            ),
        }
        verify_messages = _build_verification_prompt(text, record, field_descriptions)
        entity_schema = _build_record_schema(ENTITY_SPECS, include_ids=True)
        verify_schema = _verification_schema(entity_schema)
        verification = _request_with_retries(
            client=client,
            messages=verify_messages,
            model=args.model,
            service_tier=service_tier,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            timeout_s=args.timeout,
            schema=verify_schema,
            schema_name="verification",
        )
        corrected = verification.get("corrected") if isinstance(verification, dict) else None
        if isinstance(corrected, dict):
            corrected["verification_changes"] = verification.get("changes", [])
            record = _assign_missing_ids_to_record(corrected)

        raw_output_path.write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )

    if expander is not None:
        expanded_record = expander.expand_record(record)
        expanded_record_path = raw_output_path.with_suffix(".expanded.json")
        if expanded_record != record:
            expanded_record_path.write_text(
                json.dumps(expanded_record, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        record = expanded_record

    evidence_config = EvidenceConfig(
        embedding_model=args.embedding_model,
        nli_model=args.nli_model,
        top_k=args.evidence_top_k,
        entailment_threshold=args.evidence_entailment_threshold,
        embedding_batch_size=args.evidence_embedding_batch_size,
        nli_batch_size=args.evidence_nli_batch_size,
        cache_dir=args.evidence_cache_dir,
        device=args.evidence_device,
        show_progress=args.evidence_progress,
    )
    record = _attach_evidence_to_record(
        record,
        text=text,
        document_id=args.input_md.stem,
        config=evidence_config,
        llm_client=client,
        llm_service_tier=service_tier,
        expander=expander,
    )

    grounded_output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {grounded_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# TODO (eventually)
# finally, taking inspiration from langextract, create a visualization form for the created annotations, highlighting their spans over the markdown text, having different color schemes for the different Entities (Study/Task/Modality/Analysis/Group), and critically, allowing users to edit the value that is associated with that key. Have a button that allows you delete all evidentiary spans for that key/value and add the ability to highlight new spans for that piece of evidence.

# TODO (eventually)
# also when I looked at the prompt, enums are not being forwarded to the prompt so that the prompt knows what values it can use for a certain field.
