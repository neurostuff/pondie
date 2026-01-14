from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from information_extraction.schema_utils import _find_literal_values, _iter_model_fields


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


def _build_field_descriptions(meta_by_key: dict[str, dict[str, FieldMeta]]) -> dict[str, str]:
    return {
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
