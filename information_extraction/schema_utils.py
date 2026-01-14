from __future__ import annotations

from typing import Annotated, Any, Union, get_args, get_origin

from information_extraction import schema as schema_mod
from typing_extensions import Literal


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


def _unwrap_annotated(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        if args:
            return args[0]
    return annotation


def _unwrap_optional(annotation: Any) -> Any:
    annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _find_literal_values(annotation: Any) -> list[str] | None:
    annotation = _unwrap_optional(annotation)
    if annotation is None:
        return None
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return _find_literal_values(args[0]) if args else None
    if origin is list:
        args = get_args(annotation)
        return _find_literal_values(args[0]) if args else None
    if origin is schema_mod.ExtractedValue:
        args = get_args(annotation)
        return _find_literal_values(args[0]) if args else None
    if isinstance(annotation, type) and issubclass(annotation, schema_mod.ExtractedValue):
        metadata = getattr(annotation, "__pydantic_generic_metadata__", None) or {}
        args = metadata.get("args") or ()
        if args:
            return _find_literal_values(args[0])
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
