"""Making a record match the shape the schema declares.

The first of three kinds of fix, and the one that has to run before the others can read
anything: a slot holding a wrapper where a bare id belongs, or a bare scalar where a wrapper
belongs, is not a record the schema can be applied to. `listify_nested` and `repair_wrappers`
therefore cannot use the schema-guided `walk` -- they exist to make the schema applicable.

Each of these is a `Repair` in `fix.build_sequence`, which holds the order and the reason
each one sits where it does.
"""

from __future__ import annotations

from collections.abc import Mapping
from pondie.extraction.record import direction
from pondie import schema
from pondie.extraction.record import walk
from pondie.formats import values
from pondie.schema.reader import Schema
from typing import Any
import re


def repair_wrappers(node: Any, path: str = "") -> list[str]:
    """Put a collapsed ExtractedValue back together, and report every one.

    The wrapper is `{"extraction_status": "extracted", "value": X}`, and the model
    intermittently writes X into the status slot instead -- `"extraction_status":
    "undirected"` for a direction, `"extraction_status": 0.05` for an alpha level. Every
    such field is invalid against the schema, and the numeric ones additionally break
    `ls.py export`, whose task contract requires `llm_status` to be a string.

    The repair is unambiguous, which is why it is done rather than reported: the status
    slot has exactly two legal strings, so anything else in it was never a status. A
    value already sitting in `value` is kept and only the status is corrected; otherwise
    the misplaced payload is moved into `value`.

    Deliberately here and not in `render.normalize`: it has to hold for payloads
    already on disk, so a rebuild fixes them without paying for the extraction again.
    """

    repaired: list[str] = []
    if isinstance(node, dict):
        status = node.get("extraction_status") if "extraction_status" in node else None

        if "extraction_status" in node and status not in values.STATUSES:
            if "value" not in node or node["value"] in (None, ""):
                node["value"] = status
            node["extraction_status"] = "extracted"
            node.setdefault("value_source", "reported")
            repaired.append(f"{path or '<root>'}: status held {status!r}")
        for key, value in node.items():
            repaired += repair_wrappers(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            repaired += repair_wrappers(value, f"{path}[{index}]")
    return repaired


def listify_nested(body: dict[str, Any], sch: Schema) -> list[str]:
    """Wrap a lone object in a list wherever the schema declares a multivalued slot.

    `model_estimations[].terms` is the one that recurs: an estimation with a single
    ModelTerm comes back as the object rather than a list of one. Which of the three
    shapes a slot takes is the most confusable thing in this schema -- the prompt states
    it per line for that reason -- and this is the benign half of getting it wrong, so it
    is repaired here instead of costing a re-extraction.

    Schema-driven, and reference slots are included: a multivalued reference given one
    bare id has the same shape problem.

    Two things the walk has to get right to reach `ConnectivityDetails.seed_regions`, which
    is where 40 of the corpus's shape errors sat. A single-valued nested slot is descended
    into even though it needs no repair itself, because the slots that do are inside it --
    `Analysis.details`, `.effect` and `.inference_settings` are all single-valued. The
    recursion resolves the type designator, because `details` ranges on the abstract
    AnalysisDetails, whose only attribute is `details_type`; recursing on the declared range
    finds no `seed_regions` to repair and reports nothing.
    """

    fixed: list[str] = []

    def visit(node: Any, class_name: str, path: str) -> None:
        if not isinstance(node, dict) or values.is_field(node):
            return
        class_name = sch.designated_type(node, class_name)
        attributes = sch.attributes(class_name)
        for key, value in list(node.items()):
            attribute = attributes.get(key)
            if attribute is None:
                continue
            kind = sch.classify(key, attribute)
            if kind not in ("nested", "reference"):
                continue
            if not attribute.multivalued:
                # Nothing to repair on the slot itself, but its contents may need it.
                if kind == "nested" and isinstance(attribute.range, str):
                    visit(value, attribute["range"], f"{path}.{key}")
                continue
            if values.is_field(value):
                # A nested record list is not a multivalued scalar and has no
                # `not_reported` form: "no terms" is the empty list, not a wrapper
                # saying so. Where the model wrapped one anyway, take its `value` if it
                # carried the list and drop to empty if it only carried the excuse.
                inner = value.get("value")
                node[key] = (
                    inner
                    if isinstance(inner, list)
                    else ([inner] if isinstance(inner, dict) else [])
                )
                fixed.append(f"{path}.{key} (unwrapped {value.get('extraction_status')})")
            elif isinstance(value, dict) or (kind == "reference" and isinstance(value, str)):
                node[key] = [value]
                fixed.append(f"{path}.{key}")
            if kind == "nested":
                target = attribute.range
                if isinstance(target, str):
                    for index, item in enumerate(node[key] if isinstance(node[key], list) else []):
                        visit(item, target, f"{path}.{key}[{index}]")

    study_attributes = sch.attributes("Study")
    for attr in schema.entity_lists().values():
        attribute = study_attributes.get(attr)
        target = attribute.range if attribute is not None else None
        if isinstance(target, str):
            for index, entity in enumerate(body.get(attr, []) or []):
                visit(entity, target, f"{attr}[{index}]")
    return fixed


def unwrap_singleton_lists(body: dict[str, Any], sch: Schema) -> list[str]:
    """Unwrap a one-item list in a wrapper whose `value` is declared scalar.

    The inverse of `listify_scalars`, and the commoner direction by an order of magnitude:
    over 1,817 records 21,701 scalar wrappers held a one-item list against 1,900-odd scalars
    in list slots. It concentrates on the enum slots, because a model writing an enum member
    has just been told the field has a vocabulary and offers one from it as a list:
    `Analysis.spatial_scope` 4,470 times, `Analysis.prespecification` 4,328, `Group.species`
    2,207, `Region.region_type` 2,151.

    It matters because `values.read` returns what it finds. A consumer testing
    `spatial_scope == "whole_brain"` matches nothing against `["whole_brain"]`, and
    `spatial_scope` is the field that cost 17133391 its place in a meta-analysis.

    **Only a one-item list.** A longer one is a claim the slot cannot hold -- 730 of them,
    `spatial_scope: ['whole_brain', 'roi']` 31 times, where the two exclude each other, and
    `Region.region_type: ['anatomical', 'atlas_parcel']` 198 times, which reads more like the
    storage slot wanting `multivalued` than like an extraction error. Picking one would be
    deciding which, so those are left for `check_value_cardinality` to report.
    """

    fixed: list[str] = []
    for slot in walk.fields(body, sch):
        declared = slot.declared_value(sch)
        if declared is None or declared.multivalued or not values.is_field(slot.value):
            continue
        inner = slot.value.get("value")
        if isinstance(inner, list) and len(inner) == 1:
            slot.value["value"] = inner[0]
            fixed.append(f"{slot.path}: [{inner[0]!r}] -> {inner[0]!r}")
    return fixed


def listify_scalars(body: dict[str, Any], sch: Schema) -> list[str]:
    """Wrap a lone scalar in a list inside an `Extracted<T>List` wrapper.

    The other half of the shape confusion `listify_nested` repairs, one level down. A
    multivalued *source-derived* field is one wrapper whose `value` is a list -- the
    convention extraction-readme.md §2 leads with -- and a model that has just been told
    a wrapper holds "the value" writes the string. `interpretations` is where it recurs.

    Distinct from `listify_nested`, which repairs the slot: here the slot is right and its
    `value` is not, so the wrapper class's own `value` declaration is what decides. Walks
    from Study rather than the entity lists so a wrapper under `design.arms[]` is reached.
    """

    fixed: list[str] = []
    for slot in walk.fields(body, sch):
        declared = slot.declared_value(sch)
        if declared is None or not declared.multivalued or not values.is_field(slot.value):
            continue
        if "value" not in slot.value:
            continue
        inner = slot.value["value"]
        # A missing value is a different fault and stays visible as one; only a present
        # scalar is the shape this repairs.
        if inner is not None and not isinstance(inner, list):
            slot.value["value"] = [inner]
            fixed.append(slot.path)
    return fixed


def unwrap_plain_slots(body: dict[str, Any], sch: Schema) -> list[str]:
    """Unwrap an ExtractedValue the model put in a slot that holds a bare scalar.

    The record has two kinds of slot and they look alike from inside a model: a
    source-derived value carries `{"value": ..., "evidence": ...}`, and a cross-reference
    or native scalar carries the bare thing. Having just written twenty wrappers, the
    model writes a twenty-first, and the validator reports `must be a string, got dict`.

    Repaired rather than reported because there is nothing to decide: the wrapper's own
    `value` is the answer, and the evidence it carried was never a slot the schema has a
    place for. Only `reference` and `native` slots are touched -- an `evidence` slot is
    supposed to hold a wrapper and unwrapping it would destroy the value.
    """

    fixed: list[str] = []
    for slot in walk.slots(body, sch, kinds=("reference", "native", "evidence")):
        if slot.kind in ("reference", "native"):
            if values.is_field(slot.value) and "value" in slot.value:
                slot.owner[slot.key] = slot.value["value"]
                fixed.append(f"{slot.path}: unwrapped a wrapper into a {slot.kind} slot")
            elif values.is_field(slot.value):
                # A wrapper with no `value` says `not_reported`, which is the right encoding
                # for an evidence slot and meaningless here: a reference has no wrapper form,
                # so "not reported" is simply absence. Dropped rather than left, because the
                # validator reads it as a malformed cross-reference and the paper said
                # nothing either way.
                del slot.owner[slot.key]
                status = slot.value.get("extraction_status", "valueless")
                fixed.append(
                    f"{slot.path}: dropped an empty {status!r} wrapper from a "
                    f"{slot.kind} slot"
                )
            elif isinstance(slot.value, list):
                for index, item in enumerate(slot.value):
                    if values.is_field(item) and "value" in item:
                        slot.value[index] = item["value"]
                        fixed.append(
                            f"{slot.path}[{index}]: unwrapped a wrapper into a "
                            f"{slot.kind} slot"
                        )
        elif slot.value is not None and not isinstance(slot.value, (dict, list)):
            # The inverse slip: a bare scalar in a slot that holds an ExtractedValue. The
            # value is the model's answer and it offered no span for it, so the evidence is
            # honestly `not_found` rather than invented.
            slot.owner[slot.key] = values.wrap(
                slot.value, source="reported", evidence="not_found")
            fixed.append(
                f"{slot.path}: wrapped a bare {type(slot.value).__name__} into an "
                f"ExtractedValue")
    return fixed


#: Slot suffixes whose ExtractedValue must hold a number. Read from the wrapper's own
#: declared range rather than guessed, but the suffix is what makes the intent legible
#: at the call site.
def coerce_numeric_values(body: dict[str, Any], sch: Schema) -> list[str]:
    """Turn `"4.5"` into `4.5` where the wrapper declares a number.

    A number read off a table arrives as text and the model passes it through. The
    schema says what the slot holds, so the conversion is arithmetic; a value that is
    not a number after stripping units is left alone and stays a validator finding,
    because inventing a number is worse than reporting a string.
    """

    NUMERIC = ("float", "double", "decimal", "integer")
    fixed: list[str] = []
    for slot in walk.fields(body, sch):
        if not values.is_field(slot.value):
            continue
        declared = slot.declared_value(sch)
        # `getattr`, not `.range`: the fallback for a wrapper class that declares no `value`
        # slot was a bare `{}`, which has no `.range` and took the whole build down with an
        # AttributeError -- one paper in 89, at the last stage, after every model call it
        # needed had been paid for. `declared_value` returns None there instead.
        wants = getattr(declared, "range", None)
        inner = slot.value.get("value")
        if wants not in NUMERIC or not isinstance(inner, str):
            continue
        cleaned = re.sub(r"[^0-9eE.+-]", "", inner.strip())
        try:
            number = float(cleaned)
        except ValueError:
            continue
        slot.value["value"] = int(number) if wants == "integer" else number
        fixed.append(f"{slot.path}: {inner!r} -> {slot.value['value']}")
    return fixed


def rehome_stray_tables(body: dict[str, Any], sch: Schema) -> list[str]:
    """Move a Table the model wrote as a Study attribute into `tables`.

    Seen as `Study: attribute 'tab4' is not declared on Study` while every analysis
    referenced `tab4`. The cost is not cosmetic: the stray key is dropped on load, so
    every analysis pointing at it loses the table its coordinates are joined through,
    and the paper contributes nothing to a coordinate query.

    Only keys that some analysis actually references are moved, and only when they carry
    no `local_id` of their own -- anything else is a slot the schema does not know
    about, which is a different fault and stays reported.
    """

    declared = set(sch.attributes("Study") or {})
    referenced: set[str] = set()
    for analysis in body.get("analyses") or []:
        if isinstance(analysis, Mapping):
            cited = values.read(analysis.get("tables")) or []
            referenced |= {
                t for t in (cited if isinstance(cited, list) else [cited]) if isinstance(t, str)
            }

    moved: list[str] = []
    for key in [k for k in body if k not in declared and k in referenced]:
        stray = body.pop(key)
        entry = dict(stray) if isinstance(stray, Mapping) else {}
        entry.setdefault("local_id", key)
        body.setdefault("tables", []).append(entry)
        moved.append(f"Study.{key}: moved into tables[] as {key!r}")
    return moved


#: The fields that give a declared entity its identity. A row holding none of them names
#: nothing the next pass could build, and nothing an analysis could dangle on.
IDENTIFYING = ("local_id", "kind", "label")


def is_vacuous(entry: Any) -> bool:
    """A declared entity with every identifying field blank.

    Not the same as a malformed one. `{"local_id": null, "kind": null, "label": null}` is
    valid JSON, conforms to the shape the prompt asks for, and a list holding it is
    truthy -- so every non-emptiness check passes and it reaches a record unremarked.
    """
    if not isinstance(entry, Mapping):
        return False
    return all(not str(entry.get(field) or "").strip() for field in IDENTIFYING)


def drop_vacuous_demands(body: dict[str, Any]) -> list[str]:
    """Drop a declared entity whose local_id, kind and label are all blank.

    The `demands` pass sometimes fills the shape it was asked for with nulls rather than
    content -- see `render.vacuous_entity_demands` for the case that found it. Such a row
    is not a broken entity that could be repaired into a good one; it carries no field to
    repair from, so the only sound handling is to remove it and leave every other row as
    the pass wrote it.

    Dropping cannot orphan anything. A reference dangles by naming a `local_id`, and a row
    with no `local_id` is named by nothing, so no analysis can be pointing at it.

    A payload where *every* row is vacuous is a retry, not a repair: `postcondition_failures`
    catches it inside the pass, before this runs. Reaching here with nothing left means the
    retries were spent, and emptying the list is still right -- `satisfy` holds itself to
    this list, and holding it to a row that names no entity is worse than holding it to none.
    """
    entries = body.get("required_entities")
    if not isinstance(entries, list):
        return []
    kept = [entry for entry in entries if not is_vacuous(entry)]
    if len(kept) == len(entries):
        return []
    body["required_entities"] = kept
    return [
        f"dropped {len(entries) - len(kept)} required_entities row(s) with no local_id, "
        f"kind or label; {len(kept)} kept"
    ]
