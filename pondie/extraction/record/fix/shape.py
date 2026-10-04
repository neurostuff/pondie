"""Making a record match the shape the schema declares.

The first of three kinds of fix, and the one that has to run before the others can read
anything: a slot holding a wrapper where a bare id belongs, or a bare scalar where a wrapper
belongs, is not a record the schema can be applied to. `listify_nested` and `repair_wrappers`
therefore cannot use the schema-guided `walk` -- they exist to make the schema applicable.

Each of these is a `Repair` in `fix.build_sequence`, which holds the order and the reason
each one sits where it does.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pondie import schema
from pondie.extraction.record import walk
from pondie.formats import values
from pondie.schema.reader import Schema
from typing import Any
import re


#: `value_source` words that mean the pipeline, not the paper, produced the value.
GENERATED_WORDS = {"inferred", "derived", "computed", "calculated", "estimated", "imputed"}


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

    Two slips in a well-formed wrapper go the same way: a `value_source` the vocabulary
    lacks but plainly means `generated` ("inferred"), and an `unreported_reason` on a value
    that was reported.

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
        if node.get("extraction_status") == "extracted":
            source = node.get("value_source")
            if isinstance(source, str) and source.strip().lower() in GENERATED_WORDS:
                node["value_source"] = "generated"
                repaired.append(f"{path or '<root>'}: value_source {source!r} -> 'generated'")
            if node.pop("unreported_reason", None) is not None:
                repaired.append(f"{path or '<root>'}: extracted, so no unreported_reason")
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


#: What a slot name can be. A key outside it is debris from a malformed reply (`":{"`).
SLOT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def drop_impossible_keys(body: dict[str, Any]) -> list[str]:
    """Remove keys no slot could have, at any depth outside a field's value."""
    dropped: list[str] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, list):
            for i, item in enumerate(node):
                visit(item, f"{path}[{i}]")
        elif isinstance(node, dict) and not values.is_field(node):
            for key in list(node):
                if not SLOT_NAME.match(str(key)):
                    del node[key]
                    dropped.append(f"{path}: dropped {key!r}, not a possible slot name")
                else:
                    visit(node[key], f"{path}.{key}")

    visit(body, "Study")
    return dropped


def drop_code_filled(body: dict[str, Any], sch: Schema) -> list[str]:
    """Remove a model's value for a slot code fills (`schema.code_fills`).

    Code writes these after the merge (`is_healthy`, the normalized demographics,
    `mirror_of`, PubMed's `language` and `study_type`); a model's answer is either
    overwritten or, where nothing overwrites it, left in the wrong shape.
    """
    dropped: list[str] = []
    for slot in walk.slots(body, sch):
        if schema.code_fills(slot.owner_class, slot.key):
            del slot.owner[slot.key]
            dropped.append(f"{slot.path}: dropped a model's value for a slot code fills")
    return dropped


#: The keys that make a mapping an ExtractedValue rather than an entity.
WRAPPER_KEYS = ("extraction_status", "value_source", "unreported_reason", "evidence")


def _objects(
    node: Any, class_name: str, sch: Schema, parent: dict[str, Any] | None = None,
    parent_class: str = "", path: str = "Study",
) -> Iterator[tuple[dict[str, Any], str, dict[str, Any] | None, str, str]]:
    """(object, class, parent, parent class, path) for every object the schema nests.

    Not `walk.entities`: that skips a node carrying wrapper keys, which is a whole entity
    written as a value -- one of the faults these repairs exist for.
    """
    if not isinstance(node, dict):
        return
    class_name = sch.designated_type(node, class_name)
    attributes = sch.attributes(class_name) or {}
    if not attributes:
        return
    yield node, class_name, parent, parent_class, path
    for key, attribute in list(attributes.items()):
        if key not in node or sch.classify(key, attribute) != "nested":
            continue
        if not isinstance(attribute.range, str):
            continue
        child = node[key]
        listed = isinstance(child, list)
        for index, item in enumerate(child if listed else [child]):
            suffix = f"[{index}]" if listed else ""
            yield from _objects(item, attribute.range, sch, node, class_name, f"{path}.{key}{suffix}")


def _empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _conflicts(target: Mapping[str, Any], key: str, value: Any) -> bool:
    """Whether `target` already holds a different value at `key`, evidence aside."""
    held = target.get(key)
    return not _empty(held) and values.read(held) != values.read(value)


def _settle(target: dict[str, Any], key: str, value: Any) -> bool:
    """Put `value` at `target[key]` unless a different value is there; True if it may go.

    The same value already there makes the stray a duplicate, so it may be removed -- its
    evidence moving over when the one in place has none. A different value is two claims,
    and the caller leaves the stray where it was, still reported: nothing the model wrote
    is discarded to make a record validate.
    """
    if _conflicts(target, key, value):
        return False
    held = target.get(key)
    if _empty(held):
        target[key] = value
    elif (
        values.is_field(held)
        and values.is_field(value)
        and (held.get("evidence") or {}).get("status") != "present"
        and (value.get("evidence") or {}).get("status") == "present"
    ):
        held["evidence"] = value["evidence"]
    return True


def lift_misnested(body: dict[str, Any], sch: Schema) -> list[str]:
    """Put back what a model wrote one level from where the schema puts it.

    Three shapes, all decided by which class declares the key:

    - a parent's slot found in a child moves up to the parent (`tables` and
      `model_estimation` written under an analysis's `effect`); one the parent already
      holds is dropped, since the parent's own is the one the schema reads;
    - an object written inside one of its own slots is lifted into it: 30545239 wrote a
      factor level as `{"level": {"level": <field>, "groups": [...]}}`, leaving the term
      with no readable levels. Keys the outer object already holds win;
    - an object's slots written inside one of its field wrappers move up to it: 21078704
      put an analysis's `definition` and `prespecification` inside its `name`, 21498053 an
      effect's `statistic` inside its `kind`.

    A stray that disagrees with the value already in place is left where it is, still
    reported (`_settle`).
    """
    fixed: list[str] = []
    for node, class_name, parent, parent_class, path in list(_objects(body, "Study", sch)):
        declared = sch.attributes(class_name) or {}
        if parent is not None:
            above = sch.attributes(parent_class) or {}
            moving = [k for k in node if k not in declared and k in above]
            for key in [k for k in moving if k not in WRAPPER_KEYS]:
                if _settle(parent, key, node[key]):
                    node.pop(key)
                    fixed.append(f"{path}.{key}: moved up to the {parent_class}")
        for key, attribute in declared.items():
            inner = node.get(key)
            if values.is_field(inner):
                fixed += _lift_from_wrapper(node, key, inner, attribute, declared, sch, path)
                continue
            if (
                sch.classify(key, attribute) == "nested"
                or not isinstance(inner, dict)
                or values.is_field(inner)
                or key not in inner
                or not set(inner) <= set(declared)
                or any(_conflicts(node, k, v) for k, v in inner.items() if k != key)
            ):
                continue
            node[key] = inner.pop(key)
            for other, value in inner.items():
                _settle(node, other, value)
            fixed.append(f"{path}.{key}: a {class_name} written inside it, lifted out")
    return fixed


def _lift_from_wrapper(
    node: dict[str, Any], key: str, wrapper: dict[str, Any], attribute: Any,
    declared: Mapping[str, Any], sch: Schema, path: str,
) -> list[str]:
    """Move the owner's slots out of one of its field wrappers."""
    own = set(WRAPPER_KEYS) | {"value"}
    for wrapper_class in sch.ranges(attribute):
        own |= set(sch.attributes(wrapper_class) or {})
    fixed: list[str] = []
    for stray in [k for k in wrapper if k not in own and k in declared and k != key]:
        if _settle(node, stray, wrapper[stray]):
            wrapper.pop(stray)
            fixed.append(f"{path}.{key}.{stray}: moved out of the wrapper")
    return fixed


def drop_stray_slot_names(body: dict[str, Any], sch: Schema) -> list[str]:
    """Drop a bare string that names a slot from a list of objects.

    `design.timepoints` held `"arms"` beside its two Timepoints (25050433, 23072363): the
    next key's name, emitted as an item. A string in a list of objects is never an object,
    and one spelling a slot of the list's owner is debris rather than a misplaced id.
    """
    fixed: list[str] = []
    for node, class_name, _parent, _class, path in list(_objects(body, "Study", sch)):
        declared = sch.attributes(class_name) or {}
        for slot, attribute in declared.items():
            items = node.get(slot)
            if sch.classify(slot, attribute) != "nested" or not isinstance(items, list):
                continue
            stray = [i for i in items if isinstance(i, str) and i in declared]
            if stray:
                node[slot] = [i for i in items if not (isinstance(i, str) and i in declared)]
                fixed.append(f"{path}.{slot}: dropped {stray}, slot names rather than objects")
    return fixed


def rehome_keyed_entities(body: dict[str, Any], sch: Schema) -> list[str]:
    """Move an entity written as a key named after its own id into its list.

    Two shapes, told apart by what settles the entity's class:

    - a key some reference slot cites: `Study.tab4` while every analysis cited `tab4` as
      a table. The stray key is dropped on load, so each analysis lost the table its
      coordinates join through. The citing slot's range is the class;
    - an object of the host's own class whose `local_id` is the key: 21078704 wrote eleven
      analyses inside `a_2022_1`, keyed by their ids, and the built record held one.

    Anything else undeclared is a slot the schema does not know, and stays reported. So
    does an entity whose id its list already holds.
    """
    containers = sch.containers()
    cited: dict[str, str] = {}
    for slot in walk.slots(body, sch, kinds=("reference",)):
        named = values.read(slot.value)
        for target in named if isinstance(named, list) else [named]:
            if isinstance(target, str) and isinstance(slot.attribute.range, str):
                cited.setdefault(target, slot.attribute.range)

    moved: list[str] = []
    for node, class_name, _parent, _class, path in list(_objects(body, "Study", sch)):
        declared = sch.attributes(class_name) or {}
        for key in [k for k in node if k not in declared]:
            entry = node[key]
            own_id = entry.get("local_id") if isinstance(entry, Mapping) else None
            if key in cited and own_id in (None, key):
                target = cited[key]
            elif (
                isinstance(entry, Mapping)
                and own_id == key
                and set(entry) <= set(declared)
            ):
                target = class_name
            else:
                continue
            container = containers.get(target)
            held = body.get(container) if container else None
            if container is None or any(
                isinstance(e, Mapping) and e.get("local_id") == key for e in held or []
            ):
                continue
            node.pop(key)
            body.setdefault(container, []).append(
                {**(dict(entry) if isinstance(entry, Mapping) else {}), "local_id": key}
            )
            moved.append(f"{path}.{key}: moved into {container}[] as {key!r}")
    return moved


def unwrap_entities(body: dict[str, Any], sch: Schema) -> list[str]:
    """Strip wrapper keys from an entity written as if it were a value.

    `{"term": ..., "level": ..., "extraction_status": "extracted", "evidence": ...}` is a
    `Cell` with a wrapper's markers on it, which makes it read as a field rather than the
    entity it is. Only keys the class does not declare are removed.
    """
    fixed: list[str] = []
    for node, class_name, _parent, _class, path in _objects(body, "Study", sch):
        declared = sch.attributes(class_name) or {}
        stray = [k for k in WRAPPER_KEYS if k in node and k not in declared]
        if stray and set(node) - set(WRAPPER_KEYS):
            for key in stray:
                del node[key]
            fixed.append(f"{path}: removed wrapper keys {stray}")
    return fixed


#: Strings a model writes as a value when it means the slot's status.
STATUS_WORDS = {"not_reported", "not reported"}


def status_as_value(body: dict[str, Any], sch: Schema) -> list[str]:
    """Turn a value of `"not_reported"` into a `not_reported` field: the model put the
    status in the value, wrapped (`{"extraction_status": "extracted", "value":
    "not_reported"}`) or bare (`"direction": "not_reported"`)."""
    fixed: list[str] = []
    for slot in walk.fields(body, sch):
        field = slot.value
        if values.is_field(field):
            if field.get("extraction_status") != "extracted":
                continue
            value = field.get("value")
        else:
            value = field
        if isinstance(value, str) and value.strip().lower() in STATUS_WORDS:
            slot.owner[slot.key] = values.wrap(None, source="reported", evidence="not_found")
            fixed.append(f"{slot.path}: 'not_reported' written as a value")
    return fixed
