"""Walking a record the way the schema says it is shaped.

`values.iter_fields` finds wrappers by their marker and knows nothing about the schema, which
is right for the jobs that only need the value. Every job that needs the *declared* shape --
is this slot multivalued, what range does it hold, is it a reference or a nested entity --
had to walk the record itself, and ten functions in `builder` did: nine of them opened with
the same `if not isinstance(node, dict) or values.is_field(node): return` and eleven with the
same `designated_type` line, and they diverged in ways nothing forced them to agree on. One
passed no path. One followed lists and another did not.

So the walk is here once, as two generators, and a repair is the loop body rather than a
recursion with the loop body buried in it:

    for slot in fields(body, schema):          # every ExtractedValue, with its declaration
    for slot in references(body, schema):      # every local_id a slot points at
    for slot in slots(body, schema):           # every declared slot, whatever its kind
    for entity in entities(body, schema):      # every node with a local_id, and its class

A repair that renames an entity repoints what refers to it with `repoint`, which takes the
reference slots from the schema rather than a list of the ones its author knew about.

Generators rather than a visitor callback, because the caller then reads as a loop over the
thing it cares about. `drop_redundant_cell_levels` and `align_cell_levels` stay hand-written:
they walk analyses and model terms together, which is a join and not a traversal.

MUTATION IS SAFE. Each node's items are materialised before yielding, so a caller may assign
to `slot.owner[slot.key]` or delete it while iterating -- which is what a repair does.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from linkml_runtime.linkml_model.meta import SlotDefinition

from pondie.formats import values
from pondie.schema.reader import Schema


@dataclass(frozen=True, eq=False)
class Slot:
    """One slot of one entity, and what the schema declares it should hold."""

    #: The entity node holding it, so a caller can assign or delete.
    owner: dict[str, Any]
    owner_class: str
    key: str
    attribute: SlotDefinition
    #: What is there now: an `ExtractedValue` for a field, an id or list of ids for a
    #: reference.
    value: Any
    #: The dotted path `BuildReport` and the validator report under.
    path: str
    #: What `Schema.classify` calls it: `evidence`, `reference`, `native` or `identifier`.
    kind: str

    @property
    def range(self) -> str | None:
        declared = self.attribute.range
        return declared if isinstance(declared, str) else None

    def declared_value(self, schema: Schema) -> SlotDefinition | None:
        """The wrapper class's own `value` slot: where cardinality and range really live.

        A field's `attribute.range` is the wrapper (`ExtractedString`), not the value. Five
        repairs need the inner declaration and each reached for it differently; this is the
        one way to ask.
        """
        wrapper = self.range
        if wrapper is None:
            return None
        return (schema.attributes(wrapper) or {}).get("value")


@dataclass(frozen=True, eq=False)
class Entity:
    """One node the schema treats as an entity, with the class it is an instance of."""

    node: dict[str, Any]
    class_name: str
    path: str

    @property
    def local_id(self) -> str | None:
        found = self.node.get("local_id")
        return found if isinstance(found, str) and found else None


def _descend(
    node: Any, class_name: str, path: str, schema: Schema
) -> Iterator[tuple[dict[str, Any], str, str]]:
    """(node, class, path) for this entity and every entity nested under it."""
    if not isinstance(node, dict) or values.is_field(node):
        return
    resolved = schema.designated_type(node, class_name)
    if not schema.attributes(resolved):
        return
    yield node, resolved, path
    for key, attribute in list((schema.attributes(resolved) or {}).items()):
        if key not in node or schema.classify(key, attribute) != "nested":
            continue
        target = attribute.range
        if not isinstance(target, str):
            continue
        child = node[key]
        listed = isinstance(child, list)
        for index, item in enumerate(child if listed else [child]):
            suffix = f"[{index}]" if listed else ""
            yield from _descend(item, target, f"{path}.{key}{suffix}", schema)


def entities(body: dict, schema: Schema, *, root: str = "Study") -> Iterator[Entity]:
    """Every entity in the record, outermost first."""
    for node, class_name, path in _descend(body, root, root, schema):
        yield Entity(node=node, class_name=class_name, path=path)


def slots(
    body: dict, schema: Schema, *, kinds: tuple[str, ...] | None = None, root: str = "Study"
) -> Iterator[Slot]:
    """Every declared slot of every entity, optionally narrowed to some kinds.

    `kinds=None` yields all of them, which `unwrap_plain_slots` needs: the slip it repairs
    is a wrapper in a slot that holds a bare scalar AND a bare scalar in a slot that holds
    a wrapper, so it has to see `reference`, `native` and `evidence` together.
    """
    for node, class_name, path in _descend(body, root, root, schema):
        attributes = schema.attributes(class_name) or {}
        for key, value in list(node.items()):
            attribute = attributes.get(key)
            if attribute is None:
                continue
            kind = schema.classify(key, attribute)
            if kinds is not None and kind not in kinds:
                continue
            yield Slot(
                owner=node,
                owner_class=class_name,
                key=key,
                attribute=attribute,
                value=value,
                path=f"{path}.{key}",
                kind=kind,
            )


def fields(body: dict, schema: Schema, *, root: str = "Study") -> Iterator[Slot]:
    """Every `ExtractedValue` slot the schema declares, whatever is in it.

    Yielded even when the value is malformed, because the repairs that put a wrapper back
    into shape are exactly the ones that need to see it.
    """
    yield from slots(body, schema, kinds=("evidence",), root=root)


def references(body: dict, schema: Schema, *, root: str = "Study") -> Iterator[Slot]:
    """Every slot holding a `local_id`, or a list of them, rather than a nested entity."""
    yield from slots(body, schema, kinds=("reference",), root=root)


def declared_ids(body: dict, schema: Schema, *, root: str = "Study") -> dict[str, str]:
    """local_id -> the class that declared it.

    Schema-guided rather than a sweep of the top-level lists, which is the difference that
    matters: `ModelTerm` lives under `model_estimations[].terms` and `Condition` under
    `tasks[].conditions`, so a top-level sweep declares neither and every `Cell.term`
    reference looks dangling. `repair_references` swept the top level and
    `validate.index_ids` descended, and the two disagreed about which references resolved.
    """
    found: dict[str, str] = {}
    for entity in entities(body, schema, root=root):
        if entity.local_id:
            found[entity.local_id] = entity.class_name
    return found


def repoint(
    node: dict,
    schema: Schema,
    mapping: Mapping[str, str],
    *,
    root: str = "Study",
    target: str | None = None,
) -> int:
    """Rewrite every reference in `node`, an entity of class `root`, through `mapping`.

    `target` limits the rewrite to slots whose range is that class or a subclass of it, for
    a mapping that renames ids of one class. Returns how many references changed.
    """
    targets = None if target is None else {target, *schema.subclasses(target)}
    changed = 0
    for slot in references(node, schema, root=root):
        if targets is not None and not targets & set(schema.ranges(slot.attribute)):
            continue
        if isinstance(slot.value, str) and slot.value in mapping:
            slot.owner[slot.key] = mapping[slot.value]
            changed += 1
        elif isinstance(slot.value, list):
            new = [mapping.get(v, v) if isinstance(v, str) else v for v in slot.value]
            changed += sum(a != b for a, b in zip(new, slot.value))
            slot.owner[slot.key] = new
    return changed


def dangling_references(
    body: dict, schema: Schema, *, root: str = "Study"
) -> Iterator[tuple[Slot, int, str]]:
    """(slot, position in it, id) for every reference naming an id no entity declares.

    The one definition of "dangling": `dangling` counts it, `fix.link`'s reference repair
    and `check_local_ids`' report read it, and `apply_all` compares it around each repair.
    """
    declared = declared_ids(body, schema, root=root)
    for slot in references(body, schema, root=root):
        for position, name in enumerate(ids_of(slot.value)):
            if name not in declared:
                yield slot, position, name


def dangling(body: dict, schema: Schema, *, root: str = "Study") -> Counter[str]:
    """The referenced ids no entity declares, with how many references name each."""
    return Counter(name for _slot, _position, name in dangling_references(body, schema, root=root))


def ids_of(value: Any) -> list[str]:
    """The ids a reference slot holds, scalar or list, ignoring anything that is not one."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []
