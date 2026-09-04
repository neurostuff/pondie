"""Ask a second model what the first one missed.

`satisfy` fills the slots the demands stage asked for. What it cannot do is notice an entity
nobody asked about, or a link between two entities it filled separately -- and those are most
of what a built record lacks: a scanner named in the methods with no `Acquisition.device`
pointing at it, four ROIs an analysis searched with `regions` empty.

The sweep asks per class, with a template projected from the schema, and gives the model the
entities it may point at. Two properties of that are load-bearing:

  * **Targets are swept before the classes that reference them.** Analyses came first, being
    the central object, and that is exactly wrong for linking: four correctly named regions
    were refused because no Region existed yet, and the regions sweep ran afterwards.
  * **Candidates are listed per reference slot**, read from the schema. A fixed list of five
    classes offered the same five however the sweep was aimed, so sweeping groups -- whose
    `diagnostic_instrument` targets an Assessment -- offered no assessments, and the model
    named the instrument in prose instead. A second copy of it was then created.

`Proposer` is a protocol and the pass takes `None`: the weights are optional, and the
deterministic half of a repair is most of its value.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from pondie.schema.reader import Schema


class Proposer(Protocol):
    """Returns entities of `class_name` the paper describes, as flat dicts."""

    def propose(self, class_name: str, template: Mapping[str, Any], premise: str,
                instruction: str) -> Sequence[Mapping[str, Any]]: ...


def sweep_order(sch: Schema, keys: Sequence[str]) -> list[str]:
    """`keys` reordered so a class comes after everything it points at.

    Read from the schema's own slot kinds, so a reference added later orders itself. A cycle
    -- `Analysis.mirror_of` and the other two self-referential slots -- means neither side can
    go first, and the caller's order stands for it.
    """
    targets = {
        key: {CONTAINER.get(slot.range) for _n, slot, kind in sch.iter_slots(CLASS_OF[key])
              if kind == "reference" and isinstance(slot.range, str)}
        for key in keys if key in CLASS_OF
    }
    out: list[str] = []
    seen: set[str] = set()

    def visit(key: str, path: frozenset[str]) -> None:
        if key in seen or key in path:
            return
        for target in sorted(t for t in (targets.get(key) or set()) if t in keys):
            visit(target, path | {key})
        if key not in seen:
            seen.add(key)
            out.append(key)

    for key in keys:
        visit(key, frozenset())
    return out


def candidates(sch: Schema, record: Mapping[str, Any], class_name: str,
               label_of) -> str:
    """The entities this class may point at, one list per reference slot it declares."""
    lines = []
    for name, slot, kind in sorted(sch.iter_slots(class_name)):
        if kind != "reference" or not isinstance(slot.range, str):
            continue
        key = CONTAINER.get(slot.range)
        listed = "; ".join(
            label for entity in (record.get(key) or []) if isinstance(entity, Mapping)
            for label in [label_of(entity, slot.range)] if label
        )
        if listed:
            lines.append(f"- `{name}` may name any of these {slot.range}: {listed}")
    if not lines:
        return ""
    return ("## Entities this record already holds\n\n" + "\n".join(lines) +
            "\n\nName any of these that applies, under the slot that lists it. Name one the "
            "paper describes even if it is not listed; it will be created.\n\n")


#: Class -> the top-level record list its instances live in, and back. Both directions are
#: needed: the schema names classes and a record is keyed by container.
CONTAINER: dict[str, str] = {
    "Analysis": "analyses", "Group": "groups", "Task": "tasks", "Region": "regions",
    "Measure": "measures", "Acquisition": "acquisitions", "Device": "devices",
    "Preprocessing": "preprocessings", "ModelEstimation": "model_estimations",
    "InferenceSettings": "inference_settings", "Table": "tables",
    "Assessment": "assessments",
}
CLASS_OF: dict[str, str] = {container: cls for cls, container in CONTAINER.items()}
