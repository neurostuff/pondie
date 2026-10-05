"""Entities no analysis reaches do not go in the record.

The record is analysis-centred: an Analysis is the unit a meta-analysis pools, and every
other entity is there to describe one. An entity nothing reaches describes nothing.

WHERE IT RUNS. Twice: last at the merge (`fix.sequence`, after `mirrored`, which ADDS
analyses) and again after the repair sweep (`repair.stage`), which creates entities of its
own. Both, because `--stages` can omit repair and then the merge is the only pass that
looks -- and because running it twice is idempotent, while running it once leaves the
guarantee depending on which stages happened to run.

WHY NOT ONLY AT DEMANDS. `render.unreachable_entity_demands` holds the demands
pass to the same rule, and it is not enough: audited over 126 freshly extracted papers,
**66 of 69 orphans appear in no payload at all** -- not `tables`, `demands`, `satisfy` or
`fill` -- and **0 of the 69 were ever referenced in any payload**. They are minted by the
proposer in this stage, which creates an entity whenever the model proposes one and never
asks whether anything will point at it. Three came from `satisfy`. None came from demands,
where the post-condition already looks.

WHAT IT CATCHES, measured on that run: 50 Regions, 11 Assessments, 3 Groups, 2 Tasks. The
Regions are the clearest case -- `superior medial frontal gyrus bilaterally`, `right
inferior occipital gyrus`, `dorsal anterior cingulate cortex` -- which are where a result
was FOUND, not a region the study delimited and used. `Region` already says it is "a brain
region the study delimited and used: a connectivity seed or target, the search space an
analysis ran over", so a reported peak was never in scope and has no slot to sit in.

NOT DELETED, just not part of the record. Every drop is reported, and the report is written
beside the record, so what was removed and why stays auditable.

Tables are exempt. Their ids come from the parse rather than from a model (`ids.DERIVED`),
and a parsed row group exists whether or not an analysis cites it -- which is the very fact
`render.unconsumed_listing` is there to complain about.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, MutableMapping

from pondie.formats.values import read

#: Never swept: analyses are what reachability is measured FROM, and tables are the parse's.
SEEDS = "analyses"
EXEMPT = frozenset({"analyses", "tables"})


def _ids_in(node: Any, out: set[str]) -> None:
    """Every string a node holds, which is where a reference hides.

    Walked structurally rather than by slot name, for the reason
    `render.unreachable_entity_demands` gives: a reference can sit in a cell, a level or a
    list this function has never heard of, and enumerating slots would go stale.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key == "local_id":
                continue
            inner = read(value) if isinstance(value, Mapping) and "value" in value else value
            for item in inner if isinstance(inner, list) else [inner]:
                if isinstance(item, str):
                    out.add(item)
            _ids_in(value, out)
    elif isinstance(node, list):
        for item in node:
            _ids_in(item, out)


def reachable(record: Mapping[str, Any]) -> set[str]:
    """Every local_id an analysis reaches, following references either way, transitively.

    UNDIRECTED, because a declaration's edges point whichever way the schema stores them: a
    ModelTerm names its `model`, so an analysis naming that model never reaches the term
    walking forward. TRANSITIVE, because an analysis cites a term whose level names a
    timepoint whose arm names the group that received it, and all four are entailed.
    """
    nodes: dict[str, Any] = {}

    def collect(node: Any) -> None:
        if isinstance(node, Mapping):
            if isinstance(node.get("local_id"), str):
                nodes[node["local_id"]] = node
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(record)
    adjacent: dict[str, set[str]] = defaultdict(set)
    for local_id, node in nodes.items():
        found: set[str] = set()
        _ids_in(node, found)
        for other in (found & set(nodes)) - {local_id}:
            adjacent[local_id].add(other)
            adjacent[other].add(local_id)

    seed: set[str] = set()
    for analysis in record.get(SEEDS) or []:
        if isinstance(analysis, Mapping):
            _ids_in(analysis, seed)

    reached: set[str] = set()
    frontier = seed & set(nodes)
    while frontier:
        reached |= frontier
        nxt: set[str] = set()
        for local_id in frontier:
            nxt |= adjacent[local_id] - reached
        frontier = nxt
    return reached


def drop_unreachable(record: MutableMapping[str, Any]) -> list[str]:
    """Remove entities no analysis reaches. Returns one line per removal.

    A record with NO analyses is left alone. Reachability is measured from the analyses, so
    with none every entity is trivially unreachable and emptying the record would destroy
    an extraction rather than tidy one -- the fault there is the missing analyses, which
    `postcondition_failures` already refuses.
    """
    if not record.get(SEEDS):
        return []

    keep = reachable(record)
    notes: list[str] = []
    for container, entities in list(record.items()):
        if container in EXEMPT or not isinstance(entities, list):
            continue
        kept = []
        for entity in entities:
            local_id = entity.get("local_id") if isinstance(entity, Mapping) else None
            if not isinstance(local_id, str) or local_id in keep:
                kept.append(entity)
                continue
            label = read(entity.get("name")) if isinstance(entity, Mapping) else None
            notes.append(
                f"{container}/{local_id} dropped: no analysis reaches it"
                + (f" ({str(label)[:60]!r})" if isinstance(label, str) and label else "")
            )
        if len(kept) != len(entities):
            record[container] = kept
    return notes
