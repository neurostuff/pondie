"""Every edge the schema declares is an edge reachability follows.

An entity is dropped from the record when nothing reaches it, so a reference the walk
cannot see is an entity deleted for a link it actually had. The risk is real: of the 39
reference slots the extraction schema declares, many are NESTED -- `FactorLevel.groups`,
`Cell.term`, `AnalysisGroup.group`, `MRI.device`, `ConnectivityDetails.seed_regions` --
and an implementation that enumerated slot names would miss them and go stale the first
time one was added.

So the enumeration is driven from the schema itself rather than written down here.
"""

from __future__ import annotations

import pytest

import pondie.schema as schema_pkg
from pondie.extraction.repair.reachable import _ids_in, drop_unreachable, reachable
from pondie.schema import reader


def reference_slots() -> list[tuple[str, str, str]]:
    sch = reader.load(schema_pkg.EXTRACTION)
    classes = sch.classes if isinstance(sch.classes, dict) else {}
    return [
        (cls, name, slot.range)
        for cls in sorted(classes)
        for name, slot, kind in sch.iter_slots(cls)
        if kind == "reference" and isinstance(slot.range, str)
    ]


@pytest.mark.parametrize("cls,slot,target", reference_slots())
def test_the_walk_follows_every_reference_slot(cls: str, slot: str, target: str) -> None:
    """Placed under its own slot name, at depth, a reference is found.

    Both shapes a record uses: the bare value and the `ExtractedValue` wrapper.
    """
    for held in ("tgt_id", ["tgt_id"], {"value": ["tgt_id"], "extraction_status": "extracted"}):
        found: set[str] = set()
        _ids_in({"analyses": [{"nested": {"deeper": [{slot: held}]}}]}, found)
        assert "tgt_id" in found, f"{cls}.{slot} -> {target} was not followed ({held!r})"


def test_a_reference_in_a_slot_no_schema_declares_is_still_followed() -> None:
    """The walk is structural, not a slot list, so a slot added tomorrow needs no change
    here -- and that is the property this test pins, since the parametrised test above can
    only cover what the schema says today."""
    found: set[str] = set()
    _ids_in({"some_slot_invented_later": ["g1"]}, found)
    assert "g1" in found


def test_reachability_crosses_a_nested_holder() -> None:
    """`Cell.term` and `FactorLevel.groups` sit inside structures, not on the entity. An
    analysis reaching a term through its cells must reach what that term names."""
    record = {
        "analyses": [{"local_id": "a1", "effect": {"cells": [{"term": "trm_time"}]}}],
        "model_estimations": [
            {"local_id": "m1", "terms": [
                {"local_id": "trm_time", "levels": [{"timepoints": ["tp_post"]}]}]}
        ],
        # Timepoints are nested under `design`, which is where the walk has to find them
        "design": {"timepoints": [{"local_id": "tp_post"}]},
        "groups": [{"local_id": "g_free"}],
    }
    keep = reachable(record)

    assert {"trm_time", "tp_post"} <= keep, "the chain through a cell and a level holds"
    assert "g_free" not in keep
    assert [n for n in drop_unreachable(record) if "g_free" in n]
