"""Checks on the extraction prompt and the proposers behind it.

What was here was the second `satisfy` pass -- its gate, its merge, and the keys it revisited
-- which `Fill` replaced: a slot-level loop can say what "done" is, and an entity-shaped one
could only guess. Those tests went with it. These did not, because they are about the prompt
and the proposers rather than about the pass that was removed.
"""

from __future__ import annotations

import pytest

from pondie import paths
from pondie.extraction import recall
from pondie.extraction.prompt import render
from pondie.schema import reader

GOLD = paths.REPO / "benchmarks" / "gold" / "xevP8UDRAVh9.extraction.json"


@pytest.fixture(scope="module")
def sch():
    return reader.load(render.EXTRACTION_SCHEMA)


def wrapper(value):
    return {
        "extraction_status": "extracted",
        "value": value,
        "value_source": "reported",
        "evidence": {"status": "present"},
    }


NOT_REPORTED = {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}


def test_the_condition_vocabulary_reaches_the_model(sch):
    """`condition_kind` is the slot the vocabulary block was written for, and it lives on
    `Condition`, reachable only through `Task.conditions`. A sweep that skipped nested slots
    documented every enum except that one: 0 of 1,571 filled across 610 papers."""
    said = recall.vocabulary(sch, "Task")
    assert "task_state" in said and "control_state" in said


def test_a_half_emitted_entity_is_a_postcondition_failure():
    """A list holding a bare string is an object the model started and abandoned.

    On 21118656 `analyses` came back as one good analysis followed by 'model_vbm_ptsd_ntc'
    and 'name {' -- the local_id and first key of the whole-brain VBM analysis the paper
    reports as tested and null. Non-emptiness passed the post-condition, so nothing retried,
    and the analysis that decided whether the paper qualified was dropped downstream. The
    check has to be that entries are objects, not that the list has length.
    """
    payload = {
        "analyses": [{"local_id": "an_roi"}, "model_vbm_ptsd_ntc", "name {"],
        "required_entities": [{"local_id": "grp", "kind": "Group"}],
    }
    failures = render.postcondition_failures(payload, "demands")
    assert any("not objects" in f for f in failures), failures


def test_a_well_formed_payload_has_no_such_failure():
    """The guard must not fire on the shape it is meant to allow."""
    payload = {
        "analyses": [{"local_id": "an_roi"}],
        "required_entities": [{"local_id": "grp", "kind": "Group"}],
    }
    assert not [f for f in render.postcondition_failures(payload, "demands") if "not objects" in f]
