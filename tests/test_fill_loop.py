"""What has to hold for a slot-level fill loop to terminate on a fact rather than a hope.

The loop this replaces asked `satisfy` again and judged the answer by whether entities came
back. It could not say what "done" was, so it stopped on proxies -- nothing landed, the thin
set no smaller -- and measured on five papers it finished with three of them still thin.

The exit here is `unsettled` being empty, so these tests pin what that predicate counts and
that answering actually shrinks it. If `unsettled` is wrong the loop either never stops or
stops with work left, and neither is visible from a run that merely completes.
"""

from __future__ import annotations

import pytest

from pondie import schema
from pondie.extraction.prompt import fill
from pondie.schema import reader


@pytest.fixture(scope="module")
def sch():
    return reader.load(schema.EXTRACTION)


def group(**slots):
    return {"groups": [{"local_id": "g1", **slots}]}


def ids(rows):
    return [r["id"] for r in rows]


def test_a_value_and_a_stated_reason_both_settle_a_slot(sch):
    """The two ways of being done. A bare `not_reported` is an answer -- the status says the
    attribute was examined and nothing found -- and a loop that re-asked it would invite the
    model to overwrite a finding."""
    doc = group(
        name={"extraction_status": "extracted", "value": "MDD"},
        age_median={"extraction_status": "not_reported"},
    )
    open_ids = ids(fill.unsettled(doc, sch))
    assert "groups[g1].name" not in open_ids
    assert "groups[g1].age_median" not in open_ids


def test_undetermined_is_the_one_reason_that_leaves_a_slot_open(sch):
    """It reports on the pass, not the paper: the model could not work the slot out. That is
    the claim a further round exists to revisit, and the reason the cap is needed at all."""
    doc = group(
        age_median={"extraction_status": "not_reported", "unreported_reason": "undetermined"}
    )
    assert "groups[g1].age_median" in ids(fill.unsettled(doc, sch))


def test_a_slot_with_no_wrapper_at_all_is_open(sch):
    """`thin` counts these and nothing else. They are open for the same reason a
    `undetermined` one is: nothing has been established about them either way."""
    doc = group(name={"extraction_status": "extracted", "value": "MDD"})
    assert "groups[g1].species" in ids(fill.unsettled(doc, sch))


def test_answering_shrinks_the_open_set(sch):
    """The property the loop's termination rests on. If answering did not shrink it, the
    cap would be the only thing stopping the loop and every run would pay it in full."""
    doc = group(name={"extraction_status": "extracted", "value": "MDD"})
    before = ids(fill.unsettled(doc, sch))
    filled, reasoned, dropped = fill.apply_fill(
        doc,
        {
            "groups[g1].species": {"value": "human"},
            "groups[g1].age_median": {"unreported_reason": fill.PLAIN},
        },
        before,
    )
    after = ids(fill.unsettled(doc, sch))
    assert (filled, reasoned, dropped) == (1, 1, 0)
    assert len(after) == len(before) - 2
    assert doc["groups"][0]["species"]["value"] == "human"
    # PLAIN is the prompt's word for the ordinary case, not a schema value: it lands as the
    # bare status, which already says the attribute was examined and nothing was found.
    assert "unreported_reason" not in doc["groups"][0]["age_median"]
    assert doc["groups"][0]["age_median"]["extraction_status"] == "not_reported"


def test_an_answer_under_an_id_that_was_not_asked_is_discarded(sch):
    """A model answering under a path of its own invention is describing a record that does
    not exist. Writing it would put a field on an entity the schema never gave one."""
    doc = group(name={"extraction_status": "extracted", "value": "MDD"})
    open_ids = ids(fill.unsettled(doc, sch))
    filled, reasoned, dropped = fill.apply_fill(
        doc, {"groups[g1].invented_slot": {"value": "x"}}, open_ids
    )
    assert (filled, reasoned, dropped) == (0, 0, 1)
    assert "invented_slot" not in doc["groups"][0]


def test_an_answer_that_is_neither_a_value_nor_a_reason_is_discarded(sch):
    """Rule 2 of the prompt: never both, never neither. An empty object settles nothing and
    must not be written as though it had, or the slot leaves the open set unanswered."""
    doc = group(name={"extraction_status": "extracted", "value": "MDD"})
    open_ids = ids(fill.unsettled(doc, sch))
    filled, reasoned, dropped = fill.apply_fill(doc, {"groups[g1].species": {}}, open_ids)
    assert (filled, reasoned, dropped) == (0, 0, 1)
    assert "groups[g1].species" in ids(fill.unsettled(doc, sch))


def test_the_listing_names_the_value_type_not_the_wrapper(sch):
    """A slot's range is `ExtractedSpecies`; what the model must return is a `Species`.
    Naming the wrapper asks for the wrong shape -- the failure `render_schema` calls the
    most easily confused thing in this schema."""
    doc = group(name={"extraction_status": "extracted", "value": "MDD"})
    rows = {r["id"]: r for r in fill.unsettled(doc, sch)}
    species = rows["groups[g1].species"]
    assert species["range"] == "Species"
    assert "human" in species["vocabulary"]
    assert "human" in fill.block([species])


def test_nested_entities_are_reached(sch):
    """A Condition hangs off Task and carries its own slots. Walking only the top-level
    lists would leave every one of them permanently open, and the loop would never finish."""
    doc = {
        "tasks": [
            {
                "local_id": "t1",
                "name": {"extraction_status": "extracted", "value": "n-back"},
                "conditions": [{"local_id": "c1"}],
            }
        ]
    }
    assert any(r["id"].startswith("tasks[t1].conditions[c1].") for r in fill.unsettled(doc, sch))


def test_a_singular_nested_object_takes_no_subscript(sch):
    """`Analysis.effect` is one object, not a list of one.

    Written `effect[0]` the path went out to the model and came back to nothing: `_resolve`
    sees the bracket, demands a list, finds a dict and gives up. Every `Cell` slot hangs off
    `effect`, so on five papers this discarded every answer about a cell's `direction` while
    the residue read as "the model declined to answer" -- the reverse of what happened.
    """
    doc = {"analyses": [{"local_id": "an1", "effect": {"cells": [{"term": "t1"}]}}]}
    rows = ids(fill.unsettled(doc, sch))
    direction = "analyses[an1].effect.cells[0].direction"
    assert direction in rows, rows
    assert not any("effect[0]" in r for r in rows), "a singular nested slot took a subscript"

    filled, reasoned, dropped = fill.apply_fill(doc, {direction: {"value": "positive"}}, rows)
    assert (filled, reasoned, dropped) == (1, 0, 0)
    cell = doc["analyses"][0]["effect"]["cells"][0]
    assert cell["direction"]["value"] == "positive"


def test_every_offered_slot_can_be_written_back(sch):
    """The invariant the bug above broke, over a record shaped like a real one. `unsettled`
    and `_resolve` have to agree about addressing or the loop asks for what it cannot keep."""
    doc = {
        "analyses": [{"local_id": "an1", "effect": {"cells": [{"term": "t1"}, {"term": "t2"}]}}],
        "groups": [{"local_id": "g1"}],
        "tasks": [{"local_id": "t1", "conditions": [{"local_id": "c1"}]}],
    }
    for row in fill.unsettled(doc, sch):
        target, name = fill._resolve(doc, row["id"])
        assert target is not None, f"offered but unresolvable: {row['id']}"
        assert name, row["id"]


def test_a_bare_unwrapped_value_is_an_answer_not_an_empty_slot(sch):
    """The extraction passes emit some slots unwrapped -- `Cell.direction` arrives as
    `"positive"`, not an ExtractedValue -- and `repairs.wrappers` reshapes them at build
    time, which is after this stage runs.

    Reading the bare form as absence made the loop destructive rather than merely wasteful:
    on pMZeVGA2rQQi it offered all twelve cell directions as open and overwrote them with
    `ambiguous`, and six of those the extraction had got right. Shape is the repair pass's
    business; open means holding nothing at all.
    """
    doc = {
        "analyses": [
            {"local_id": "an1", "effect": {"cells": [{"term": "t1", "direction": "positive"}]}}
        ]
    }
    open_ids = ids(fill.unsettled(doc, sch))
    assert "analyses[an1].effect.cells[0].direction" not in open_ids, open_ids


def test_a_held_value_is_never_overwritten_even_if_it_is_offered(sch):
    """The second guard. `unsettled` should never offer an answered slot, but it did once,
    and the writer is the last place to stop a correct value being replaced by a reason."""
    doc = {
        "analyses": [
            {"local_id": "an1", "effect": {"cells": [{"term": "t1", "direction": "positive"}]}}
        ]
    }
    path = "analyses[an1].effect.cells[0].direction"
    filled, reasoned, dropped = fill.apply_fill(
        doc, {path: {"unreported_reason": "ambiguous"}}, [path]
    )
    assert (filled, reasoned, dropped) == (0, 0, 1)
    assert doc["analyses"][0]["effect"]["cells"][0]["direction"] == "positive"


def test_an_undetermined_slot_is_still_revisable(sch):
    """The guard must not close the one door the loop needs: `undetermined` is the model
    saying it could not tell, and a later round exists to replace exactly that."""
    doc = group(
        age_median={"extraction_status": "not_reported", "unreported_reason": "undetermined"}
    )
    path = "groups[g1].age_median"
    filled, reasoned, dropped = fill.apply_fill(doc, {path: {"value": 41}}, [path])
    assert (filled, reasoned, dropped) == (1, 0, 0)
    assert doc["groups"][0]["age_median"]["value"] == 41


def test_a_reason_outside_the_vocabulary_leaves_the_slot_untouched(sch):
    """Refusing an answer and settling the slot anyway is worse than either alone.

    The first cut wrote the bare `not_reported` before checking the reason, so a rejected
    answer still closed the slot -- and closed it as plain silence, a claim the model had not
    made. Nothing revisits a settled slot, so the loop would have recorded silence on the
    strength of an answer it had just thrown away.
    """
    doc = group(name={"extraction_status": "extracted", "value": "MDD"})
    open_ids = ids(fill.unsettled(doc, sch))
    filled, reasoned, dropped = fill.apply_fill(
        doc, {"groups[g1].age_mean": {"unreported_reason": "silent"}}, open_ids
    )
    assert (filled, reasoned, dropped) == (0, 0, 1)
    assert "age_mean" not in doc["groups"][0]
    assert "groups[g1].age_mean" in ids(fill.unsettled(doc, sch))


def test_plain_silence_is_written_as_the_bare_status(sch):
    """There is no schema token for the ordinary case, because `not_reported` already says
    the attribute was examined and the source carries no value. The prompt still needs a name
    for it -- a model offered only the unusual reasons reaches for the nearest one -- so it
    answers `PLAIN` and that lands as the status alone."""
    doc = group(name={"extraction_status": "extracted", "value": "MDD"})
    open_ids = ids(fill.unsettled(doc, sch))
    filled, reasoned, dropped = fill.apply_fill(
        doc, {"groups[g1].age_mean": {"unreported_reason": fill.PLAIN}}, open_ids
    )
    assert (filled, reasoned, dropped) == (0, 1, 0)
    held = doc["groups"][0]["age_mean"]
    assert held == {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}
    assert "groups[g1].age_mean" not in ids(fill.unsettled(doc, sch))
    assert fill.PLAIN not in fill.VOCABULARY
