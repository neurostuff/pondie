"""What has to hold for a second extraction pass to be worth making.

The pass costs a model call on ~80% of papers, so the gate that fires it and the merge that
lands its answers are both load-bearing, and both shipped wrong. The gate has now been wrong
twice in opposite directions -- once counting a category that never occurs, once counting a
correct silence as a hole -- and neither was catchable, because no test named `thin`. The
one test that drove the stage end to end returned an empty payload, on which every possible
implementation returns the same answer.

Each test here pins one of those failures with an input that distinguishes it.
"""

from __future__ import annotations

import json

import pytest

from pondie import paths
from pondie.extraction import recall, recall_server
from pondie.extraction.prompt import render
from pondie.schema import reader

GOLD = paths.REPO / "benchmarks" / "gold" / "xevP8UDRAVh9.extraction.json"


@pytest.fixture(scope="module")
def sch():
    return reader.load(render.EXTRACTION_SCHEMA)


@pytest.fixture(scope="module")
def keys(sch):
    return render.priority_keys(sch)


def wrapper(value):
    return {"extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": {"status": "present"}}


NOT_REPORTED = {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}


def test_the_priority_keys_come_from_the_schema(sch, keys):
    """A written-out ("tasks", "groups") is a second copy of what `containers()` knows."""
    assert keys == [sch.containers()[name] for name in render.PRIORITY]


def test_a_gold_record_is_not_thin(keys):
    """A sparse paper extracted perfectly is not a paper extracted lazily.

    `as_field` drops `value` from any wrapper that is not `extracted`, so counting "has a
    value" scores a correct `not_reported` -- the model looked, the paper was silent -- the
    same as never looking. Under that counting this hand-curated record read 42% empty and
    would have bought itself a second opinion it cannot benefit from.
    """
    gold = json.loads(GOLD.read_text("utf-8"))
    assert not render.thin(gold, keys)


def test_a_record_stripped_of_its_answers_is_thin(keys):
    """The other direction, so the test above cannot pass by `thin` always being False."""
    gold = json.loads(GOLD.read_text("utf-8"))
    stripped = {key: [{"local_id": e.get("local_id")} for e in (gold.get(key) or [])]
                for key in keys}
    assert render.thin(stripped, keys)


def test_an_empty_payload_is_not_thin(keys):
    """Nothing to be thin about. This is the input the old stage test used, which is why it
    could not tell the two broken gates apart."""
    assert not render.thin({}, keys)


def test_a_task_with_no_conditions_counts_as_unanswered(sch, keys):
    """The pass exists for 16038771, where not one condition was described.

    A nested list that is absent or empty iterates nothing, so a Task carrying no conditions
    contributed zero holes and the paper the feature was written for did not trigger it.
    """
    with_none = {"tasks": [{"local_id": "tsk", "name": wrapper("picture viewing")}]}
    filled, empty = render._expected(sch, "Task", with_none["tasks"][0])
    assert empty, "a task with no conditions must count its missing conditions"


def test_fill_empty_adds_a_condition_the_first_pass_omitted(keys):
    """Matching nested items by `local_id` and skipping the unmatched drops exactly the
    entity the pass was sent to find."""
    payload = {"tasks": [{"local_id": "tsk", "conditions": [
        {"local_id": "cnd_neutral", "name": wrapper("Neutral")}]}]}
    second = {"tasks": [{"local_id": "tsk", "conditions": [
        {"local_id": "cnd_erotic", "name": wrapper("Erotic"),
         "condition_kind": wrapper("task_state")}]}]}
    landed, dropped = render.fill_empty(payload, second, ["tasks"])
    names = [render.fields.read(c.get("name"))
             for c in payload["tasks"][0]["conditions"]]
    assert "Erotic" in names, f"the new condition was dropped; got {names}"
    assert landed and not dropped


def test_merge_does_not_move_one_distribution_entry_onto_another():
    """`CategoryDistribution` has no `local_id`, so keying it by one maps every entry to "".

    Only the last target entry survives that mapping and every source entry looks it up, so
    the male count lands on the female entry wherever female's is empty. That is a wrong
    value written into a record, which is worse than the omission this pass exists to fix.
    """
    target = {"local_id": "grp", "sex_distribution": [
        {"category": wrapper("male"), "count": wrapper(8)},
        {"category": wrapper("female")}]}
    source = {"local_id": "grp", "sex_distribution": [
        {"category": wrapper("male"), "count": wrapper(8)},
        {"category": wrapper("female"), "count": wrapper(3)}]}
    render._merge_empty(target, source)
    female = [e for e in target["sex_distribution"]
              if render.fields.read(e.get("category")) == "female"]
    assert len(female) == 1
    got = render.fields.read(female[0].get("count"))
    assert got in (None, 3), f"female count became {got}, the male entry's value"


def test_a_not_reported_claim_is_not_overwritten():
    """`not_reported` is a positive claim that the paper is silent, not an empty slot.

    It carries no `value` key, so a merge testing the value reads it as absent and lets a
    second opinion replace it -- the one direction a pass that promises "add and never
    overwrite" must not move.
    """
    target = {"local_id": "grp", "education_summary": dict(NOT_REPORTED)}
    render._merge_empty(target, {"local_id": "grp",
                                 "education_summary": wrapper("12 years")})
    assert target["education_summary"]["extraction_status"] == "not_reported"


def test_entities_under_unmatched_ids_are_reported_not_dropped_silently():
    """A pass answering under ids it invented has every answer discarded, and "filled 0" is
    also what a paper needing nothing reports. The two must be distinguishable."""
    payload = {"groups": [{"local_id": "grp_controls", "name": wrapper("controls")}]}
    second = {"groups": [{"local_id": "grp_hc", "age_mean": wrapper(24.1)}]}
    landed, dropped = render.fill_empty(payload, second, ["groups"])
    assert (landed, dropped) == (0, 1)


def test_the_priority_prompt_offers_only_the_keys_it_renders(sch):
    """Rule 2 names the top-level keys to emit. Naming eleven while describing four classes
    asks for lists the pass never rendered -- the "model invents the shape" failure -- and
    everything outside tasks/groups is discarded by `fill_empty` regardless."""
    prompt = render.build_prompt("A paper.", "priority", True, "")
    names, _study_keep = render.mode_classes(sch, "priority")
    offered = [k for k in sch.containers().values() if f" {k}," in prompt.system
               or prompt.system.count(f"{k},") or prompt.system.endswith(k)]
    for key in render.priority_keys(sch):
        assert key in prompt.system
    for key in sch.containers().values():
        if key in render.priority_keys(sch) or key == "tables":
            continue
        assert f"{key}" not in prompt.system.split("These keys go at the TOP LEVEL")[1][:400], \
            f"{key} is offered as a payload key but its class is not rendered"


def test_the_condition_vocabulary_reaches_the_model(sch):
    """`condition_kind` is the slot the vocabulary block was written for, and it lives on
    `Condition`, reachable only through `Task.conditions`. A sweep that skipped nested slots
    documented every enum except that one: 0 of 1,571 filled across 610 papers."""
    said = recall.vocabulary(sch, "Task")
    assert "task_state" in said and "control_state" in said


def test_both_proposers_share_one_propose():
    """The two differed by one type annotation, so wiring the vocabulary in meant making the
    same edit twice by hand -- and a fix applied to one copy is invisible until a run
    disagrees with itself."""
    assert recall.NuExtract.propose is recall_server.NuExtractServer.propose
