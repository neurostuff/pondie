"""The `normalized` sidecar: what a corpus-level normalizer writes, and the one-off migration.

The closed fields are in test_normalization.py beside their lexicons. These are the two
writers that do not decide one value at a time: `medical_condition.annotate`, whose rows come
from a corpus run, and the migration, whose rows come from records and tables written before
the sidecar existed. Both are checked against study_schema's own extraction model.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from study_schema.models import extraction

from pondie.normalization.medical_condition import Link, annotate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import migrate_normalized_sidecar as M  # noqa: E402


def _v(value):
    return {"extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": {"status": "not_found"}}


def _valid(wrapper: dict) -> None:
    """Every entry is a `Normalization` the PR-branch model accepts."""
    for entry in wrapper["normalized"]:
        extraction.Normalization.model_validate(entry)


def test_a_comorbid_item_gets_an_entry_per_head_and_vocabulary():
    group = {"local_id": "grp_p", "medical_condition": _v(
        ["Alzheimer's disease", "depression and anxiety", "healthy older adults"])}
    links = [
        Link(study="S", group="grp_p", text="Alzheimer's disease", item=0,
             head="Alzheimer's disease", mondo="MONDO:0004975", label="Alzheimer disease",
             method="lexical", score=1.0, onvoc="Alzheimer's Disease", onvoc_id="ONVOC:1",
             onvoc_via="crosswalk"),
        Link(study="S", group="grp_p", text="depression and anxiety", item=1,
             head="depression", mondo="MONDO:0002050", label="depressive disorder",
             method="retrieval", score=0.97, onvoc="Mood Disorder", onvoc_id="ONVOC:2",
             onvoc_via="ancestor"),
        Link(study="S", group="grp_p", text="depression and anxiety", item=1,
             head="anxiety", method="review", score=0.9),
        Link(study="S", group="grp_p", text="healthy older adults", item=2,
             sentinel="NO_CONDITION", scope="assertion"),
        Link(study="OTHER", group="grp_p", text="x", item=0, mondo="MONDO:1", method="lexical"),
    ]
    record = {"groups": [group]}
    tally = annotate(record, links, "S")
    entries = group["medical_condition"]["normalized"]

    assert [(e["item"], e["vocabulary"], e["status"], e.get("head")) for e in entries] == [
        (0, "MONDO", "mapped", None),
        (0, "ONVOC", "mapped", None),
        (1, "MONDO", "mapped", "depression"),
        (1, "ONVOC", "mapped", "depression"),
        (1, "MONDO", "unmapped", "anxiety"),
        (2, "MONDO", "absent", None),
    ]
    assert entries[3]["rollup"] is True, "ONVOC through an ancestor is broader than the paper"
    assert "rollup" not in entries[1]
    assert entries[4]["confidence"] == 0.9 and entries[4]["method"] == "review"
    assert group["medical_condition"]["value"][1] == "depression and anxiety"
    assert tally == {"set": 6, "unmapped": 1}
    _valid(group["medical_condition"])

    annotate(record, [], "S")
    assert "normalized" not in group["medical_condition"], "a group with no row is cleared"


def test_the_migration_moves_siblings_and_attaches_table_rows():
    group = {
        "local_id": "grp_ad",
        "age_mean": _v(71.0),
        "age_unit": _v("years"),
        "age_unit_normalized": _v("years"),
        "medical_condition": _v(["AD", "MCI"]),
        "sex_distribution": [{"category": _v("men"), "category_normalized": _v("MALE")},
                             {"category": _v("?"), "category_normalized": _v("UNKNOWN")}],
    }
    nan = float("nan")
    medical = {"S": [
        {"group": "grp_ad", "index": 0.0, "route": "lexical", "score": 1.0,
         "mondo": "MONDO:0004975", "label": "Alzheimer disease", "onvoc": "Dementia",
         "onvoc_id": "ONVOC:9", "query_via": "rollup"},
        {"group": "grp_ad", "index": 1.0, "route": "review", "score": 0.88, "mondo": nan,
         "label": nan, "onvoc": nan, "onvoc_id": nan, "query_via": "none"},
        {"group": "grp_ad", "index": nan, "route": "lexical", "score": 1.0, "mondo": "M:1"},
    ]}
    task = {"local_id": "tsk_odd", "name": _v("oddball")}
    tasks = {"S": [{"task": "tsk_odd", "canonical": "hedtsk_oddball",
                    "canonical_name": "Oddball Task", "decided_by": "citation",
                    "qualifiers": [{"text": "visual", "axis": "stimulus_modality"}]}]}
    body = {"groups": [group], "tasks": [task]}
    tally: Counter = Counter()

    M.migrate(body, "S", medical, tasks, "astra_norm.groups_medical snap",
              "astra_norm.tasks snap", tally)

    assert "age_unit_normalized" not in group
    [age] = group["age_unit"]["normalized"]
    assert (age["vocabulary"], age["value"]) == ("AgeUnit", "years")
    assert age["mapping_tool"] == "pondie.normalization.age_unit pre-sidecar"
    male, unknown = (c["category"]["normalized"][0] for c in group["sex_distribution"])
    assert all("category_normalized" not in c for c in group["sex_distribution"])
    assert (male["status"], male["value"]) == ("mapped", "MALE")
    assert unknown["status"] == "unmapped" and "value" not in unknown

    medical_entries = group["medical_condition"]["normalized"]
    assert [(e["item"], e["vocabulary"], e["status"]) for e in medical_entries] == [
        (0, "MONDO", "mapped"), (0, "ONVOC", "mapped"), (1, "MONDO", "unmapped")]
    assert medical_entries[1]["rollup"] is True
    assert tally["medical rows skipped, no index"] == 1

    [canonical] = task["name"]["normalized"]
    assert canonical["value"] == "hedtsk_oddball"
    assert canonical["also_read"] == ["references"], "the citation route read the references"
    assert canonical["qualifiers"] == ["visual"], "the table's {text, axis} keeps its words"
    for wrapper in (group["age_unit"], group["medical_condition"], task["name"],
                    *(c["category"] for c in group["sex_distribution"])):
        _valid(wrapper)


def test_a_migrated_entry_is_replaced_by_the_next_normalizer_run():
    """`pre-sidecar` stands in for the commit nobody recorded. The module name is the same,
    so the first `apply_normalized` over a migrated record replaces it."""
    from pondie import normalization

    group = {"age_mean": _v(30.0), "age_unit": _v("years"), "age_unit_normalized": _v("years")}
    body = {"groups": [group]}
    M.move_siblings(body, Counter())
    normalization.apply_normalized(body)

    [entry] = group["age_unit"]["normalized"]
    assert not entry["mapping_tool"].endswith("pre-sidecar")
    assert entry["method"] == "lexical"


def test_a_task_named_by_the_lexicon_read_nothing_else():
    task = {"local_id": "tsk_s", "name": _v("Stroop task")}
    tasks = [{"task": "tsk_s", "canonical": "hedtsk_stroop", "decided_by": "name"}]
    M.attach_tasks({"tasks": [task]}, tasks, "astra_norm.tasks snap", Counter())

    [canonical] = task["name"]["normalized"]
    assert "also_read" not in canonical
    _valid(task["name"])
