"""What the selection must not get wrong.

`pondie select` is one of the three pipelines the package advertises and had no test at all.
It raised `TypeError` on every real record for as long as `coordinate_space.resolve` has
returned a `Decision` rather than a dict, and nothing noticed.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter

import pytest

from pondie import paths
from pondie.formats import parse_keys
from pondie.query.engine import Result, Selection, select

CANDIDATES = str(paths.REPO / "benchmarks" / "candidate" / "*.extraction.json")


@pytest.fixture(scope="module")
def result() -> Result:
    return select(Selection(records=(CANDIDATES,)))


def test_selecting_over_real_records_does_not_raise(result):
    """The shipped candidate set is 16 real records; the verb ran on none of them."""
    assert result.seen_papers, "the glob found records"
    assert isinstance(result.rows, list)


def test_the_funnel_says_why_each_analysis_was_dropped(result):
    """A count with no reason is not a funnel."""
    assert result.lost, "nothing survives this checkout, so everything is accounted for"
    for reason, count in result.lost.items():
        assert count > 0 and reason.strip()
    # A space rejection says how the space was decided, not just what it was: the reason is
    # the difference between "the paper said so" and "we could not tell".
    spaces = [r for r in result.lost if r.startswith("space=")]
    assert spaces and all("(" in r for r in spaces), spaces


def test_a_missing_stage_one_parse_is_not_blamed_on_the_extractor(result):
    """Two different problems: a corpus that was never synced, and a record whose key does
    not join. Reporting both as "no joinable row group" sent the reader to the wrong place.
    """
    assert (
        "no joinable row group" not in result.lost or "no stage-1 parse synced" in result.lost
    )


def test_an_analysis_with_no_sample_size_is_dropped_rather_than_given_one(tmp_path):
    """NiMARE weights by sample size, so an invented one changes the pooled result.

    It used to substitute 30, with no comment, no measurement and no count.

    Asserted on `poolable()`, which is the decision, rather than on `to_studyset()`, which
    is NiMARE's conversion of it. The rule is this package's and runs everywhere; the
    conversion is theirs and would only have been checked where NiMARE is installed.
    """
    rows = [
        {"study": "S1", "points": [[1.0, 2.0, 3.0]], "n": None},
        {"study": "S2", "points": [[4.0, 5.0, 6.0]], "n": 20},
    ]
    outcome = Result(Selection(records=(CANDIDATES,)), rows, Counter(), {"S2"}, {"S1", "S2"})
    studies = outcome.poolable()

    assert outcome.lost["no sample size, so it cannot be weighted"] == 1
    assert [s["id"] for s in studies] == ["S2"], "only the weightable study is pooled"
    assert studies[0]["analyses"][0]["metadata"]["sample_sizes"] == [20]
    assert studies[0]["analyses"][0]["points"] == [
        {"coordinates": [4.0, 5.0, 6.0], "space": "MNI"}
    ]


def test_the_studyset_is_one_analysis_per_study_with_every_coordinate(tmp_path):
    """`to_studyset`'s whole claim: "One analysis per study. `combine_analyses()` pools."

    A study contributing two selected analyses must arrive as ONE analysis carrying both
    sets of coordinates -- pooling them is the point of a coordinate meta-analysis, and two
    analyses from one paper would weight that paper twice. The coordinates themselves must
    survive the round trip intact; they are the data.
    """
    rows = [
        {"study": "S1", "points": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "n": 20},
        {"study": "S1", "points": [[7.0, 8.0, 9.0]], "n": 20},
        {"study": "S2", "points": [[-2.0, -4.0, -6.0]], "n": 31},
        {"study": "S3", "points": [[0.0, 0.0, 0.0]], "n": None},
    ]
    outcome = Result(
        Selection(records=(CANDIDATES,)), rows, Counter(), {"S1", "S2"}, {"S1", "S2", "S3"}
    )
    studyset = outcome.to_studyset()
    analyses = {s.id: s.analyses for s in studyset.studies}

    assert set(analyses) == {"S1", "S2"}, "the unweightable study is absent, not invented"
    assert [len(a) for a in analyses.values()] == [1, 1], "one analysis per study"
    assert sorted(tuple(p.coordinates) for a in analyses["S1"] for p in a.points) == [
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
        (7.0, 8.0, 9.0),
    ], "both analyses' foci pooled"
    assert [tuple(p.coordinates) for p in analyses["S2"][0].points] == [(-2.0, -4.0, -6.0)]
    assert sorted(
        a.metadata["sample_sizes"][0] for group in analyses.values() for a in group
    ) == [20, 31]
    assert outcome.lost["no sample size, so it cannot be weighted"] == 1


def test_building_the_studyset_never_touches_nimares_deprecated_dataset(recwarn):
    """`Dataset` is removed in NiMARE 1.0, so the pipeline must not route through it.

    This went through `Studyset.from_dataset(Dataset(...))` and emitted two FutureWarnings
    per call. A comment saying "native" would rot; the warning is the thing that tells us.
    """
    rows = [{"study": "S1", "points": [[1.0, 2.0, 3.0]], "n": 20}]
    outcome = Result(Selection(records=(CANDIDATES,)), rows, Counter(), {"S1"}, {"S1"})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        outcome.to_studyset()


def test_a_joined_analysis_reaches_the_row_without_a_synced_corpus(tmp_path, monkeypatch):
    """The `Decision` subscript bug, pinned where a clean clone can see it.

    `resolved["space"]` on line 341 raised `TypeError` on every analysis that got that far,
    and the tests above only caught it once a corpus was synced: without one, `stage1` is
    absent, every analysis is dropped at "no stage-1 parse synced", and the offending line
    is never reached. So the suite passed on a clean checkout and failed on a working one --
    exactly backwards. This builds the parse it needs in `tmp_path` instead.
    """
    from pondie import paths
    from pondie.query import engine

    parse = {
        "analyses": [
            {
                "name": "Patients > controls",
                "table_id": "t1",
                "table_number": 1,
                "points": [{"coordinates": [10.0, -20.0, 30.0], "space": "MNI", "values": []}],
            }
        ]
    }
    stage1 = tmp_path / "S1" / "stage1" / "analyses.json"
    stage1.parent.mkdir(parents=True)
    stage1.write_text(json.dumps(parse))
    monkeypatch.setattr(paths, "stage1", lambda study, **_: stage1)

    key = parse_keys.parse_keys(parse["analyses"])[0]
    record = {
        "analyses": [
            {
                "local_id": "a1",
                "name": {"extraction_status": "extracted", "value": "Patients > controls"},
                "source_table_analysis": {"extraction_status": "extracted", "value": key},
                "coordinate_space": {"extraction_status": "extracted", "value": "MNI"},
                "spatial_scope": {"extraction_status": "extracted", "value": "whole_brain"},
            }
        ]
    }
    written = tmp_path / "S1.extraction.json"
    written.write_text(json.dumps(record))

    outcome = engine.select(Selection(records=(str(written),)))

    assert outcome.rows, f"nothing reached a row; funnel says {dict(outcome.lost)}"
    row = outcome.rows[0]
    # The assertion that fails on the bug: a `Decision` carries `.value`, and subscripting
    # one raises rather than returning the space.
    assert row["space"] == "MNI"
    assert row["points"] == [[10.0, -20.0, 30.0]]


def _one_analysis_record(tmp_path, monkeypatch, group_slots, link_n):
    """A record with one joinable, poolable analysis over one cohort."""
    from pondie import paths

    parse = {
        "analyses": [
            {
                "name": "Patients > controls",
                "table_id": "t1",
                "table_number": 1,
                "points": [{"coordinates": [10.0, -20.0, 30.0], "space": "MNI", "values": []}],
            }
        ]
    }
    stage1 = tmp_path / "S1" / "stage1" / "analyses.json"
    stage1.parent.mkdir(parents=True, exist_ok=True)
    stage1.write_text(json.dumps(parse))
    monkeypatch.setattr(paths, "stage1", lambda study, **_: stage1)

    group = {"local_id": "g1"}
    group.update(
        {k: {"extraction_status": "extracted", "value": v} for k, v in group_slots.items()}
    )
    link = {"group": "g1"}
    if link_n is not None:
        link["n"] = {"extraction_status": "extracted", "value": link_n}

    record = {
        "groups": [group],
        "analyses": [
            {
                "local_id": "a1",
                "name": {"extraction_status": "extracted", "value": "Patients > controls"},
                "source_table_analysis": {
                    "extraction_status": "extracted",
                    "value": parse_keys.parse_keys(parse["analyses"])[0],
                },
                "coordinate_space": {"extraction_status": "extracted", "value": "MNI"},
                "spatial_scope": {"extraction_status": "extracted", "value": "whole_brain"},
                "groups": [link],
            }
        ],
    }
    written = tmp_path / "S1.extraction.json"
    written.write_text(json.dumps(record))
    return select(Selection(records=(str(written),)))


def test_the_analysed_n_wins_over_the_cohort_total(tmp_path, monkeypatch):
    """`AnalysisGroup.n` is what the analysis used; the cohort total is only a ceiling."""
    outcome = _one_analysis_record(
        tmp_path, monkeypatch, {"acquired_count": 40, "enrolled_count": 50}, link_n=31
    )
    assert outcome.rows[0]["n"] == 31
    assert outcome.rows[0]["n_source"] == ["analysis"]


def test_a_missing_analysed_n_falls_back_to_the_cohort_it_ran_on(tmp_path, monkeypatch):
    """The defect this fixes: one contrast carried n and the study's others did not.

    Three of four contrasts in a real record named the same two cohorts and left `n` empty,
    so they were dropped for want of a number the same record already held twice over.
    """
    outcome = _one_analysis_record(
        tmp_path, monkeypatch, {"acquired_count": 40, "enrolled_count": 50}, link_n=None
    )
    assert outcome.rows[0]["n"] == 40, "acquired, not enrolled: enrolment precedes the scanner"
    assert outcome.rows[0]["n_source"] == ["acquired_count"]
    assert [s["id"] for s in outcome.poolable()] == ["S1"], "and it pools"


def test_enrolled_is_the_last_resort_not_the_first(tmp_path, monkeypatch):
    outcome = _one_analysis_record(tmp_path, monkeypatch, {"enrolled_count": 50}, link_n=None)
    assert outcome.rows[0]["n"] == 50
    assert outcome.rows[0]["n_source"] == ["enrolled_count"]


def test_a_cohort_that_reports_no_size_at_all_is_still_not_given_one(tmp_path, monkeypatch):
    """The fallback reaches for a reported number; it does not invent one.

    A record whose paper never states a size stays unweightable, which is the rule the
    substituted-30 default broke.
    """
    outcome = _one_analysis_record(tmp_path, monkeypatch, {}, link_n=None)
    assert outcome.rows[0]["n"] is None
    assert outcome.rows[0]["n_source"] == []
    assert outcome.poolable() == []
    assert outcome.lost["no sample size, so it cannot be weighted"] == 1


def test_the_funnel_says_when_a_weight_came_from_a_cohort_total(tmp_path, monkeypatch):
    """An inferred weight that nothing reports is the silent failure, not the inference."""
    outcome = _one_analysis_record(tmp_path, monkeypatch, {"acquired_count": 40}, link_n=None)
    assert "weighted on a cohort total" in outcome.funnel()
    assert "acquired_count" in outcome.funnel()


def test_an_arm_is_the_comparator_whenever_it_says_so(tmp_path):
    """`sham tPEMF` and `active tPEMF` differ by one word, and the wrong call inverts a map."""
    from pondie.normalization import UNKNOWN
    from pondie.normalization.arm_role import ACTIVE, CONTROL, role

    assert role("placebo") == CONTROL
    assert role("sham stimulation") == CONTROL
    assert role("normal saline placebo") == CONTROL, "its own name gives it away"
    assert role("no intervention") == CONTROL, "a comparator need not say placebo"
    assert role("active tPEMF") == ACTIVE, "the control rule is decisive, not greedy"
    assert role("0.5 mg/kg ketamine") == ACTIVE
    assert role("LDLPFC rTMS") == ACTIVE
    # Not every arm is one or the other, and pretending otherwise is how foci land in the
    # opposite map. LPS is an inflammatory challenge; MDD is a diagnosis, not an arm.
    assert role("LPS") == UNKNOWN
    assert role("MDD") == UNKNOWN
    assert role("") == UNKNOWN


def _arm_record(tmp_path, monkeypatch, arms, levels, cells):
    from pondie import paths

    parse = {
        "analyses": [
            {
                "name": "A",
                "table_id": "t1",
                "table_number": 1,
                "points": [{"coordinates": [1.0, 2.0, 3.0], "space": "MNI", "values": []}],
            }
        ]
    }
    stage1 = tmp_path / "S1" / "stage1" / "analyses.json"
    stage1.parent.mkdir(parents=True, exist_ok=True)
    stage1.write_text(json.dumps(parse))
    monkeypatch.setattr(paths, "stage1", lambda study, **_: stage1)

    def ex(v):
        return {"extraction_status": "extracted", "value": v}

    record = {
        "design": {"arms": [{"local_id": i, "name": ex(n)} for i, n in arms.items()]},
        "model_estimations": [
            {
                "local_id": "m1",
                "terms": [
                    {
                        "local_id": "trm",
                        "levels": [{"level": ex(lv), "arms": [a]} for lv, a in levels.items()],
                    }
                ],
            }
        ],
        "groups": [{"local_id": "g1", "acquired_count": ex(20)}],
        "analyses": [
            {
                "local_id": "a1",
                "name": ex("A"),
                "source_table_analysis": ex(parse_keys.parse_keys(parse["analyses"])[0]),
                "coordinate_space": ex("MNI"),
                "spatial_scope": ex("whole_brain"),
                "groups": [{"group": "g1", "n": ex(20)}],
                "effect": {
                    "cells": [
                        {"term": "trm", "level": ex(lv), "direction": ex(d)}
                        for lv, d in cells.items()
                    ]
                },
            }
        ],
    }
    written = tmp_path / "S1.extraction.json"
    written.write_text(json.dumps(record))
    return written


def test_the_two_arm_directions_are_not_the_same_selection(tmp_path, monkeypatch):
    """The bug this implements away: `direction` was declared and never read, so a query
    for treatment>control and one for control>treatment returned identical rows.
    """
    written = _arm_record(
        tmp_path,
        monkeypatch,
        arms={"arm_a": "citalopram", "arm_c": "placebo"},
        levels={"drug": "arm_a", "pbo": "arm_c"},
        cells={"drug": "positive", "pbo": "negative"},
    )
    kept = select(Selection(records=(str(written),), arm_contrast="active_over_control"))
    other = select(Selection(records=(str(written),), arm_contrast="control_over_active"))

    assert len(kept.rows) == 1, dict(kept.lost)
    assert other.rows == [], "the same analysis cannot serve both directions"
    assert other.lost["arm contrast runs active_over_control"] == 1


def test_an_unsigned_arm_contrast_is_dropped_from_both(tmp_path, monkeypatch):
    """`undirected` asserts the source did not sign it; a side cannot be invented for it."""
    written = _arm_record(
        tmp_path,
        monkeypatch,
        arms={"arm_a": "ketamine", "arm_c": "saline"},
        levels={"drug": "arm_a", "pbo": "arm_c"},
        cells={"drug": "undirected", "pbo": "undirected"},
    )
    for way in ("active_over_control", "control_over_active"):
        outcome = select(Selection(records=(str(written),), arm_contrast=way))
        assert outcome.rows == []
        assert outcome.lost["no signed arm contrast"] == 1


def test_a_contrast_between_cohorts_is_not_an_arm_contrast(tmp_path, monkeypatch):
    """Patients-vs-controls is signed and is not a treatment contrast. Selecting it as one
    would pool a diagnosis difference into a map labelled as a drug effect.
    """
    written = _arm_record(
        tmp_path,
        monkeypatch,
        arms={"arm_a": "MDD", "arm_c": "healthy controls"},
        levels={"mdd": "arm_a", "hc": "arm_c"},
        cells={"mdd": "positive", "hc": "negative"},
    )
    outcome = select(Selection(records=(str(written),), arm_contrast="active_over_control"))
    assert outcome.rows == [], "neither side classifies as an arm, so it is not selectable"
    assert outcome.lost["no signed arm contrast"] == 1


def _exposure_record(
    tmp_path, monkeypatch, *, arms=(), timepoints=(), levels, cells, group_arm=None
):
    """A record with one poolable analysis whose levels name arms and/or timepoints."""
    from pondie import paths

    parse = {
        "analyses": [
            {
                "name": "A",
                "table_id": "t1",
                "table_number": 1,
                "points": [{"coordinates": [1.0, 2.0, 3.0], "space": "MNI", "values": []}],
            }
        ]
    }
    stage1 = tmp_path / "S1" / "stage1" / "analyses.json"
    stage1.parent.mkdir(parents=True, exist_ok=True)
    stage1.write_text(json.dumps(parse))
    monkeypatch.setattr(paths, "stage1", lambda study, **_: stage1)

    def ex(v):
        return {"extraction_status": "extracted", "value": v}

    group = {"local_id": "g1", "acquired_count": ex(20)}
    if group_arm:
        group["arm"] = group_arm

    record = {
        "design": {
            "arms": [{"local_id": i, "name": ex(n)} for i, n in arms],
            "timepoints": [
                {"local_id": i, "name": ex(i), "relation_to_intervention": ex(r)}
                for i, r in timepoints
            ],
        },
        "model_estimations": [
            {
                "local_id": "m1",
                "terms": [
                    {
                        "local_id": "trm",
                        "levels": [{"level": ex(lv), **link} for lv, link in levels.items()],
                    }
                ],
            }
        ],
        "groups": [group],
        "analyses": [
            {
                "local_id": "a1",
                "name": ex("A"),
                "source_table_analysis": ex(parse_keys.parse_keys(parse["analyses"])[0]),
                "coordinate_space": ex("MNI"),
                "spatial_scope": ex("whole_brain"),
                "groups": [{"group": "g1", "n": ex(20)}],
                "effect": {
                    "cells": [
                        {"term": "trm", "level": ex(lv), "direction": ex(d)}
                        for lv, d in cells.items()
                    ]
                },
            }
        ],
    }
    written = tmp_path / "S1.extraction.json"
    written.write_text(json.dumps(record))
    return written


def test_a_parallel_group_trial_reaches_the_filter_through_group_arm(tmp_path, monkeypatch):
    """The half of the schema the first implementation missed.

    `FactorLevel.arms` is the crossover route. A parallel-group trial allocates whole
    cohorts, so its arm arrives via `Group.arm`, and reading only the first route makes
    every parallel-group trial invisible -- which is most randomised imaging trials.
    """
    written = _exposure_record(
        tmp_path,
        monkeypatch,
        arms=(("arm_a", "citalopram"), ("arm_c", "placebo")),
        levels={"drug": {"groups": ["g1"]}, "pbo": {"arms": ["arm_c"]}},
        cells={"drug": "positive", "pbo": "negative"},
        group_arm="arm_a",
    )
    outcome = select(Selection(records=(str(written),), treatment_exposure="increase"))
    assert len(outcome.rows) == 1, dict(outcome.lost)
    assert outcome.rows[0]["route"] == "arm"


def test_before_and_after_is_a_treatment_contrast_too(tmp_path, monkeypatch):
    """Read from `Timepoint.relation_to_intervention`, not from what a level is called."""
    written = _exposure_record(
        tmp_path,
        monkeypatch,
        timepoints=(("tp_pre", "pre_intervention"), ("tp_post", "post_intervention")),
        levels={
            "week eight": {"timepoints": ["tp_post"]},
            "scan 1": {"timepoints": ["tp_pre"]},
        },
        cells={"week eight": "positive", "scan 1": "negative"},
    )
    outcome = select(Selection(records=(str(written),), treatment_exposure="increase"))
    assert len(outcome.rows) == 1, dict(outcome.lost)
    assert outcome.rows[0]["route"] == "time", "neither level name says pre or post"
    flipped = select(Selection(records=(str(written),), treatment_exposure="decrease"))
    assert flipped.rows == []


def test_a_before_after_change_inside_the_sham_arm_is_not_a_treatment_effect(
    tmp_path, monkeypatch
):
    """It is a placebo or repetition effect, and pooling it inflates the treatment map."""
    written = _exposure_record(
        tmp_path,
        monkeypatch,
        arms=(("arm_c", "sham stimulation"),),
        timepoints=(("tp_pre", "pre_intervention"), ("tp_post", "post_intervention")),
        levels={"after": {"timepoints": ["tp_post"]}, "before": {"timepoints": ["tp_pre"]}},
        cells={"after": "positive", "before": "negative"},
        group_arm="arm_c",
    )
    outcome = select(Selection(records=(str(written),), treatment_exposure="increase"))
    assert outcome.rows == []
    assert outcome.lost["not a treatment-exposure contrast"] == 1


def test_a_during_intervention_occasion_is_neither_pole(tmp_path, monkeypatch):
    """`during_intervention` is a real relation and is not folded onto before or after."""
    written = _exposure_record(
        tmp_path,
        monkeypatch,
        timepoints=(("tp_mid", "during_intervention"), ("tp_pre", "pre_intervention")),
        levels={"mid": {"timepoints": ["tp_mid"]}, "base": {"timepoints": ["tp_pre"]}},
        cells={"mid": "positive", "base": "negative"},
    )
    for way in ("increase", "decrease"):
        assert select(Selection(records=(str(written),), treatment_exposure=way)).rows == []
