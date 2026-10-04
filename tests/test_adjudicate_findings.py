"""The validator findings the adjudicator settles, and the ones settled without it.

From the 55-paper PTSD run: 77 ROI analyses naming no region, 8 ROI corrections likewise,
16 cell levels no declared level spells, and 6 effect kinds their cells contradict.
"""

import pytest

from pondie.extraction.models import Cost, ModelReply
from pondie.extraction.record.fix import link
from pondie.extraction.record.validate import EXTRACTION_SCHEMA, Validator
from pondie.extraction.repair import stage
from pondie.formats import values
from pondie.schema import reader

TEXT = (
    "Hippocampal and amygdala volumes were manually traced on each T1 image. "
    "Left hippocampal volume was smaller in PTSD than in healthy controls. "
    "The PTSD group showed greater activation than the control group."
)


def _v(value):
    return {"extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": {"status": "not_found"}}


def _sch():
    return reader.load(EXTRACTION_SCHEMA)


def _answering(*resolutions):
    calls = []

    def caller(call, *, paper, stage):
        calls.append(call)
        return ModelReply(payload={"resolutions": list(resolutions)}, cost=Cost(calls=1))

    return caller, calls


def _adjudicate(record, caller):
    report = stage.Report()
    stage.adjudicate(record, _sch(), TEXT, caller, study_id="S1", model="m", report=report)
    return report


def _roi_record():
    return {
        "regions": [{"local_id": "reg_amygdala", "name": _v("amygdala"),
                     "definition_method": _v("anatomical_a_priori")}],
        "analyses": [{"local_id": "ana_hc", "name": _v("Hippocampal volume"),
                      "spatial_scope": _v("roi")}],
    }


# --- ROI scope with no regions -------------------------------------------------------------


def test_an_roi_analysis_naming_no_region_is_asked_which_regions():
    cases = stage.contradictions(_roi_record(), _sch())
    assert [(c.slot, c.names) for c in cases] == [("spatial_scope", "regions")]
    assert "Hippocampal volume" in cases[0].question


def test_the_regions_named_are_linked_and_the_missing_ones_declared():
    record = _roi_record()
    caller, _ = _answering({
        "id": "analyses/ana_hc/regions", "value": "roi",
        "quote": "Hippocampal and amygdala volumes were manually traced on each T1 image.",
        "regions": [{"name": "hippocampus", "definition_method": "anatomical_a_priori"},
                    {"name": "amygdala", "definition_method": "anatomical_a_priori"}],
    })
    _adjudicate(record, caller)
    regions = record["analyses"][0]["regions"]
    assert "reg_amygdala" in regions and len(regions) == 2
    assert len(record["regions"]) == 2, "the amygdala was already held, the hippocampus is new"
    validator = Validator(_sch(), None)
    validator.check_record(record)
    assert not [e for e in validator.errors if "must name the regions" in e]



def test_a_region_whose_definition_the_paper_omits_carries_the_slot_as_not_reported():
    """It answered "the paper's own words" when the prompt offered no way to say nothing."""
    record = _roi_record()
    caller, _ = _answering({
        "id": "analyses/ana_hc/regions", "value": "roi",
        "quote": "Left hippocampal volume was smaller in PTSD than in healthy controls.",
        "regions": [{"name": "left hippocampus", "definition_method": "not_reported"}],
    })
    _adjudicate(record, caller)
    created = record["regions"][-1]
    assert created["definition_method"]["extraction_status"] == "not_reported"
    validator = Validator(_sch(), None)
    validator.check_record(record)
    assert not [e for e in validator.errors if "regions[1]" in e]

def test_an_roi_analysis_that_searched_the_whole_brain_is_rescoped():
    record = _roi_record()
    caller, _ = _answering({
        "id": "analyses/ana_hc/regions", "value": "whole_brain",
        "quote": "Left hippocampal volume was smaller in PTSD than in healthy controls.",
    })
    _adjudicate(record, caller)
    assert values.read(record["analyses"][0]["spatial_scope"]) == "whole_brain"
    assert not record["analyses"][0].get("regions")


# --- levels and kinds ----------------------------------------------------------------------


def _effect_record(level, kind):
    return {
        "model_estimations": [{"local_id": "mod_a", "terms": [{
            "local_id": "trm_group", "name": _v("group"), "type": _v("categorical"),
            "levels": [{"level": _v("PTSD")}, {"level": _v("control group")}]}]}],
        "analyses": [{"local_id": "ana_1", "name": _v("PTSD > control"),
                      "model_estimation": "mod_a",
                      "effect": {"kind": _v(kind), "cells": [
                          {"term": "trm_group", "level": _v(level), "direction": _v("positive")},
                          {"term": "trm_group", "level": _v("PTSD")}]}}],
    }


def test_a_level_no_declared_level_spells_is_put_as_a_choice_of_the_declared_ones():
    record = _effect_record("healthy controls", "contrast")
    [case] = [c for c in stage.contradictions(record, _sch()) if c.slot == "level"]
    assert case.options == ("PTSD", "control group")
    caller, _ = _answering({
        "id": case.id, "value": "control group",
        "quote": "The PTSD group showed greater activation than the control group.",
    })
    _adjudicate(record, caller)
    cell = record["analyses"][0]["effect"]["cells"][0]
    assert values.read(cell["level"]) == "control group"
    assert cell["level"]["evidence"]["status"] == "present"


def test_an_effect_kind_its_cells_contradict_offers_both_and_writes_the_answer():
    record = _effect_record("control group", "interaction")
    [case] = [c for c in stage.contradictions(record, _sch()) if c.slot == "kind"]
    assert case.options[1] == "interaction" and case.options[0] != "interaction"
    caller, _ = _answering({
        "id": case.id, "value": case.options[0],
        "quote": "The PTSD group showed greater activation than the control group.",
    })
    report = _adjudicate(record, caller)
    assert values.read(record["analyses"][0]["effect"]["kind"]) == case.options[0]
    assert report.adjudicated == [f"{case.id}: {case.options[0]}"]


def test_every_case_goes_in_one_call():
    record = _effect_record("healthy controls", "interaction")
    record.update(_roi_record() | {"analyses": record["analyses"] + _roi_record()["analyses"]})
    assert len(stage.contradictions(record, _sch())) == 3
    caller, calls = _answering()
    _adjudicate(record, caller)
    assert len(calls) == 1


# --- settled without a model ---------------------------------------------------------------


@pytest.mark.parametrize("level, group, dropped", [
    ("PTSD group", "recent onset PTSD", True),
    ("non PTSD group", "non PTSD subjects", True),
    ("non-PTSD", "adults without PTSD", True),
    ("PTSD", "non-PTSD controls", False),
    ("PTSD", "trauma survivors", False),
])
def test_a_level_restating_the_analysis_only_group_is_dropped(level, group, dropped):
    """23155380: `'PTSD group'` on a CAPS correlation run within the PTSD group."""
    body = {
        "groups": [{"local_id": "grp_a", "name": _v(group)}],
        "model_estimations": [{"local_id": "mod_a", "terms": [
            {"local_id": "trm_caps", "name": _v("CAPS"), "type": _v("continuous")}]}],
        "analyses": [{"local_id": "ana_1", "model_estimation": "mod_a",
                      "groups": [{"group": "grp_a"}],
                      "effect": {"cells": [{"term": "trm_caps", "level": _v(level)}]}}],
    }
    link.drop_redundant_cell_levels(body)
    assert ("level" not in body["analyses"][0]["effect"]["cells"][0]) is dropped



@pytest.mark.parametrize("direction, dropped", [(None, True), ("undirected", True),
                                                ("negative", False)])
def test_a_level_on_an_unsigned_product_column_is_dropped(direction, dropped):
    """21418787: `level: 'PTSD'` on a group x BAI F-test. A signed one is left: there the
    level says which group's slope the sign describes."""
    cell = {"term": "trm_x", "level": _v("PTSD")}
    if direction:
        cell["direction"] = _v(direction)
    body = {
        "model_estimations": [{"local_id": "mod_a", "terms": [
            {"local_id": "trm_group", "name": _v("group"), "type": _v("categorical"),
             "levels": [{"level": _v("PTSD")}, {"level": _v("MDD")}]},
            {"local_id": "trm_bai", "name": _v("BAI"), "type": _v("continuous")},
            {"local_id": "trm_x", "name": _v("group x BAI"), "type": _v("continuous"),
             "interaction_with": ["trm_group", "trm_bai"]}]}],
        "analyses": [{"local_id": "ana_1", "model_estimation": "mod_a",
                      "effect": {"cells": [cell]}}],
    }
    link.drop_redundant_cell_levels(body)
    assert ("level" not in cell) is dropped


def _moderation(level, direction):
    cell = {"term": "trm_x", "level": _v(level)}
    if direction:
        cell["direction"] = _v(direction)
    return {
        "model_estimations": [{"local_id": "mod_a", "terms": [
            {"local_id": "trm_group", "name": _v("group"), "type": _v("categorical"),
             "levels": [{"level": _v("PTSD")}, {"level": _v("control")}]},
            {"local_id": "trm_age", "name": _v("age"), "type": _v("continuous")},
            {"local_id": "trm_x", "name": _v("group x age"), "type": _v("continuous"),
             "interaction_with": ["trm_group", "trm_age"]}]}],
        "analyses": [{"local_id": "ana_1", "name": _v("group x age"),
                      "model_estimation": "mod_a", "effect": {"cells": [cell]}}],
    }


@pytest.mark.parametrize("level, direction, fault", [
    ("PTSD", "negative", None),
    ("control", "positive", None),
    ("patients", "negative", "components ('PTSD', 'control')"),
    ("PTSD", "undirected", "only when signed"),
])
def test_a_signed_product_cell_may_name_a_level_of_its_categorical_component(
    level, direction, fault
):
    """25212487: "age negatively predicted GMV in PTSD youth" is `negative` with `level: PTSD`:
    without the level the sign has no reference."""
    validator = Validator(_sch(), None)
    validator.check_record(_moderation(level, direction))
    found = [e for e in validator.errors if e.endswith(".level") or ".level:" in e]
    assert (found == []) if fault is None else (len(found) == 1 and fault in found[0])


def test_a_product_cells_unmatched_level_is_put_as_a_choice_of_component_levels():
    [case] = [c for c in stage.contradictions(_moderation("patients", "negative"), _sch())
              if c.slot == "level"]
    assert case.options == ("PTSD", "control")


def _group_by_time(direction="positive", ordered=True, extra_cells=()):
    def level(name, order):
        entry = {"level": _v(name)}
        if ordered:
            entry["order"] = _v(order)
        return entry

    return {
        "model_estimations": [{"local_id": "mod_a", "terms": [
            {"local_id": "trm_group", "name": _v("group"), "type": _v("categorical"),
             "levels": [{"level": _v("TD")}, {"level": _v("PTSD")}]},
            {"local_id": "trm_time", "name": _v("time"), "type": _v("categorical"),
             "levels": [level("follow-up", 2), level("baseline", 1)]},
            {"local_id": "trm_gxt", "name": _v("group by time"), "type": _v("continuous"),
             "interaction_with": ["trm_group", "trm_time"]}]}],
        "analyses": [{"local_id": "ana_1", "name": _v("Group x Time"),
                      "model_estimation": "mod_a", "effect": {"kind": _v("interaction"), "cells": [
                          {"term": "trm_gxt", "level": _v("PTSD"), "direction": _v(direction)},
                          *extra_cells]}}],
    }


def _signs(record):
    return {(c["term"], values.read(c["level"])): values.read(c["direction"])
            for c in record["analyses"][0]["effect"]["cells"]}


def test_a_product_of_two_factors_may_not_name_a_level():
    """Two factors cross in their own cells, so the level there is reported, not accepted."""
    validator = Validator(_sch(), None)
    validator.check_record(_group_by_time())
    assert any("two factors cross in their own cells" in e for e in validator.errors)


@pytest.mark.parametrize("direction, ptsd, td", [("positive", "positive", "negative"),
                                                 ("negative", "negative", "positive")])
def test_a_signed_product_of_two_factors_is_rewritten_as_crossed_cells(direction, ptsd, td):
    """30343133: "GMV decreased over time in TD youths, whereas youths with PTSD showed
    slightly increasing GMV" was `{term: group-by-time, level: PTSD, direction: positive}`."""
    from pondie.extraction.record.effect import derive_effect_kind, terms_in_scope

    record = _group_by_time(direction)
    assert link.cross_products_of_factors(record)
    assert _signs(record) == {("trm_group", "PTSD"): ptsd, ("trm_group", "TD"): td,
                              ("trm_time", "follow-up"): "positive",
                              ("trm_time", "baseline"): "negative"}
    cells = record["analyses"][0]["effect"]["cells"]
    terms = terms_in_scope("mod_a", {"mod_a": record["model_estimations"][0]})
    assert derive_effect_kind(cells, terms)[0] == "interaction"
    validator = Validator(_sch(), None)
    validator.check_record(record)
    assert not [e for e in validator.errors if "level" in e or "effect.kind" in e]


@pytest.mark.parametrize("record", [
    _group_by_time(ordered=False),
    _group_by_time(extra_cells=[{"term": "trm_time", "level": _v("baseline"),
                                 "direction": _v("held")}]),
], ids=["unordered", "factor-already-celled"])
def test_a_product_of_factors_is_left_where_the_reading_is_not_certain(record):
    before = _signs(record)
    assert link.cross_products_of_factors(record) == []
    assert _signs(record) == before

def test_a_value_cannot_be_wrapped_with_not_applicable_evidence():
    """The schema reserves it for not_reported fields; seven writers got it wrong."""
    with pytest.raises(ValueError):
        values.wrap("x", source="generated", evidence="not_applicable")
    assert values.wrap(None, source="reported", evidence="not_applicable")[
        "extraction_status"] == "not_reported"
