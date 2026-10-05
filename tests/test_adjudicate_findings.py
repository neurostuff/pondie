"""The validator findings the adjudicator settles, and the ones settled without it.

From the 55-paper PTSD run: 77 ROI analyses naming no region, 8 ROI corrections likewise,
16 cell levels no declared level spells, and 6 effect kinds their cells contradict.
"""

import json

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
    "The PTSD group showed greater activation than the control group. "
    "There was no significant difference in hippocampal volume between the groups. "
    "Grey matter volume was reduced in PTSD relative to controls."
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

def _cited(value, quote):
    span = stage.span_tools.resolve(TEXT, quote).as_record()
    return {"extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": {"status": "present", "sets": [{"spans": [span]}]}}


def test_a_cited_scope_is_rescoped_by_a_cited_answer_among_the_options():
    """27082610: total-volume comparisons cited as `roi` with no region; every cited
    `whole_brain` answer was refused as losing the warrant."""
    record = _roi_record()
    record["analyses"][0]["spatial_scope"] = _cited("roi", "Left hippocampal volume")
    caller, _ = _answering({
        "id": "analyses/ana_hc/regions", "value": "whole_brain",
        "quote": "Left hippocampal volume was smaller in PTSD than in healthy controls.",
    })
    report = _adjudicate(record, caller)
    assert values.read(record["analyses"][0]["spatial_scope"]) == "whole_brain", report.adjudicated


def test_a_cited_value_outside_the_options_is_still_not_coerced():
    """12853571: a cited compound scope is not collapsed to one enum value."""
    from pondie.extraction.repair.guard import Edit, refuses_losing_the_warrant

    entity = {"correction_scope": _cited("whole volume and small volumes", "Hippocampal")}
    edit = Edit({}, entity, "correction_scope", "whole_brain", TEXT, "Hippocampal",
                choices=("whole_brain", "roi"))
    assert refuses_losing_the_warrant(edit) is not None


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


NULL = "There was no significant difference in hippocampal volume between the groups."


def _null_contrast():
    """22952599, 23021615, 28549317: a null two-group comparison stated as a contrast, its
    cells `undirected` -- which derives `omnibus`."""
    record = _effect_record("control group", "contrast")
    for cell in record["analyses"][0]["effect"]["cells"]:
        cell["direction"] = _v("undirected")
    return record


def _kind_case(record):
    [case] = [c for c in stage.contradictions(record, _sch()) if c.slot == "kind"]
    return case


def test_keeping_a_contrast_marks_its_undirected_signs_not_reported():
    record = _null_contrast()
    case = _kind_case(record)
    assert case.options == ("omnibus", "contrast") and "not reported" in case.question
    caller, _ = _answering({"id": case.id, "value": "contrast", "quote": NULL})
    report = _adjudicate(record, caller)
    cells = record["analyses"][0]["effect"]["cells"]
    assert [c["direction"]["extraction_status"] for c in cells] == ["not_reported"] * 2
    validator = Validator(_sch(), None)
    validator.check_record(record)
    assert not [e for e in validator.errors if "direction" in e or "effect.kind" in e]
    assert not [c for c in stage.contradictions(record, _sch()) if c.slot == "kind"]
    assert report.adjudicated[0].startswith(f"{case.id}: kept contrast; cells[0].direction")


def test_answering_omnibus_leaves_the_undirected_cells_as_they_are():
    record = _null_contrast()
    case = _kind_case(record)
    _adjudicate(record, _answering({"id": case.id, "value": "omnibus", "quote": NULL})[0])
    effect = record["analyses"][0]["effect"]
    assert values.read(effect["kind"]) == "omnibus"
    assert [values.read(c["direction"]) for c in effect["cells"]] == ["undirected"] * 2


def test_an_answer_that_leaves_the_contradiction_standing_is_reported_so():
    # Keeping `interaction` cannot add the cells an interaction needs.
    record = _effect_record("control group", "interaction")
    case = _kind_case(record)
    quote = "The PTSD group showed greater activation than the control group."
    caller, _ = _answering({"id": case.id, "value": "interaction", "quote": quote})
    report = _adjudicate(record, caller)
    assert report.adjudicated == [
        f"{case.id}: kept interaction",
        f"{case.id}: still contradicted after the answer",
    ]


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



def test_a_factors_cells_filed_on_a_product_move_to_the_factor():
    """28287194: `Patients -, Healthy controls +` on group-by-time beside `T2: held` is
    patients below controls at T2 -- a crossing of group, held within one occasion."""
    from pondie.extraction.record.effect import derive_effect_kind, terms_in_scope

    record = _group_by_time()
    cells = record["analyses"][0]["effect"]["cells"]
    cells[:] = [{"term": "trm_gxt", "level": _v("PTSD"), "direction": _v("negative")},
                {"term": "trm_gxt", "level": _v("TD"), "direction": _v("positive")},
                {"term": "trm_time", "level": _v("follow-up"), "direction": _v("held")}]
    record["analyses"][0]["effect"]["kind"] = _v("contrast")
    assert link.cross_products_of_factors(record)
    assert [c["term"] for c in cells] == ["trm_group", "trm_group", "trm_time"]
    terms = terms_in_scope("mod_a", {"mod_a": record["model_estimations"][0]})
    assert derive_effect_kind(cells, terms)[0] == "contrast"


def test_a_moderations_cells_are_not_moved_to_its_factor():
    """A product with a continuous term is a moderation; its cells stay on it."""
    record = _moderation("PTSD", "negative")
    cells = record["analyses"][0]["effect"]["cells"]
    cells.append({"term": "trm_x", "level": _v("control"), "direction": _v("positive")})
    link.cross_products_of_factors(record)
    assert {c["term"] for c in cells} == {"trm_x"}

@pytest.mark.parametrize("record", [
    _group_by_time(ordered=False),
    _group_by_time(extra_cells=[{"term": "trm_time", "level": _v("baseline"),
                                 "direction": _v("held")}]),
], ids=["unordered", "factor-already-celled"])
def test_a_product_of_factors_is_left_where_the_reading_is_not_certain(record):
    before = _signs(record)
    assert link.cross_products_of_factors(record) == []
    assert _signs(record) == before


def _seeds(borrowed=("trm_gxa",)):
    """Two seed models; the extractor declared the design under the first only."""
    return {
        "model_estimations": [
            {"local_id": "mod_vmpfc", "terms": [
                {"local_id": "trm_group", "name": _v("group"), "type": _v("categorical"),
                 "levels": [{"level": _v("PTSD")}, {"level": _v("TD")}]},
                {"local_id": "trm_age", "name": _v("age"), "type": _v("continuous"),
                 "variation_level": _v("between_subject")},
                {"local_id": "trm_gxa", "name": _v("group x age"), "type": _v("continuous"),
                 "interaction_with": ["trm_group", "trm_age"]}]},
            {"local_id": "mod_other", "terms": [
                {"local_id": "trm_sex", "name": _v("sex"), "type": _v("categorical"),
                 "levels": [{"level": _v("F")}, {"level": _v("M")}]}]},
            {"local_id": "mod_pcc", "terms": []},
        ],
        "analyses": [{"local_id": "ana_pcc", "name": _v("group x age (PCC)"),
                      "model_estimation": "mod_pcc", "effect": {
                          "kind": _v("interaction"),
                          "cells": [{"term": t, "direction": _v("negative")} for t in borrowed]}}],
    }


def test_a_cell_outside_its_model_chain_derives_no_kind_and_raises_no_kind_case():
    """30343133: the cell's term was unresolvable, so it derived `simple_effect` against a
    stated `interaction`, and the adjudicator was asked a question with no right answer."""
    from pondie.extraction.record.effect import UNRESOLVED_TERM, derive_effect_kind

    record = _seeds()
    cells = record["analyses"][0]["effect"]["cells"]
    assert derive_effect_kind(cells, {})[0] == UNRESOLVED_TERM
    assert not [c for c in stage.contradictions(record, _sch()) if c.slot == "kind"]
    validator = Validator(_sch(), None)
    validator.check_record(record)
    assert not [e for e in validator.errors if "effect.kind" in e]
    assert any("does not reach" in e for e in validator.errors)


def test_an_empty_model_gets_the_terms_its_analyses_borrow_from_one_other():
    record = _seeds()
    assert link.fill_empty_models(record, _sch())
    pcc = record["model_estimations"][2]
    assert [t["local_id"] for t in pcc["terms"]] == [
        "mod_pcc.trm_group", "mod_pcc.trm_age", "mod_pcc.trm_gxa"]
    assert pcc["terms"][2]["interaction_with"] == ["mod_pcc.trm_group", "mod_pcc.trm_age"]
    assert record["analyses"][0]["effect"]["cells"][0]["term"] == "mod_pcc.trm_gxa"
    assert [t["local_id"] for t in record["model_estimations"][0]["terms"]][0] == "trm_group"
    validator = Validator(_sch(), None)
    validator.check_record(record)
    assert not [e for e in validator.errors if "does not reach" in e or "effect.kind" in e]



def test_an_empty_model_does_not_take_a_term_another_analysis_tests():
    """17923164: one model held the IES total, intrusion and avoidance scores; the intrusion
    model takes intrusion and the covariate, not the total its neighbour tested."""
    record = _seeds()
    donor = record["model_estimations"][0]
    donor["terms"].append({"local_id": "trm_total", "name": _v("IES total"),
                           "type": _v("continuous")})
    record["analyses"].append({"local_id": "ana_total", "model_estimation": "mod_vmpfc",
                               "effect": {"cells": [{"term": "trm_total",
                                                     "direction": _v("negative")}]}})
    link.fill_empty_models(record, _sch())
    copied = [t["local_id"] for t in record["model_estimations"][2]["terms"]]
    assert copied == ["mod_pcc.trm_group", "mod_pcc.trm_age", "mod_pcc.trm_gxa"]

def test_the_donor_is_the_one_model_declaring_every_borrowed_term():
    """21338692: `age of first use` was declared by two models and `years of use` by one of
    them, so one model holds everything the empty model's analysis cells."""
    record = _seeds(borrowed=("trm_gxa", "trm_age"))
    record["model_estimations"][1]["terms"].append(
        {"local_id": "trm_age", "name": _v("age"), "type": _v("continuous")})
    assert link.fill_empty_models(record, _sch())
    assert "mod_pcc.trm_gxa" in [t["local_id"] for t in record["model_estimations"][2]["terms"]]



def test_donors_declaring_the_same_design_are_one_donor():
    """16701903 declared an identical `group` factor in two models."""
    record = _seeds(borrowed=("trm_sex",))
    record["model_estimations"][0]["terms"].append(
        json.loads(json.dumps(record["model_estimations"][1]["terms"][0])))
    record["model_estimations"][0]["terms"] = [record["model_estimations"][0]["terms"][-1]]
    assert link.fill_empty_models(record, _sch())
    assert [t["local_id"] for t in record["model_estimations"][2]["terms"]] == ["mod_pcc.trm_sex"]


def test_donors_declaring_different_designs_are_none():
    record = _seeds(borrowed=("trm_sex",))
    other = json.loads(json.dumps(record["model_estimations"][1]["terms"][0]))
    # The levels written inside their own `level` slot, as 26682697 did: comparing
    # designs must not need anything hashable.
    other["levels"] = [{"level": {"level": _v("female")}}, {"level": {"level": _v("male")}}]
    record["model_estimations"][0]["terms"] = [other]
    assert link.fill_empty_models(record, _sch()) == []

def test_an_empty_model_borrowing_from_two_models_is_left():
    """No model declares both borrowed terms: no one design the record states to copy."""
    record = _seeds(borrowed=("trm_gxa", "trm_sex"))
    assert link.fill_empty_models(record, _sch()) == []
    assert record["model_estimations"][2]["terms"] == []


def test_an_analysis_without_a_model_takes_the_one_declaring_its_terms():
    """17892884: the cells named `mod_repeated_hippocampus`'s group term."""
    record = _seeds()
    record["analyses"][0]["model_estimation"] = None
    record["analyses"][0]["effect"]["cells"] = [{"term": "trm_sex", "direction": _v("negative")}]
    link.infer_missing_models(record)
    assert record["analyses"][0]["model_estimation"] == "mod_other"


def test_an_roi_correction_takes_the_regions_its_analyses_agree_on():
    from pondie.extraction.record.fix import derive

    body = {"inference_settings": [{"local_id": "inf_svc", "correction_scope": _v("roi")}],
            "analyses": [{"local_id": "a", "inference_settings": ["inf_svc"],
                          "regions": ["reg_amy"]},
                         {"local_id": "b", "inference_settings": "inf_svc",
                          "regions": ["reg_amy"]}]}
    derive.derive_correction_regions(body)
    assert body["inference_settings"][0]["correction_regions"] == ["reg_amy"]
    body["analyses"][1]["regions"] = ["reg_hip"]
    body["inference_settings"][0].pop("correction_regions")
    assert derive.derive_correction_regions(body) == []


def test_a_level_matches_through_an_abbreviation_this_paper_defines():
    """26347628 declared `HC`, celled `healthy controls`, and defined `HCs`."""
    record = _effect_record("healthy controls", "contrast")
    term = record["model_estimations"][0]["terms"][0]
    term["levels"] = [{"level": _v("PTSD")}, {"level": _v("HC")}]
    link.align_cell_levels(record, "We scanned 20 healthy controls (HCs) and 20 patients.")
    assert values.read(record["analyses"][0]["effect"]["cells"][0]["level"]) == "HC"
    other = _effect_record("healthy controls", "contrast")
    other["model_estimations"][0]["terms"][0]["levels"] = [{"level": _v("PTSD")},
                                                           {"level": _v("HC")}]
    assert link.align_cell_levels(other, "No abbreviations are defined here.") == []

def test_a_value_cannot_be_wrapped_with_not_applicable_evidence():
    """The schema reserves it for not_reported fields; seven writers got it wrong."""
    with pytest.raises(ValueError):
        values.wrap("x", source="generated", evidence="not_applicable")
    assert values.wrap(None, source="reported", evidence="not_applicable")[
        "extraction_status"] == "not_reported"


def _partial():
    """17825801: one diagnosis x exposure design, fitted unadjusted and adjusted for age."""
    def term(tid, name, kind):
        return {"local_id": tid, "name": _v(name), "type": _v(kind)}
    return {
        "model_estimations": [
            {"local_id": "mod_unadj", "terms": [term("trm_dx", "Diagnosis", "categorical"),
                                                term("trm_exp", "Exposure", "categorical"),
                                                term("trm_icv", "ICV", "continuous")]},
            {"local_id": "mod_age", "terms": [term("trm_dx", "Diagnosis", "categorical"),
                                              term("trm_age", "age", "continuous")]},
        ],
        "analyses": [{"local_id": "a_age", "model_estimation": "mod_age", "effect": {"cells": [
            {"term": "trm_dx", "direction": _v("positive")},
            {"term": "trm_exp", "direction": _v("negative")}]}}],
    }


def test_a_partial_model_gets_the_celled_term_it_lacks_and_not_the_donors_covariates():
    record = _partial()
    assert link.complete_partial_models(record, _sch())
    terms = [t["local_id"] for t in record["model_estimations"][1]["terms"]]
    assert terms == ["trm_dx", "trm_age", "mod_age.trm_exp"]
    cells = record["analyses"][0]["effect"]["cells"]
    assert [c["term"] for c in cells] == ["trm_dx", "mod_age.trm_exp"]
    validator = Validator(_sch(), None)
    validator.check_record(record)
    assert not [e for e in validator.errors if "does not reach" in e]


def test_donors_that_disagree_leave_a_partial_model_alone():
    record = _partial()
    other = json.loads(json.dumps(record["model_estimations"][0]))
    other["local_id"] = "mod_other"
    other["terms"][1]["name"] = _v("Combat exposure, three levels")
    record["model_estimations"].append(other)
    assert link.complete_partial_models(record, _sch()) == []


def test_a_model_with_its_own_same_named_term_is_repointed_not_given_a_copy():
    """19794316: `mod_group_wholebrain` declares its own `group` term; its cells named the
    ROI model's `trm_group`. Run before `cell_terms`, the partial-model repair copied
    `trm_group` in beside it."""
    from pondie.extraction.record import fix

    def term(tid):
        return {"local_id": tid, "name": _v("group"), "type": _v("categorical"),
                "levels": [{"level": _v("PTSD")}, {"level": _v("control")}]}
    record = {
        "model_estimations": [{"local_id": "mod_roi", "terms": [term("trm_group")]},
                              {"local_id": "mod_wb", "terms": [term("trm_group_wb")]}],
        "analyses": [{"local_id": "a_wb", "model_estimation": "mod_wb", "effect": {"cells": [
            {"term": "trm_group", "level": _v("PTSD"), "direction": _v("positive")},
            {"term": "trm_group", "level": _v("control"), "direction": _v("negative")}]}}],
    }
    fix.apply_all(record, fix.Context(schema=_sch()), stage="merged")
    [wb] = [m for m in record["model_estimations"] if m["local_id"] == "mod_wb"]
    assert [t["local_id"] for t in wb["terms"]] == ["trm_group_wb"]
    assert {c["term"] for c in record["analyses"][0]["effect"]["cells"]} == {"trm_group_wb"}


def _named(name, ptsd, control):
    record = _effect_record("control group", "contrast")
    analysis = record["analyses"][0]
    analysis["name"] = _v(name)
    analysis["effect"]["cells"] = [
        {"term": "trm_group", "level": _v("PTSD"), "direction": _v(ptsd)},
        {"term": "trm_group", "level": _v("control group"), "direction": _v(control)}]
    return record


REDUCED = "Grey matter volume was reduced in PTSD relative to controls."


@pytest.mark.parametrize("name, ptsd, control, raised", [
    ("PTSD < controls", "positive", "negative", True),      # 21418787: signs reversed
    ("PTSD < controls", "negative", "positive", False),
    ("PTSD vs controls, P < 0.01 corrected", "positive", "negative", False),  # a threshold
])
def test_a_name_stating_the_reverse_of_the_cells_is_a_case(name, ptsd, control, raised):
    cases = [c for c in stage.contradictions(_named(name, ptsd, control), _sch())
             if c.id.endswith("direction")]
    assert bool(cases) is raised


def test_answering_name_reverses_the_cells_signs():
    record = _named("PTSD < controls", "positive", "negative")
    [case] = [c for c in stage.contradictions(record, _sch()) if c.id.endswith("direction")]
    caller, _ = _answering({"id": case.id, "value": "name", "quote": REDUCED})
    report = _adjudicate(record, caller)
    cells = record["analyses"][0]["effect"]["cells"]
    assert [values.read(c["direction"]) for c in cells] == ["negative", "positive"]
    assert not any("still contradicted" in line for line in report.adjudicated)


def test_answering_cells_reverses_the_names_comparison():
    """30343133: the parse titled it 'PTSD > TD', the cells agreed, the name said '<'."""
    record = _named("PTSD < controls", "positive", "negative")
    [case] = [c for c in stage.contradictions(record, _sch()) if c.id.endswith("direction")]
    caller, _ = _answering({"id": case.id, "value": "cells", "quote": REDUCED})
    _adjudicate(record, caller)
    assert values.read(record["analyses"][0]["name"]) == "PTSD > controls"
    assert values.read(record["analyses"][0]["effect"]["cells"][0]["direction"]) == "positive"


def test_keeping_whole_brain_clears_the_regions_beside_it():
    """33169525: five analyses answered `kept whole_brain` and kept their regions, so the
    contradiction stood after every answer."""
    record = _roi_record()
    analysis = record["analyses"][0]
    analysis["spatial_scope"] = _v("whole_brain")
    analysis["regions"] = ["reg_amygdala"]
    [case] = [c for c in stage.contradictions(record, _sch()) if c.slot == "spatial_scope"]
    caller, _ = _answering({"id": case.id, "value": "whole_brain", "quote": REDUCED})
    report = _adjudicate(record, caller)
    assert analysis["regions"] == []
    assert report.adjudicated == [f"{case.id}: kept whole_brain; regions cleared"]


def test_reversing_a_name_keeps_its_threshold():
    """Review: 'PTSD > HC (p < 0.001)' answered 'cells' became 'PTSD < HC (p > 0.001)'."""
    from pondie.extraction.repair.stage import _reverse_name
    analysis = {"name": _v("PTSD > HC (p < 0.001)")}
    _reverse_name(analysis)
    assert values.read(analysis["name"]) == "PTSD < HC (p < 0.001)"


def test_a_copy_of_a_copied_term_is_scoped_once():
    """Audit: complete_partial_models did not drop a donor's scope prefix, so a term copied
    from a model fill_empty_models had filled became `mod_c.mod_b.trm_exp`."""
    record = _partial()
    record["model_estimations"].append({"local_id": "mod_c", "terms": [
        {"local_id": "trm_age", "name": _v("age"), "type": _v("continuous")}]})
    record["model_estimations"][0]["local_id"] = "mod_b"
    record["model_estimations"][0]["terms"][1]["local_id"] = "mod_b.trm_exp"
    record["analyses"].append({"local_id": "a_c", "model_estimation": "mod_c", "effect": {
        "cells": [{"term": "mod_b.trm_exp", "direction": _v("positive")}]}})
    link.complete_partial_models(record, _sch())
    [mod_c] = [m for m in record["model_estimations"] if m["local_id"] == "mod_c"]
    assert "mod_c.trm_exp" in [t["local_id"] for t in mod_c["terms"]]
