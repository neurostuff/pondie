"""What a repair pass may not write.

Each case is a regression that shipped, named by the paper it was found on. The guards were
written against these and nothing else, so a test that stops failing means a guard was
undone rather than that the case got easier.
"""

from __future__ import annotations

import pytest

from pondie import schema
from pondie.extraction.record import guards
from pondie.formats import values
from pondie.schema import reader


@pytest.fixture(scope="module")
def sch():
    return reader.load(schema.STORAGE)


def field(value, evidence=None):
    return {"extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": evidence or {"status": "not_found"}}


def cited(value, quote):
    return field(value, {"status": "present",
                         "sets": [{"source": "model_quote", "spans": [{"text": quote}]}]})


def edit(sch, class_name, entity, slot, value, record=None):
    return guards.Edit(schema=sch, record=record or {}, entity=entity,
                       class_name=class_name, slot=slot, value=value)


def why(refusals):
    return " ".join(r.why for r in refusals)


# --------------------------------------------------------------------------------- values


def test_an_edit_that_only_shortens_is_refused(sch):
    """22952599: "compared to traumatized controls." became "compared to traumatized"."""
    entity = {"local_id": "a1", "definition": cited(
        "Decreased gray matter volume in PTSD patients compared to traumatized controls.",
        "Decreased gray matter volume in PTSD patients compared to traumatized controls.")}
    e = edit(sch, "Analysis", entity, "definition",
             "Decreased gray matter volume in PTSD patients compared to traumatized")
    assert "shortens" in why(guards.refusals(e))


def test_an_edit_that_extends_and_keeps_its_span_is_allowed(sch):
    """23021615: the restored full sentence was already the cited span."""
    quote = ("Relative to the non-PTSD group, the PTSD group showed reduced gray matter in "
             "the same large cluster comprising the sgACC, caudate, and hypothalamus "
             "( Fig. 3 A, B).")
    entity = {"local_id": "a1", "definition": cited(
        "Relative to the non-PTSD group, the PTSD group showed reduced gray matter", quote)}
    e = edit(sch, "Analysis", entity, "definition",
             "Relative to the non-PTSD group, the PTSD group showed reduced gray matter in "
             "the same large cluster comprising the sgACC, caudate, and hypothalamus.")
    assert guards.refusals(e) == []


def test_an_edit_that_drops_the_warrant_is_refused(sch):
    """12853571: a cited, true "whole volume analyzed and a priori small volumes" was
    coerced to the bare enum "whole_brain", losing the small-volume half."""
    entity = {"local_id": "i1", "correction_scope": cited(
        "whole volume analyzed and a priori small volumes",
        "Correction was applied to the whole volume analyzed and to a priori small volumes.")}
    e = edit(sch, "InferenceSettings", entity, "correction_scope", "whole_brain")
    assert "warrant" in why(guards.refusals(e))


def test_one_value_does_not_replace_several(sch):
    """16701903 acquires MP-RAGE at TE 4.4 ms and FLASH at TE 5 ms."""
    entity = {"local_id": "acq", "echo_time_seconds": field([0.0044, 0.005])}
    e = edit(sch, "MRI", entity, "echo_time_seconds", 0.0044)
    assert "several values with one" in why(guards.refusals(e))


# ---------------------------------------------------------------------------- scope pairs


@pytest.mark.parametrize("scope,regions,refused", [
    ("roi", [], True),                  # 19996042: a restriction with nothing named
    ("roi", ["reg_x"], False),
    ("whole_brain", ["reg_x"], True),   # 11950456: whole-brain beside a named region
    ("whole_brain", [], False),
])
def test_a_scope_and_the_regions_beside_it_must_agree(sch, scope, regions, refused):
    entity = {"local_id": "i1", "correction_regions": list(regions)}
    e = edit(sch, "InferenceSettings", entity, "correction_scope", scope)
    assert bool(guards.refusals(e)) is refused


def test_a_whole_brain_analysis_is_not_given_regions_to_search(sch):
    entity = {"local_id": "a1", "spatial_scope": field("whole_brain"), "regions": []}
    e = edit(sch, "Analysis", entity, "regions", ["reg_sgacc"])
    assert "not restricted to a region" in why(guards.refusals(e))


# ----------------------------------------------------------------------------- references


def test_nothing_references_itself(sch):
    """27082610, 19942229: `inputs_from` resolved to the model being edited."""
    entity = {"local_id": "mod_adc"}
    e = edit(sch, "ModelEstimation", entity, "inputs_from", ["mod_adc"])
    assert "names the entity it is written on" in why(guards.refusals(e))


def test_repointing_may_not_orphan_the_terms_a_cell_names(sch):
    """19942229: `a_793_1` was moved to a model that does not reach `trm_group_r_nr`."""
    record = {"model_estimations": [
        {"local_id": "mod_a", "terms": [{"local_id": "t_a"}]},
        {"local_id": "mod_b", "terms": [{"local_id": "t_b"}]}]}
    entity = {"local_id": "a1", "model_estimation": "mod_a",
              "effect": {"cells": [{"term": "t_a"}]}}
    away = edit(sch, "Analysis", entity, "model_estimation", "mod_b", record)
    assert "does not reach" in why(guards.refusals(away))
    home = edit(sch, "Analysis", entity, "model_estimation", "mod_a", record)
    assert guards.refusals(home) == []


def test_every_guard_is_registered_and_named_once():
    names = [g.name for g in guards.GUARDS]
    assert len(names) == len(set(names))
    assert all(g.what and g.check for g in guards.GUARDS)


def test_a_repair_that_damages_the_record_is_reported(sch):
    """The check that did not exist while 665 findings accumulated across fifteen records."""
    from pondie.extraction.record.validate import Validator

    before = {"analyses": [{"local_id": "a1", "name": field("VBM")}]}
    after = {"analyses": [{"local_id": "a1", "name": field("VBM"),
                           "correction_scope": field("roi")}]}
    validator = Validator(sch, None)
    assert any("correction_scope" in line for line in validator.diff(before, after))
    assert validator.diff(before, before) == []


def test_a_class_is_swept_after_what_it_points_at(sch):
    """16508348: analyses were swept first, so four correctly named regions were refused for
    having no target, and the regions sweep ran afterwards."""
    from pondie.extraction import recall

    order = recall.sweep_order(sch, ["analyses", "groups", "inference_settings", "regions"])
    assert order.index("regions") < order.index("analyses")
    assert set(order) == {"analyses", "groups", "inference_settings", "regions"}


# --------------------------------------------------------------------------- the write path


def test_a_reference_gains_without_losing_what_was_there(sch):
    """12853571: `assessments` was replaced by four new ids, dropping `asm_caps` -- the CAPS
    total score, which is the one thing that correlation is of."""
    from pondie.extraction.record import edit as edit_module

    record = {"assessments": [{"local_id": "asm_caps", "name": field("CAPS total score")},
                              {"local_id": "asm_ies", "name": field("impact of event scale")}],
              "analyses": [{"local_id": "a1", "name": field("correlation"),
                            "assessments": ["asm_caps"]}]}
    log = edit_module.apply(sch, record, "Analysis", record["analyses"][0],
                            {"assessments": ["impact of event scale"]})
    assert record["analyses"][0]["assessments"] == ["asm_caps", "asm_ies"]
    assert log.changed


def test_a_reference_list_holds_each_target_once(sch):
    """23021615: four preprocessing names all resolved to one entity, written four times."""
    from pondie.extraction.record import edit as edit_module

    record = {"preprocessings": [{"local_id": "prp_vbm", "name": field("VBM pipeline")}],
              "model_estimations": [{"local_id": "m1", "name": field("group model")}]}
    edit_module.apply(sch, record, "ModelEstimation", record["model_estimations"][0],
                      {"preprocessing": ["VBM pipeline"] * 3})
    assert record["model_estimations"][0]["preprocessing"] == ["prp_vbm"]


def test_a_value_that_will_not_fit_its_slot_is_refused_not_coerced(sch):
    """28416565: `is_healthy` was given the word, and `bool("false")` is True."""
    from pondie.extraction.record import edit as edit_module

    record = {"groups": [{"local_id": "g1", "name": field("patients")}]}
    log = edit_module.apply(sch, record, "Group", record["groups"][0],
                            {"is_healthy": "mostly"})
    assert "is_healthy" not in record["groups"][0]
    assert any(r.slot == "is_healthy" for r in log.refused)

    edit_module.apply(sch, record, "Group", record["groups"][0], {"is_healthy": "no"})
    assert values.read(record["groups"][0]["is_healthy"]) is False


def test_references_are_written_before_the_values_that_guard_against_them(sch):
    """11950456: the scope landed beside a named region because the guard on the regions
    side ran while the scope was still unset, and the scope was set afterwards."""
    from pondie.extraction.record import edit as edit_module

    record = {"regions": [{"local_id": "reg_stg", "name": field("superior temporal gyrus"),
                           "definition_method": field("anatomical_a_priori")}],
              "inference_settings": [{"local_id": "i1"}]}
    log = edit_module.apply(sch, record, "InferenceSettings", record["inference_settings"][0],
                            {"correction_scope": "whole_brain",
                             "correction_regions": ["superior temporal gyrus"]})
    assert record["inference_settings"][0]["correction_regions"] == ["reg_stg"]
    assert "correction_scope" not in record["inference_settings"][0]
    assert any(r.slot == "correction_scope" for r in log.refused)


# ------------------------------------------------------------------------- orchestration


def test_repair_is_in_the_sequence_and_does_nothing_unless_asked(tmp_path):
    """It wants a GPU for the local models and a call for the adjudication. A run that asks
    for neither should pay for neither, so being in the sequence has to be free."""
    from pondie.extraction.models import Settings, StageName
    from pondie.extraction.stages import Repair, sequence

    settings = Settings(payloads=tmp_path, records=tmp_path, model="m")
    assert StageName.repair in [s.name for s in sequence(settings)]

    class Stub:
        study_id = "p"

    outcome = Repair().run(paper=Stub(), settings=settings, caller=None)
    assert outcome.skipped and "neither" in (outcome.reason or "")


def test_repair_reports_what_it_introduced(sch, tmp_path):
    """A finding the pass caused is a defect in the pass, not in the paper."""
    from pondie.extraction import repair as repair_pass

    record = {"analyses": [{"local_id": "a1", "name": field("VBM")}]}
    report = repair_pass.run(record, "", sch, study_id="p")
    assert report.introduced == []
    assert report.summary().startswith("wrote 0")


def test_only_a_settleable_contradiction_reaches_the_model(sch):
    """A case is adjudicable when it can be put as "choose one of these and quote the
    sentence". A dangling reference cannot, and is the largest group by count."""
    from pondie.extraction import repair as repair_pass

    contradictory = {
        "regions": [{"local_id": "reg_stg", "name": field("superior temporal gyrus"),
                     "definition_method": field("anatomical_a_priori")}],
        "inference_settings": [{"local_id": "i1",
                                "correction_scope": field("whole_brain"),
                                "correction_regions": ["reg_stg"]}]}
    cases = repair_pass.contradictions(contradictory, sch)
    assert len(cases) == 1
    assert cases[0].slot == "correction_scope"
    assert "superior temporal gyrus" in cases[0].question
    assert "roi" in cases[0].options

    consistent = {"inference_settings": [{"local_id": "i1",
                                          "correction_scope": field("whole_brain"),
                                          "correction_regions": []}]}
    assert repair_pass.contradictions(consistent, sch) == []
