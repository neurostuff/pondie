"""What a repair pass may not write.

Each case is a regression that shipped, named by the paper it was found on. The guards were
written against these and nothing else, so a test that stops failing means a guard was
undone rather than that the case got easier.
"""

from __future__ import annotations

import json

import pytest

from pondie import schema
from pondie.extraction.record import edit as edit_module
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
    return edit_module.Edit(record=record or {}, entity=entity, slot=slot, value=value)


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
    assert "shortens" in why(edit_module.refusals(e))


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
    assert edit_module.refusals(e) == []


def test_an_edit_that_drops_the_warrant_is_refused(sch):
    """12853571: a cited, true "whole volume analyzed and a priori small volumes" was
    coerced to the bare enum "whole_brain", losing the small-volume half."""
    entity = {"local_id": "i1", "correction_scope": cited(
        "whole volume analyzed and a priori small volumes",
        "Correction was applied to the whole volume analyzed and to a priori small volumes.")}
    e = edit(sch, "InferenceSettings", entity, "correction_scope", "whole_brain")
    assert "warrant" in why(edit_module.refusals(e))


def test_one_value_does_not_replace_several(sch):
    """16701903 acquires MP-RAGE at TE 4.4 ms and FLASH at TE 5 ms."""
    entity = {"local_id": "acq", "echo_time_seconds": field([0.0044, 0.005])}
    e = edit(sch, "MRI", entity, "echo_time_seconds", 0.0044)
    assert "several values with one" in why(edit_module.refusals(e))


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
    assert bool(edit_module.refusals(e)) is refused


def test_a_whole_brain_analysis_is_not_given_regions_to_search(sch):
    entity = {"local_id": "a1", "spatial_scope": field("whole_brain"), "regions": []}
    e = edit(sch, "Analysis", entity, "regions", ["reg_sgacc"])
    assert "not restricted to a region" in why(edit_module.refusals(e))


# ----------------------------------------------------------------------------- references


def test_nothing_references_itself(sch):
    """27082610, 19942229: `inputs_from` resolved to the model being edited."""
    entity = {"local_id": "mod_adc"}
    e = edit(sch, "ModelEstimation", entity, "inputs_from", ["mod_adc"])
    assert "names the entity it is written on" in why(edit_module.refusals(e))


def test_repointing_may_not_orphan_the_terms_a_cell_names(sch):
    """19942229: `a_793_1` was moved to a model that does not reach `trm_group_r_nr`."""
    record = {"model_estimations": [
        {"local_id": "mod_a", "terms": [{"local_id": "t_a"}]},
        {"local_id": "mod_b", "terms": [{"local_id": "t_b"}]}]}
    entity = {"local_id": "a1", "model_estimation": "mod_a",
              "effect": {"cells": [{"term": "t_a"}]}}
    away = edit(sch, "Analysis", entity, "model_estimation", "mod_b", record)
    assert "does not reach" in why(edit_module.refusals(away))
    home = edit(sch, "Analysis", entity, "model_estimation", "mod_a", record)
    assert edit_module.refusals(home) == []


def test_every_guard_is_registered_and_documented():
    """The list is the specification: a reviewer reads it to know what stops a bad write,
    and `refusals` runs all of them so one write reports every reason it was rejected."""
    assert len(edit_module.GUARDS) == len(set(edit_module.GUARDS))
    assert all(check.__doc__ for check in edit_module.GUARDS)


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


@pytest.fixture
def corpus(tmp_path):
    """A real `Paper` on disk, not a stub.

    The stub this replaces had a `text()` method where `Paper.text` is a property returning a
    Path, so the stage's `paper.text()` raised TypeError on its first line of real work and
    the suite stayed green. A fake that duck-types the contract wrongly tests the fake.
    """
    from pondie import paths
    from pondie.extraction.models import Flavour, Paper

    root = tmp_path / "corpus"
    study = "p"
    text_path = paths.text(study, Flavour.pubget, root)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("Images were acquired on a 3 T scanner.\n", encoding="utf-8")
    records = tmp_path / "records"
    records.mkdir()
    # A record carrying a contradiction, so the adjudication path is actually reached: a
    # whole-brain correction naming the region it was restricted to.
    (records / f"{study}.extraction.json").write_text(json.dumps({
        "regions": [{"local_id": "reg_stg", "name": field("superior temporal gyrus"),
                     "definition_method": field("anatomical_a_priori")}],
        "inference_settings": [{"local_id": "i1",
                                "correction_scope": field("whole_brain"),
                                "correction_regions": ["reg_stg"]}]}))
    return Paper(study_id=study, root=root), records


def test_repair_runs_by_default_and_can_be_turned_off(tmp_path, corpus):
    """On by default, both halves, and independent -- so a machine with no GPU still gets
    the adjudication, and a run that wants neither can say so."""
    from pondie.extraction.models import Settings, StageName
    from pondie.extraction.stages import Repair, sequence

    paper, records = corpus
    default = Settings(payloads=tmp_path / "pay", records=records, model="m")
    assert default.repair and default.adjudicate
    assert StageName.repair in [s.name for s in sequence(default)]

    off = Settings(payloads=tmp_path / "pay", records=records, model="m",
                   repair=False, adjudicate=False)
    outcome = Repair().run(paper=paper, settings=off, caller=None)
    assert outcome.skipped and "neither" in (outcome.reason or "")


def test_the_stage_runs_against_a_real_paper(tmp_path, corpus):
    """The stage had never been executed: `paper.text()` raised TypeError immediately, the
    driver swallowed it, and every paper was reported failed."""
    from pondie.extraction.models import Cost, ModelReply, Settings
    from pondie.extraction.stages import Repair

    paper, records = corpus

    asked = []

    def caller(call, *, paper, stage):
        # A real ModelReply. The fake that returned a bare dict hid `reply.payload` being
        # read as `reply.body` -- an attribute of the MalformedReply exception, not of a
        # reply.
        asked.append(call)
        return ModelReply(payload={"resolutions": [
            {"id": "inference_settings/i1/correction_scope", "value": "roi",
             "quote": "Images were acquired on a 3 T scanner."}]}, cost=Cost())

    settings = Settings(payloads=tmp_path / "pay", records=records, model="m", repair=False)
    outcome = Repair().run(paper=paper, settings=settings, caller=caller)
    assert outcome.ok, outcome.reason
    assert asked, "the contradiction never reached the model"
    written = json.loads((records / "p.extraction.json").read_text())
    resolved = written["inference_settings"][0]["correction_scope"]
    assert values.read(resolved) == "roi"
    assert resolved["evidence"]["status"] == "present"


def test_repair_reports_what_it_introduced(sch, tmp_path):
    """A finding the pass caused is a defect in the pass, not in the paper."""
    from pondie.extraction import repair as repair_pass

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


def test_a_template_offers_the_slots_a_class_declares(sch):
    """`local_id` on every class, not only Analysis: without it the model can name an entity
    but never address one, so every correction had to be matched by label."""
    from pondie.extraction.recall import template_for

    template = template_for(sch, "Region")
    fields = template["regions"][0]
    assert list(fields)[0] == "local_id"
    assert "name" in fields
    # a closed vocabulary reaches the model as its values, not as a free string
    assert "atlas" in fields["definition_method"]
    # an open one keeps the enum branch rather than degrading to "string"
    assert "anatomical" in fields["region_type"]


def test_a_reference_slot_is_offered_by_name_not_as_a_nested_record(sch):
    from pondie.extraction.recall import template_for

    fields = template_for(sch, "Analysis")["analyses"][0]
    assert fields["regions"] == ["verbatim-string"]
    assert fields["measure"] == "verbatim-string"


def test_the_call_carries_a_directive_naming_what_to_list():
    """16508348: the same template and premise returned nothing without one, and three
    correct regions with it. A template says what an answer must look like, not what
    question it answers."""
    from pondie.extraction.recall import directive

    assert "brain region" in directive("Region")
    assert "statistical analysis" in directive("Analysis")
    assert "tied to an analysis" in directive("Group")


# -------------------------------------------------------------------------------- creation


def test_a_region_the_proposal_fully_specifies_is_created(sch):
    """The live proposer returns definition_method with the name, so a Region is
    constructible as valid -- hippocampus, on 16508348."""
    from pondie.extraction.record import edit as edit_module

    record = {"regions": []}
    entity, why = edit_module.create(sch, record, "Region", {
        "name": "hippocampus", "definition_method": "anatomical_a_priori",
        "region_type": "anatomical"})
    assert entity is not None, why
    assert entity["local_id"] == "reg_hippocampus"
    assert values.read(entity["definition_method"]) == "anatomical_a_priori"


def test_an_entity_that_could_not_be_valid_is_refused_by_the_slots_it_lacks(sch):
    """Analysis requires eight slots including `effect`, a nested structure no flat template
    carries. The refusal names them, so making analyses creatable is a matter of supplying
    what the message asks for rather than of changing a policy."""
    from pondie.extraction.record import edit as edit_module

    entity, why = edit_module.create(sch, {"analyses": []}, "Analysis",
                                     {"name": "PTSD < controls", "definition": "a contrast"})
    assert entity is None
    assert "table parse" in why or "effect" in why


def test_ids_nobody_chooses_are_not_chosen(sch):
    """A Table id comes from the parse, so an invented one would not match the table the
    parse produced. An Analysis id is minted only where there is no parse to take one
    from -- see `test_an_analysis_reported_only_in_prose_can_be_named`."""
    from pondie.extraction.record import ids

    assert ids.mint("Table", "Table 2", set()) is None
    assert ids.mint("Region", "left amygdala", set()) == "reg_left_amygdala"
    assert ids.mint("Region", "left amygdala", {"reg_left_amygdala"}) == "reg_left_amygdala_2"


def test_the_prompt_and_the_repair_pass_share_one_id_convention():
    """Two copies of a convention is one copy and one drift."""
    from pondie.extraction.record import ids

    table = ids.prefix_table()
    assert "reg_   Region" in table and "asm_   Assessment" in table
    assert all(prefix in table for prefix in ids.PREFIX.values())


def test_a_proposal_the_paper_does_not_support_is_not_written(sch):
    """Step 2 of the pass, which was documented and not wired: the checker was threaded
    through and never called, so proposals were written ungrounded."""
    from pondie.extraction.evidence import grounding

    class Reject:
        def score(self, claims):
            return [0.02] * len(claims)

    class Accept:
        def score(self, claims):
            return [0.95] * len(claims)

    proposals = [{"name": "hippocampus", "definition_method": "anatomical_a_priori"}]
    refused: list = []
    assert grounding.supported(proposals, "Region", "text", Reject(), 0.5, refused) == []
    assert refused and "does not support" in refused[0].why

    kept = grounding.supported(proposals, "Region", "text", Accept(), 0.5, [])
    assert kept == proposals


def test_an_entity_is_judged_by_what_it_is_not_by_its_name_alone(sch):
    """"group VBM t-tests" was scored unsupported for a paper saying "t-tests with
    statistical parametric mapping (SPM5)" -- the phrase was the extractor's, not the
    paper's."""
    from pondie.extraction.evidence import grounding

    said = grounding.describe("ModelEstimation", {
        "name": "group VBM t-tests", "model_family": "glm", "software": "SPM5"})
    assert "group VBM t-tests" in said
    assert "SPM5" in said and "glm" in said


# ------------------------------------------------------------------- grounding what can be


@pytest.mark.parametrize("slot,node,expected", [
    ("magnetic_strength", {"value": "3 T", "value_source": "reported"}, True),
    ("recruitment_method", {"value": "a clinic", "value_source": "reported"}, True),
    # a judgement about the method, not a thing the paper says
    ("spatial_scope", {"value": "whole_brain", "value_source": "reported"}, False),
    ("prespecification", {"value": "exploratory", "value_source": "reported"}, False),
    ("direction", {"value": "negative", "value_source": "reported"}, False),
    # the record's own marker for a value it produced rather than read
    ("definition", {"value": "a contrast", "value_source": "generated"}, False),
    # an address; the paper never says "reg_hippocampus"
    ("local_id", {"value": "reg_hippocampus", "value_source": "reported"}, False),
])
def test_only_a_field_a_sentence_could_support_is_grounded(slot, node, expected):
    from pondie.extraction.evidence import grounding

    assert grounding.groundable(slot, node) is expected


def test_a_span_that_supports_something_else_is_dropped_and_the_value_kept(sch):
    """A doubted citation is reported and left where it is.

    Deleting it destroyed 46% of all spans across a six-paper sample, 36% of them sentences
    containing the value verbatim. The pass this was ported from replaced a span only when a
    proposer found a better one, so total support could only rise."""
    from pondie.extraction.evidence import grounding

    class Reject:
        # Below `DOUBT_BELOW`, which the calibration put at 0.02: at the 0.2 first chosen
        # from three hand-read examples, 58% of true citations were flagged too.
        def score(self, claims):
            return [0.005] * len(claims)

    quote = "The authors thank the Dipartimento per i Rapporti Internazionali."
    record = {"acquisitions": [{"local_id": "acq", "magnetic_strength": cited("3 T", quote)}]}
    refused: list = []
    weak = grounding.review_spans(record, Reject(), refused)

    node = record["acquisitions"][0]["magnetic_strength"]
    assert values.read(node) == "3 T"
    assert node["evidence"]["status"] == "present", "the citation must survive the doubt"
    assert node["evidence"]["sets"][0]["spans"][0]["text"] == quote
    assert weak and weak[0][1] == 0.005
    assert any("left in place" in r.why for r in refused)


def test_a_span_that_does_support_its_value_is_left_alone(sch):
    from pondie.extraction.evidence import grounding

    class Accept:
        def score(self, claims):
            return [0.93] * len(claims)

    record = {"acquisitions": [{"local_id": "acq", "magnetic_strength": cited(
        "3 T", "Images were acquired on a 3 T scanner.")}]}
    grounding.review_spans(record, Accept(), [])
    assert record["acquisitions"][0]["magnetic_strength"]["evidence"]["status"] == "present"


def test_one_instrument_under_two_names_is_not_created_twice(sch):
    """12853571: "clinician-administered PTSD scale (CAPS)" minted a second copy of
    `asm_caps` ("CAPS total score"), and analyses then linked to the copy."""
    from pondie.extraction.record import edit as edit_module

    class Abbrev:
        def expand(self, short):
            return {"CAPS": "clinician-administered PTSD scale",
                    "PTSD": "posttraumatic stress disorder"}.get(short)

    assert edit_module.same_entity(
        "CAPS total score", "clinician-administered PTSD scale (CAPS)", Abbrev())
    assert not edit_module.same_entity("PTSD checklist", "PTSD symptom scale", Abbrev())


def test_an_analysis_reported_only_in_prose_can_be_named(sch):
    """16038682 reports three peaks in a sentence and has no coordinate table at all.
    Refusing to name such an analysis is refusing to record it."""
    from pondie.extraction.record import ids

    assert ids.mint("Analysis", "PTSD < controls", set()) == "ana_ptsd_controls"
    assert ids.mint("Table", "Table 2", set()) is None


def test_a_multivalued_slot_keeps_its_values_separate(sch):
    """`str()` of a list is the list's repr, so a slot given ["a", "b"] took the single
    string "['a', 'b']" -- one bogus value where two belong, legal enough to pass the
    validator."""
    assert values.cast(sch, "Group", "inclusion_criteria", ["right-handed", "aged 25-45"]) \
        == ["right-handed", "aged 25-45"]
    # all or nothing: one element that will not cast refuses the whole list
    assert values.cast(sch, "Group", "medications", ["fluoxetine", 42]) == ["fluoxetine", "42"]


def test_an_instrument_already_in_the_record_is_not_minted_again(sch):
    """The dedupe has to run before the id is minted: stems differ where labels agree, so
    "CAPS total score" and "clinician-administered PTSD scale (CAPS)" collided nowhere."""
    from pondie.extraction.record import edit as edit_module

    class Abbrev:
        def expand(self, short):
            return {"CAPS": "clinician-administered PTSD scale",
                    "PTSD": "posttraumatic stress disorder"}.get(short)

    record = {"assessments": [{"local_id": "asm_caps", "name": field("CAPS total score")}]}
    entity, why = edit_module.create(
        sch, record, "Assessment",
        {"name": "clinician-administered PTSD scale (CAPS)"}, "", Abbrev())
    assert entity is None
    assert "already holds" in why and "asm_caps" in why


def test_a_nested_slot_is_not_stringified(sch):
    """`Analysis.groups` holds AnalysisGroup objects; casting one would make it a string."""
    from pondie.extraction.record import edit as edit_module

    record = {"analyses": [{"local_id": "a1", "name": field("contrast")}]}
    edit_module.apply(sch, record, "Analysis", record["analyses"][0],
                      {"groups": [{"group": "grp_ptsd"}]})
    assert "groups" not in record["analyses"][0]


def test_the_analysis_directive_is_not_circular():
    """"List every statistical analysis ... used by one of its statistical analyses" asks
    the sweep to find analyses by their relation to analyses."""
    from pondie.extraction.recall import directive

    said = directive("Analysis")
    assert "tied to an analysis" not in said
    assert "tested comparison" in said
    assert "tied to an analysis" in directive("Region")


def test_a_repaired_record_says_it_was_repaired(sch):
    """A repaired record is not the record the extractor produced, and leaving the extractor
    metadata alone makes two records that differ look comparable."""
    from pondie.extraction import repair as repair_pass

    untouched = {"analyses": [{"local_id": "a1", "name": field("VBM")}]}
    repair_pass.run(untouched, "", sch, study_id="p")
    assert "repaired_by" not in untouched.get("extraction_metadata", {})

    changed = {
        "regions": [{"local_id": "reg_stg", "name": field("superior temporal gyrus"),
                     "definition_method": field("anatomical_a_priori")}],
        "inference_settings": [{"local_id": "i1", "correction_scope": field("whole_brain"),
                                "correction_regions": ["reg_stg"]}]}

    class Caller:
        def __call__(self, call, *, paper, stage):
            from pondie.extraction.models import Cost, ModelReply
            return ModelReply(payload={"resolutions": [
                {"id": "inference_settings/i1/correction_scope", "value": "roi",
                 "quote": "A region of interest analysis was performed."}]}, cost=Cost())

    repair_pass.run(changed, "A region of interest analysis was performed.", sch,
                    study_id="p", caller=Caller(), model="m")
    assert changed["extraction_metadata"]["repaired_by"] == repair_pass.REPAIRER


def test_a_slot_of_a_subclass_is_written_against_that_subclass(sch):
    """An acquisition is an `MRI` by type designator, and `magnetic_field_strength_tesla` is
    a slot of that subclass. Written against the container's declared class it is an
    attribute `Acquisition` does not have -- three of three spot-checked papers."""
    from pondie.extraction.record import edit as edit_module

    designator = sch.type_designator("Acquisition")
    record = {"acquisitions": [{"local_id": "acq", designator: "MRI",
                                "name": field("structural scan")}]}
    edit_module.apply(sch, record, "Acquisition", record["acquisitions"][0],
                      {"magnetic_field_strength_tesla": "3"})
    assert "magnetic_field_strength_tesla" in record["acquisitions"][0]


def test_an_edit_to_an_existing_entity_is_not_asked_to_justify_its_existence(sch):
    """26424424: the model returned every ROI of all three analyses by their exact ids, and
    the existence gate threw the proposals away before the edit was attempted -- 61 refusals
    and all but one of the links. An entity the extractor already found does not need its
    existence re-established."""
    from pondie.extraction import repair as repair_pass

    class RejectEverything:
        def score(self, claims):
            return [0.0] * len(claims)

    record = {
        "regions": [{"local_id": "r_ains", "name": field("left aINS"),
                     "definition_method": field("anatomical_a_priori")}],
        "analyses": [{"local_id": "an_group_gmv", "name": field("group contrast"),
                      "spatial_scope": field("roi"), "regions": []}]}

    class Proposer:
        def propose(self, sch_, class_name, premise, instruction):
            if class_name != "Analysis":
                return []
            return [{"local_id": "an_group_gmv", "regions": ["left aINS"]}]

        def ask(self, template, instruction, premise, what=""):
            return {}

    report = repair_pass.run(record, "", sch, study_id="p",
                             proposer=Proposer(), checker=RejectEverything())
    assert record["analyses"][0]["regions"] == ["r_ains"], report.refused


def test_the_type_designator_is_never_rewritten(sch):
    """19914045: the repair wrote `acquisition_type` through the ExtractedValue wrapper that
    every other native slot gets, leaving a dict in a slot declared `string`. The class was
    already resolved from that designator, so rewriting it re-types the entity after every
    other slot in the same proposal has been checked against the old class."""
    from pondie.extraction.record import edit as edit_module

    designator = sch.type_designator("Acquisition")
    entity = {"local_id": "acq", designator: "MRI", "name": field("structural scan")}
    record = {"acquisitions": [entity]}
    edit_module.apply(sch, record, "Acquisition", entity,
                      {designator: "PET", "magnetic_field_strength_tesla": "3"})
    assert entity[designator] == "MRI", "the designator must survive the edit untouched"
    assert "magnetic_field_strength_tesla" in entity, "the subclass slot must still land"


def test_a_starved_proposer_is_reported_not_silently_empty(sch):
    """Eight workers sharing one card OOMed every sweep on every full-length paper for an
    hour. Each recorded `0 written, 0 refused`, which reads as a pass with nothing to do --
    the stubs, whose premises were already under the floor, were the only ones that worked."""
    from pondie.extraction import recall
    from pondie.extraction import repair as repair_pass

    class Starving:
        def propose(self, sch, class_name, premise, instruction):
            raise recall.Starved(f"{class_name}: out of memory at a 6000-character premise")

    record = {"analyses": [{"local_id": "an", "name": field("a contrast")}]}
    report = repair_pass.run(record, "Methods. A contrast was computed.", sch,
                             study_id="p", proposer=Starving())
    assert report.refused, "a starved sweep must leave a trace a reviewer can see"
    assert any("out of memory" in r.why for r in report.refused)
    assert not report.written


def test_the_local_models_take_fewer_workers_than_the_stages_above(sch):
    """The gate is per process and per limit, so every worker thread waits on the same one.
    A gate built per call would let eight workers interleave inside one card."""
    from pondie.extraction import repair as repair_pass

    assert repair_pass.gate(2) is repair_pass.gate(2), "one semaphore per limit, per process"
    assert repair_pass.gate(2) is not repair_pass.gate(3)

    gate = repair_pass.gate(2)
    assert gate.acquire(blocking=False) and gate.acquire(blocking=False)
    assert not gate.acquire(blocking=False), "the third paper must wait"
    gate.release(), gate.release()


def test_the_default_is_one_paper_in_the_models_at_a_time(tmp_path):
    """Settings has to be right on beast without being told: the failure it prevents is
    silent, so a run that forgets the flag looks like a run with nothing to repair."""
    from pondie.extraction.models import Settings

    assert Settings(payloads=tmp_path, records=tmp_path, model="m").repair_workers == 1


def test_a_numeric_value_is_not_judged_against_prose(sch):
    """"echo time seconds is 0.004" against "TE = 4 ms" reads as unsupported however the
    paper wrote it. The pass this was ported from measured prose claims at 0.571 and numeric
    claims at 0.114 and excluded numerics for that reason; scoring them anyway took 100% of
    `echo_time_seconds`, `height_threshold_value` and `clusterwise_threshold_value`."""
    from pondie.extraction.evidence import grounding

    class Reject:
        def score(self, claims):
            return [0.01] * len(claims)

    record = {"acquisitions": [{"local_id": "acq", "echo_time_seconds": cited(
        "0.004", "Images were acquired with TE = 4 ms.")}]}
    refused: list = []
    assert grounding.review_spans(record, Reject(), refused) == []
    assert refused == [], "a number must not be scored against the prose that states it"


def test_a_claim_names_the_entity_and_where_in_it_the_leaf_sits(sch):
    """`effect.cells[0].level` read "level is African American." -- a fragment naming
    nothing. 44% of `level` spans were discarded on claims like that."""
    from pondie.extraction.evidence import grounding

    record = {"analyses": [{"local_id": "an", "name": field("AA versus CC smokers"),
                            "effect": {"cells": [{"level": field("African American")}]}}]}
    claim = grounding.claim_for(record, "analyses[0].effect.cells[0].level",
                                "African American")
    assert "AA versus CC smokers" in claim, claim
    assert "contrast cell 1" in claim, claim
    assert claim.endswith("the level is African American.")


def test_the_papers_own_abbreviation_is_written_beside_the_acronym(sch):
    """The value says "African American" and the sentence says "AA", so the checker entails
    nothing and scores 0.016. The paper defines the pair; `repair.run` already builds the
    table for `same_entity`."""
    from pondie.extraction.evidence import grounding

    class Store:
        def expand(self, short, paper=""):
            return {"AA": "African American", "CC": "Caucasian"}.get(short)

    span = "AA smokers showed greater activation than CC smokers"
    out = grounding.expand(span, Store(), "16759342")
    assert "AA (African American)" in out and "CC (Caucasian)" in out
    assert grounding.expand(span, None) == span


def _relocating(monkeypatch, sentences, old_score, new_score):
    """A proposer that offers `sentences`, and a checker that scores the incumbent
    `old_score` and the replacement `new_score`."""
    from pondie.extraction.evidence import relocate

    class Proposer:
        def ask(self, template, instruction, premise, what=""):
            tags = template["fields"][0]["field_id"]
            return {"fields": [{"field_id": tags[0], "supporting_sentences": sentences}]}

    class Checker:
        def __init__(self):
            self.calls = 0

        def score(self, claims):
            self.calls += 1
            return [new_score if self.calls == 1 else old_score] * len(claims)

    return relocate, Proposer(), Checker()


def test_a_better_sentence_replaces_the_one_it_beats(sch, monkeypatch):
    """The half that made the original numbers good: ask for the sentence that does support
    the value, and swap only on a strict improvement, so total support can only rise."""
    doc = "Methods. Images were acquired on a 3 T Siemens scanner. We thank the department."
    record = {"acquisitions": [{"local_id": "acq", "modality": cited(
        "3 T", "We thank the department.")}]}
    relocate, proposer, checker = _relocating(
        monkeypatch, ["Images were acquired on a 3 T Siemens scanner."], 0.04, 0.91)

    refused: list = []
    improved = relocate.relocate(record, doc, doc, [("acquisitions[0].modality", 0.04)],
                                 proposer, checker, refused)

    span = record["acquisitions"][0]["modality"]["evidence"]["sets"][0]["spans"][0]
    assert improved == ["acquisitions[0].modality"]
    assert span["text"] == "Images were acquired on a 3 T Siemens scanner."
    assert doc[span["start_char"]:span["end_char"]] == span["text"]


def test_a_replacement_that_is_no_better_is_not_made(sch, monkeypatch):
    """`new <= old` keeps the incumbent. Without the re-score the pass could make evidence
    worse while reporting that it repaired it."""
    doc = "Methods. Images were acquired on a 3 T Siemens scanner. Something else entirely."
    quote = "Images were acquired on a 3 T Siemens scanner."
    record = {"acquisitions": [{"local_id": "acq", "modality": cited("3 T", quote)}]}
    relocate, proposer, checker = _relocating(
        monkeypatch, ["Something else entirely."], 0.80, 0.30)

    refused: list = []
    improved = relocate.relocate(record, doc, doc, [("acquisitions[0].modality", 0.80)],
                                 proposer, checker, refused)

    assert improved == []
    assert record["acquisitions"][0]["modality"]["evidence"]["sets"][0]["spans"][0]["text"] \
        == quote
    assert any("no better sentence" in r.why for r in refused)


def test_a_field_with_no_citation_is_contested_even_though_it_scored_nothing(sch):
    """A `not_found` field says no sentence was ever located for it -- the clearest case for
    going to look. Reached only through a scoring fallback before, which left 96 fields
    across five papers that nothing asked about."""
    from pondie.extraction.evidence import relocate

    record = {"acquisitions": [{"local_id": "acq",
                                "modality": field("3 T")}]}
    rows = relocate.contested(record, weak=[])
    assert [r.path for r in rows] == ["acquisitions[0].modality"]
    assert rows[0].premise == ""


def test_every_leaf_of_a_projected_template_admits_nothing():
    """A grammar with no way to say "nothing here" does not decline -- it emits the best
    string it can. On 16759342, a paper declaring no arms whose every group the extractor
    left empty, a non-nullable schema filled `Group.arm` with 'smoking' and 'non-smoking'
    under `strict: true` AND `strict: false` alike; nullable answers null under both."""
    from pondie.extraction.recall import template_for
    from pondie.extraction.recall_server import json_schema_for
    from pondie.extraction.record.validate import EXTRACTION_SCHEMA

    # The extraction schema, because that is what the proposer projects. The storage schema
    # declares slots the extraction one does not, and a template built from it is not the
    # template a run sends.
    sch = reader.load(EXTRACTION_SCHEMA)

    def leaves(node, path="", bad=None):
        bad = [] if bad is None else bad
        if not isinstance(node, dict):
            return bad
        if "enum" in node:
            if None not in node["enum"]:
                bad.append(f"{path}: closed enum cannot express nothing")
            return bad
        kind = node.get("type")
        if isinstance(kind, str) and kind not in ("object", "array"):
            bad.append(f"{path}: {kind!r} is not nullable")
        if kind == "object":
            if node.get("required"):
                bad.append(f"{path}: required {node['required']}")
            for name, child in (node.get("properties") or {}).items():
                leaves(child, f"{path}.{name}", bad)
        for child in node.get("anyOf") or []:
            leaves(child, path, bad)
        if isinstance(node.get("items"), dict):
            leaves(node["items"], f"{path}[]", bad)
        return bad

    for class_name in ("Group", "Analysis", "Region", "InferenceSettings"):
        schema = json_schema_for(template_for(sch, class_name))
        assert leaves(schema, class_name) == [], leaves(schema, class_name)


def test_an_enum_carries_null_inside_itself(sch):
    """A type union cannot widen a closed `enum`, so nullability there is a second
    mechanism -- and one an enum branch that forgets it would silently lose."""
    from pondie.extraction.recall_server import json_schema_for

    schema = json_schema_for(["whole_brain", "roi"])
    assert None in schema["anyOf"][0]["enum"]


def test_the_server_proposer_asks_for_a_strict_grammar():
    """Strict was off while the invention above was blamed on it. The schema was the cause;
    a strict grammar over a nullable schema both guarantees parseable output and lets the
    model answer nothing."""
    from pondie.extraction.recall_server import NuExtractServer

    assert NuExtractServer()._strict is True


def test_the_served_proposer_is_the_default(tmp_path):
    """Chosen for the failure it ends rather than the speed: a free-running decoder that
    repeats a completed object until `max_tokens` parses to nothing and reads as the model
    declining to answer, and no prompt change fixes that from inside the process."""
    from pondie.extraction.models import Settings

    settings = Settings(payloads=tmp_path, records=tmp_path, model="m")
    assert settings.proposer_url.startswith("http")


def test_an_address_nothing_answers_falls_back_and_says_so(monkeypatch, capsys):
    """A run that quietly used the other proposer produces a different record, and nothing
    in the output distinguishes the two -- so the fallback is announced, not silent."""
    from pondie.extraction import repair as repair_pass
    from pondie.extraction.recall_server import NuExtractServer

    monkeypatch.setattr(NuExtractServer, "reachable", lambda self, timeout=3.0: False)
    monkeypatch.setattr("pondie.extraction.recall.NuExtract", lambda **kw: "LOCAL")
    monkeypatch.setattr("pondie.extraction.evidence.grounding.MiniCheck",
                        lambda *a, **k: "CHECKER")

    repair_pass.models.cache_clear()
    proposer, checker = repair_pass.models("", 1, "http://127.0.0.1:9/v1", "nu")

    assert proposer == "LOCAL" and checker == "CHECKER"
    assert "no server at" in capsys.readouterr().err
    repair_pass.models.cache_clear()


def test_a_reachable_server_is_used_without_loading_anything_locally(monkeypatch):
    """The point of serving it: ~5 GB of weights stay out of every worker."""
    from pondie.extraction import repair as repair_pass
    from pondie.extraction.recall_server import NuExtractServer

    monkeypatch.setattr(NuExtractServer, "reachable", lambda self, timeout=3.0: True)
    monkeypatch.setattr("pondie.extraction.recall.NuExtract",
                        lambda **kw: pytest.fail("must not load the in-process model"))
    monkeypatch.setattr("pondie.extraction.evidence.grounding.MiniCheck",
                        lambda *a, **k: "CHECKER")

    repair_pass.models.cache_clear()
    proposer, _checker = repair_pass.models("", 1, "http://127.0.0.1:8311/v1", "nu")
    assert isinstance(proposer, NuExtractServer)
    repair_pass.models.cache_clear()


def test_the_reachability_probe_asks_the_right_address():
    """`rsplit` on the completions URL left `/v1/chat/models`, which 404s -- so a live server
    read as absent and the run silently loaded the in-process model instead. The tests above
    stub `reachable`, so only this one can catch it."""
    from pondie.extraction.recall_server import NuExtractServer

    server = NuExtractServer(base_url="http://host:8311/v1")
    assert server._url == "http://host:8311/v1/chat/completions"
    assert server._base + "/models" == "http://host:8311/v1/models"

    trailing = NuExtractServer(base_url="http://host:8311/v1/")
    assert trailing._base + "/models" == "http://host:8311/v1/models"


class _DeadEngine:
    """A server whose engine has gone, then comes back after a restart."""

    def __init__(self, deaths=1):
        self.deaths, self.calls, self.restarts = deaths, 0, 0

    def post(self, _request):
        self.calls += 1
        if self.calls <= self.deaths:
            from pondie.extraction.recall_server import EngineDied
            raise EngineDied("EngineCore encountered an issue")
        return '{"groups": [{"local_id": "g1"}]}'


def _served(monkeypatch, engine, recovers=True):
    from pondie.extraction.recall_server import NuExtractServer

    server = NuExtractServer()
    monkeypatch.setattr(server, "_post", engine.post)
    def recover():
        engine.restarts += 1
        return recovers
    monkeypatch.setattr(server, "recover", recover)
    return server


def test_a_dead_engine_is_restarted_and_the_call_retried(monkeypatch):
    """An engine that dies at paper 300 would otherwise fail every paper after it."""
    engine = _DeadEngine(deaths=1)
    server = _served(monkeypatch, engine)

    payload = server.ask({"groups": [{"local_id": "string"}]}, "find groups", "Methods.")

    assert engine.restarts == 1, "must try to bring the server back"
    assert payload == {"groups": [{"local_id": "g1"}]}, "and retry the call"


def test_a_second_death_on_one_call_is_not_retried_again(monkeypatch):
    """Once is the server; twice on the same request is the request. Retrying further just
    takes the server down for every other paper in the run."""
    from pondie.extraction.recall_server import EngineDied

    engine = _DeadEngine(deaths=2)
    server = _served(monkeypatch, engine)

    with pytest.raises(EngineDied):
        server.ask({"groups": [{"local_id": "string"}]}, "find groups", "Methods.")
    assert engine.restarts == 1, "one recovery per call, not a loop"


def test_a_paper_that_killed_the_engine_is_named_and_skipped(sch, monkeypatch):
    """Logged and skipped: the record still gets the deterministic half of the pass, and the
    run does not spend another engine on it."""
    from pondie.extraction import repair as repair_pass
    from pondie.extraction.recall_server import EngineDied

    class Killer:
        def propose(self, *_a, **_k):
            raise EngineDied("EngineCore encountered an issue")

        def ask(self, *_a, **_k):
            raise EngineDied("EngineCore encountered an issue")

    repair_pass.POISON.discard("p_dead")
    record = {"analyses": [{"local_id": "an", "name": field("a contrast")}]}
    first = repair_pass.run(record, "Methods. A contrast.", sch, study_id="p_dead",
                            proposer=Killer())
    assert "p_dead" in repair_pass.POISON
    assert any("engine died" in r.why for r in first.refused)

    second = repair_pass.run(record, "Methods. A contrast.", sch, study_id="p_dead",
                             proposer=Killer())
    assert any("killed the proposer engine earlier" in r.why for r in second.refused)
    repair_pass.POISON.discard("p_dead")


def test_a_server_that_is_gone_counts_as_a_dead_engine(monkeypatch):
    """A server whose engine died answers 500; one that died outright refuses the
    connection. Only the first was caught, so killing the server produced `URLError:
    Connection refused`, no recovery was attempted, and the paper failed. `HTTPError`
    subclasses `URLError`, so the order of the clauses matters."""
    import urllib.error

    from pondie.extraction.recall_server import EngineDied, NuExtractServer

    server = NuExtractServer()
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            urllib.error.URLError("[Errno 111] Connection refused")))
    with pytest.raises(EngineDied):
        server._post({"model": "nu"})


def test_a_bad_request_is_still_not_a_dead_engine(monkeypatch):
    """A 400 must not trigger a restart: the server is fine and the request is not."""
    import io
    import urllib.error

    from pondie.extraction.recall_server import EngineDied, NuExtractServer

    def refuse(*_a, **_k):
        raise urllib.error.HTTPError("u", 400, "Bad Request", {},
                                     io.BytesIO(b'{"error":"malformed"}'))

    server = NuExtractServer()
    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(RuntimeError) as caught:
        server._post({"model": "nu"})
    assert not isinstance(caught.value, EngineDied)


def test_a_served_proposer_is_not_throttled_by_the_local_gate(sch):
    """The gate bounds this process's card. Holding it across a sweep that waits on the
    network serialises eight workers over a card the proposer never touches."""
    from pondie.extraction import repair as repair_pass

    held = []

    class Served:
        local = False

        def propose(self, *_a, **_k):
            held.append(repair_pass.gate(1).acquire(blocking=False))
            if held[-1]:
                repair_pass.gate(1).release()
            return []

        def ask(self, *_a, **_k):
            return {}

    record = {"analyses": [{"local_id": "an", "name": field("a contrast")}]}
    repair_pass.run(record, "Methods. A contrast.", sch, study_id="p", proposer=Served())

    assert held, "the sweep must have run"
    assert all(held), "the gate must be free while a served proposer is working"


def test_an_in_process_proposer_still_holds_the_gate(sch):
    """Unchanged where it matters: two of these in one card is the contention the gate was
    added for."""
    from pondie.extraction import repair as repair_pass

    held = []

    class Local:
        local = True

        def propose(self, *_a, **_k):
            got = repair_pass.gate(1).acquire(blocking=False)
            held.append(got)
            if got:
                repair_pass.gate(1).release()
            return []

        def ask(self, *_a, **_k):
            return {}

    record = {"analyses": [{"local_id": "an", "name": field("a contrast")}]}
    repair_pass.run(record, "Methods. A contrast.", sch, study_id="p2", proposer=Local())

    assert held, "the sweep must have run"
    assert not any(held), "an in-process proposer must hold the gate for the whole pass"


def test_the_checker_is_still_bounded_when_the_proposer_is_served(sch):
    """It is then the only thing in this process on a card, so it is what the gate is for."""
    from pondie.extraction import repair as repair_pass

    seen = []

    class Checker:
        def score(self, claims):
            seen.append(repair_pass.gate(1).acquire(blocking=False))
            if seen[-1]:
                repair_pass.gate(1).release()
            return [0.9] * len(claims)

    class Served:
        local = False

        def propose(self, *_a, **_k):
            return [{"local_id": "an", "name": "a better contrast"}]

        def ask(self, *_a, **_k):
            return {}

    record = {"analyses": [{"local_id": "an", "name": cited("a contrast", "A contrast.")}]}
    repair_pass.run(record, "Methods. A contrast was computed.", sch, study_id="p3",
                    proposer=Served(), checker=Checker())

    assert seen, "the checker must have been called"
    assert not any(seen), "and it must hold the gate while scoring"


def test_the_evidence_stage_asks_for_no_card(sch):
    """Network in one stage, card in another. A stage that spends tokens *and* holds a card
    can have neither half improved without paying for the other: when the retriever got 5.4x
    faster mid-run, 353 already-extracted papers could not take the improvement without
    re-running the quote pass, and the corpus ended up built two ways."""
    import inspect

    from pondie.extraction import stages

    source = inspect.getsource(stages.Evidence)
    assert "_retriever" not in source, "the locator belongs to repair now"
    assert "reranker" not in source, "and nothing here should touch one"
    assert "load_reranker" in inspect.getsource(stages.Repair)


def test_a_retrieved_sentence_must_also_beat_the_incumbent(sch):
    """It used to write a second span whenever it cleared its own gate, with no refusal
    recorded anywhere. As a candidate it obeys the one rule that cannot lower support."""
    from pondie.extraction.evidence import relocate as relocate_module

    doc = "Methods. Images were acquired on a 3 T Siemens scanner. We thank the department."
    good = "Images were acquired on a 3 T Siemens scanner."

    class Unit:
        text = good

    class Reranker:
        pass

    class Checker:
        def __init__(self):
            self.calls = 0

        def score(self, claims):
            self.calls += 1
            return [0.9 if self.calls == 1 else 0.04] * len(claims)

    monkey = {"unit": Unit()}
    relocate_module.retrieval = None  # not used; locate is patched below
    record = {"acquisitions": [{"local_id": "acq", "name": field("scan"),
                                "modality": cited("3 T", "We thank the department.")}]}

    import pondie.extraction.evidence.retrieval as retrieval_module
    original = retrieval_module.locate
    retrieval_module.locate = lambda *_a, **_k: monkey["unit"]
    try:
        refused: list = []
        improved = relocate_module.relocate(
            record, doc, doc, [("acquisitions[0].modality", 0.04)],
            None, Checker(), refused, reranker=Reranker(), units=[Unit()])
    finally:
        retrieval_module.locate = original

    assert "acquisitions[0].modality" in improved
    assert record["acquisitions"][0]["modality"]["evidence"]["sets"][0]["spans"][0]["text"] \
        == good


def test_the_retriever_alone_is_enough_to_run_the_pass(sch):
    """`relocate` used to require a proposer. With the locator moved into repair, a host
    with a reranker and no NuExtract still improves evidence."""
    import inspect

    from pondie.extraction.evidence import relocate as relocate_module

    source = inspect.getsource(relocate_module.relocate)
    assert "if proposer is not None else ()" in source, \
        "the proposer loop must be skippable"


class _Findings:
    def __init__(self):
        self.errors, self.warnings = [], []

    def error(self, path, message):
        self.errors.append((path, message))

    def warn(self, path, message):
        self.warnings.append((path, message))


def test_a_value_said_to_be_reported_needs_a_sentence():
    """11549754 carries `measures.family` = electrophysiology, reported, not_found -- on a
    BOLD fMRI study whose own identifiers read `mod_fmri` and `mea_neural_response`. The
    record contradicts itself and nothing said so."""
    from pondie.extraction.record import rules

    record = {"measures": [{"local_id": "m", "family": {
        "extraction_status": "extracted", "value": "electrophysiology",
        "value_source": "reported", "evidence": {"status": "not_found"}}}]}
    found = _Findings()
    rules.check_value_source_honesty(record, found)

    assert len(found.warnings) == 1, found.warnings
    assert "generated" in found.warnings[0][1]
    assert found.errors == [], "a reading may be right; this is a warning"


def test_a_generated_value_without_a_sentence_is_fine():
    """`generated` is the schema's own word for a value the pipeline reasoned to. Saying so
    is the fix, so it must not then be flagged."""
    from pondie.extraction.record import rules

    record = {"measures": [{"local_id": "m", "family": {
        "extraction_status": "extracted", "value": "electrophysiology",
        "value_source": "generated", "evidence": {"status": "not_found"}}}]}
    found = _Findings()
    rules.check_value_source_honesty(record, found)
    assert found.warnings == []


def test_a_field_that_could_never_have_had_a_sentence_is_not_flagged():
    """Unfiltered this fired 1,978 times over 200 records, on table literals, on record
    addresses, and on values read off the method. `groundable` already knows all three."""
    from pondie.extraction.record import rules

    def claimed(value):
        return {"extraction_status": "extracted", "value": value,
                "value_source": "reported", "evidence": {"status": "not_found"}}

    found = _Findings()
    rules.check_value_source_honesty(
        {"tables": [{"local_id": "t", "caption": claimed("Table 1. Peaks"),
                     "source_table_analysis": claimed("3#1")}]},
        found)
    assert found.warnings == [], found.warnings


def test_a_reasoned_value_claimed_as_reported_is_flagged():
    """`grounding` exempts these from scoring because a paper does not write down that a
    scope was `roi`. That is exactly why one asserted as `reported` with no sentence is
    worth seeing -- a conclusion wearing the label of a quotation. All four wrong values
    found by hand on this corpus were of that shape."""
    from pondie.extraction.record import rules

    def claimed(value):
        return {"extraction_status": "extracted", "value": value,
                "value_source": "reported", "evidence": {"status": "not_found"}}

    found = _Findings()
    rules.check_value_source_honesty(
        {"analyses": [{"local_id": "a", "spatial_scope": claimed("roi")}]}, found)
    assert len(found.warnings) == 1, found.warnings


def test_a_measure_the_scanner_could_not_have_produced():
    """The error is only in the pair: either field alone reads fine."""
    from pondie.extraction.record import rules

    record = {
        "acquisitions": [{"local_id": "a", "modality": field("fMRI")}],
        "measures": [{"local_id": "m", "family": field("electrophysiology")}],
    }
    found = _Findings()
    rules.check_modality_measures(record, found)
    assert len(found.warnings) == 1
    assert "electrophysiology" in found.warnings[0][1]

    ok = {
        "acquisitions": [{"local_id": "a", "modality": field("fMRI")}],
        "measures": [{"local_id": "m", "family": field("functional_bold")}],
    }
    clean = _Findings()
    rules.check_modality_measures(ok, clean)
    assert clean.warnings == []


def test_two_modalities_do_not_forbid_each_other_s_measures():
    """A study with both an sMRI and an fMRI acquisition produces both structural and BOLD
    measures. Unioning the forbidden sets would reject each for belonging to the other."""
    from pondie.extraction.record import rules

    record = {
        "acquisitions": [{"local_id": "a1", "modality": field("fMRI")},
                         {"local_id": "a2", "modality": field("sMRI")}],
        "measures": [{"local_id": "m1", "family": field("functional_bold")},
                     {"local_id": "m2", "family": field("structural_morphometry")}],
    }
    found = _Findings()
    rules.check_modality_measures(record, found)
    assert found.warnings == [], found.warnings


def test_a_breakdown_that_does_not_sum_to_its_own_denominator():
    """Checked against the denominator the entries themselves declare -- `Group` has no `n`,
    which is what an earlier version of this rule assumed, so it could never fire."""
    from pondie.extraction.record import rules

    def entry(count, denominator):
        return {"count": field(count), "denominator": field(denominator)}

    over = {"groups": [{"local_id": "g",
                        "sex_distribution": [entry(12, 20), entry(14, 20)]}]}
    found = _Findings()
    rules.check_counts_add_up(over, found)
    assert len(found.warnings) == 1
    assert "sum to 26" in found.warnings[0][1] and "denominator of 20" in found.warnings[0][1]

    ok = {"groups": [{"local_id": "g",
                      "sex_distribution": [entry(12, 20), entry(8, 20)]}]}
    clean = _Findings()
    rules.check_counts_add_up(ok, clean)
    assert clean.warnings == []

    # A paper reporting one category of two sums below the base by design. Every one of the
    # nine this fired on across the corpus was that, and none was an error.
    partial = {"groups": [{"local_id": "g", "sex_distribution": [entry(16, 30)]}]}
    quiet = _Findings()
    rules.check_counts_add_up(partial, quiet)
    assert quiet.warnings == []


def test_an_enrolment_funnel_that_grows():
    """approached, consented, enrolled, acquired -- each a subset of the one before, by the
    schema's own definitions, so the sequence cannot increase."""
    from pondie.extraction.record import rules

    record = {"groups": [{"local_id": "g", "enrolled_count": field(20),
                          "acquired_count": field(24)}]}
    found = _Findings()
    rules.check_counts_add_up(record, found)
    assert len(found.warnings) == 1
    assert "subset of the one before" in found.warnings[0][1]

    ok = {"groups": [{"local_id": "g", "approached_count": field(40),
                      "enrolled_count": field(24), "acquired_count": field(20)}]}
    clean = _Findings()
    rules.check_counts_add_up(ok, clean)
    assert clean.warnings == []


def test_the_new_rules_are_registered():
    """A rule not in `RULES` runs nowhere."""
    from pondie.extraction.record import rules

    names = {rule.name for rule in rules.RULES}
    assert {"value_source_honesty", "modality_measures", "counts_add_up"} <= names


def test_the_doubt_threshold_is_set_where_the_data_puts_it():
    """0.2 was chosen from three hand-read examples. Over 2,025 sentences the extraction
    model chose as a warrant and 2,025 it did not, it flags 58% of the true ones -- 83 on a
    single paper, a list nobody opens. At 0.02 three quarters of what it flags is genuinely
    unsupported and there is a quarter as much of it.

    Nothing is removed on this score at any threshold: dropping 90% of bad citations costs
    58% of good ones, and keeping 90% of good ones keeps two thirds of the bad."""
    from pondie.extraction.evidence import grounding

    assert grounding.DOUBT_BELOW == 0.02
    assert grounding.PRUNE_BELOW == grounding.DOUBT_BELOW, "the old name still resolves"


def test_a_minted_id_becomes_a_label_a_paper_could_contain():
    """`Measure`, `Acquisition`, `Device` and `ModelEstimation` declare no `name`, and only
    `Device` has no usable fallback either, so `label_of` returned the raw id for 336 of
    1,032 entities over eighty papers. Nothing matches `dev_siemens_trio` in a paper, so
    `resolve`, `same_entity` and the locator's entity bonus were all working blind."""
    from pondie.extraction.record.edit import from_local_id

    assert from_local_id("dev_siemens_trio") == "siemens trio"
    assert from_local_id("mea_cerebral_blood_flow") == "cerebral blood flow"
    assert from_local_id("mod_group_regression") == "group regression"
    assert from_local_id("acq_fmri") == "fmri"


def test_a_derived_label_too_short_to_be_a_name_is_refused():
    """`mea_fa` would offer "fa", which appears inside "factor" and "surface"."""
    from pondie.extraction.record.edit import from_local_id

    assert from_local_id("mea_fa") == ""
    assert from_local_id("dev_ge") == ""


def test_the_raw_id_survives_when_nothing_can_be_derived():
    """A label is better than no label for a report, and the id is what it always was."""
    from pondie.extraction.record.edit import label_of

    assert label_of({"local_id": "mea_fa"}) == "mea_fa"


def test_a_declared_name_still_wins_over_the_id():
    """Derivation is the fourth fallback, not a replacement: `acq_fmri` yields "fmri", the
    modality rather than what the paper calls that acquisition."""
    from pondie.extraction.record.edit import label_of

    assert label_of({"local_id": "acq_fmri", "name": field("resting-state scan")}) \
        == "resting-state scan"
    assert label_of({"local_id": "mod_glm", "model_type": field("mixed effects")}) \
        == "mixed effects"
    assert label_of({"local_id": "dev_siemens_trio"}) == "siemens trio"


def test_a_one_word_derived_label_cannot_merge_two_entities():
    """`same_entity` needs two words in common, so "fmri" and "fmri" never merge two
    acquisitions on the strength of a modality they share."""
    from pondie.extraction.record.edit import same_entity

    assert not same_entity("fmri", "fmri")
    assert same_entity("siemens trio", "siemens trio scanner")


def test_a_short_derived_label_does_not_match_inside_a_word():
    """The locator's entity bonus is word-bounded, which is what makes a derived label safe
    to hand it at all."""
    from pondie.extraction.evidence import retrieval

    units = ["A factor analysis of the surface data was performed.",
             "Images were acquired on a Siemens Trio scanner."]
    assert retrieval.entity_hits(units, "fa") == []
    assert retrieval.entity_hits(units, "siemens trio") == [1]


def test_a_replacement_that_is_part_of_what_it_replaces_is_refused(sch):
    """The proposer returns a quote cut mid-word often enough that this fired twice in
    fifteen changes over three papers: `hrf_model` lost "and temporally smoothed the data."
    to "and temporally smo", and `software` went from two complete spans to one ending at
    "2.1 x 2.1 x 7 mm,". Both scored well -- a prefix of a sentence says most of what the
    sentence says -- so the score cannot be what stops it."""
    from pondie.extraction.evidence import relocate as relocate_module

    full = "We used a delayed boxcar model and temporally smoothed the data."
    doc = f"Methods. {full} Results follow."

    class Proposer:
        def ask(self, template, instruction, premise, what=""):
            tags = template["fields"][0]["field_id"]
            return {"fields": [{"field_id": tags[0],
                                "supporting_sentences": ["We used a delayed boxcar model "
                                                         "and temporally smo"]}]}

        def propose(self, *_a, **_k):
            return []

    class Checker:
        def score(self, claims):
            return [0.9] * len(claims)      # the fragment would win on score

    record = {"model_estimations": [{"local_id": "m",
                                     "hrf_model": cited("delayed boxcar model", full)}]}
    refused: list = []
    improved = relocate_module.relocate(
        record, doc, doc, [("model_estimations[0].hrf_model", 0.01)],
        Proposer(), Checker(), refused)

    assert improved == [], "a fragment of the incumbent is not an improvement"
    assert record["model_estimations"][0]["hrf_model"]["evidence"]["sets"][0]["spans"][0][
        "text"] == full
    assert any("part of the sentence it would replace" in r.why for r in refused)


def test_a_genuinely_different_sentence_still_replaces(sch):
    """The guard is containment, not length: a better sentence may be shorter."""
    from pondie.extraction.evidence import relocate as relocate_module

    doc = ("Methods. Images were acquired on a 3 T Siemens scanner. "
           "The authors thank the department for its support.")

    class Proposer:
        def ask(self, template, instruction, premise, what=""):
            tags = template["fields"][0]["field_id"]
            return {"fields": [{"field_id": tags[0], "supporting_sentences":
                                ["Images were acquired on a 3 T Siemens scanner."]}]}

        def propose(self, *_a, **_k):
            return []

    class Checker:
        def __init__(self):
            self.calls = 0

        def score(self, claims):
            self.calls += 1
            return [0.9 if self.calls == 1 else 0.02] * len(claims)

    record = {"acquisitions": [{"local_id": "a", "modality": cited(
        "3 T", "The authors thank the department for its support.")}]}
    refused: list = []
    improved = relocate_module.relocate(
        record, doc, doc, [("acquisitions[0].modality", 0.02)],
        Proposer(), Checker(), refused)
    assert improved == ["acquisitions[0].modality"]


def test_a_list_of_numbers_is_a_number():
    """`str([6, 6, 6])` is "[6, 6, 6]", whose brackets fail the pattern, so a smoothing
    kernel, a voxel size and an echo-time pair all read as prose and were sent to a checker
    that cannot judge a number."""
    from pondie.extraction.evidence.grounding import is_numeric

    assert is_numeric([6, 6, 6])
    assert is_numeric([2.1, 2.1, 7])
    assert is_numeric("0.005")
    assert is_numeric(3)
    assert not is_numeric(["left aINS", "right OFC"])
    assert not is_numeric("6 mm FWHM")
    assert not is_numeric([])
    assert not is_numeric(True), "a flag is not a measurement"


def test_a_numeric_field_is_not_contested(sch):
    """On 26424424 the one citation that stated the value -- "smoothing of normalized gray
    matter (GM) tissue maps with a 6 mm 3 FWHM Gaussian filter." -- was replaced on score
    noise by "Data were preprocessed according to default toolbox settings: bias
    correction;", which does not mention smoothing at all."""
    from pondie.extraction.evidence import relocate as relocate_module

    stated = ("smoothing of normalized gray matter (GM) tissue maps with a "
              "6 mm 3 FWHM Gaussian filter.")
    record = {"preprocessings": [{"local_id": "prp", "smoothing_fwhm_mm": {
        "extraction_status": "extracted", "value": [6, 6, 6], "value_source": "reported",
        "evidence": {"status": "present", "sets": [{"spans": [{"text": stated}]}]}}}]}

    rows = relocate_module.contested(record, weak=[("preprocessings[0].smoothing_fwhm_mm",
                                                    0.01)])
    assert rows == [], "a number cannot be judged by entailment, so it is not contested"
