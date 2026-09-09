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
    return {
        "extraction_status": "extracted",
        "value": value,
        "value_source": "reported",
        "evidence": evidence or {"status": "not_found"},
    }


def cited(value, quote):
    return field(
        value,
        {"status": "present", "sets": [{"source": "model_quote", "spans": [{"text": quote}]}]},
    )


def edit(sch, class_name, entity, slot, value, record=None):
    return edit_module.Edit(record=record or {}, entity=entity, slot=slot, value=value)


def why(refusals):
    return " ".join(r.why for r in refusals)


# --------------------------------------------------------------------------------- values


def test_an_edit_that_only_shortens_is_refused(sch):
    """22952599: "compared to traumatized controls." became "compared to traumatized"."""
    entity = {
        "local_id": "a1",
        "definition": cited(
            "Decreased gray matter volume in PTSD patients compared to traumatized controls.",
            "Decreased gray matter volume in PTSD patients compared to traumatized controls.",
        ),
    }
    e = edit(
        sch,
        "Analysis",
        entity,
        "definition",
        "Decreased gray matter volume in PTSD patients compared to traumatized",
    )
    assert "shortens" in why(edit_module.refusals(e))


def test_an_edit_that_extends_and_keeps_its_span_is_allowed(sch):
    """23021615: the restored full sentence was already the cited span."""
    quote = (
        "Relative to the non-PTSD group, the PTSD group showed reduced gray matter in "
        "the same large cluster comprising the sgACC, caudate, and hypothalamus "
        "( Fig. 3 A, B)."
    )
    entity = {
        "local_id": "a1",
        "definition": cited(
            "Relative to the non-PTSD group, the PTSD group showed reduced gray matter", quote
        ),
    }
    e = edit(
        sch,
        "Analysis",
        entity,
        "definition",
        "Relative to the non-PTSD group, the PTSD group showed reduced gray matter in "
        "the same large cluster comprising the sgACC, caudate, and hypothalamus.",
    )
    assert edit_module.refusals(e) == []


def test_an_edit_that_drops_the_warrant_is_refused(sch):
    """12853571: a cited, true "whole volume analyzed and a priori small volumes" was
    coerced to the bare enum "whole_brain", losing the small-volume half."""
    entity = {
        "local_id": "i1",
        "correction_scope": cited(
            "whole volume analyzed and a priori small volumes",
            "Correction was applied to the whole volume analyzed and to a priori small volumes.",
        ),
    }
    e = edit(sch, "InferenceSettings", entity, "correction_scope", "whole_brain")
    assert "warrant" in why(edit_module.refusals(e))


def test_one_value_does_not_replace_several(sch):
    """16701903 acquires MP-RAGE at TE 4.4 ms and FLASH at TE 5 ms."""
    entity = {"local_id": "acq", "echo_time_seconds": field([0.0044, 0.005])}
    e = edit(sch, "MRI", entity, "echo_time_seconds", 0.0044)
    assert "drops values" in why(edit_module.refusals(e))


# ---------------------------------------------------------------------------- scope pairs


@pytest.mark.parametrize(
    "scope,regions,refused",
    [
        ("roi", [], True),  # 19996042: a restriction with nothing named
        ("roi", ["reg_x"], False),
        ("whole_brain", ["reg_x"], True),  # 11950456: whole-brain beside a named region
        ("whole_brain", [], False),
    ],
)
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
    record = {
        "model_estimations": [
            {"local_id": "mod_a", "terms": [{"local_id": "t_a"}]},
            {"local_id": "mod_b", "terms": [{"local_id": "t_b"}]},
        ]
    }
    entity = {
        "local_id": "a1",
        "model_estimation": "mod_a",
        "effect": {"cells": [{"term": "t_a"}]},
    }
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
    after = {
        "analyses": [{"local_id": "a1", "name": field("VBM"), "correction_scope": field("roi")}]
    }
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

    record = {
        "assessments": [
            {"local_id": "asm_caps", "name": field("CAPS total score")},
            {"local_id": "asm_ies", "name": field("impact of event scale")},
        ],
        "analyses": [
            {"local_id": "a1", "name": field("correlation"), "assessments": ["asm_caps"]}
        ],
    }
    log = edit_module.apply(
        sch, record, "Analysis", record["analyses"][0], {"assessments": ["impact of event scale"]}
    )
    assert record["analyses"][0]["assessments"] == ["asm_caps", "asm_ies"]
    assert log.changed


def test_a_reference_list_holds_each_target_once(sch):
    """23021615: four preprocessing names all resolved to one entity, written four times."""
    from pondie.extraction.record import edit as edit_module

    record = {
        "preprocessings": [{"local_id": "prp_vbm", "name": field("VBM pipeline")}],
        "model_estimations": [{"local_id": "m1", "name": field("group model")}],
    }
    edit_module.apply(
        sch,
        record,
        "ModelEstimation",
        record["model_estimations"][0],
        {"preprocessing": ["VBM pipeline"] * 3},
    )
    assert record["model_estimations"][0]["preprocessing"] == ["prp_vbm"]


def test_a_value_that_will_not_fit_its_slot_is_refused_not_coerced(sch):
    """28416565: `is_healthy` was given the word, and `bool("false")` is True."""
    from pondie.extraction.record import edit as edit_module

    record = {"groups": [{"local_id": "g1", "name": field("patients")}]}
    log = edit_module.apply(sch, record, "Group", record["groups"][0], {"is_healthy": "mostly"})
    assert "is_healthy" not in record["groups"][0]
    assert any(r.slot == "is_healthy" for r in log.refused)

    edit_module.apply(sch, record, "Group", record["groups"][0], {"is_healthy": "no"})
    assert values.read(record["groups"][0]["is_healthy"]) is False


def test_references_are_written_before_the_values_that_guard_against_them(sch):
    """11950456: the scope landed beside a named region because the guard on the regions
    side ran while the scope was still unset, and the scope was set afterwards."""
    from pondie.extraction.record import edit as edit_module

    record = {
        "regions": [
            {
                "local_id": "reg_stg",
                "name": field("superior temporal gyrus"),
                "definition_method": field("anatomical_a_priori"),
            }
        ],
        "inference_settings": [{"local_id": "i1"}],
    }
    log = edit_module.apply(
        sch,
        record,
        "InferenceSettings",
        record["inference_settings"][0],
        {"correction_scope": "whole_brain", "correction_regions": ["superior temporal gyrus"]},
    )
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
    (records / f"{study}.extraction.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "local_id": "reg_stg",
                        "name": field("superior temporal gyrus"),
                        "definition_method": field("anatomical_a_priori"),
                    }
                ],
                "inference_settings": [
                    {
                        "local_id": "i1",
                        "correction_scope": field("whole_brain"),
                        "correction_regions": ["reg_stg"],
                    }
                ],
            }
        )
    )
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

    off = Settings(
        payloads=tmp_path / "pay", records=records, model="m", repair=False, adjudicate=False
    )
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
        return ModelReply(
            payload={
                "resolutions": [
                    {
                        "id": "inference_settings/i1/correction_scope",
                        "value": "roi",
                        "quote": "Images were acquired on a 3 T scanner.",
                    }
                ]
            },
            cost=Cost(),
        )

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

    record = {"analyses": [{"local_id": "a1", "name": field("VBM")}]}
    report = repair_pass.run(record, "", sch, study_id="p")
    assert report.introduced == []
    assert report.summary().startswith("wrote 0")


def test_only_a_settleable_contradiction_reaches_the_model(sch):
    """A case is adjudicable when it can be put as "choose one of these and quote the
    sentence". A dangling reference cannot, and is the largest group by count."""
    from pondie.extraction import repair as repair_pass

    contradictory = {
        "regions": [
            {
                "local_id": "reg_stg",
                "name": field("superior temporal gyrus"),
                "definition_method": field("anatomical_a_priori"),
            }
        ],
        "inference_settings": [
            {
                "local_id": "i1",
                "correction_scope": field("whole_brain"),
                "correction_regions": ["reg_stg"],
            }
        ],
    }
    cases = repair_pass.contradictions(contradictory, sch)
    assert len(cases) == 1
    assert cases[0].slot == "correction_scope"
    assert "superior temporal gyrus" in cases[0].question
    assert "roi" in cases[0].options

    consistent = {
        "inference_settings": [
            {"local_id": "i1", "correction_scope": field("whole_brain"), "correction_regions": []}
        ]
    }
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
    entity, why = edit_module.create(
        sch,
        record,
        "Region",
        {
            "name": "hippocampus",
            "definition_method": "anatomical_a_priori",
            "region_type": "anatomical",
        },
    )
    assert entity is not None, why
    assert entity["local_id"] == "reg_hippocampus"
    assert values.read(entity["definition_method"]) == "anatomical_a_priori"


def test_an_entity_that_could_not_be_valid_is_refused_by_the_slots_it_lacks(sch):
    """Analysis requires eight slots including `effect`, a nested structure no flat template
    carries. The refusal names them, so making analyses creatable is a matter of supplying
    what the message asks for rather than of changing a policy."""
    from pondie.extraction.record import edit as edit_module

    entity, why = edit_module.create(
        sch, {"analyses": []}, "Analysis", {"name": "PTSD < controls", "definition": "a contrast"}
    )
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


# ------------------------------------------------------------------- grounding what can be


def test_one_instrument_under_two_names_is_not_created_twice(sch):
    """12853571: "clinician-administered PTSD scale (CAPS)" minted a second copy of
    `asm_caps` ("CAPS total score"), and analyses then linked to the copy."""
    from pondie.extraction.record import edit as edit_module

    class Abbrev:
        def expand(self, short):
            return {
                "CAPS": "clinician-administered PTSD scale",
                "PTSD": "posttraumatic stress disorder",
            }.get(short)

    assert edit_module.same_entity(
        "CAPS total score", "clinician-administered PTSD scale (CAPS)", Abbrev()
    )
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
    assert values.cast(sch, "Group", "inclusion_criteria", ["right-handed", "aged 25-45"]) == [
        "right-handed",
        "aged 25-45",
    ]
    # all or nothing: one element that will not cast refuses the whole list
    assert values.cast(sch, "Group", "medications", ["fluoxetine", 42]) == ["fluoxetine", "42"]


def test_an_instrument_already_in_the_record_is_not_minted_again(sch):
    """The dedupe has to run before the id is minted: stems differ where labels agree, so
    "CAPS total score" and "clinician-administered PTSD scale (CAPS)" collided nowhere."""
    from pondie.extraction.record import edit as edit_module

    class Abbrev:
        def expand(self, short):
            return {
                "CAPS": "clinician-administered PTSD scale",
                "PTSD": "posttraumatic stress disorder",
            }.get(short)

    record = {"assessments": [{"local_id": "asm_caps", "name": field("CAPS total score")}]}
    entity, why = edit_module.create(
        sch,
        record,
        "Assessment",
        {"name": "clinician-administered PTSD scale (CAPS)"},
        "",
        Abbrev(),
    )
    assert entity is None
    assert "already holds" in why and "asm_caps" in why


def test_a_nested_slot_is_not_stringified(sch):
    """`Analysis.groups` holds AnalysisGroup objects; casting one would make it a string."""
    from pondie.extraction.record import edit as edit_module

    record = {"analyses": [{"local_id": "a1", "name": field("contrast")}]}
    edit_module.apply(
        sch, record, "Analysis", record["analyses"][0], {"groups": [{"group": "grp_ptsd"}]}
    )
    assert "groups" not in record["analyses"][0]


def test_the_analysis_directive_is_not_circular():
    """ "List every statistical analysis ... used by one of its statistical analyses" asks
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
        "regions": [
            {
                "local_id": "reg_stg",
                "name": field("superior temporal gyrus"),
                "definition_method": field("anatomical_a_priori"),
            }
        ],
        "inference_settings": [
            {
                "local_id": "i1",
                "correction_scope": field("whole_brain"),
                "correction_regions": ["reg_stg"],
            }
        ],
    }

    class Caller:
        def __call__(self, call, *, paper, stage):
            from pondie.extraction.models import Cost, ModelReply

            return ModelReply(
                payload={
                    "resolutions": [
                        {
                            "id": "inference_settings/i1/correction_scope",
                            "value": "roi",
                            "quote": "A region of interest analysis was performed.",
                        }
                    ]
                },
                cost=Cost(),
            )

    repair_pass.run(
        changed,
        "A region of interest analysis was performed.",
        sch,
        study_id="p",
        caller=Caller(),
        model="m",
    )
    assert changed["extraction_metadata"]["repaired_by"] == repair_pass.REPAIRER


def test_a_slot_of_a_subclass_is_written_against_that_subclass(sch):
    """An acquisition is an `MRI` by type designator, and `magnetic_field_strength_tesla` is
    a slot of that subclass. Written against the container's declared class it is an
    attribute `Acquisition` does not have -- three of three spot-checked papers."""
    from pondie.extraction.record import edit as edit_module

    designator = sch.type_designator("Acquisition")
    record = {
        "acquisitions": [{"local_id": "acq", designator: "MRI", "name": field("structural scan")}]
    }
    edit_module.apply(
        sch,
        record,
        "Acquisition",
        record["acquisitions"][0],
        {"magnetic_field_strength_tesla": "3"},
    )
    assert "magnetic_field_strength_tesla" in record["acquisitions"][0]


def test_the_type_designator_is_never_rewritten(sch):
    """19914045: the repair wrote `acquisition_type` through the ExtractedValue wrapper that
    every other native slot gets, leaving a dict in a slot declared `string`. The class was
    already resolved from that designator, so rewriting it re-types the entity after every
    other slot in the same proposal has been checked against the old class."""
    from pondie.extraction.record import edit as edit_module

    designator = sch.type_designator("Acquisition")
    entity = {"local_id": "acq", designator: "MRI", "name": field("structural scan")}
    record = {"acquisitions": [entity]}
    edit_module.apply(
        sch,
        record,
        "Acquisition",
        entity,
        {designator: "PET", "magnetic_field_strength_tesla": "3"},
    )
    assert entity[designator] == "MRI", "the designator must survive the edit untouched"
    assert "magnetic_field_strength_tesla" in entity, "the subclass slot must still land"


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

    record = {
        "measures": [
            {
                "local_id": "m",
                "family": {
                    "extraction_status": "extracted",
                    "value": "electrophysiology",
                    "value_source": "reported",
                    "evidence": {"status": "not_found"},
                },
            }
        ]
    }
    found = _Findings()
    rules.check_value_source_honesty(record, found)

    assert len(found.warnings) == 1, found.warnings
    assert "generated" in found.warnings[0][1]
    assert found.errors == [], "a reading may be right; this is a warning"


def test_a_generated_value_without_a_sentence_is_fine():
    """`generated` is the schema's own word for a value the pipeline reasoned to. Saying so
    is the fix, so it must not then be flagged."""
    from pondie.extraction.record import rules

    record = {
        "measures": [
            {
                "local_id": "m",
                "family": {
                    "extraction_status": "extracted",
                    "value": "electrophysiology",
                    "value_source": "generated",
                    "evidence": {"status": "not_found"},
                },
            }
        ]
    }
    found = _Findings()
    rules.check_value_source_honesty(record, found)
    assert found.warnings == []


def test_a_field_that_could_never_have_had_a_sentence_is_not_flagged():
    """Unfiltered this fired 1,978 times over 200 records, on table literals, on record
    addresses, and on values read off the method. `groundable` already knows all three."""
    from pondie.extraction.record import rules

    def claimed(value):
        return {
            "extraction_status": "extracted",
            "value": value,
            "value_source": "reported",
            "evidence": {"status": "not_found"},
        }

    found = _Findings()
    rules.check_value_source_honesty(
        {
            "tables": [
                {
                    "local_id": "t",
                    "caption": claimed("Table 1. Peaks"),
                    "source_table_analysis": claimed("3#1"),
                }
            ]
        },
        found,
    )
    assert found.warnings == [], found.warnings


def test_a_reasoned_value_claimed_as_reported_is_flagged():
    """`grounding` exempts these from scoring because a paper does not write down that a
    scope was `roi`. That is exactly why one asserted as `reported` with no sentence is
    worth seeing -- a conclusion wearing the label of a quotation. All four wrong values
    found by hand on this corpus were of that shape."""
    from pondie.extraction.record import rules

    def claimed(value):
        return {
            "extraction_status": "extracted",
            "value": value,
            "value_source": "reported",
            "evidence": {"status": "not_found"},
        }

    found = _Findings()
    rules.check_value_source_honesty(
        {"analyses": [{"local_id": "a", "spatial_scope": claimed("roi")}]}, found
    )
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
        "acquisitions": [
            {"local_id": "a1", "modality": field("fMRI")},
            {"local_id": "a2", "modality": field("sMRI")},
        ],
        "measures": [
            {"local_id": "m1", "family": field("functional_bold")},
            {"local_id": "m2", "family": field("structural_morphometry")},
        ],
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

    over = {"groups": [{"local_id": "g", "sex_distribution": [entry(12, 20), entry(14, 20)]}]}
    found = _Findings()
    rules.check_counts_add_up(over, found)
    assert len(found.warnings) == 1
    assert "sum to 26" in found.warnings[0][1] and "denominator of 20" in found.warnings[0][1]

    ok = {"groups": [{"local_id": "g", "sex_distribution": [entry(12, 20), entry(8, 20)]}]}
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

    record = {
        "groups": [{"local_id": "g", "enrolled_count": field(20), "acquired_count": field(24)}]
    }
    found = _Findings()
    rules.check_counts_add_up(record, found)
    assert len(found.warnings) == 1
    assert "subset of the one before" in found.warnings[0][1]

    ok = {
        "groups": [
            {
                "local_id": "g",
                "approached_count": field(40),
                "enrolled_count": field(24),
                "acquired_count": field(20),
            }
        ]
    }
    clean = _Findings()
    rules.check_counts_add_up(ok, clean)
    assert clean.warnings == []


def test_the_new_rules_are_registered():
    """A rule not in `RULES` runs nowhere."""
    from pondie.extraction.record import rules

    names = {rule.name for rule in rules.RULES}
    assert {"value_source_honesty", "modality_measures", "counts_add_up"} <= names


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

    assert (
        label_of({"local_id": "acq_fmri", "name": field("resting-state scan")})
        == "resting-state scan"
    )
    assert (
        label_of({"local_id": "mod_glm", "model_type": field("mixed effects")}) == "mixed effects"
    )
    assert label_of({"local_id": "dev_siemens_trio"}) == "siemens trio"


def test_a_one_word_derived_label_cannot_merge_two_entities():
    """`same_entity` needs two words in common, so "fmri" and "fmri" never merge two
    acquisitions on the strength of a modality they share."""
    from pondie.extraction.record.edit import same_entity

    assert not same_entity("fmri", "fmri")
    assert same_entity("siemens trio", "siemens trio scanner")


def test_a_value_the_pass_could_not_place_is_marked_generated(sch):
    """Marked `reported` regardless, the pass asserted the source said things it may not
    have. Nine of thirteen findings on the first paper where the proposer could write values
    at all were that pairing -- species, recruitment_method, is_healthy, spatial_scope, each
    `reported` with no sentence, which is the shape `check_value_source_honesty` exists to
    catch and which repair was producing itself.

    A value the locator could not cite but the paper plainly contains is still written --
    that is a locator failure, not an invention -- and it is still honestly `generated`.
    "advertisement" is under `_wrap`'s twenty-character floor, so no span is even looked
    for, which is exactly the case the document check has to keep.
    """
    from pondie.extraction.record import edit as edit_module

    record = {"groups": [{"local_id": "g", "name": field("patients")}]}
    entity = record["groups"][0]
    edit_module.apply(
        sch,
        record,
        "Group",
        entity,
        {"recruitment_method": "advertisement"},
        text="Methods. Participants answered an advertisement.",
    )

    written = entity["recruitment_method"]
    assert written["evidence"]["status"] == "not_found"
    assert written["value_source"] == "generated", "no sentence, so not reported"


def test_a_value_the_pass_did_place_stays_reported(sch):
    """The label follows the evidence, so a value with a span keeps its provenance."""
    from pondie.extraction.record import edit as edit_module

    quote = "Participants were recruited by newspaper advertisement in the local area."
    record = {"groups": [{"local_id": "g", "name": field("patients")}]}
    entity = record["groups"][0]
    edit_module.apply(
        sch, record, "Group", entity, {"recruitment_method": quote}, text=f"Methods. {quote}"
    )

    written = entity["recruitment_method"]
    assert written["evidence"]["status"] == "present"
    assert written["value_source"] == "reported"


def test_the_repair_stage_edits_against_the_schema_it_checks_against():
    """`Table.coordinate_space` exists in storage and not in extraction, so editing against
    one and validating against the other offered the proposer slots the record may not
    carry. Harmless while every template held only `local_id`; four invalid writes on the
    first paper once they did not."""
    import inspect

    from pondie.extraction import stages

    source = inspect.getsource(stages.Repair.run)
    assert "reader.load(EXTRACTION_SCHEMA)" in source
    assert "schema.STORAGE" not in source


def test_a_wrapper_is_resolved_to_what_it_wraps(sch):
    """`Group.is_healthy` declares `ExtractedBoolean`, so reading `range` directly gives a
    class name and concludes "reference". Two consumers did that independently: `nu_type`
    offered nothing but `local_id` for every class, and `cast` skipped the coercion branch
    its own docstring names."""
    from pondie.extraction.record.validate import EXTRACTION_SCHEMA
    from pondie.schema import reader

    schema = reader.load(EXTRACTION_SCHEMA)
    assert schema.value_ranges(schema.attributes("Group")["is_healthy"]) == ["boolean"]
    assert schema.value_ranges(schema.attributes("Group")["acquired_count"]) == ["integer"]
    assert schema.value_ranges(schema.attributes("Region")["name"]) == ["string"]
    # An open vocabulary keeps both branches.
    assert set(schema.value_ranges(schema.attributes("Condition")["condition_kind"])) == {
        "ConditionKind",
        "string",
    }


def test_a_string_answer_lands_in_the_type_its_slot_declares(sch):
    """The model answers in the paper's words: "true" for a boolean, "31" for a count. Eight
    of eleven findings on the first paper where the proposer could write values were this --
    `ExtractedBoolean.value must be a boolean, got str`."""
    from pondie.extraction.record.validate import EXTRACTION_SCHEMA
    from pondie.formats import values as value_tools
    from pondie.schema import reader

    schema = reader.load(EXTRACTION_SCHEMA)
    assert value_tools.cast(schema, "Group", "is_healthy", "true") is True
    assert value_tools.cast(schema, "Group", "acquired_count", "31") == 31
    assert value_tools.cast(schema, "Group", "age_mean", "24.6") == 24.6
    # And an answer that will not fit is still refused rather than coerced.
    assert value_tools.cast(schema, "Group", "acquired_count", "about twenty") is None
    assert value_tools.cast(schema, "Group", "is_healthy", "mostly") is None


def test_a_nested_object_gains_prose_it_was_missing(sch):
    """The template began offering `Task.conditions` before `apply` could write them, so the
    proposer was asked and its answer discarded. Prose it can place in the paper lands."""
    from pondie.extraction.record import edit as edit_module

    said = "the neutral condition showed household objects matched for visual complexity"
    record = {
        "tasks": [
            {
                "local_id": "tsk",
                "name": field("picture viewing"),
                "conditions": [{"local_id": "c1", "name": field("Neutral")}],
            }
        ]
    }
    entity = record["tasks"][0]
    edit_module.apply(
        sch,
        record,
        "Task",
        entity,
        {"conditions": [{"local_id": "c1", "description": said}]},
        text=f"Methods. In this study {said}, presented in blocks.",
    )

    written = entity["conditions"][0]["description"]
    assert values.read(written) == said
    assert written["evidence"]["status"] == "present"


def test_a_nested_classification_is_left_to_the_pass_that_read_the_paper(sch):
    """An enum term is vocabulary, not a quote, so it can never be placed -- which is the
    line to draw. `satisfy` classifies, having read the whole document, and got `Neutral`
    right on 16038771; this sweep, asked the same from a template, answered `fixation` for
    three picture-viewing conditions."""
    from pondie.extraction.record import edit as edit_module

    record = {
        "tasks": [
            {
                "local_id": "tsk",
                "name": field("picture viewing"),
                "conditions": [{"local_id": "c2", "name": field("Disgust")}],
            }
        ]
    }
    entity = record["tasks"][0]
    log = edit_module.apply(
        sch,
        record,
        "Task",
        entity,
        {"conditions": [{"local_id": "c2", "condition_kind": "fixation"}]},
        text="Methods. Disgust pictures were shown in thirty-second blocks.",
    )

    assert values.read(entity["conditions"][0].get("condition_kind")) is None
    assert any("nothing in the paper places this value" in r.why for r in log.refused)


def test_a_nested_object_keeps_what_it_already_had(sch):
    """An extracted value with a sentence behind it outranks a proposal without one, which
    is what stops a second pass quietly rewriting the first."""
    from pondie.extraction.record import edit as edit_module

    kept = cited("control_state", "a neutral condition served as the comparison")
    record = {
        "tasks": [
            {
                "local_id": "tsk",
                "name": field("picture viewing"),
                "conditions": [
                    {"local_id": "c1", "name": field("Neutral"), "condition_kind": kept}
                ],
            }
        ]
    }
    entity = record["tasks"][0]
    edit_module.apply(
        sch,
        record,
        "Task",
        entity,
        {"conditions": [{"local_id": "c1", "condition_kind": "task_state"}]},
        text="Methods. Something else entirely.",
    )

    assert values.read(entity["conditions"][0]["condition_kind"]) == "control_state"


def test_a_nested_object_the_record_does_not_have_is_not_invented(sch):
    """Completing what `satisfy` left thin is not the same as adding a condition the paper
    never ran, and this pass is in no position to tell the difference."""
    from pondie.extraction.record import edit as edit_module

    record = {
        "tasks": [
            {
                "local_id": "tsk",
                "name": field("picture viewing"),
                "conditions": [{"local_id": "c1", "name": field("Neutral")}],
            }
        ]
    }
    entity = record["tasks"][0]
    edit_module.apply(
        sch,
        record,
        "Task",
        entity,
        {"conditions": [{"local_id": "c9", "name": "Fixation", "condition_kind": "fixation"}]},
        text="Methods. A fixation cross was shown between blocks.",
    )

    assert [values.read(c["name"]) for c in entity["conditions"]] == ["Neutral"]


def test_a_structure_a_flat_reply_cannot_carry_is_still_left_alone(sch):
    """`Analysis.effect` nests cells nesting statistics. `recall.flat` is what separates the
    two cases, and it must keep saying no to this one."""
    from pondie.extraction.recall import flat

    assert flat(sch, "Condition")
    assert not flat(sch, "Effect")
