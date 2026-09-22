"""What a repair pass may not write.

Each case is a regression that shipped, named by the paper it was found on. The guards were
written against these and nothing else, so a test that stops failing means a guard was
undone rather than that the case got easier.
"""

from __future__ import annotations

import json

import pytest

from pondie import schema
from pondie.extraction.record import fix
from pondie.extraction.repair import guard as edit_module
from pondie.formats import values
from pondie.schema import reader


def unwarranted(value, evidence=None):
    """A reported value with `evidence.status: not_found` unless a caller supplies a set --
    the state a proposal arrives in, and what the guards are deciding about."""
    return {
        "extraction_status": "extracted",
        "value": value,
        "value_source": "reported",
        "evidence": evidence or {"status": "not_found"},
    }


def cited(value, quote):
    return unwarranted(
        value,
        {"status": "present", "sets": [{"source": "model_quote", "spans": [{"text": quote}]}]},
    )


def edit(storage_schema, class_name, entity, slot, value, record=None):
    return edit_module.Edit(record=record or {}, entity=entity, slot=slot, value=value)


def why(refusals):
    return " ".join(r.why for r in refusals)


# --------------------------------------------------------------------------------- values


def test_an_edit_that_only_shortens_is_refused(storage_schema):
    """22952599: "compared to traumatized controls." became "compared to traumatized"."""
    entity = {
        "local_id": "a1",
        "definition": cited(
            "Decreased gray matter volume in PTSD patients compared to traumatized controls.",
            "Decreased gray matter volume in PTSD patients compared to traumatized controls.",
        ),
    }
    e = edit(
        storage_schema,
        "Analysis",
        entity,
        "definition",
        "Decreased gray matter volume in PTSD patients compared to traumatized",
    )
    assert "shortens" in why(edit_module.refusals(e))


def test_an_edit_that_extends_and_keeps_its_span_is_allowed(storage_schema):
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
        storage_schema,
        "Analysis",
        entity,
        "definition",
        "Relative to the non-PTSD group, the PTSD group showed reduced gray matter in "
        "the same large cluster comprising the sgACC, caudate, and hypothalamus.",
    )
    assert edit_module.refusals(e) == []


def test_an_edit_that_drops_the_warrant_is_refused(storage_schema):
    """12853571: a cited, true "whole volume analyzed and a priori small volumes" was
    coerced to the bare enum "whole_brain", losing the small-volume half."""
    entity = {
        "local_id": "i1",
        "correction_scope": cited(
            "whole volume analyzed and a priori small volumes",
            "Correction was applied to the whole volume analyzed and to a priori small volumes.",
        ),
    }
    e = edit(storage_schema, "InferenceSettings", entity, "correction_scope", "whole_brain")
    assert "warrant" in why(edit_module.refusals(e))


def test_one_value_does_not_replace_several(storage_schema):
    """16701903 acquires MP-RAGE at TE 4.4 ms and FLASH at TE 5 ms."""
    entity = {"local_id": "acq", "echo_time_seconds": unwarranted([0.0044, 0.005])}
    e = edit(storage_schema, "MRI", entity, "echo_time_seconds", 0.0044)
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
def test_a_scope_and_the_regions_beside_it_must_agree(storage_schema, scope, regions, refused):
    entity = {"local_id": "i1", "correction_regions": list(regions)}
    e = edit(storage_schema, "InferenceSettings", entity, "correction_scope", scope)
    assert bool(edit_module.refusals(e)) is refused


def test_a_whole_brain_analysis_is_not_given_regions_to_search(storage_schema):
    entity = {"local_id": "a1", "spatial_scope": unwarranted("whole_brain"), "regions": []}
    e = edit(storage_schema, "Analysis", entity, "regions", ["reg_sgacc"])
    assert "not restricted to a region" in why(edit_module.refusals(e))


# ----------------------------------------------------------------------------- references


def test_nothing_references_itself(storage_schema):
    """27082610, 19942229: `inputs_from` resolved to the model being edited."""
    entity = {"local_id": "mod_adc"}
    e = edit(storage_schema, "ModelEstimation", entity, "inputs_from", ["mod_adc"])
    assert "names the entity it is written on" in why(edit_module.refusals(e))


def test_repointing_may_not_orphan_the_terms_a_cell_names(storage_schema):
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
    away = edit(storage_schema, "Analysis", entity, "model_estimation", "mod_b", record)
    assert "does not reach" in why(edit_module.refusals(away))
    home = edit(storage_schema, "Analysis", entity, "model_estimation", "mod_a", record)
    assert edit_module.refusals(home) == []


def test_every_guard_is_registered_and_documented():
    """The list is the specification: a reviewer reads it to know what stops a bad write,
    and `refusals` runs all of them so one write reports every reason it was rejected."""
    assert len(edit_module.GUARDS) == len(set(edit_module.GUARDS))
    assert all(check.__doc__ for check in edit_module.GUARDS)


def test_a_repair_that_damages_the_record_is_reported(storage_schema):
    """The check that did not exist while 665 findings accumulated across fifteen records."""
    from pondie.extraction.record.validate import Validator

    before = {"analyses": [{"local_id": "a1", "name": unwarranted("VBM")}]}
    after = {
        "analyses": [
            {"local_id": "a1", "name": unwarranted("VBM"), "correction_scope": unwarranted("roi")}
        ]
    }
    validator = Validator(storage_schema, None)
    assert any("correction_scope" in line for line in validator.diff(before, after))
    assert validator.diff(before, before) == []


def test_a_class_is_swept_after_what_it_points_at(storage_schema):
    """16508348: analyses were swept first, so four correctly named regions were refused for
    having no target, and the regions sweep ran afterwards."""
    from pondie.extraction.repair import propose as recall

    order = recall.sweep_order(
        storage_schema, ["analyses", "groups", "inference_settings", "regions"]
    )
    assert order.index("regions") < order.index("analyses")
    assert set(order) == {"analyses", "groups", "inference_settings", "regions"}


# --------------------------------------------------------------------------- the write path


def test_a_reference_gains_without_losing_what_was_there(storage_schema):
    """12853571: `assessments` was replaced by four new ids, dropping `asm_caps` -- the CAPS
    total score, which is the one thing that correlation is of."""
    from pondie.extraction.repair import guard as edit_module

    record = {
        "assessments": [
            {"local_id": "asm_caps", "name": unwarranted("CAPS total score")},
            {"local_id": "asm_ies", "name": unwarranted("impact of event scale")},
        ],
        "analyses": [
            {"local_id": "a1", "name": unwarranted("correlation"), "assessments": ["asm_caps"]}
        ],
    }
    log = edit_module.apply(
        storage_schema,
        record,
        "Analysis",
        record["analyses"][0],
        {"assessments": ["impact of event scale"]},
    )
    assert record["analyses"][0]["assessments"] == ["asm_caps", "asm_ies"]
    assert log.changed


def test_a_reference_list_holds_each_target_once(storage_schema):
    """23021615: four preprocessing names all resolved to one entity, written four times."""
    from pondie.extraction.repair import guard as edit_module

    record = {
        "preprocessings": [{"local_id": "prp_vbm", "name": unwarranted("VBM pipeline")}],
        "model_estimations": [{"local_id": "m1", "name": unwarranted("group model")}],
    }
    edit_module.apply(
        storage_schema,
        record,
        "ModelEstimation",
        record["model_estimations"][0],
        {"preprocessing": ["VBM pipeline"] * 3},
    )
    assert record["model_estimations"][0]["preprocessing"] == ["prp_vbm"]


def test_a_value_that_will_not_fit_its_slot_is_refused_not_coerced(storage_schema):
    """28416565: a boolean slot was given the word, and `bool("false")` is True.

    Was `Group.is_healthy` until that slot became derived and left the extraction schema;
    `tfce_used` is the same shape and still asked for."""
    from pondie.extraction.repair import guard as edit_module

    record = {"groups": [{"local_id": "g1", "name": unwarranted("patients")}]}
    log = edit_module.apply(
        storage_schema, record, "Group", record["groups"][0], {"is_healthy": "mostly"}
    )
    assert "is_healthy" not in record["groups"][0]
    assert any(r.slot == "is_healthy" for r in log.refused)

    edit_module.apply(storage_schema, record, "Group", record["groups"][0], {"is_healthy": "no"})
    assert values.read(record["groups"][0]["is_healthy"]) is False


def test_references_are_written_before_the_values_that_guard_against_them(storage_schema):
    """11950456: the scope landed beside a named region because the guard on the regions
    side ran while the scope was still unset, and the scope was set afterwards."""
    from pondie.extraction.repair import guard as edit_module

    record = {
        "regions": [
            {
                "local_id": "reg_stg",
                "name": unwarranted("superior temporal gyrus"),
                "definition_method": unwarranted("anatomical_a_priori"),
            }
        ],
        "inference_settings": [{"local_id": "i1"}],
    }
    log = edit_module.apply(
        storage_schema,
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
                        "name": unwarranted("superior temporal gyrus"),
                        "definition_method": unwarranted("anatomical_a_priori"),
                    }
                ],
                "inference_settings": [
                    {
                        "local_id": "i1",
                        "correction_scope": unwarranted("whole_brain"),
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


def test_repair_reports_what_it_introduced(storage_schema, tmp_path):
    """A finding the pass caused is a defect in the pass, not in the paper."""
    from pondie.extraction import repair as repair_pass

    record = {"analyses": [{"local_id": "a1", "name": unwarranted("VBM")}]}
    report = repair_pass.run(record, "", storage_schema, study_id="p")
    assert report.introduced == []
    assert report.summary().startswith("wrote 0")


def test_only_a_settleable_contradiction_reaches_the_model(storage_schema):
    """A case is adjudicable when it can be put as "choose one of these and quote the
    sentence". A dangling reference cannot, and is the largest group by count."""
    from pondie.extraction import repair as repair_pass

    contradictory = {
        "regions": [
            {
                "local_id": "reg_stg",
                "name": unwarranted("superior temporal gyrus"),
                "definition_method": unwarranted("anatomical_a_priori"),
            }
        ],
        "inference_settings": [
            {
                "local_id": "i1",
                "correction_scope": unwarranted("whole_brain"),
                "correction_regions": ["reg_stg"],
            }
        ],
    }
    cases = repair_pass.contradictions(contradictory, storage_schema)
    assert len(cases) == 1
    assert cases[0].slot == "correction_scope"
    assert "superior temporal gyrus" in cases[0].question
    assert "roi" in cases[0].options

    consistent = {
        "inference_settings": [
            {
                "local_id": "i1",
                "correction_scope": unwarranted("whole_brain"),
                "correction_regions": [],
            }
        ]
    }
    assert repair_pass.contradictions(consistent, storage_schema) == []


def test_a_template_offers_the_slots_a_class_declares(storage_schema):
    """`local_id` on every class, not only Analysis: without it the model can name an entity
    but never address one, so every correction had to be matched by label."""
    from pondie.extraction.repair.propose import template_for

    template = template_for(storage_schema, "Region")
    fields = template["regions"][0]
    assert list(fields)[0] == "local_id"
    assert "name" in fields
    # a closed vocabulary reaches the model as its values, not as a free string
    assert "atlas" in fields["definition_method"]
    # an open one keeps the enum branch rather than degrading to "string"
    assert "anatomical" in fields["region_type"]


def test_a_reference_slot_is_offered_by_name_not_as_a_nested_record(storage_schema):
    from pondie.extraction.repair.propose import template_for

    fields = template_for(storage_schema, "Analysis")["analyses"][0]
    assert fields["regions"] == ["verbatim-string"]
    assert fields["measure"] == "verbatim-string"


def test_the_call_carries_a_directive_naming_what_to_list():
    """16508348: the same template and premise returned nothing without one, and three
    correct regions with it. A template says what an answer must look like, not what
    question it answers."""
    from pondie.extraction.repair.propose import directive

    assert "brain region" in directive("Region")
    assert "statistical analysis" in directive("Analysis")
    assert "tied to an analysis" in directive("Group")


# -------------------------------------------------------------------------------- creation


def test_a_region_the_proposal_fully_specifies_is_created(storage_schema):
    """The live proposer returns definition_method with the name, so a Region is
    constructible as valid -- hippocampus, on 16508348."""
    from pondie.extraction.repair import guard as edit_module

    record = {"regions": []}
    entity, why = edit_module.create(
        storage_schema,
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


def test_an_acquisition_is_minted_with_the_subclass_its_modality_names(storage_schema):
    """A proposal carries the modality and nothing else, because `acquisition_type` is
    deterministic -- the schema says it is "derived by the mapper from the modality value's
    `instantiates`, never extracted" and the extraction projection drops it.

    Counting it as a required slot refused every acquisition the sweep proposed. Deriving it
    is what makes the minted entity the right class: without the designator an MEG record
    resolves to the base `Acquisition`, and `apply` then checks each later write against the
    base class, where every modality-specific parameter is undeclared.
    """
    from pondie.extraction.repair import guard as edit_module

    for modality, expected in (("MEG", "OtherModality"), ("SPECT", "OtherModality"),
                               ("fMRI", "MRI"), ("PET", "PET")):
        entity, why = edit_module.create(
            storage_schema, {"acquisitions": []}, "Acquisition",
            {"name": f"{modality} scan", "modality": modality},
        )
        assert entity is not None, why
        # Bare, not wrapped: the designator is a plain string, and the record states it.
        assert entity["acquisition_type"] == expected
        assert values.read(entity["modality"]) == modality


def test_a_region_keeps_a_definition_method_the_vocabulary_does_not_cover(storage_schema):
    """`definition_method` is required and open, so an unlisted method costs the word rather
    than the Region. Closed, `create` answered "Region would be missing definition_method"
    and the name, region_type and description the proposal also carried went with it."""
    from pondie.extraction.repair import guard as edit_module

    entity, why = edit_module.create(
        storage_schema, {"regions": []}, "Region",
        {"name": "left amygdala", "definition_method": "hand drawn by an expert"},
    )
    assert entity is not None, why
    assert values.read(entity["definition_method"]) == "hand drawn by an expert"


def test_an_entity_that_could_not_be_valid_is_refused_by_the_slots_it_lacks(storage_schema):
    """Analysis requires eight slots including `effect`, a nested structure no flat template
    carries. The refusal names them, so making analyses creatable is a matter of supplying
    what the message asks for rather than of changing a policy."""
    from pondie.extraction.repair import guard as edit_module

    entity, why = edit_module.create(
        storage_schema,
        {"analyses": []},
        "Analysis",
        {"name": "PTSD < controls", "definition": "a contrast"},
    )
    assert entity is None
    assert "table parse" in why or "effect" in why


def test_ids_nobody_chooses_are_not_chosen(storage_schema):
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
    # Every class a model mints an id for, and only those. A `DERIVED` class has its ids
    # assigned before the prompt is built and handed over by name, so printing its prefix
    # would read as permission to invent one that points at nothing.
    assert all(prefix in table for name, prefix in ids.PREFIX.items() if name not in ids.DERIVED)
    assert all(name not in table for name in ids.DERIVED)


# ------------------------------------------------------------------- grounding what can be


def test_one_instrument_under_two_names_is_not_created_twice(storage_schema):
    """12853571: "clinician-administered PTSD scale (CAPS)" minted a second copy of
    `asm_caps` ("CAPS total score"), and analyses then linked to the copy."""
    from pondie.extraction.repair import guard as edit_module

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


def test_an_analysis_reported_only_in_prose_can_be_named(storage_schema):
    """16038682 reports three peaks in a sentence and has no coordinate table at all.
    Refusing to name such an analysis is refusing to record it."""
    from pondie.extraction.record import ids

    assert ids.mint("Analysis", "PTSD < controls", set()) == "ana_ptsd_controls"
    assert ids.mint("Table", "Table 2", set()) is None


def test_a_multivalued_slot_keeps_its_values_separate(storage_schema):
    """`str()` of a list is the list's repr, so a slot given ["a", "b"] took the single
    string "['a', 'b']" -- one bogus value where two belong, legal enough to pass the
    validator."""
    assert values.cast(
        storage_schema, "Group", "inclusion_criteria", ["right-handed", "aged 25-45"]
    ) == [
        "right-handed",
        "aged 25-45",
    ]
    # all or nothing: one element that will not cast refuses the whole list
    assert values.cast(storage_schema, "Group", "medications", ["fluoxetine", 42]) == [
        "fluoxetine",
        "42",
    ]


def test_an_instrument_already_in_the_record_is_not_minted_again(storage_schema):
    """The dedupe has to run before the id is minted: stems differ where labels agree, so
    "CAPS total score" and "clinician-administered PTSD scale (CAPS)" collided nowhere."""
    from pondie.extraction.repair import guard as edit_module

    class Abbrev:
        def expand(self, short):
            return {
                "CAPS": "clinician-administered PTSD scale",
                "PTSD": "posttraumatic stress disorder",
            }.get(short)

    record = {"assessments": [{"local_id": "asm_caps", "name": unwarranted("CAPS total score")}]}
    entity, why = edit_module.create(
        storage_schema,
        record,
        "Assessment",
        {"name": "clinician-administered PTSD scale (CAPS)"},
        "",
        Abbrev(),
    )
    assert entity is None
    assert "already holds" in why and "asm_caps" in why


def test_a_nested_slot_is_not_stringified(storage_schema):
    """`Analysis.groups` holds AnalysisGroup objects; casting one would make it a string."""
    from pondie.extraction.repair import guard as edit_module

    record = {"analyses": [{"local_id": "a1", "name": unwarranted("contrast")}]}
    edit_module.apply(
        storage_schema,
        record,
        "Analysis",
        record["analyses"][0],
        {"groups": [{"group": "grp_ptsd"}]},
    )
    assert "groups" not in record["analyses"][0]


def test_the_analysis_directive_is_not_circular():
    """ "List every statistical analysis ... used by one of its statistical analyses" asks
    the sweep to find analyses by their relation to analyses."""
    from pondie.extraction.repair.propose import directive

    said = directive("Analysis")
    assert "tied to an analysis" not in said
    assert "tested comparison" in said
    assert "tied to an analysis" in directive("Region")


def test_a_repaired_record_says_it_was_repaired(storage_schema):
    """A repaired record is not the record the extractor produced, and leaving the extractor
    metadata alone makes two records that differ look comparable."""
    from pondie.extraction import repair as repair_pass

    untouched = {"analyses": [{"local_id": "a1", "name": unwarranted("VBM")}]}
    repair_pass.run(untouched, "", storage_schema, study_id="p")
    assert "repaired_by" not in untouched.get("extraction_metadata", {})

    changed = {
        "regions": [
            {
                "local_id": "reg_stg",
                "name": unwarranted("superior temporal gyrus"),
                "definition_method": unwarranted("anatomical_a_priori"),
            }
        ],
        "inference_settings": [
            {
                "local_id": "i1",
                "correction_scope": unwarranted("whole_brain"),
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
        storage_schema,
        study_id="p",
        caller=Caller(),
        model="m",
    )
    assert changed["extraction_metadata"]["repaired_by"] == repair_pass.REPAIRER


def test_a_slot_of_a_subclass_is_written_against_that_subclass(storage_schema):
    """An acquisition is an `MRI` by type designator, and `magnetic_field_strength_tesla` is
    a slot of that subclass. Written against the container's declared class it is an
    attribute `Acquisition` does not have -- three of three spot-checked papers."""
    from pondie.extraction.repair import guard as edit_module

    designator = storage_schema.type_designator("Acquisition")
    record = {
        "acquisitions": [
            {"local_id": "acq", designator: "MRI", "name": unwarranted("structural scan")}
        ]
    }
    edit_module.apply(
        storage_schema,
        record,
        "Acquisition",
        record["acquisitions"][0],
        {"magnetic_field_strength_tesla": "3"},
    )
    assert "magnetic_field_strength_tesla" in record["acquisitions"][0]


def test_the_type_designator_is_never_rewritten(storage_schema):
    """19914045: the repair wrote `acquisition_type` through the ExtractedValue wrapper that
    every other native slot gets, leaving a dict in a slot declared `string`. The class was
    already resolved from that designator, so rewriting it re-types the entity after every
    other slot in the same proposal has been checked against the old class."""
    from pondie.extraction.repair import guard as edit_module

    designator = storage_schema.type_designator("Acquisition")
    entity = {"local_id": "acq", designator: "MRI", "name": unwarranted("structural scan")}
    record = {"acquisitions": [entity]}
    edit_module.apply(
        storage_schema,
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
    """A paper does not write down that a scope was `roi`, so a checker asked for the
    sentence behind one is asking for a sentence that does not exist. That is exactly why
    one asserted as `reported` with no sentence is worth seeing -- a conclusion wearing the
    label of a quotation. All four wrong values found by hand on this corpus were of that
    shape."""
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
    """The error is only in the pair: either unwarranted alone reads fine."""
    from pondie.extraction.record import rules

    record = {
        "acquisitions": [{"local_id": "a", "modality": unwarranted("fMRI")}],
        "measures": [{"local_id": "m", "family": unwarranted("electrophysiology")}],
    }
    found = _Findings()
    rules.check_modality_measures(record, found)
    assert len(found.warnings) == 1
    assert "electrophysiology" in found.warnings[0][1]

    ok = {
        "acquisitions": [{"local_id": "a", "modality": unwarranted("fMRI")}],
        "measures": [{"local_id": "m", "family": unwarranted("functional_bold")}],
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
            {"local_id": "a1", "modality": unwarranted("fMRI")},
            {"local_id": "a2", "modality": unwarranted("sMRI")},
        ],
        "measures": [
            {"local_id": "m1", "family": unwarranted("functional_bold")},
            {"local_id": "m2", "family": unwarranted("structural_morphometry")},
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
        return {"count": unwarranted(count), "denominator": unwarranted(denominator)}

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
        "groups": [
            {"local_id": "g", "enrolled_count": unwarranted(20), "acquired_count": unwarranted(24)}
        ]
    }
    found = _Findings()
    rules.check_counts_add_up(record, found)
    assert len(found.warnings) == 1
    assert "subset of the one before" in found.warnings[0][1]

    ok = {
        "groups": [
            {
                "local_id": "g",
                "approached_count": unwarranted(40),
                "enrolled_count": unwarranted(24),
                "acquired_count": unwarranted(20),
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
    from pondie.extraction.repair.guard import from_local_id

    assert from_local_id("dev_siemens_trio") == "siemens trio"
    assert from_local_id("mea_cerebral_blood_flow") == "cerebral blood flow"
    assert from_local_id("mod_group_regression") == "group regression"
    assert from_local_id("acq_fmri") == "fmri"


def test_a_derived_label_too_short_to_be_a_name_is_refused():
    """`mea_fa` would offer "fa", which appears inside "factor" and "surface"."""
    from pondie.extraction.repair.guard import from_local_id

    assert from_local_id("mea_fa") == ""
    assert from_local_id("dev_ge") == ""


def test_the_raw_id_survives_when_nothing_can_be_derived():
    """A label is better than no label for a report, and the id is what it always was."""
    from pondie.extraction.record.ids import label_of

    assert label_of({"local_id": "mea_fa"}) == "mea_fa"


def test_a_declared_name_still_wins_over_the_id():
    """Derivation is the fourth fallback, not a replacement: `acq_fmri` yields "fmri", the
    modality rather than what the paper calls that acquisition."""
    from pondie.extraction.record.ids import label_of

    assert (
        label_of({"local_id": "acq_fmri", "name": unwarranted("resting-state scan")})
        == "resting-state scan"
    )
    assert (
        label_of({"local_id": "mod_glm", "model_type": unwarranted("mixed effects")})
        == "mixed effects"
    )
    assert label_of({"local_id": "dev_siemens_trio"}) == "siemens trio"


def test_a_one_word_derived_label_cannot_merge_two_entities():
    """`same_entity` needs two words in common, so "fmri" and "fmri" never merge two
    acquisitions on the strength of a modality they share."""
    from pondie.extraction.repair.guard import same_entity

    assert not same_entity("fmri", "fmri")
    assert same_entity("siemens trio", "siemens trio scanner")


def test_a_value_the_pass_could_not_place_is_marked_generated(storage_schema):
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
    from pondie.extraction.repair import guard as edit_module

    record = {"groups": [{"local_id": "g", "name": unwarranted("patients")}]}
    entity = record["groups"][0]
    edit_module.apply(
        storage_schema,
        record,
        "Group",
        entity,
        {"recruitment_method": "advertisement"},
        text="Methods. Participants answered an advertisement.",
    )

    written = entity["recruitment_method"]
    assert written["evidence"]["status"] == "not_found"
    assert written["value_source"] == "generated", "no sentence, so not reported"


def test_a_value_the_pass_did_place_stays_reported(storage_schema):
    """The label follows the evidence, so a value with a span keeps its provenance."""
    from pondie.extraction.repair import guard as edit_module

    quote = "Participants were recruited by newspaper advertisement in the local area."
    record = {"groups": [{"local_id": "g", "name": unwarranted("patients")}]}
    entity = record["groups"][0]
    edit_module.apply(
        storage_schema,
        record,
        "Group",
        entity,
        {"recruitment_method": quote},
        text=f"Methods. {quote}",
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


def test_a_wrapper_is_resolved_to_what_it_wraps(storage_schema):
    """`InferenceSettings.tfce_used` declares `ExtractedBoolean`, so reading `range` gives a
    class name and concludes "reference". Two consumers did that independently: `nu_type`
    offered nothing but `local_id` for every class, and `cast` skipped the coercion branch
    its own docstring names."""
    from pondie.extraction.record.validate import EXTRACTION_SCHEMA
    from pondie.schema import reader

    schema = reader.load(EXTRACTION_SCHEMA)
    assert schema.value_ranges(schema.attributes("InferenceSettings")["tfce_used"]) == ["boolean"]
    assert schema.value_ranges(schema.attributes("Group")["acquired_count"]) == ["integer"]
    assert schema.value_ranges(schema.attributes("Region")["name"]) == ["string"]
    # An open vocabulary keeps both branches.
    assert set(schema.value_ranges(schema.attributes("Condition")["condition_kind"])) == {
        "ConditionKind",
        "string",
    }


def test_a_string_answer_lands_in_the_type_its_slot_declares(storage_schema):
    """The model answers in the paper's words: "true" for a boolean, "31" for a count. Eight
    of eleven findings on the first paper where the proposer could write values were this --
    `ExtractedBoolean.value must be a boolean, got str`."""
    from pondie.extraction.record.validate import EXTRACTION_SCHEMA
    from pondie.formats import values as value_tools
    from pondie.schema import reader

    schema = reader.load(EXTRACTION_SCHEMA)
    assert value_tools.cast(schema, "InferenceSettings", "tfce_used", "true") is True
    assert value_tools.cast(schema, "Group", "acquired_count", "31") == 31
    assert value_tools.cast(schema, "Group", "age_mean", "24.6") == 24.6
    # And an answer that will not fit is still refused rather than coerced.
    assert value_tools.cast(schema, "Group", "acquired_count", "about twenty") is None
    assert value_tools.cast(schema, "InferenceSettings", "tfce_used", "mostly") is None


def test_a_nested_object_gains_prose_it_was_missing(storage_schema):
    """The template began offering `Task.conditions` before `apply` could write them, so the
    proposer was asked and its answer discarded. Prose it can place in the paper lands."""
    from pondie.extraction.repair import guard as edit_module

    said = "the neutral condition showed household objects matched for visual complexity"
    record = {
        "tasks": [
            {
                "local_id": "tsk",
                "name": unwarranted("picture viewing"),
                "conditions": [{"local_id": "c1", "name": unwarranted("Neutral")}],
            }
        ]
    }
    entity = record["tasks"][0]
    edit_module.apply(
        storage_schema,
        record,
        "Task",
        entity,
        {"conditions": [{"local_id": "c1", "description": said}]},
        text=f"Methods. In this study {said}, presented in blocks.",
    )

    written = entity["conditions"][0]["description"]
    assert values.read(written) == said
    assert written["evidence"]["status"] == "present"


def test_a_nested_classification_is_left_to_the_pass_that_read_the_paper(storage_schema):
    """An enum term is vocabulary, not a quote, so it can never be placed -- which is the
    line to draw. `satisfy` classifies, having read the whole document, and got `Neutral`
    right on 16038771; this sweep, asked the same from a template, answered `fixation` for
    three picture-viewing conditions."""
    from pondie.extraction.repair import guard as edit_module

    record = {
        "tasks": [
            {
                "local_id": "tsk",
                "name": unwarranted("picture viewing"),
                "conditions": [{"local_id": "c2", "name": unwarranted("Disgust")}],
            }
        ]
    }
    entity = record["tasks"][0]
    log = edit_module.apply(
        storage_schema,
        record,
        "Task",
        entity,
        {"conditions": [{"local_id": "c2", "condition_kind": "fixation"}]},
        text="Methods. Disgust pictures were shown in thirty-second blocks.",
    )

    assert values.read(entity["conditions"][0].get("condition_kind")) is None
    assert any("nothing in the paper places this value" in r.why for r in log.refused)


def test_a_nested_object_keeps_what_it_already_had(storage_schema):
    """An extracted value with a sentence behind it outranks a proposal without one, which
    is what stops a second pass quietly rewriting the first."""
    from pondie.extraction.repair import guard as edit_module

    kept = cited("control_state", "a neutral condition served as the comparison")
    record = {
        "tasks": [
            {
                "local_id": "tsk",
                "name": unwarranted("picture viewing"),
                "conditions": [
                    {"local_id": "c1", "name": unwarranted("Neutral"), "condition_kind": kept}
                ],
            }
        ]
    }
    entity = record["tasks"][0]
    edit_module.apply(
        storage_schema,
        record,
        "Task",
        entity,
        {"conditions": [{"local_id": "c1", "condition_kind": "task_state"}]},
        text="Methods. Something else entirely.",
    )

    assert values.read(entity["conditions"][0]["condition_kind"]) == "control_state"


def test_a_nested_object_the_record_does_not_have_is_not_invented(storage_schema):
    """Completing what `satisfy` left thin is not the same as adding a condition the paper
    never ran, and this pass is in no position to tell the difference."""
    from pondie.extraction.repair import guard as edit_module

    record = {
        "tasks": [
            {
                "local_id": "tsk",
                "name": unwarranted("picture viewing"),
                "conditions": [{"local_id": "c1", "name": unwarranted("Neutral")}],
            }
        ]
    }
    entity = record["tasks"][0]
    edit_module.apply(
        storage_schema,
        record,
        "Task",
        entity,
        {"conditions": [{"local_id": "c9", "name": "Fixation", "condition_kind": "fixation"}]},
        text="Methods. A fixation cross was shown between blocks.",
    )

    assert [values.read(c["name"]) for c in entity["conditions"]] == ["Neutral"]


def test_a_structure_a_flat_reply_cannot_carry_is_still_left_alone(storage_schema):
    """`Analysis.effect` nests cells nesting statistics. `recall.flat` is what separates the
    two cases, and it must keep saying no to this one."""
    from pondie.extraction.repair.propose import flat

    assert flat(storage_schema, "Condition")
    assert not flat(storage_schema, "Effect")
