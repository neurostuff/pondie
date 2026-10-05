"""Record formatting faults found by validating pipeline output, one test per cause.

Each was systematic: the same fault on most records of a run, so no record validated clean.
"""

import json

from pondie import normalization, schema
from pondie.extraction.models import (
    PaperOutcome,
    RunReport,
    StageName,
    StageOutcome,
)
from pondie.extraction.prompt import fill, render
from pondie.extraction.record.fix import shape
from pondie.extraction.record.validate import Validator
from pondie.schema import reader


def _v(value, source="reported"):
    return {"extraction_status": "extracted", "value": value, "value_source": source,
            "evidence": {"status": "not_found"}}


def _group():
    return {
        "local_id": "grp_ptsd",
        "name": _v("PTSD"),
        "medical_condition": _v(["posttraumatic stress disorder"]),
        "age_unit": _v("years"),
        "age_mean": _v(34.5),
        "sex_distribution": [{"category": _v("female"), "count": _v(8)}],
        "population_characteristics": _v(["combat veterans", "right-handed"]),
    }


def _errors(record):
    validator = Validator(reader.load(schema.EXTRACTION), None)
    validator.check_record(record)
    return validator.errors


# --- code-filled slots ---------------------------------------------------------------------


def test_code_filled_slots_validate():
    """`is_healthy`, the normalized demographics and `other_characteristics` were written
    as extracted with `not_applicable` evidence (or `derived`, no evidence), which the
    schema forbids: 354 of 565 errors on one 55-paper run."""
    record = {"local_id": "S1", "groups": [_group()]}
    normalization.apply_derived(record)
    group = record["groups"][0]
    for slot in ("is_healthy", "age_unit_normalized", "other_characteristics"):
        assert group[slot]["value_source"] == "generated", slot
        assert group[slot]["evidence"] == {"status": "not_found"}, slot
    assert group["sex_distribution"][0]["category_normalized"]["evidence"]["status"] == "not_found"
    bad = [e for e in _errors(record) if "not_applicable" in e or "not declared" in e]
    assert bad == []


def test_the_prompt_does_not_ask_for_slots_code_fills():
    system = render.build_prompt("A paper.", "single", False, "").system
    for slot in ("is_healthy", "age_unit_normalized", "category_normalized", "mirror_of",
                 "other_characteristics", "study_type"):
        assert f"`{slot}`" not in system, slot
    lists = system.split("TOP LEVEL of the object and nowhere else:", 1)[1].split("\n", 1)[0]
    assert "language" not in lists and "study_type" not in lists
    assert "`acquisition_type`" in system, "a type designator is the model's to name"


def test_fill_does_not_ask_for_slots_code_fills():
    payload = {"groups": [{"local_id": "grp_a", "name": _v("A")}]}
    asked = {row["id"].rsplit(".", 1)[-1] for row in fill.unsettled(payload, reader.load(
        schema.EXTRACTION))}
    assert "is_healthy" not in asked and "age_unit_normalized" not in asked
    assert "medical_condition" in asked


def test_a_models_value_for_a_slot_code_fills_is_dropped():
    """The model's `language` wrapper survived as "must be a string, got dict"."""
    body = {"language": [_v("English")], "groups": [{"local_id": "g", "is_healthy": _v(True)}]}
    dropped = shape.drop_code_filled(body, reader.load(schema.EXTRACTION))
    assert "language" not in body and "is_healthy" not in body["groups"][0]
    assert len(dropped) == 2


# --- reply shape ---------------------------------------------------------------------------


def test_an_analysis_slot_filed_inside_effect_moves_up():
    body = {"analyses": [{"local_id": "ana_1", "effect": {"cells": [], "tables": ["tbl2"],
                                                          "model_estimation": "mod_a"},
                          "model_estimation": "mod_b"}]}
    notes = shape.lift_misnested(body, reader.load(schema.EXTRACTION))
    analysis = body["analyses"][0]
    assert analysis["tables"] == ["tbl2"]
    assert analysis["model_estimation"] == "mod_b", "the analysis's own value stays"
    assert analysis["effect"]["model_estimation"] == "mod_a", "a conflicting one is kept"
    assert len(notes) == 1


def test_a_cell_written_as_a_wrapper_reads_as_a_cell():
    cell = {"term": "trm_group", "level": _v("PTSD"), "direction": _v("negative"),
            "extraction_status": "extracted", "evidence": {"status": "not_found"}}
    body = {"analyses": [{"local_id": "ana_1", "effect": {"cells": [cell]}}]}
    shape.unwrap_entities(body, reader.load(schema.EXTRACTION))
    assert set(cell) == {"term", "level", "direction"}



def test_a_factor_level_written_inside_its_own_level_is_lifted():
    """30545239 and 16684342: `{"level": {"level": <field>, "groups": [...]}}` left the term
    with no readable level. Keys already on the outer level (`order`) stay."""
    level = {"level": {"level": _v("PTSD"), "groups": ["grp_ptsd"]}, "order": _v(1)}
    body = {"model_estimations": [{"local_id": "mod_a", "terms": [
        {"local_id": "trm_group", "name": _v("group"), "type": _v("categorical"),
         "levels": [level]}]}]}
    shape.lift_misnested(body, reader.load(schema.EXTRACTION))
    assert level == {"level": _v("PTSD"), "groups": ["grp_ptsd"], "order": _v(1)}


def test_a_parents_slot_in_any_child_moves_up():
    """Not only an analysis's slots under `effect`: an Effect's `kind` filed in a cell."""
    effect = {"cells": [{"term": "trm_group", "kind": _v("contrast")}]}
    body = {"analyses": [{"local_id": "ana_1", "effect": effect}]}
    shape.lift_misnested(body, reader.load(schema.EXTRACTION))
    assert effect == {"cells": [{"term": "trm_group"}], "kind": _v("contrast")}


def test_entities_written_inside_a_sibling_keyed_by_their_ids_join_their_list():
    """21078704 wrote eleven analyses inside `a_2022_1`; the built record held one."""
    inner = {"local_id": "ana_medial_years", "name": _v("medial frontal x years")}
    body = {"analyses": [{"local_id": "a_2022_1", "name": _v("cluster"),
                          "ana_medial_years": inner}]}
    moved = shape.rehome_keyed_entities(body, reader.load(schema.EXTRACTION))
    assert [a["local_id"] for a in body["analyses"]] == ["a_2022_1", "ana_medial_years"]
    assert "ana_medial_years" not in body["analyses"][0] and len(moved) == 1


def test_a_keyed_entity_whose_id_is_already_held_stays_reported():
    body = {"analyses": [
        {"local_id": "a_1", "a_2": {"local_id": "a_2", "name": _v("copy")}},
        {"local_id": "a_2", "name": _v("original")}]}
    assert shape.rehome_keyed_entities(body, reader.load(schema.EXTRACTION)) == []
    assert "a_2" in body["analyses"][0]


def test_any_entity_written_as_a_value_loses_its_wrapper_keys():
    level = {"level": _v("PTSD"), "extraction_status": "extracted",
             "evidence": {"status": "not_found"}}
    body = {"model_estimations": [{"local_id": "mod_a", "terms": [
        {"local_id": "trm_group", "name": _v("group"), "levels": [level]}]}]}
    shape.unwrap_entities(body, reader.load(schema.EXTRACTION))
    assert level == {"level": _v("PTSD")}


def test_an_objects_slots_written_inside_its_field_wrapper_move_out():
    """21078704 put an analysis's `definition` inside its `name`; 21498053 an effect's
    `statistic` inside its `kind`."""
    name = {**_v("cluster"), "definition": _v("a regression")}
    effect = {"kind": {**_v("contrast"), "statistic": _v("t")}, "cells": []}
    body = {"analyses": [{"local_id": "ana_1", "name": name, "effect": effect}]}
    shape.lift_misnested(body, reader.load(schema.EXTRACTION))
    assert name == _v("cluster") and body["analyses"][0]["definition"] == _v("a regression")
    assert effect["kind"] == _v("contrast") and effect["statistic"] == _v("t")


def test_a_stray_that_contradicts_what_is_there_stays_put():
    """Nothing the model wrote is discarded to make a record validate."""
    name = {**_v("cluster"), "definition": _v("one reading")}
    body = {"analyses": [{"local_id": "ana_1", "name": name, "definition": _v("another")}]}
    shape.lift_misnested(body, reader.load(schema.EXTRACTION))
    assert name["definition"] == _v("one reading")
    assert body["analyses"][0]["definition"] == _v("another")



def test_a_stray_with_the_same_value_is_a_duplicate_and_keeps_its_evidence():
    """21078704's `outcome` sat in both places with one value and different evidence."""
    cited = {**_v("significant_effect"),
             "evidence": {"status": "present", "sets": [{"spans": [{"text": "p < .05"}]}]}}
    name = {**_v("cluster"), "outcome": cited}
    body = {"analyses": [{"local_id": "ana_1", "name": name,
                          "outcome": _v("significant_effect")}]}
    shape.lift_misnested(body, reader.load(schema.EXTRACTION))
    assert "outcome" not in name
    assert body["analyses"][0]["outcome"]["evidence"]["status"] == "present"

def test_a_slot_name_in_a_list_of_objects_is_dropped():
    """25050433: `design.timepoints` held `"arms"` beside two Timepoints."""
    tp = {"local_id": "tp_1", "name": _v("baseline")}
    body = {"design": {"timepoints": [tp, "arms"]}}
    shape.drop_stray_slot_names(body, reader.load(schema.EXTRACTION))
    assert body["design"]["timepoints"] == [tp]


def test_a_wrapper_meaning_generated_or_carrying_a_moot_reason_is_repaired():
    field = {**_v(2.0, "inferred"), "unreported_reason": "not_stated"}
    shape.repair_wrappers({"x": field})
    assert field == _v(2.0, "generated")

def test_not_reported_written_as_a_value_becomes_the_status():
    """`Cell.direction: 'not_reported'` is not a permissible direction."""
    cell = {"term": "trm_group", "direction": _v("not_reported", "generated")}
    body = {"analyses": [{"local_id": "ana_1", "effect": {"cells": [cell]}}]}
    shape.status_as_value(body, reader.load(schema.EXTRACTION))
    assert cell["direction"]["extraction_status"] == "not_reported"
    assert cell["direction"]["evidence"] == {"status": "not_applicable"}


def test_not_reported_written_as_a_bare_value_becomes_the_status():
    """16701903 wrote `"direction": "not_reported"` with no wrapper at all."""
    cell = {"term": "trm_group", "direction": "not_reported"}
    body = {"analyses": [{"local_id": "ana_1", "effect": {"cells": [cell]}}]}
    shape.status_as_value(body, reader.load(schema.EXTRACTION))
    assert cell["direction"]["extraction_status"] == "not_reported"


def test_a_top_level_key_no_slot_could_have_is_dropped():
    reply = {"analyses": [], "}rayele": "x", "tab4": {"caption": _v("Peaks")}}
    payload, notes = render.normalize(reply, "single")
    assert "}rayele" not in payload and "}rayele" not in payload.get("study", {})
    assert "tab4" in payload["study"], "a valid name stays, for rehome_keyed_entities"
    assert any("not a possible slot name" in n for n in notes)



def test_a_key_no_slot_could_have_is_dropped_at_any_depth():
    """21498053 left `":{": ""` on an analysis."""
    note = _v("text")
    note["value"] = {"free": 1}
    body = {"analyses": [{"local_id": "ana_1", ":{": "", "notes": note}]}
    dropped = shape.drop_impossible_keys(body)
    assert body["analyses"][0] == {"local_id": "ana_1", "notes": note}
    assert dropped == ["Study.analyses[0]: dropped ':{', not a possible slot name"]

# --- PubMed --------------------------------------------------------------------------------


def test_build_fills_language_and_study_type_from_pubmed(monkeypatch):
    from pondie.extraction import pubmed, stages

    asked = []

    def summaries(pmids, **_):
        asked.extend(pmids)
        return {"123": {"study_type": ["Journal Article"], "language": ["eng"]}}

    monkeypatch.setattr(pubmed, "summaries", summaries)
    record = {"local_id": "123"}
    note = stages._fill_from_pubmed(record, "123")
    assert record["language"] == ["eng"] and record["study_type"] == ["Journal Article"]
    assert "pubmed:" in note
    assert stages._fill_from_pubmed({"local_id": "S1"}, "S1") == "" and asked == ["123"]


# --- visibility ----------------------------------------------------------------------------


def test_the_run_summary_counts_valid_records_and_names_the_commonest_fault():
    def built(study, errors):
        return PaperOutcome(study_id=study, outcomes=(
            StageOutcome(stage=StageName.build, study_id=study, validation_errors=errors),))

    report = RunReport(papers=(
        built("A", ()),
        built("B", ("Study.groups[0].x: bad 'one'", "Study.groups[1].x: bad 'two'")),
        built("C", ("Study.groups[3].x: bad 'three'",)),
    ))
    summary = report.summary()
    assert "records valid: 1/3" in summary
    assert "    3  Study.groups[i].x: bad '…'" in summary


def test_a_rebuilt_payload_with_the_old_faults_validates(tmp_path):
    """The fixes run at `build`, so payloads already on disk are repaired by a rebuild."""
    from pondie.extraction.record.builder import merge_payloads
    from pondie.extraction.record.fix import Context, apply_all, AT_MERGE

    payload = {"language": [_v("English")], "groups": [_group()],
               "analyses": [{"local_id": "ana_1", "effect": {"cells": [], "tables": []}}]}
    (tmp_path / "single.json").write_text(json.dumps(payload))
    body, _ = merge_payloads(tmp_path)
    apply_all(body, Context(schema=reader.load(schema.EXTRACTION)), stage=AT_MERGE)
    assert "language" not in body
    assert "tables" not in body["analyses"][0]["effect"]


def test_a_type_designator_is_not_treated_as_code_filled():
    """`acquisition_type` is `deterministic` in storage but the model names it; dropping it
    left every MRI-specific slot undeclared and the required designator missing."""
    assert not schema.code_fills("Acquisition", "acquisition_type")
    body = {"acquisitions": [{"local_id": "acq_t1", "acquisition_type": "MRI"}]}
    assert shape.drop_code_filled(body, reader.load(schema.EXTRACTION)) == []

