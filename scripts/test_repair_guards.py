"""Every failure mode the repair pass guards against, as a test that fails without the guard.

Each test names the paper it was found on. That is not decoration: these are all regressions
of behaviour that shipped, and the record of which document exhibited it is what makes a
future change checkable against the same evidence. The differential validation that found
most of them (`introduced`) runs inside the loop, so a guard that stops working shows up on
the paper that first needed it.

Run: pytest scripts/pondie_arm/test_repair_guards.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import repair_loop as R  # noqa: E402
import render_record as RR  # noqa: E402


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def schema():
    from pondie import schema as ps
    from pondie.schema import reader as rd

    return rd.load(ps.STORAGE)


@pytest.fixture(scope="module", autouse=True)
def wired(schema):
    """The module-level tables the guards read, filled as `main` fills them."""
    from nuextract_recall import CLASSES

    keys = {**R.study_keys(schema), **CLASSES}
    R.REF_SLOTS = {k: R.reference_slots(schema, c) for k, c in CLASSES.items()}
    R.DECLARED = {k: R.declared_slots(schema, c) for k, c in keys.items()}
    R.REQUIRED = {k: R.unfillable(schema, c) for k, c in keys.items()}
    R.ENUMS = {n: set(getattr(e, "permissible_values", {}) or {})
               for n, e in schema.enums.items()}
    R.RANGES = {k: {n: getattr(sl, "range", None)
                    for n, sl in (getattr(schema.classes.get(c), "attributes", None)
                                  or {}).items()}
                for k, c in keys.items()}
    return True


def field(value, evidence=None):
    return {"extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": evidence or {"status": "not_found"}}


def cited(value, quote, at=0):
    """`at` is the span's offset in the document, which is what orders the passage list."""
    return field(value, {"status": "present",
                         "sets": [{"source": "model_quote",
                                   "spans": [{"text": quote, "start_char": at,
                                              "end_char": at + len(quote)}]}]})


# --------------------------------------------------------------------------- evidence


def test_an_edit_that_extends_a_value_keeps_the_span_that_still_supports_it():
    """23021615: `definition` was corrected from a clause to the paper's full sentence and
    marked unsupported, because every edit discarded its evidence unconditionally."""

    quote = ("Relative to the non-PTSD group, the PTSD group showed reduced gray matter in "
             "the same large cluster comprising the sgACC, caudate, and hypothalamus "
             "( Fig. 3 A, B).")
    node = cited("Relative to the non-PTSD group, the PTSD group showed reduced gray matter",
                 quote)
    longer = ("Relative to the non-PTSD group, the PTSD group showed reduced gray matter in "
              "the same large cluster comprising the sgACC, caudate, and hypothalamus.")
    assert R.carried_evidence(node, longer)["status"] == "present"


def test_an_edit_to_something_the_span_does_not_say_loses_it():
    node = cited("3 T", "Scanning used a 3 T magnet.")
    assert R.carried_evidence(node, "Hippocampal volume differed")["status"] == "not_found"


def test_a_field_with_no_prior_evidence_gains_none_from_an_edit():
    assert R.carried_evidence(None, "some value")["status"] == "not_found"


# --------------------------------------------------------------------------- typing


@pytest.mark.parametrize("cls,slot,value,expected", [
    ("groups", "is_healthy", "true", True),
    ("groups", "is_healthy", "No", False),
    ("groups", "is_healthy", "mostly", None),          # 28416565: str into a boolean
    ("analyses", "prespecification", "exploratory", "exploratory"),
    ("analyses", "prespecification", "post-hoc", None),  # 28888350: not in the vocabulary
    ("regions", "definition_method", "atlas", "atlas"),
    ("regions", "definition_method", "hand drawn", None),
    ("regions", "region_type", "gray matter", "gray matter"),   # open vocabulary: allowed
])
def test_a_value_is_written_only_in_the_type_and_vocabulary_its_slot_declares(
        cls, slot, value, expected):
    assert R.typed(value, cls, slot) == expected


# --------------------------------------------------------------------------- identity


def test_one_instrument_under_two_names_is_one_entity(monkeypatch):
    """12853571: "clinician-administered PTSD scale (CAPS)" minted a second copy of
    `asm_caps` ("CAPS total score"), and analyses then linked to the copy."""

    class Abbrev:
        def expand(self, short):
            return "clinician-administered PTSD scale" if short == "CAPS" else (
                "posttraumatic stress disorder" if short == "PTSD" else None)

    ab = Abbrev()
    assert R.same_entity("CAPS total score", "clinician-administered PTSD scale (CAPS)", ab)


def test_two_instruments_sharing_an_abbreviation_stay_two(monkeypatch):
    class Abbrev:
        def expand(self, short):
            return "posttraumatic stress disorder" if short == "PTSD" else None

    assert not R.same_entity("PTSD checklist", "PTSD symptom scale", Abbrev())


def test_a_minted_id_uses_the_address_prefix_its_class_declares():
    record = {"regions": [{"local_id": "reg_acc"}]}
    assert R.mint_id(record, "regions", "right orbitofrontal cortex", 0).startswith("reg_")
    assert R.mint_id(record, "model_estimations", "Regression analysis", 0).startswith("mod_")


def test_a_minted_id_does_not_collide_with_one_already_taken():
    record = {"regions": [{"local_id": "reg_hippocampus"}]}
    assert R.mint_id(record, "regions", "hippocampus", 0) != "reg_hippocampus"


# --------------------------------------------------------------------------- references


def test_a_slot_is_swept_after_what_it_points_at():
    """16508348: analyses were swept first, so four correctly named regions were refused for
    having no target, and the regions sweep ran afterwards."""

    order = R.sweep_order(["analyses", "groups", "inference_settings", "regions"])
    assert order.index("regions") < order.index("analyses")


def test_nothing_references_itself():
    """27082610, 19942229: `inputs_from` resolved to the model being edited."""

    record = {"model_estimations": [{"local_id": "m1", "name": field("ANCOVA")}]}
    target = record["model_estimations"][0]
    R.apply_edit(target, {"local_id": "m1", "inputs_from": ["ANCOVA"]},
                 "model_estimations", record, 0)
    assert "m1" not in (target.get("inputs_from") or [])


def test_repointing_an_analysis_may_not_orphan_the_terms_its_cells_name():
    """19942229: `a_793_1` was moved to a model that does not reach `trm_group_r_nr`."""

    record = {
        "model_estimations": [
            {"local_id": "mod_a", "terms": [{"local_id": "t_a"}], "name": field("A")},
            {"local_id": "mod_b", "terms": [{"local_id": "t_b"}], "name": field("B")}],
        "analyses": [{"local_id": "a1", "name": field("contrast"),
                      "model_estimation": "mod_a",
                      "effect": {"cells": [{"term": "t_a"}]}}]}
    analysis = record["analyses"][0]
    assert R.orphans_cell_terms(analysis, "mod_b", record)
    assert not R.orphans_cell_terms(analysis, "mod_a", record)


def test_a_multivalued_reference_gains_without_losing_what_was_there():
    """12853571: `assessments` was replaced by four new ids, dropping `asm_caps` -- the CAPS
    total score, which is the one thing that correlation is of."""

    record = {"assessments": [{"local_id": "asm_caps", "name": field("CAPS total score")},
                              {"local_id": "asm_ies", "name": field("impact of event scale")}],
              "analyses": [{"local_id": "a1", "name": field("correlation"),
                            "assessments": ["asm_caps"]}]}
    target = record["analyses"][0]
    R.apply_edit(target, {"local_id": "a1", "assessments": ["impact of event scale"]},
                 "analyses", record, 0)
    assert target["assessments"] == ["asm_caps", "asm_ies"]


def test_a_reference_list_holds_each_target_once():
    """23021615: four preprocessing names all resolved to `prp_vbm`, written four times."""

    record = {"preprocessings": [{"local_id": "prp_vbm", "name": field("VBM pipeline")}],
              "model_estimations": [{"local_id": "m1", "name": field("group model")}]}
    target = record["model_estimations"][0]
    R.apply_edit(target, {"local_id": "m1",
                          "preprocessing": ["VBM pipeline", "VBM pipeline", "VBM pipeline"]},
                 "model_estimations", record, 0)
    assert target["preprocessing"] == ["prp_vbm"]


# --------------------------------------------------------------------------- scope pairs


@pytest.mark.parametrize("scope,regions,written", [
    ("roi", [], None),                    # 19996042: a restriction with nothing named
    ("roi", ["reg_x"], "roi"),
    ("whole_brain", ["reg_x"], None),     # 11950456: whole-brain beside a named region
    ("whole_brain", [], "whole_brain"),
])
def test_a_scope_and_the_regions_beside_it_must_agree(scope, regions, written):
    record = {"regions": [{"local_id": "reg_x", "name": field("amygdala"),
                           "definition_method": field("atlas")}],
              "inference_settings": [{"local_id": "i1",
                                      "correction_regions": list(regions)}]}
    target = record["inference_settings"][0]
    R.apply_edit(target, {"local_id": "i1", "correction_scope": scope},
                 "inference_settings", record, 0)
    node = target.get("correction_scope")
    assert (node.get("value") if isinstance(node, dict) else None) == written


def test_a_whole_brain_analysis_is_not_given_regions_to_search():
    record = {"regions": [{"local_id": "reg_x", "name": field("sgACC"),
                           "definition_method": field("atlas")}],
              "analyses": [{"local_id": "a1", "name": field("VBM"),
                            "spatial_scope": field("whole_brain"), "regions": []}]}
    target = record["analyses"][0]
    R.apply_edit(target, {"local_id": "a1", "regions": ["sgACC"]}, "analyses", record, 0)
    assert target["regions"] == []


def test_a_slot_is_written_only_on_a_class_that_declares_it():
    """23021615: `correction_scope` was written onto three analyses; it belongs to
    InferenceSettings, which those analyses already referenced."""

    record = {"analyses": [{"local_id": "a1", "name": field("VBM")}]}
    target = record["analyses"][0]
    R.apply_edit(target, {"local_id": "a1", "correction_scope": "roi"}, "analyses", record, 0)
    assert "correction_scope" not in target


def test_an_edit_that_only_shortens_a_value_is_not_a_correction():
    """22952599: "compared to traumatized controls." became "compared to traumatized"."""

    record = {"analyses": [{"local_id": "a1", "name": field("contrast"),
                            "definition": cited(
                                "Decreased gray matter volume in PTSD patients compared to "
                                "traumatized controls.",
                                "Decreased gray matter volume in PTSD patients compared to "
                                "traumatized controls.")}]}
    target = record["analyses"][0]
    R.apply_edit(target, {"local_id": "a1",
                          "definition": "Decreased gray matter volume in PTSD patients "
                                        "compared to traumatized"},
                 "analyses", record, 0)
    assert target["definition"]["value"].endswith("controls.")


# --------------------------------------------------------------------------- foci


def test_an_analysis_is_not_handed_the_last_table_for_want_of_a_better_match():
    """22952599: "ROI analysis of hippocampus and amygdala" was bound to the whole-brain
    table, taking its premotor and parietal peaks."""

    unclaimed = [{"_rg_id": "t2#0", "table_id": "t2", "name": "PTSD vs control",
                  "points": [{"coordinates": [53, -1, 35]}]}]
    assert R.bind_foci({"name": "ROI analysis of hippocampus and amygdala"}, unclaimed) is None


def test_a_row_group_naming_no_coordinates_is_still_a_row_group(tmp_path):
    """A table section reading "n.s." reports an analysis that found nothing. Dropping those
    made 142 of the corpus's 1,822 row groups unbindable."""

    stage1 = tmp_path / "1" / "stage1"
    stage1.mkdir(parents=True)
    (stage1 / "analyses.json").write_text(json.dumps({"analyses": [
        {"table_id": "t1", "name": "CT (Alcohol) > CCT (Alcohol) (0 voxels)", "points": []},
        {"table_id": "t1", "name": "with peaks", "points": [{"coordinates": [1, 2, 3]}]}]}))
    rows = R.stage1_analyses(tmp_path, "1")
    assert len(rows) == 2
    assert rows[0]["_rg_id"] != rows[1]["_rg_id"]


# --------------------------------------------------------------------------- record shape


def test_the_loops_own_bookkeeping_does_not_travel_in_the_record():
    record = {"analyses": [{"local_id": "a1", "_provenance": {"source": "nuextract3"},
                            "effect": {"cells": [{"_provenance": {"x": 1}}]}}]}
    audit = R.strip_provenance(record)
    assert "_provenance" not in json.dumps(record)
    assert set(audit) == {"analyses[0]", "analyses[0].effect.cells[0]"}


def test_a_repair_that_damages_the_record_is_reported():
    """The check that did not exist while 665 violations accumulated across 15 records."""

    before = {"analyses": [{"local_id": "a1", "name": field("VBM")}]}
    after = json.loads(json.dumps(before))
    after["analyses"][0]["correction_scope"] = field("roi")   # not a slot on Analysis
    assert any("correction_scope" in line for line in R.introduced(before, after))
    assert R.introduced(before, before) == []


# --------------------------------------------------------------------------- rendering


def test_a_sentence_supporting_many_fields_is_written_once_and_cited_by_id():
    shared = "Images were acquired on a 3 T Siemens Trio scanner."
    # The MPRAGE sentence sits earlier in the paper than the scanner one, so it is `s1`
    # however the walk happens to reach the fields.
    record = {"acquisitions": [{
        "local_id": "acq1",
        "magnetic_strength": cited("3 T", shared, at=900),
        "scanner": cited("Siemens Trio", shared, at=900),
        "sequence": cited("MPRAGE", "A T1-weighted MPRAGE sequence was used.", at=100)}]}
    out = RR.render(record, "full", "1", corpus=None, show_users=True)
    assert out.count(shared) == 1
    body = out.split("### Supporting passages")[0]
    # ids run in document order, and both fields citing one sentence cite one id
    assert "sequence: MPRAGE  [s1]" in body
    assert "magnetic strength: 3 T  [s2]" in body
    assert "scanner: Siemens Trio  [s2]" in body
    assert "\x00" not in out


def test_the_arm_with_no_evidence_carries_no_passage_list():
    record = {"acquisitions": [{"local_id": "acq1",
                                "magnetic_strength": cited("3 T", "Scanning used 3 T.")}]}
    assert "Supporting passages" not in RR.render(record, "none", "1", corpus=None)


def test_a_correction_may_not_shorten_a_list():
    """16701903 acquires two sequences, MP-RAGE at TE 4.4 ms and FLASH at TE 5 ms, and
    `echo_time_seconds` held both. A single-value correction cast into `[value]` fixed one
    type error by dropping half the data."""

    node = {"extraction_status": "extracted", "value": [0.0044, 0.005],
            "value_source": "reported", "evidence": {"status": "present"}}
    assert R.would_shorten(node, "0.0044")
    assert not R.would_shorten({**node, "value": [0.0044]}, "0.005")
    assert not R.would_shorten({**node, "value": "3 T"}, "1.5 T")


@pytest.mark.parametrize("old,new,expected", [
    (38, "42", 42),                 # 20673548: a corrected count is still a count
    (38, "forty-two", 38),          # unconvertible: keep what was there
    (0.0044, "0.005", 0.005),
    (True, "no", False),
    ("3 T", "1.5 T", "1.5 T"),      # a string correction passes through
])
def test_a_correction_keeps_the_type_the_value_had(old, new, expected):
    """The evidence pass writes into a node resolved by path, so it knows no class or slot
    and cannot consult the schema. The old value's own type is enough."""
    assert R.like(old, new) == expected
