"""An edit must not cost the record the warrant it already had.

`refuses_losing_the_warrant` allows an edit when an existing span still contains the new
value, and its docstring promises the old spans are kept. They were not: the write rebuilt
the wrapper through `_wrap`, which searches for a span only when the value is at least
twenty characters long. Every count and every mean age is shorter than that, so the rebuild
replaced a verified span with `not_found` and demoted `reported` to `generated` -- for an
edit the guard had just certified as still supported.

Measured on 18823721: 26 fields kept their value and lost their citation that way, against
8 fields that gained one. The pass subtracted.
"""

from __future__ import annotations

import pytest

from pondie import schema
from pondie.extraction.record import edit
from pondie.schema import reader

PAPER = ("Participants were 12 opioid-dependent patients (mean age = 44.5 years, "
         "S.D. = 3.9) recruited from a detoxification unit.")


@pytest.fixture(scope="module")
def sch():
    return reader.load(schema.EXTRACTION)


def cited(value, text, source="reported"):
    return {"extraction_status": "extracted", "value": value, "value_source": source,
            "evidence": {"status": "present",
                         "sets": [{"source": "model_quote",
                                   "spans": [{"text": text, "start_char": 0,
                                              "end_char": len(text)}]}]}}


def test_re_proposing_the_same_value_is_not_an_edit(sch):
    """The commonest case, and the one that did the damage: the proposer offers a value the
    record already holds. Rewriting it can only lose what warranted it."""
    entity = {"local_id": "grp_a", "age_mean": cited(44.5, "mean age = 44.5 years")}
    log = edit.apply(sch, {"groups": [entity]}, "Group", entity, {"age_mean": 44.5}, PAPER)
    assert entity["age_mean"]["evidence"]["status"] == "present"
    assert entity["age_mean"]["value_source"] == "reported"
    assert not log.written, "a no-op was recorded as a write"
    assert any("already recorded" in r.why for r in log.refused)


def test_an_edit_the_old_span_still_warrants_inherits_it(sch):
    """`refuses_losing_the_warrant` lets this through *because* the span still contains the
    value. Dropping the span afterwards makes the guard's certification meaningless."""
    entity = {"local_id": "grp_a",
              "age_mean": cited(44.0, "mean age = 44.5 years")}
    edit.apply(sch, {"groups": [entity]}, "Group", entity, {"age_mean": 44.5}, PAPER)
    node = entity["age_mean"]
    assert node["value"] == 44.5
    assert node["evidence"]["status"] == "present", "the warrant was thrown away"
    assert node["evidence"]["sets"][0]["spans"][0]["text"] == "mean age = 44.5 years"
    assert node["value_source"] == "reported"


def test_a_value_no_span_supports_is_still_honestly_ungrounded(sch):
    """The inheritance must not manufacture a warrant. A value the old span does not
    contain gets what it earns: `not_found`, and `generated` rather than `reported`."""
    entity = {"local_id": "grp_a", "age_mean": cited(44.5, "mean age = 44.5 years")}
    edit.apply(sch, {"groups": [entity]}, "Group", entity, {"age_mean": 61.2}, PAPER)
    node = entity["age_mean"]
    if node["value"] == 61.2:                      # a guard may refuse it outright, which is fine
        assert node["evidence"]["status"] == "not_found"
        assert node["value_source"] == "generated"


def test_an_absent_field_is_still_filled(sch):
    """The counterweight: none of this may turn repair into a pass that writes nothing."""
    entity = {"local_id": "grp_a"}
    log = edit.apply(sch, {"groups": [entity]}, "Group", entity, {"age_mean": 44.5}, PAPER)
    assert entity["age_mean"]["value"] == 44.5
    assert log.written


def test_a_one_element_list_is_still_a_list(sch):
    """`["DSM-IV heroin dependence"] -> "heroin dependence"` on 18823721 dropped both the
    diagnostic system and the list type. `refuses_truncation` compared strings and saw a
    list; `refuses_shortening_a_list` required more than one element. It passed both, and
    with evidence inheritance it would have passed the warrant gate too -- a record made
    worse while every check reported green."""
    entity = {"local_id": "grp_a",
              "medical_condition": cited(["DSM-IV heroin dependence"],
                                         "diagnosed with DSM-IV heroin dependence")}
    log = edit.apply(sch, {"groups": [entity]}, "Group", entity,
                     {"medical_condition": "heroin dependence"}, PAPER)
    assert entity["medical_condition"]["value"] == ["DSM-IV heroin dependence"]
    assert not log.written


def test_a_list_slot_is_written_as_a_list(sch):
    """Multiplicity on the extraction schema lives in the range name, not on the attribute:
    `medications` ranges on `ExtractedStringList`, whose own `multivalued` is False. Reading
    it raw wrote bare strings into four list slots -- four of the five findings the pass
    introduced on 18823721."""
    entity = {"local_id": "grp_a"}
    edit.apply(sch, {"groups": [entity]}, "Group", entity,
               {"medications": "methadone"}, PAPER)
    assert entity["medications"]["value"] == ["methadone"]


def test_a_digit_inside_a_number_is_not_a_warrant(sch):
    """`acquired_count: 12 -> 1` inherited "consisted of 12 opioid-dependent patients",
    because "1" is inside "12". The span said the opposite of the value it was made to
    warrant, and the edit passed every gate. Numbers are compared as numbers."""
    entity = {"local_id": "grp_a",
              "acquired_count": cited(12, "consisted of 12 opioid-dependent patients")}
    edit.apply(sch, {"groups": [entity]}, "Group", entity, {"acquired_count": 1}, PAPER)
    node = entity["acquired_count"]
    if node["value"] == 1:
        assert node["evidence"]["status"] == "not_found", "a digit substring bought a span"


def test_extending_a_grounded_list_is_allowed(sch):
    """`["SPM2"] -> ["SPM2", "FSL"]` with both named in the paper is the edit this pass
    exists for. `_bare` stringified the list repr, so extension looked unwarranted while
    dropping a value looked fine -- exactly backwards."""
    text = "Analysis used SPM2 and FSL."
    entity = {"local_id": "prp", "software": cited(["SPM2"], "Analysis used SPM2 and FSL.")}
    edit.apply(sch, {"preprocessings": [entity]}, "Preprocessing", entity,
               {"software": ["SPM2", "FSL"]}, text)
    assert entity["software"]["value"] == ["SPM2", "FSL"]
    assert entity["software"]["evidence"]["status"] == "present"


def test_shortening_a_list_is_refused_whatever_shape_it_arrives_in(sch):
    """`shape` resolves multiplicity through the wrapper now, so the new value is always a
    list and the old `isinstance` test never fired -- switching the guard off on the very
    slot it was written for."""
    entity = {"local_id": "prp",
              "software": cited(["SPM2", "FSL"], "Analysis used SPM2 and FSL.")}
    log = edit.apply(sch, {"preprocessings": [entity]}, "Preprocessing", entity,
                     {"software": ["SPM2"]}, "Analysis used SPM2 and FSL.")
    assert entity["software"]["value"] == ["SPM2", "FSL"]
    assert any("drops values" in r.why for r in log.refused)


def test_an_exclusive_reference_is_not_copied_to_a_second_entity(sch):
    """One target set belongs to one entity on an exclusive slot; the second copy is refused.

    18823721: the pass wrote the same four questionnaires -- ASI, OCDUS, DDQ and SHAPS -- to
    both groups as `diagnostic_instrument`, the slot the schema describes as the assessment
    that established THIS group's condition. Two of the four were administered to the
    patients only, and none of them established a diagnosis.
    """
    record = {
        "groups": [{"local_id": "grp_patients", "name": _named("patients")},
                   {"local_id": "grp_controls", "name": _named("controls")}],
        "assessments": [{"local_id": "asm_caps", "name": _named("CAPS")}],
    }
    claimed: dict = {}
    first = edit.apply(
        sch, record, "Group", record["groups"][0],
        {"local_id": "grp_patients", "diagnostic_instrument": ["CAPS"]},
        PAPER, None, claimed)
    second = edit.apply(
        sch, record, "Group", record["groups"][1],
        {"local_id": "grp_controls", "diagnostic_instrument": ["CAPS"]},
        PAPER, None, claimed)
    assert record["groups"][0]["diagnostic_instrument"] == ["asm_caps"]
    assert "diagnostic_instrument" not in record["groups"][1]
    assert [s for s, _v in first.written] == ["diagnostic_instrument"]
    assert not second.written
    assert "the same targets were just written to grp_patients" in second.refused[0].why


def test_a_shared_reference_on_an_ordinary_slot_is_still_written(sch):
    """Sharing is how the other reference slots work, and refusing it would break them.

    Over twelve papers the pass made fifteen shared-target writes -- six analyses on one
    SCID, three on one cue task, two model estimations on one preprocessing -- and every one
    is correct. Only the slots in `EXCLUSIVE_REFERENCES` are refused.
    """
    record = {
        "model_estimations": [{"local_id": "mod_one", "name": _named("first level")},
                              {"local_id": "mod_two", "name": _named("second level")}],
        "preprocessings": [{"local_id": "prp_fmri", "name": _named("fmri preprocessing")}],
    }
    claimed: dict = {}
    for entity, local_id in zip(record["model_estimations"], ("mod_one", "mod_two")):
        edit.apply(sch, record, "ModelEstimation", entity,
                   {"local_id": local_id, "preprocessing": ["fmri preprocessing"]},
                   PAPER, None, claimed)
    assert record["model_estimations"][0]["preprocessing"] == ["prp_fmri"]
    assert record["model_estimations"][1]["preprocessing"] == ["prp_fmri"]


def _named(label):
    return {"extraction_status": "extracted", "value": label, "value_source": "reported",
            "evidence": {"status": "not_found"}}
