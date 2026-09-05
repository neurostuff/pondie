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
