"""Every evidence set says which locator found it.

The two sets `apply_evidence` can attach were already two different locators, but only by
position in the list -- nothing downstream could say which one warranted a value, or count
how often each was right. `source` makes that legible, and these tests pin the labels,
because a mislabelled set is worse than an unlabelled one: it asserts a provenance the
sentence does not have.
"""

from __future__ import annotations

import pytest

from pondie.extraction.evidence import quote as qz


def field(value="3 T"):
    return {"extraction_status": "extracted", "value": value, "value_source": "reported"}


def payload():
    return {"acquisitions": [{"local_id": "acq1", "magnetic_strength": field()}]}


def sources(doc):
    (acq,) = doc["acquisitions"]
    return [s["source"] for s in acq["magnetic_strength"]["evidence"]["sets"]]


def test_a_model_quote_is_labelled_as_one():
    doc = payload()
    qz.apply_evidence(doc, {"acquisitions[0].magnetic_strength": "Scanning used a 3 T magnet."})
    assert sources(doc) == ["model_quote"]


def test_a_literally_located_quote_is_not_labelled_a_model_quote():
    """`literal_quotes` settles a field before any model is asked, so calling the result a
    `model_quote` claims a reading that never happened -- on the one field whose purpose is
    telling the locators apart, and which a reviewer weighs a span by."""
    doc = payload()
    path = "acquisitions[0].magnetic_strength"
    qz.apply_evidence(doc, {path: "Scanning used a 3 T magnet."}, literal={path})
    assert sources(doc) == ["literal_match"]


def test_the_literal_locator_finds_a_value_that_occurs_once_and_skips_a_repeated_one():
    """Uniqueness is the safety condition. A value the document states in one place has a
    determined sentence; one it states in three does not, and goes to the model as before."""
    text = (
        "Participants were scanned on a 3 T magnet.\n"
        "Analysis used SPM12 throughout.\n"
        "We report SPM12 defaults, and SPM12 was also used for preprocessing.\n"
    )
    doc = {
        "acquisitions": [
            {
                "local_id": "acq",
                "magnetic_strength": {
                    "extraction_status": "extracted",
                    "value": "3 T",
                    "value_source": "reported",
                },
                "software": {
                    "extraction_status": "extracted",
                    "value": "SPM12",
                    "value_source": "reported",
                },
            }
        ]
    }
    got = qz.literal_quotes(doc, text)
    assert "acquisitions[0].magnetic_strength" in got, got
    assert "3 T" in got["acquisitions[0].magnetic_strength"]
    assert "acquisitions[0].software" not in got, "SPM12 occurs three times; not determined"


def test_the_literal_locator_will_not_match_inside_a_longer_number():
    """`spans._tolerant_pattern` anchors nothing, so a bare `3` hits the `3` of `13`. A
    value that only appears as part of another token is not present at all."""
    text = "Thirteen volunteers (n = 13) completed the protocol without incident.\n"
    doc = {
        "groups": [
            {
                "local_id": "g",
                "size": {
                    "extraction_status": "extracted",
                    "value": "3",
                    "value_source": "reported",
                },
            }
        ]
    }
    assert qz.literal_quotes(doc, text) == {}


def test_a_field_no_locator_placed_carries_no_set_to_label():
    doc = payload()
    qz.apply_evidence(doc, {})
    (acq,) = doc["acquisitions"]
    evidence = acq["magnetic_strength"]["evidence"]
    assert evidence["status"] == "not_found"
    assert "sets" not in evidence
