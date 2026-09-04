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


def test_a_retrieved_passage_is_labelled_as_the_retriever(monkeypatch):
    """The union set exists to recover fields the model did not quote, so its label has to
    survive the path where it is the *only* set as well as the path where it is second."""

    class Unit:
        text = "Images were acquired on a 3 T Siemens Trio."

    monkeypatch.setattr(qz.retrieval, "locate", lambda *a, **k: Unit())

    doc = payload()
    qz.apply_evidence(doc, {}, reranker=object(), units=["x"])
    assert sources(doc) == ["retriever"]

    doc = payload()
    qz.apply_evidence(doc, {"acquisitions[0].magnetic_strength": "Scanning used a 3 T magnet."},
                      reranker=object(), units=["x"])
    assert sources(doc) == ["model_quote", "retriever"]


def test_the_retriever_does_not_relabel_the_passage_the_model_already_quoted(monkeypatch):
    """A duplicate is not a second warrant, and counting it as one would inflate the
    retriever's apparent contribution by exactly the fields it agreed with."""

    quoted = "Scanning used a 3 T magnet."

    class Unit:
        text = quoted

    monkeypatch.setattr(qz.retrieval, "locate", lambda *a, **k: Unit())
    doc = payload()
    qz.apply_evidence(doc, {"acquisitions[0].magnetic_strength": quoted},
                      reranker=object(), units=["x"])
    assert sources(doc) == ["model_quote"]


def test_a_field_no_locator_placed_carries_no_set_to_label():
    doc = payload()
    qz.apply_evidence(doc, {})
    (acq,) = doc["acquisitions"]
    evidence = acq["magnetic_strength"]["evidence"]
    assert evidence["status"] == "not_found"
    assert "sets" not in evidence


def test_a_duplicate_survives_no_difference_in_whitespace(monkeypatch):
    """The retriever's unit and the model's quote do not agree on whitespace.

    On 19914045 the model copied the paper's non-breaking spaces and the retriever's unit
    carried ordinary ones, so the exact-substring guard called one sentence two passages and
    the field kept two sets holding the same text.
    """

    quoted = "Duration of the disorder ranged from 1 to 63 months (mean = 10.7 months)."

    class Unit:
        text = "Duration of the disorder ranged from 1 to 63 months (mean = 10.7 months)."

    monkeypatch.setattr(qz.retrieval, "locate", lambda *a, **k: Unit())
    doc = payload()
    qz.apply_evidence(doc, {"acquisitions[0].magnetic_strength": quoted},
                      reranker=object(), units=["x"])
    assert sources(doc) == ["model_quote"]
