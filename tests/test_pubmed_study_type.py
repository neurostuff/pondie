"""`Study.study_type` and `Study.language`, filled from PubMed rather than from the paper.

Six of neurometabench's sixteen meta-analyses exclude by publication type -- "editorial
letters, case-reports, systematic reviews, meta-analyses" -- and the slot was `None` on all
1,817 committed records with no writer anywhere in the package, so the criterion could not
be written. "English language" is stated by at least six of them and had no slot at all.
The network is not exercised here; `summaries` is the seam and `fill` is what has the logic
worth testing.
"""

from __future__ import annotations

import pytest

from pondie.extraction import pubmed


def test_fill_writes_the_types_verbatim():
    """The slot's description asks for PubMed's own strings, so nothing is mapped."""
    record = {"local_id": "35664889"}
    changed = pubmed.fill(record, {"35664889": ["Journal Article", "Review"]})
    assert record["study_type"] == ["Journal Article", "Review"]
    assert changed and "Review" in changed[0]


def test_a_pmid_the_lookup_did_not_reach_is_left_alone():
    """Writing an empty list would assert PubMed assigns the article no type, and it always
    assigns at least `Journal Article`. "We could not ask" is not "it has none"."""
    record = {"local_id": "35664889"}
    assert pubmed.fill(record, {}) == []
    assert "study_type" not in record


def test_a_record_whose_id_is_not_a_pmid_is_left_alone():
    record = {"local_id": "2abntY3hQSyq"}
    assert pubmed.fill(record, {"2abntY3hQSyq": ["Review"]}) == []
    assert "study_type" not in record


def test_filling_twice_is_not_a_change():
    record = {"local_id": "1", "study_type": ["Review"]}
    assert pubmed.fill(record, {"1": ["Review"]}) == []


def test_the_exclusion_set_names_the_types_the_criteria_name():
    """One definition of "not original data", so a query does not carry its own."""
    for kind in ("Review", "Systematic Review", "Meta-Analysis", "Editorial", "Letter",
                 "Case Reports", "Comment"):
        assert kind in pubmed.NOT_ORIGINAL_RESEARCH
    for kind in ("Journal Article", "Randomized Controlled Trial", "Observational Study",
                 "Comparative Study", "Clinical Trial"):
        assert kind not in pubmed.NOT_ORIGINAL_RESEARCH


def test_the_slot_is_declared_on_the_extraction_side():
    """It is `deterministic` in storage, so the projection drops it, and the code that
    fills it runs on the extraction side -- the `mirror_of` case. Without the declaration
    the fill writes an attribute the validator does not know."""
    from pondie import schema
    from pondie.schema import reader

    extraction = reader.load(schema.EXTRACTION)
    slot = extraction.attributes("Study").get("study_type")
    assert slot is not None, "study_type must be declared for a filled record to validate"
    assert slot.multivalued
    assert extraction.classify("study_type", slot) == "native", (
        "PubMed is not the paper, so there is no span to cite and no wrapper to justify")


def test_is_healthy_is_declared_and_uses_a_permissible_value_source():
    """`ValueSource` offers `reported` and `generated`. `apply` wrote `derived`, which is
    neither, so every group it touched was a validation error."""
    from pondie import schema
    from pondie.extraction.record.validate import _VALUE_SOURCE
    from pondie.normalization.is_healthy import apply
    from pondie.schema import reader

    extraction = reader.load(schema.EXTRACTION)
    assert extraction.attributes("Group").get("is_healthy") is not None

    record = {"groups": [{"local_id": "g1", "medical_condition": {
        "value": ["obesity"], "extraction_status": "extracted"}}]}
    apply(record)
    written = record["groups"][0]["is_healthy"]
    assert written["value"] is False
    assert written["value_source"] in _VALUE_SOURCE


@pytest.mark.parametrize("ids,expect", [
    (["123", "not-a-pmid", "", None, 456], ["123", "456"]),
])
def test_only_numeric_ids_are_asked_for(ids, expect, monkeypatch):
    seen: list[list[str]] = []

    def fake_open(url, data=None, timeout=0):  # noqa: ARG001
        import urllib.parse
        query = urllib.parse.parse_qs(data.decode())
        seen.append(query["id"][0].split(","))

        class R:
            def read(self):
                return b'{"result": {"uids": []}}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return R()

    monkeypatch.setattr(pubmed.urllib.request, "urlopen", fake_open)
    monkeypatch.setattr(pubmed.time, "sleep", lambda _s: None)
    pubmed.summaries(ids)
    assert seen == [expect]


def test_fill_writes_language_from_the_same_answer():
    """One `esummary` call carries both fields, so the second costs no second request."""
    record = {"local_id": "34400176"}
    changed = pubmed.fill(record, {"34400176": {"study_type": ["Journal Article"],
                                                "language": ["eng"]}})
    assert record["study_type"] == ["Journal Article"]
    assert record["language"] == ["eng"]
    assert len(changed) == 2


def test_fill_still_takes_the_shape_publication_types_returns():
    """That function was the whole module once and callers still hold its output."""
    record = {"local_id": "1"}
    assert pubmed.fill(record, {"1": ["Review"]})
    assert record["study_type"] == ["Review"] and "language" not in record


def test_an_empty_field_is_not_written():
    """PubMed indexes every article in at least one language; an empty list would be a
    claim that it did not."""
    record = {"local_id": "1"}
    pubmed.fill(record, {"1": {"study_type": ["Review"], "language": []}})
    assert "language" not in record


def test_language_is_declared_on_the_extraction_side():
    from pondie import schema
    from pondie.schema import reader

    extraction = reader.load(schema.EXTRACTION)
    slot = extraction.attributes("Study").get("language")
    assert slot is not None, "language must be declared for a filled record to validate"
    assert slot.multivalued
    assert extraction.classify("language", slot) == "native"
