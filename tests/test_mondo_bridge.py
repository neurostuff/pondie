"""MONDO as identity, ONVOC as the query grain, and the derivation between them.

"Do the two agree, or are they the same concept at two levels" is answered as an
identifier relation with a path behind it, not by comparing labels. Rationale:
docs/condition-normalization.md.

Skipped without `data/vocab/mondo.json`; `python -m pondie.vocabularies.fetch mondo`.
"""

from __future__ import annotations

import pytest

from pondie.normalization import medical_condition as mc
from pondie.vocabularies.mondo import MONDO, load_mondo, onvoc_crosswalk

needs_mondo = pytest.mark.skipif(not MONDO.is_file(), reason="MONDO release not fetched")


@pytest.fixture(scope="module")
def vocab():
    return load_mondo()


@pytest.fixture(scope="module")
def crosswalk():
    return onvoc_crosswalk()


@needs_mondo
def test_a_missing_release_says_what_to_run(tmp_path) -> None:
    """The loader read a path nothing in the repository ever wrote."""
    with pytest.raises(FileNotFoundError, match="pondie.vocabularies.fetch"):
        load_mondo(tmp_path / "absent.json")


@needs_mondo
def test_the_node_carries_the_crosswalks_a_clinical_system_asks_in(vocab) -> None:
    """Choosing MONDO is not choosing against SNOMED: the node carries both ids."""
    node = vocab.exact("schizophrenia")
    assert vocab.curie(node) == "MONDO:0005090"
    assert vocab.umls[node] == "C0036341"
    assert vocab.sctid[node] == "58214004"


@needs_mondo
def test_a_synonym_is_indexed_and_reachable(vocab) -> None:
    """102,615 surface forms against 32,109 labels, and the difference is what papers write."""
    assert len(vocab.forms) > 3 * len(vocab.labels)
    assert vocab.exact("Alzheimer disease") == vocab.exact("Alzheimer's disease")


@needs_mondo
def test_a_subtype_reaches_its_onvoc_term_by_walking_up(vocab, crosswalk) -> None:
    """ONVOC has `Dementia` and nothing below it, so bvFTD has no crosswalk of its own.

    Walking MONDO's ancestors to the nearest node that has one reaches it, and the path
    is kept so the claim can be checked rather than believed.
    """
    node = vocab.exact("behavioral variant of frontotemporal dementia")
    onvoc_id, label, via, path = mc.bridge(node, vocab, crosswalk)
    assert (label, via) == ("Dementia", "ancestor")
    assert onvoc_id == "ONVOC:0000190"
    assert any("dementia (MONDO:0001627)" == step for step in path)


@needs_mondo
def test_a_term_with_its_own_tie_does_not_walk(vocab, crosswalk) -> None:
    assert mc.bridge(vocab.exact("schizophrenia"), vocab, crosswalk)[2] == "crosswalk"


@pytest.fixture(scope="module")
def onvoc():
    from pondie.vocabularies.onvoc import load_onvoc

    return load_onvoc().scoped(("disorders",))


@needs_mondo
def test_walking_up_recovers_the_coverage_gaps_the_flat_route_reported(
    vocab, crosswalk, onvoc
) -> None:
    """`alcohol dependence` and `nicotine dependence` are not gaps after all.

    normalization-layer.md lists them as ONVOC coverage gaps at 106 and 101 studies,
    measured against ONVOC's flat surface. Their ancestors reach `Substance Dependence`.
    """
    for term in ("alcohol dependence", "nicotine dependence", "cocaine dependence"):
        _id, label, via, _path = mc.bridge(vocab.exact(term), vocab, crosswalk, onvoc)
        assert (label, via) == ("Substance Dependence", "ancestor"), term


@needs_mondo
def test_a_term_onvoc_has_is_never_proposed_back_to_it(vocab, crosswalk, onvoc) -> None:
    """ONVOC carries these two and the crosswalk ties neither, so an identifier-only
    bridge would propose terms the vocabulary already has."""
    for term, expected in (
        ("post-traumatic stress disorder", "Post-Traumatic Stress Disorder"),
        ("Huntington disease", "Huntington's Disease"),
    ):
        _id, label, via, _path = mc.bridge(vocab.exact(term), vocab, crosswalk, onvoc)
        assert (label, via) == (expected, "name"), term


@needs_mondo
def test_a_crosswalk_row_that_cannot_be_right_decides_nothing(vocab, crosswalk, onvoc) -> None:
    """`MONDO:0005148` is listed under both Type 1 and Type 2 Diabetes Mellitus, so
    taking the first row seen made every type 2 cohort a type 1 cohort."""
    assert "MONDO:0005148" not in crosswalk
    for term in ("type 1 diabetes mellitus", "type 2 diabetes mellitus"):
        _id, label, _via, _path = mc.bridge(vocab.exact(term), vocab, crosswalk, onvoc)
        assert label.lower() == term.lower()


@needs_mondo
def test_a_condition_onvoc_cannot_name_at_any_grain_is_a_proposal(
    vocab, crosswalk, onvoc
) -> None:
    """What is left after all four layers: a real gap, and the evidence for a term."""
    node = vocab.exact("anorexia nervosa")
    _id, label, via, _path = mc.bridge(node, vocab, crosswalk, onvoc)
    assert (label, via) == ("", "")


@needs_mondo
def test_the_lexical_layer_prefers_the_form_the_paper_wrote(vocab) -> None:
    """`variants` promises "most faithful first" and the dict comprehension kept the LAST."""
    assert mc._lexical("schizophrenia", vocab) == vocab.exact("schizophrenia")


@needs_mondo
def test_a_record_becomes_one_row_per_head_with_both_grains(tmp_path, monkeypatch) -> None:
    """The whole route, with the encoder stubbed so the plumbing is what is under test.

    Six groups covering every outcome: a subtype that bridges by ancestor, a value that
    denies one condition while naming another, an absence from the string, an absence the
    `is_healthy` flag supplies for a non-answer, two comorbidities in one group, and a
    value nothing matches.
    """
    import json

    class Stub:
        def __init__(self, vocab):
            self.vocab = vocab

        def build(self):
            return self

        def nearest(self, queries, top=5):
            return [(0, 0.10, [1, 2]) for _ in queries]

    monkeypatch.setattr(mc, "Index", Stub)
    (tmp_path / "S1.extraction.json").write_text(json.dumps({"study": {
        "local_id": "S1",
        "groups": [
            {"local_id": "g1", "medical_condition": {
                "value": ["behavioral variant of frontotemporal dementia"],
                "extraction_status": "extracted"}},
            {"local_id": "g2", "medical_condition": {
                "value": ["no neurological or psychiatric disorder"],
                "extraction_status": "extracted"}},
            {"local_id": "g3", "medical_condition": {
                "value": ["schizophrenia; no substance abuse"],
                "extraction_status": "extracted"}},
            {"local_id": "g4", "is_healthy": {"value": True}, "medical_condition": {
                "value": ["unknown"], "extraction_status": "extracted"}},
            {"local_id": "g5", "medical_condition": {
                "value": ["nicotine dependence", "type 2 diabetes mellitus"],
                "extraction_status": "extracted"}},
            {"local_id": "g6", "medical_condition": {
                "value": ["zzzq nonexistent condition"], "extraction_status": "extracted"}},
        ]}}))

    out = mc.normalize((str(tmp_path / "*.extraction.json"),))
    by_group = {row.group: row for row in out.links}

    assert by_group["g1"].mondo == "MONDO:0017160"
    assert (by_group["g1"].onvoc, by_group["g1"].onvoc_via) == ("Dementia", "ancestor")
    assert by_group["g1"].sctid == "716994006"

    assert by_group["g3"].head == "schizophrenia"
    assert by_group["g3"].denied == ("substance abuse",)

    assert by_group["g2"].sentinel == "NO_CONDITION"
    # The flag's one job: the string said nothing either way, so it supplied the absence.
    assert (by_group["g4"].sentinel, by_group["g4"].scope) == ("NO_CONDITION", "flag")
    assert out.cohorts["absent by flag"] == 1

    assert {row.head for row in out.links if row.group == "g5"} == {
        "nicotine dependence", "type 2 diabetes mellitus"
    }
    assert by_group["g6"].method == "rejected" and not by_group["g6"].matched

    assert ("behavioral variant of frontotemporal dementia", "Dementia") in {
        (gap["label"], gap["under"]) for gap in out.grain_gaps(minimum=1)
    }
