"""What the normalization and cross-corpus query must not get wrong.

A wrong mapping is worse than a missing one. A missing mapping is visible -- the row says
no match and someone looks. A wrong one is queried across a corpus and believed, and the
paper's own wording that would have exposed it is sitting in a field nobody re-reads. So
most of these tests are about refusing, not about matching.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

import pytest


from pipeline import normalize as nz  # noqa: E402
from pipeline import query as q  # noqa: E402


@pytest.fixture(scope="module")
def onvoc():
    return nz.load_onvoc()


# --- surface forms ----------------------------------------------------------

def test_a_parenthetical_acronym_is_its_own_candidate():
    got = nz.variants("Autism Diagnostic Observation Schedule (ADOS)")
    assert "ADOS" in got
    assert "Autism Diagnostic Observation Schedule" in got


def test_laterality_is_stripped_but_only_after_the_whole_phrase_is_tried():
    got = nz.variants("left anterior insula")
    assert got[0] == "left anterior insula"
    assert any("insula" == v.strip().lower() for v in got)


def test_a_phrase_of_only_qualifiers_keeps_its_words():
    # Stripping every content word would leave nothing to look up, and an empty query
    # matches whatever is shortest.
    assert nz.variants("left right") == ["left right"]


# --- acronyms ---------------------------------------------------------------

def test_an_apostrophe_does_not_manufacture_an_initial():
    # Folding `Alzheimer's Disease` leaves a stray `s`, which turned a two-word name
    # into the three-letter ASD -- a different disorder entirely.
    assert nz.acronym("Alzheimer's Disease") == ""


def test_domain_nouns_carry_a_letter():
    # `disorder` and `scale` are exactly the words a clinical acronym is built from.
    assert nz.acronym("Autism Spectrum Disorder") == "asd"


def test_a_two_word_label_has_no_acronym():
    assert nz.acronym("Drug Use") == ""


def test_an_uncorroborated_acronym_is_refused(onvoc):
    # ONVOC has exactly one label whose initials are MDD and it is Mood Dysregulation
    # Disorder, while a paper writing MDD means Major Depressive Disorder.
    record = {"local_id": "S1", "groups": [{"name": {"value": "MDD"}}]}
    mapped = nz.normalize(record, {"ONVOC": onvoc})
    assert [m.matched for m in mapped] == [False]


def test_an_acronym_the_record_spells_out_is_accepted(onvoc):
    record = {"local_id": "S1", "groups": [
        {"name": {"value": "ASD"},
         "description": {"value": "children with autism spectrum disorder"}}]}
    mapped = [m for m in nz.normalize(record, {"ONVOC": onvoc}) if m.path == "groups.name"]
    assert mapped[0].matched and mapped[0].method == "acronym"


# --- branch routing ---------------------------------------------------------

def test_a_test_is_not_matched_to_a_psychological_concept(onvoc):
    # `Wechsler Abbreviated Scale of Intelligence` contains the word `Intelligence`, and
    # an unscoped lookup returns that concept confidently and wrongly.
    record = {"local_id": "S1", "assessments": [
        {"name": {"value": "Wechsler Abbreviated Scale of Intelligence (WASI-IV)"}}]}
    mapped = nz.normalize(record, {"ONVOC": onvoc})
    assert all(m.concept is None or m.concept.branch == "Tests" for m in mapped)


def test_a_group_may_draw_from_disorders_or_populations(onvoc):
    scoped = onvoc.scoped(("disorders", "population"))
    assert len(scoped) < len(onvoc)
    concept, _method, _others = scoped.match("patients with major depressive disorder")
    assert concept is not None and "Depress" in concept.label


def test_an_agent_is_looked_up_only_among_drugs(onvoc):
    scoped = onvoc.scoped(("drugs",))
    assert all(c.branch in nz.BRANCHES["drugs"] for c in scoped.concepts)


# --- morphology -------------------------------------------------------------

def test_a_stem_bridges_depression_to_depressive_disorder(onvoc):
    concept, _method, _others = onvoc.scoped(("disorders",)).match("depression")
    assert concept is not None and "Depress" in concept.label


def test_an_ambiguous_stem_is_not_guessed():
    a = nz.Concept("1", "Alpha Thing", "V")
    b = nz.Concept("2", "Alpha Things", "V")
    vocabulary = nz.Vocabulary("V", [a, b])
    # Both stem alike, so the stem cannot decide and must not.
    assert vocabulary.by_stem == {}


# --- the treatment/control query --------------------------------------------

def _trial(levels, kinds=("pharmacological", "placebo")):
    return {
        "local_id": "S1",
        "design": {"arms": [
            {"local_id": "a1", "name": {"value": "escitalopram"},
             "arm_kind": {"value": kinds[0]}, "agent": {"value": "escitalopram"}},
            {"local_id": "a2", "name": {"value": "placebo"},
             "arm_kind": {"value": kinds[1]}, "agent": {"value": "saline"}}]},
        "analyses": [{"local_id": "an1", "name": {"value": "drug > placebo"},
                      "effect": {"cells": [
                          {"level": {"value": levels[0]},
                           "direction": {"value": "positive"}},
                          {"level": {"value": levels[1]},
                           "direction": {"value": "negative"}}]}}]}


def test_an_intervention_against_a_comparator_is_found():
    found = list(q.treatment_contrasts(_trial(("escitalopram", "placebo"))))
    assert len(found) == 1
    assert found[0].intervention.name == "escitalopram"
    assert found[0].comparator.kind == "placebo"
    # The direction reported is the intervention's, which is the only reading that
    # survives pooling -- each paper names its contrast whichever way round it likes.
    assert found[0].direction == "positive"


def test_a_trial_with_no_comparator_arm_yields_nothing():
    assert list(q.treatment_contrasts(
        _trial(("escitalopram", "placebo"),
               kinds=("pharmacological", "pharmacological")))) == []


def test_a_group_contrast_is_not_a_treatment_contrast():
    record = _trial(("patients", "healthy controls"))
    assert list(q.treatment_contrasts(record)) == []


def test_a_level_naming_two_arms_places_neither():
    record = _trial(("escitalopram", "placebo"))
    # Give both arms the same name so the level is ambiguous.
    record["design"]["arms"][1]["name"]["value"] = "escitalopram"
    assert list(q.treatment_contrasts(record)) == []


@pytest.mark.parametrize("kind,expected", [
    ("pharmacological", "intervention"), ("stimulation", "intervention"),
    ("active_comparator", "intervention"), ("placebo", "comparator"),
    ("sham", "comparator"), ("usual_care", "comparator"),
    ("no_intervention", "comparator"), ("", None)])
def test_every_arm_kind_has_a_side(kind, expected):
    assert q.role(kind) == expected


def test_levels_are_matched_by_words_not_similarity():
    assert not q.same("men", "women")
    assert q.same("REAL", "the REAL group")


# --- abbreviations ----------------------------------------------------------

from pipeline import abbreviations as ab  # noqa: E402


def test_a_definition_in_brackets_is_mined():
    got = ab.mine("We used the Autism Diagnostic Observation Schedule (ADOS) throughout.")
    assert got.get("ADOS") == "Autism Diagnostic Observation Schedule"


def test_a_manufacturer_string_is_not_an_abbreviation():
    # A detector looking for `long form (SF)` also finds `(Philips Medical Systems, Best,
    # The Netherlands)`, whose letters happen to fit.
    assert not ab.Abbreviations.plausible("Systems, Best, The Netherlands")
    assert ab.Abbreviations.plausible("dorsolateral prefrontal cortex")


def test_the_canonical_expansion_is_the_one_seen_most():
    store = ab.Abbreviations()
    for _ in range(3):
        store.learn("the echo-planar imaging (EPI) sequence")
    store.learn("the echoplanar imaging (EPI) sequence")
    assert store.expand("EPI") == "echo-planar imaging"


def test_spelling_variants_are_not_reported_as_disagreements():
    store = ab.Abbreviations()
    store.learn("the Brodmann Area (BA) map")
    store.learn("the Brodmann Areas (BA) map")
    assert store.disagreements() == []


def test_a_genuine_conflict_is_reported():
    store = ab.Abbreviations()
    store.learn("the fractional anisotropy (FA) map")
    store.learn("a flip angle (FA) of ninety degrees")
    assert [s for s, _v in store.disagreements()] == ["fa"]


def test_a_papers_own_definition_beats_the_corpus():
    # `FA` is fractional anisotropy in a diffusion paper and flip angle in an acquisition
    # section. A corpus-wide expansion is wrong for whichever paper meant the other.
    corpus = ab.Abbreviations()
    for _ in range(5):
        corpus.learn("the fractional anisotropy (FA) map")
    assert corpus.expand("FA") == "fractional anisotropy"
    paper = corpus.for_paper("images used a flip angle (FA) of 90 degrees")
    assert paper.expand("FA") == "flip angle"
    # and the corpus store is untouched
    assert corpus.expand("FA") == "fractional anisotropy"


def test_a_curated_entry_is_not_overwritten_by_a_mined_one():
    store = ab.Abbreviations()
    store.add("SSRI", "selective serotonin reuptake inhibitor", "curated")
    store.learn("a superficial siderosis of the retina interface (SSRI)")
    assert store.expand("SSRI") == "selective serotonin reuptake inhibitor"


def test_expansion_reaches_a_vocabulary_the_acronym_cannot(onvoc):
    store = ab.Abbreviations()
    store.add("dlPFC", "dorsolateral prefrontal cortex", "curated")
    record = {"local_id": "S1", "regions": [{"name": {"value": "left dlPFC parcel"}}]}
    without = nz.normalize(record, {"ONVOC": onvoc})
    with_store = nz.normalize(record, {"ONVOC": onvoc}, store)
    assert [m.expansions for m in with_store] == [("dorsolateral prefrontal cortex",)]
    # ONVOC has no dorsolateral entry, so this one still does not match -- but the
    # expansion is now recorded, which is what makes it a usable term proposal.
    assert not without[0].matched


# --- new-term candidates ----------------------------------------------------

def test_unmatched_values_become_counted_candidates():
    rows = [nz.Mapping("A", "assessments.name", "Beck Depression Inventory", None),
            nz.Mapping("B", "assessments.name", "beck depression inventory (BDI)", None),
            nz.Mapping("C", "assessments.name", "Something Else", None)]
    got = nz.candidates(rows, minimum=2)
    assert len(got) == 1
    assert got[0].support == 2
    # The longest surface form is kept: it is the most informative proposal.
    assert got[0].text == "beck depression inventory (BDI)"
    assert got[0].branch_group == "tests"


def test_a_matched_value_is_not_a_candidate(onvoc):
    concept = onvoc.concepts[0]
    rows = [nz.Mapping("A", "groups.name", "x", concept, "exact")]
    assert nz.candidates(rows) == []
