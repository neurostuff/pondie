"""The condition linker's exact layers, on cases the production catalog got wrong.

Each case is a string ns-pond's catalog held and the 2026-11-27 audit (ns-pond PR 259)
graded wrong or unmapped. None of these needs the MONDO release: the vocabulary is built here.
"""

from __future__ import annotations

import json

import pytest

from pondie.normalization import medical_condition as mc
from pondie.vocabularies.abbreviations import Abbreviations, expansions_in
from pondie.vocabularies.folding import variants
from pondie.vocabularies.mondo import load_mondo
from pondie.vocabularies.phrases import NO_CONDITION, triage


@pytest.mark.parametrize(
    "value",
    [
        "Attention-deficit/hyperactivity disorder",
        "attention deficit/hyperactivity disorder",
        "avoidant/restrictive food intake disorder",
        "Fragile X-associated tremor/ataxia syndrome",
    ],
)
def test_a_slash_inside_one_name_does_not_split_it(value: str) -> None:
    """Split, ADHD became "attention-deficit", which retrieval placed at the inattentive
    subtype: 572 wrong ids."""
    assert triage(value).heads == (value,)


def test_a_slash_between_two_conditions_still_splits() -> None:
    assert triage("Overweight/obesity").heads == ("Overweight", "obesity")
    assert triage("Alcohol and/or cannabis abuse").heads == ("Alcohol", "cannabis abuse")


def test_a_substance_named_once_is_written_onto_both_conditions() -> None:
    """The second head was a bare "dependence", which names no substance."""
    assert triage("past alcohol abuse/dependence").heads == ("alcohol abuse", "alcohol dependence")
    assert triage("Cannabis abuse or dependence").heads == (
        "Cannabis abuse",
        "Cannabis dependence",
    )


@pytest.mark.parametrize(
    "value, heads",
    [
        # retrieval accepted the whole head as *panic disorder 2*, an OMIM locus
        ("Current panic disorder in 2", ("panic disorder",)),
        # the slash in the count split the value before anything removed it
        ("Cognitive impairment in 18/44 patients", ("Cognitive impairment",)),
        ("Diabetes mellitus: 9 (23.7%)", ("Diabetes mellitus",)),
        ("Alcohol use disorder in 27.0% of participants", ("Alcohol use disorder",)),
        ("Major depressive disorder in a subset of participants", ("Major depressive disorder",)),
        ("Psychopathy in 43 participants meeting PCL-R total score ≥30", ("Psychopathy",)),
        # the count ends where the list of conditions resumes
        (
            "Physical illness in 28 participants, related to hypertension or substance abuse",
            ("Physical illness, related to hypertension", "substance abuse"),
        ),
    ],
)
def test_a_count_is_not_part_of_the_name(value: str, heads: tuple[str, ...]) -> None:
    assert triage(value).heads == heads


@pytest.mark.parametrize(
    "value",
    ["Normal hearing", "Neurotypical", "Typically developed controls", "Normal vision",
     "Cognitively healthy in 40 participants"],
)
def test_a_control_group_the_corpus_writes_is_an_absence(value: str) -> None:
    """2,185 items like these were written as unmapped conditions with role `primary`."""
    assert triage(value).kind == NO_CONDITION


def test_an_onset_qualifier_comes_off_whole() -> None:
    """`adolescent` alone was lifted, leaving "onset schizophrenia"."""
    assert triage("Adolescent-onset schizophrenia").heads == ("schizophrenia",)


def test_a_value_that_is_one_known_name_is_one_head() -> None:
    """"without" is a negation cue and `chronic` a course qualifier; inside a name they are
    neither. Without `known` triage cannot tell, and says so by splitting."""
    names = {"migraine without aura", "chronic fatigue syndrome"}

    def known(text: str) -> bool:
        return text.lower() in names

    assert triage("Migraine without aura", known).heads == ("Migraine without aura",)
    assert triage("Chronic fatigue syndrome", known).heads == ("Chronic fatigue syndrome",)
    assert triage("Chronic fatigue syndrome").heads == ("fatigue syndrome",)


def test_a_closed_prefix_reaches_the_hyphenated_form() -> None:
    """MONDO writes *post-traumatic stress disorder*; 398 PTSD items wrote "posttraumatic"."""
    assert "post traumatic stress disorder" in variants("Posttraumatic stress disorder")
    assert "posttraumatic stress disorder" in variants("post-traumatic stress disorder")


def test_a_slash_short_form_the_paper_defined_is_expanded_whole() -> None:
    """The miner keeps `/` in a short form and the lookup used to drop it, so a paper's own
    "(AD/HD)" was read as `AD` and `HD`."""
    text = (
        "Children with attention-deficit/hyperactivity disorder (AD/HD) and oppositional "
        "defiant disorder (ODD) or conduct disorder (CD) were recruited."
    )
    store = Abbreviations().for_paper(text, "p1")
    assert [short for short, _ in expansions_in("AD/HD in 46% of patients", store)] == ["AD/HD"]
    assert [short for short, _ in expansions_in("ODD/CD", store)] == ["ODD", "CD"]


def test_an_onvoc_iri_is_written_as_the_crosswalk_writes_it() -> None:
    """The name layers wrote the release's IRI, so one concept carried two ids."""
    assert mc.onvoc_curie("https://w3id.org/onvoc/ONVOC_0000243") == "ONVOC:0000243"
    assert mc.onvoc_curie("ONVOC:0000243") == "ONVOC:0000243"


@pytest.fixture
def tiny(tmp_path):
    """A six-node MONDO: a disorder and its numbered subtype sharing a form, a susceptibility
    entry, a disease a modifier narrows, a locus with a clinical synonym, and the node mild
    cognitive impairment falls under."""

    def node(i: str, label: str, *synonyms: str) -> dict:
        return {
            "id": f"http://purl.obolibrary.org/obo/MONDO_{i}",
            "type": "CLASS",
            "lbl": label,
            "meta": {"synonyms": [{"pred": "hasExactSynonym", "val": s} for s in synonyms]},
        }

    graph = {
        "graphs": [
            {
                "nodes": [
                    # the numbered node is listed first, so `setdefault` would have kept it
                    node("9000002", "cerebral arteriopathy, type 1", "CADASIL"),
                    node("9000001", "cerebral arteriopathy", "CADASIL"),
                    node("9000003", "hemorrhage, intracerebral, susceptibility to", "ICH"),
                    node("9000004", "Parkinson disease"),
                    node("9000005", "Alzheimer disease 2", "late-onset Alzheimer disease"),
                    # the real id: `curated/condition-extension.tsv` names it as MCI's parent
                    node("0002039", "cognitive disorder"),
                ],
                "edges": [],
            }
        ]
    }
    path = tmp_path / "mondo.json"
    path.write_text(json.dumps(graph))
    return load_mondo(path)


def test_a_form_a_subtype_shares_belongs_to_the_disorder(tiny) -> None:
    assert tiny.curie(mc._lexical("CADASIL", tiny)) == "MONDO:9000001"


def test_a_susceptibility_entry_is_not_reached_by_its_abbreviation(tiny) -> None:
    """`ICH` named intracerebral haemorrhage in every paper that wrote it."""
    assert mc._lexical("ICH", tiny) is None


def test_a_leading_modifier_is_tried_off_only_after_the_whole_head_missed(tiny) -> None:
    node, method = mc._placed("Idiopathic Parkinson disease", tiny)
    assert (tiny.curie(node), method) == ("MONDO:9000004", "trimmed")
    assert mc._placed("Parkinson disease", tiny)[1] == "lexical"
    # every modifier off first, so the broad disease wins, then one fewer: taking both off
    # "early progressive" alone lost *progressive supranuclear palsy*
    assert mc._trimmed("Early progressive supranuclear palsy") == [
        "supranuclear palsy", "progressive supranuclear palsy",
    ]


def test_the_whole_name_check_does_not_take_a_numbered_node_for_an_unnumbered_name(tiny) -> None:
    """"late-onset Alzheimer disease" is a synonym of *Alzheimer disease 2* alone; kept whole
    it reached the locus, split as usual it reaches the disease."""
    is_name = mc.known(tiny)
    assert not is_name("late-onset Alzheimer disease")
    assert is_name("cerebral arteriopathy, type 1")
    assert is_name("CADASIL")


def test_a_curated_row_whose_node_this_release_lacks_is_skipped(tiny) -> None:
    """The alias table names real MONDO ids; none is in this six-node release."""
    assert tiny.aliases == {}


def test_a_condition_mondo_lacks_is_placed_under_the_node_it_falls_under(tiny) -> None:
    """Mild cognitive impairment (2,125 production items) has no MONDO node."""
    for head in ("Mild cognitive impairment", "amnestic MCI", "Progressive mild cognitive impairment"):
        node, method = mc._placed(head, tiny)
        assert (tiny.curie(node), method) == ("MONDO:0002039", "extension"), head
        assert mc.extension_of(head).id == "mild_cognitive_impairment"
    assert mc.known(tiny)("Amnestic mild cognitive impairment")


def test_a_condition_mondo_lacks_is_written_at_both_grains() -> None:
    """The MONDO entry names the parent, marked broader, so a MONDO search still finds the
    group; the extension entry names the condition itself."""
    row = mc.Link(study="s", group="g", text="aMCI in 3 participants", head="aMCI",
                  mondo="MONDO:0002039", label="cognitive disorder", method="extension", score=1.0)
    written = [(e["vocabulary"], e.get("value"), e.get("rollup")) for e in mc.entries(row, "tool")]
    assert written == [("MONDO", "MONDO:0002039", True),
                       ("condition_extension", "mild_cognitive_impairment", None)]


def test_mild_is_a_severity_except_in_mild_cognitive_impairment() -> None:
    assert triage("Subjective cognitive decline or mild cognitive impairment").heads == (
        "Subjective cognitive decline", "mild cognitive impairment",
    )
    assert triage("mild depression").heads == ("depression",)


def test_mci_due_to_a_disease_is_both_conditions() -> None:
    """A group with MCI due to AD has AD, so the disease is a head of its own."""
    assert triage("Mild cognitive impairment due to Alzheimer's disease").heads == (
        "Mild cognitive impairment", "Alzheimer's disease",
    )
    assert triage("amnestic MCI due to AD").heads == ("amnestic MCI", "AD")
