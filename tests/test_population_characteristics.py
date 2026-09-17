"""The partition rule, on the strings that produced it.

Every case here is a value observed in the corpus or in one of the three schema-test
extraction rounds; none is invented. The rounds are what made the rule necessary -- asking
for selectivity in the slot description left 24%, then 14%, then 7% non-selective entries --
so the values they produced are the regression set.
"""

from __future__ import annotations

import pytest

from pondie.normalization import population_characteristics as pc

#: The four categories the user named as unwanted, plus the forms the corpus sweep found for
#: each rule. These must leave `population_characteristics`.
MOVES = [
    ("normal-weight", "weight"),
    ("Healthy weight children", "weight"),
    ("normal body mass index", "weight"),
    ("right-hand dominant", "handedness"),
    ("Right-handed", "handedness"),
    ("Right-handedness", "handedness"),
    ("typically developing", "development"),
    ("Typically developing children and adolescents", "development"),
    ("healthy", "health"),
    ("Healthy controls", "health"),
    ("physically healthy", "health"),
    ("no neurological or psychiatric disorder", "no_condition"),
    ("No history of head trauma", "no_condition"),
    ("No serious medical conditions", "no_condition"),
    ("no chronic conditions", "no_condition"),
    ("No DSM-IV Axis I disorder", "no_condition"),
    ("Normal or corrected-to-normal vision", "senses"),
    ("not colour-blind", "senses"),
    ("adequate hearing", "senses"),
    ("Cognitively normal", "cognition"),
    ("normal intelligence", "cognition"),
    ("Native English speaker", "language"),
    ("English-speaking", "language"),
    ("Fluent in English", "language"),
    ("Eligible for MRI", "mri_eligibility"),
    ("No contraindications for MRI", "mri_eligibility"),
    ("Not pregnant", "mri_eligibility"),
    ("Provided written informed consent", "consent"),
]

#: Selective traits. Every one is a real entry from a schema-test round or a curated corpus
#: term, and a rule that moves any of them has destroyed the field's only purpose.
KEEPS = [
    "smokers",
    "non-abstinent",
    "Otherwise healthy adult smokers",
    "Mean smoking rate was 23.3 cigarettes per day",
    "stable at maximal lifetime weight for at least 6 months",
    "Cocaine use disorders; current cocaine dependence in 8 participants",
    "Heavy drinkers",
    "non–treatment-seeking",
    "social drinkers",
    "long-term mindfulness meditation practice",
    "at least three years of daily practice",
    "undergraduates",
    "sedentary",
    "college athletes",
    "heavy caffeine consumers",
    "obese",
    "overweight",
    "bilinguals",
    "professional musicians",
    "veterans",
    "shift workers",
    "vegetarian",
]

#: The two asymmetries from the module docstring. A blunter rule gets each backwards, and
#: getting them backwards moves the trait a study recruited for.
ASYMMETRIC = [
    ("Right-handed", pc.NORMATIVE),
    ("left-handed", pc.KEPT),
    ("mixed-handed", pc.KEPT),
    ("ambidextrous", pc.KEPT),
    ("no history of neurological illness", pc.NORMATIVE),
    ("no history of smoking", pc.KEPT),
    ("no substance abuse", pc.KEPT),
    ("cannabis use less than 50 times", pc.KEPT),
    ("No PTSD diagnosis", pc.KEPT),
    ("no post-traumatic stress disorder", pc.KEPT),
    ("no significant re-experiencing, avoidance, or hyperarousal symptoms", pc.KEPT),
]


@pytest.mark.parametrize("text,rule", MOVES)
def test_non_selective_values_move(text: str, rule: str) -> None:
    verdict = pc.normalize(text)
    assert verdict.kind == pc.NORMATIVE, f"{text!r} stayed ({verdict.rule})"
    assert verdict.rule == rule


@pytest.mark.parametrize("text", KEEPS)
def test_selective_traits_stay(text: str) -> None:
    verdict = pc.normalize(text)
    assert verdict.kind == pc.KEPT, f"{text!r} moved via {verdict.rule}"


@pytest.mark.parametrize("text,kind", ASYMMETRIC)
def test_the_two_asymmetries(text: str, kind: str) -> None:
    assert pc.normalize(text).kind == kind


@pytest.mark.parametrize("text", ["not reported", "N/A", "none", "No specific characteristics",
                                  "unremarkable", "", "   "])
def test_content_free_entries_are_dropped_not_moved(text: str) -> None:
    """`other_characteristics` keeps track of values; there is nothing here to keep."""
    assert pc.normalize(text).kind == pc.EMPTY


def test_matching_is_full_string_not_substring() -> None:
    """The case that decides the whole design.

    `search(r"healthy")` moves "Otherwise healthy adult smokers" and loses the cohort's
    defining trait. Reducing to a core and matching it in full keeps it.
    """
    assert pc.core("Otherwise healthy adult smokers") == "healthy adult smokers"
    assert pc.normalize("Otherwise healthy adult smokers").kind == pc.KEPT
    assert pc.normalize("otherwise healthy").kind == pc.NORMATIVE


def test_core_strips_generic_person_nouns_from_both_ends() -> None:
    assert pc.core("Healthy weight children") == "healthy weight"
    assert pc.core("All participants were Caucasian") == "caucasian"
    assert pc.core("typically developing controls") == "typically developing"


def test_apply_partitions_a_group_and_marks_the_moved_side_derived() -> None:
    record = {
        "groups": [
            {
                "local_id": "g1",
                "population_characteristics": {
                    "value": ["Heavy drinkers", "right-handed", "Heavy drinkers",
                              "normal or corrected-to-normal vision", "not reported"],
                    "extraction_status": "extracted",
                    "value_source": "reported",
                    "evidence": [{"sentence_ids": ["s1"]}],
                },
            }
        ]
    }
    tally = pc.apply(record)
    group = record["groups"][0]
    assert group["population_characteristics"]["value"] == ["Heavy drinkers"]
    assert group["other_characteristics"]["value"] == [
        "right-handed", "normal or corrected-to-normal vision"
    ]
    # The kept side keeps the model's evidence, because it is evidence for those values.
    assert group["population_characteristics"]["value_source"] == "reported"
    assert group["population_characteristics"]["evidence"] == [{"sentence_ids": ["s1"]}]
    # The moved side is the rule's answer, not the model's, and says so.
    assert group["other_characteristics"]["value_source"] == "derived"
    assert tally == {"groups": 1, "kept": 1, "moved": 2, "dropped": 1, "deduped": 1}


def test_apply_leaves_an_unread_field_alone() -> None:
    """Unread is not empty. A group nobody read is not a group with no characteristics."""
    record = {"groups": [{"local_id": "g1"},
                         {"local_id": "g2",
                          "population_characteristics": {"value": None,
                                                         "extraction_status": "not_reported"}}]}
    assert pc.apply(record)["groups"] == 0
    assert "other_characteristics" not in record["groups"][0]
    assert "other_characteristics" not in record["groups"][1]


def test_apply_does_not_create_the_slot_when_nothing_moves() -> None:
    record = {"groups": [{"population_characteristics": {"value": ["heavy drinkers"],
                                                         "extraction_status": "extracted"}}]}
    pc.apply(record)
    assert "other_characteristics" not in record["groups"][0]


def test_every_curated_task_term_survives() -> None:
    """157 curated task names, 0 moved. A task name is never a cohort trait."""
    import csv
    from pathlib import Path

    path = Path("/home/jdkent/projects/autonima-results/experiments/record_arms/data"
                "/vocab_tasks_name.csv")
    if not path.is_file():
        pytest.skip("curated task vocabulary not on this host")
    moved = [
        form
        for row in csv.DictReader(path.open())
        for form in [row["canonical_label"], *row["variants"].split(" | ")]
        if pc.normalize(form).kind == pc.NORMATIVE
    ]
    assert moved == []
