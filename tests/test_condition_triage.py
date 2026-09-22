"""What a `medical_condition` value asserts, before any vocabulary is asked.

Every case here was a wrong answer in production, from a gate that was one regex anchored
at the start of the value and tested before the split. Why each one matters:
docs/condition-normalization.md.
"""

from __future__ import annotations

import pytest

from pondie.normalization._negation import available
from pondie.vocabularies.phrases import NO_CONDITION, NOT_READ, triage

needs_parser = pytest.mark.skipif(not available(), reason="negation scope needs spaCy")


@pytest.mark.parametrize(
    "value",
    [
        "no neurological or psychiatric disorder",
        "absence of major depressive disorder",
        "No clinically significant cognitive impairment No dementia",
        "no history of major depression or other mental illness",
        "free of Axis I disorders",
        "HIV-negative",
        "drug-free",
    ],
)
@needs_parser
def test_a_value_that_denies_everything_names_nothing(value: str) -> None:
    """The gate's original job, plus the nominal (`absence of X`) and suffixed
    (`HIV-negative`) forms it could not do."""
    assert triage(value).kind == NO_CONDITION
    assert triage(value).heads == ()


@needs_parser
def test_a_negation_inside_one_conjunct_does_not_reach_the_others() -> None:
    """`Parkinson's disease; no dementia` is one diagnosis and one denial, not two."""
    triaged = triage("Parkinson's disease; no dementia")
    assert triaged.heads == ("Parkinson's disease",)
    assert triaged.denied == ("dementia",)


@needs_parser
def test_a_negation_scopes_over_the_coordination_it_governs() -> None:
    """Parse the whole value BEFORE splitting it, because a cue scopes over an `or`.

    Split first and the second conjunct arrives with no cue left in it.
    """
    assert triage("no history of major depression or other mental illness").heads == ()


@needs_parser
def test_a_mid_string_negation_is_found_where_an_anchored_one_is_not() -> None:
    """A real corpus value, and one the leading anchor read as a sick cohort."""
    value = (
        "physically healthy adults without severe neurodevelopmental, "
        "neuropsychiatric, or neurologic disorders"
    )
    assert triage(value).kind == NO_CONDITION


@pytest.mark.parametrize(
    ("value", "head"),
    [
        ("healthy smokers", "healthy smokers"),
        ("healthy obese adults", "healthy obese adults"),
    ],
)
def test_healthy_in_front_of_a_condition_is_not_an_absence(value: str, head: str) -> None:
    """`healthy` was an unanchored alternative of the negation pattern, so the addiction
    and obesity literature's "healthy" swallowed the condition beside it."""
    assert triage(value).heads == (head,)


@pytest.mark.parametrize(
    "value", ["healthy", "healthy controls", "healthy older adults", "Healthy, non-clinical population", "none"]
)
def test_a_bare_assertion_of_wellness_is_an_absence(value: str) -> None:
    """The other side of the same anchor: what IS only wellness words still matches."""
    assert triage(value).kind == NO_CONDITION


@pytest.mark.parametrize("value", ["not reported", "unknown", "n/a", "not applicable"])
def test_a_non_answer_is_not_a_healthy_cohort(value: str) -> None:
    """The third state: `is_healthy` is unset for an unread cohort, not True."""
    assert triage(value).kind == NOT_READ


@pytest.mark.parametrize(
    ("value", "heads", "qualifiers"),
    [
        ("treatment-resistant major depression", ("major depression",), ("treatment-resistant",)),
        ("first-episode schizophrenia", ("schizophrenia",), ("first-episode",)),
        ("schizophrenia or bipolar disorder", ("schizophrenia", "bipolar disorder"), ()),
        ("Schizophrenia patients", ("Schizophrenia",), ()),
    ],
)
def test_the_parse_does_not_rewrite_the_value(value, heads, qualifiers) -> None:
    """A tokenizer does not round-trip, so the scope is cut by character offset.

    Rejoining turned `treatment-resistant` into `treatment - resistant`, which the
    qualifier pattern no longer matches.
    """
    triaged = triage(value)
    assert triaged.heads == heads
    assert triaged.qualifiers == qualifiers


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        ("no dementia", ""),
        ("schizophrenia", "schizophrenia"),
        ("schizophrenia, no substance abuse", "schizophrenia"),
        ("free of Axis I disorders", ""),
        ("HIV-negative", ""),
        ("drug-free", ""),
    ],
)
def test_the_cue_window_holds_up_without_a_parse(value: str, kept: str) -> None:
    """The fallback is the whole negation layer on a host with no spaCy, so it is tested
    on its own rather than only behind the parse."""
    from pondie.normalization._negation import cue_forward_scope

    assert cue_forward_scope(value) == kept


def test_triage_still_gates_when_no_parser_is_installed(monkeypatch) -> None:
    """Degrading to the cue window must not degrade to no gate at all."""
    from pondie.normalization import _negation

    monkeypatch.setattr(_negation, "scope", lambda text: None)
    assert triage("no neurological or psychiatric disorder").kind == NO_CONDITION
    assert triage("Parkinson's disease; no dementia").heads == ("Parkinson's disease",)
    assert triage("schizophrenia").heads == ("schizophrenia",)
