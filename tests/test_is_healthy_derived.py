"""`Group.is_healthy` is derived, and cannot disagree with `medical_condition`."""

from __future__ import annotations

import pytest

from pondie.normalization.is_healthy import apply, derive, is_condition


def group(value, status="extracted", healthy=None):
    g = {"medical_condition": {"value": value, "extraction_status": status}}
    if healthy is not None:
        g["is_healthy"] = {"value": healthy, "extraction_status": "extracted",
                           "value_source": "reported"}
    return g


@pytest.mark.parametrize(
    "value,expected",
    [
        (["nicotine dependence"], False),
        (["obesity"], False),
        (["Substance Dependence (cocaine)"], False),
        (["heavy drinking"], False),
        # a positive assertion of wellness is not a condition
        (["Healthy controls"], True),
        (["healthy"], True),
        (["Healthy, non-clinical population"], True),
        (["normal volunteers"], True),
        (["none"], True),
        # negations are the absence of a condition, which triage already knows
        (["no history of drug abuse"], True),
        (["No Axis I psychiatric disorder"], True),
        # the model looked and found nothing
        ([], True),
        # a comorbidity beside an assertion of health is still a condition
        (["healthy", "obesity"], False),
    ],
)
def test_derivation(value, expected):
    assert derive(group(value)) is expected


@pytest.mark.parametrize("status", ["not_reported", "not_applicable", "unknown"])
def test_unread_is_not_healthy(status):
    """Unset, not False. A cohort nobody read is not a sick one."""
    assert derive(group([], status=status)) is None


def test_absent_field_is_unset():
    assert derive({}) is None
    assert derive({"medical_condition": None}) is None


def test_apply_overrules_a_contradicting_model_value():
    """The case the change exists for: "healthy smokers" with nicotine dependence."""
    record = {"groups": [group(["nicotine dependence"], healthy=True)]}
    tally = apply(record)
    assert record["groups"][0]["is_healthy"]["value"] is False
    # `generated`, not `derived`. `ValueSource` offers `reported` and `generated` and
    # nothing else, so `derived` was a validation error on every group this touched, and
    # the enum's gloss for `generated` -- "Created by the extraction system" -- is this.
    assert record["groups"][0]["is_healthy"]["value_source"] == "generated"
    assert tally["overruled"] == 1


def test_apply_removes_the_flag_when_nothing_was_read():
    record = {"groups": [group([], status="not_reported", healthy=True)]}
    apply(record)
    assert "is_healthy" not in record["groups"][0]


def test_apply_agrees_where_the_model_was_right():
    record = {"groups": [group(["obesity"], healthy=False)]}
    assert apply(record)["agreed"] == 1


def test_condition_predicate_handles_junk():
    assert is_condition(None) is False
    assert is_condition("") is False
    assert is_condition("   ") is False
