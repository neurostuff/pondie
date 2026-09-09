"""`Effect.kind` is a second statement of a shape the cells already state.

The point of holding it twice is that one statement cannot be checked. These pin the
derivation against §3's ordered steps, and the rule that compares the two.

The case the rule exists for is real and is used as the fixture below: `ngDTY5BgJUuX`'s three
classification analyses carry `DecodingDetails` and two cells with no direction, which derives
`contrast` with its direction lost, while the paper reports "significant above-chance
cross-classification accuracies" -- one signed cell, a `simple_effect`. Reviewers left every
one of those cells unsigned and signed the genuine contrast in the same corpus, so the review
pass agreed there was no direction to give and still nothing said the shape was wrong.
"""

from __future__ import annotations

from pondie.extraction.record.effect import (
    NO_LABEL,
    derive_effect_kind,
    terms_in_scope,
)
from pondie.extraction.record.rules import check_effect_kind


def field(value):
    return {"extraction_status": "extracted", "value": value}


def unreported():
    return {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}


class Findings:
    def __init__(self):
        self.errors, self.warnings = [], []

    def error(self, path, message):
        self.errors.append((path, message))

    def warning(self, path, message):
        self.warnings.append((path, message))


CLASSES = {
    "local_id": "trm-class",
    "name": field("classification task"),
    "type": field("categorical"),
    "variation_level": field("within_subject"),
    "levels": [{"level": field("synchronous")}, {"level": field("asynchronous")}],
}


def decoding_record(cells):
    """`ngDTY5BgJUuX`'s shape: a decoding analysis over a two-level classification term."""
    return {
        "model_estimations": [{"local_id": "me", "terms": [CLASSES]}],
        "analyses": [
            {
                "local_id": "analysis-10",
                "model_estimation": "me",
                "details": {"details_type": "DecodingDetails"},
                "effect": {"cells": cells},
            }
        ],
    }


# --- the derivation --------------------------------------------------------


def test_one_signed_cell_is_a_simple_effect():
    """§5.9: the class the reported accuracy is for takes the positive cell, and the other
    takes none, because absence is the zero weight."""
    terms = {"trm-class": CLASSES}
    kind, _why = derive_effect_kind(
        [{"term": "trm-class", "level": field("synchronous"), "direction": field("positive")}],
        terms,
    )
    assert kind == "simple_effect"


def test_two_signed_cells_on_one_term_are_a_contrast():
    terms = {"trm-class": CLASSES}
    kind, _why = derive_effect_kind(
        [
            {"term": "trm-class", "level": field("synchronous"), "direction": field("positive")},
            {"term": "trm-class", "level": field("asynchronous"), "direction": field("negative")},
        ],
        terms,
    )
    assert kind == "contrast"


def test_cells_with_no_direction_at_all_describe_no_test():
    """Which is what the five decoding analyses hold today."""
    terms = {"trm-class": CLASSES}
    kind, why = derive_effect_kind(
        [
            {"term": "trm-class", "level": field("synchronous"), "direction": unreported()},
            {"term": "trm-class", "level": field("asynchronous"), "direction": unreported()},
        ],
        terms,
    )
    assert kind == "contrast" and "direction" in why


# --- the rule --------------------------------------------------------------


def test_a_stated_kind_that_matches_its_cells_is_silent():
    record = decoding_record(
        [{"term": "trm-class", "level": field("synchronous"), "direction": field("positive")}]
    )
    record["analyses"][0]["effect"]["kind"] = field("simple_effect")
    findings = Findings()
    check_effect_kind(record, findings)
    assert findings.errors == []


def test_the_decoding_analysis_encoded_as_a_contrast_is_reported():
    """The real defect: `DecodingDetails`, two directionless cells, and a kind that says the
    thing the paper actually reports. Before `kind` existed there was nothing to compare."""
    record = decoding_record(
        [
            {"term": "trm-class", "level": field("synchronous"), "direction": unreported()},
            {"term": "trm-class", "level": field("asynchronous"), "direction": unreported()},
        ]
    )
    record["analyses"][0]["effect"]["kind"] = field("simple_effect")
    findings = Findings()
    check_effect_kind(record, findings)
    assert findings.errors, "a stated simple_effect over a two-level contrast must be reported"
    path, message = findings.errors[0]
    assert path == "analyses[0].effect.kind"
    assert "simple_effect" in message and "contrast" in message


def test_an_unstated_kind_disagrees_with_nothing():
    """The slot is new, so most records do not carry it yet. Absence is not a contradiction."""
    record = decoding_record(
        [
            {"term": "trm-class", "level": field("synchronous"), "direction": unreported()},
            {"term": "trm-class", "level": field("asynchronous"), "direction": unreported()},
        ]
    )
    findings = Findings()
    check_effect_kind(record, findings)
    assert findings.errors == []


def test_cells_that_derive_nothing_are_reported_whatever_the_slot_says():
    """A derivation of `none` is a defect before anything is compared against it."""
    record = decoding_record([])
    findings = Findings()
    check_effect_kind(record, findings)
    assert findings.errors and findings.errors[0][0] == "analyses[0].effect.cells"
    assert NO_LABEL not in findings.errors[0][1], "report the reason, not the label"


def test_the_scope_walk_is_the_one_in_effect():
    """The rule derives against `terms_in_scope`, so a cell naming a lower stage's term is
    judged against the same set `check_cell_terms` uses."""
    models = {
        "group": {"local_id": "group", "inputs_from": ["subject"], "terms": []},
        "subject": {"local_id": "subject", "terms": [CLASSES]},
    }
    assert "trm-class" in terms_in_scope("group", models)
