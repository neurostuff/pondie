"""`Group.sex_distribution[].category` -> the sex a count is reported for.

OTHER holds a reported category outside the binary.

Why: docs/normalization-rationale.md, "sex_distribution and handedness_distribution".
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule, apply_to_distribution

MALE, FEMALE = "MALE", "FEMALE"
VALUES = (MALE, FEMALE, OTHER, UNKNOWN)

RULES = (
    # The anchors are deliberate -- "men and women" is not MALE -- and a trailing role
    # noun is not a second category: "male patients" and "female patients" are 4 values.
    Rule.of(FEMALE,
            r"^\s*(?:fe[\s-]?male[s]?|women|woman|girls?|\bF\b)"
            r"(?:\s+(?:patients?|subjects?|participants?|controls?|volunteers?))?\s*$"),
    Rule.of(MALE,
            r"^\s*(?:male[s]?|men|man|boys?|\bM\b)"
            r"(?:\s+(?:patients?|subjects?|participants?|controls?|volunteers?))?\s*$"),
    Rule.of(OTHER, r"non[\s-]?binary|\bother\b|transgender|intersex"),
)

FIELD = ClosedField("groups.sex_distribution.category", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report


def apply(record: dict) -> dict[str, int]:
    """Set `category_normalized` on every `sex_distribution` entry. Tally of what changed."""

    return apply_to_distribution(record, "sex_distribution", normalize)
