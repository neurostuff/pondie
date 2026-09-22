"""`Group.sex_distribution[].category` -> the sex a count is reported for.

OTHER holds a reported category outside the binary.

Why: docs/normalization-rationale.md, "sex_distribution and handedness_distribution".
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

MALE, FEMALE = "MALE", "FEMALE"
VALUES = (MALE, FEMALE, OTHER, UNKNOWN)

RULES = (
    Rule.of(FEMALE, r"^\s*(?:fe[\s-]?male[s]?|women|woman|girls?|\bF\b)\s*$"),
    Rule.of(MALE, r"^\s*(?:male[s]?|men|man|boys?|\bM\b)\s*$"),
    Rule.of(OTHER, r"non[\s-]?binary|\bother\b|transgender|intersex"),
)

FIELD = ClosedField("groups.sex_distribution.category", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report
