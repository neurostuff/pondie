"""`Group.handedness_distribution[].category` -> the handedness a count is for.

Why: docs/normalization-rationale.md, "sex_distribution and handedness_distribution".
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

RIGHT, LEFT, AMBIDEXTROUS = "RIGHT", "LEFT", "AMBIDEXTROUS"
VALUES = (RIGHT, LEFT, AMBIDEXTROUS, OTHER, UNKNOWN)

RULES = (
    Rule.of(AMBIDEXTROUS, r"ambidext|\bmixed[\s-]?hand"),
    # `R` and `L` alone, anchored: one value each here, and unambiguous in a handedness
    # category. Unanchored they would fire inside any sentence that has an R in it.
    Rule.of(RIGHT, r"(?<!non[\s-])(?<!not )\bright\b|non[\s-]?left[\s-]?hand|^\s*R\s*$"),
    Rule.of(LEFT, r"(?<!non[\s-])(?<!not )\bleft\b|^\s*L\s*$"),
)

VOCABULARY = "HandednessCategory"

FIELD = ClosedField("groups.handedness_distribution.category", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report

