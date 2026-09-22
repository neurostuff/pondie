"""`Analysis.prespecification` -> preregistered, exploratory or UNKNOWN.

Whether the contrast was planned before the data were seen. The targets are the
schema's own permissible values. No OTHER.

Why, with the measurements: docs/normalization-rationale.md, "prespecification".
"""

from __future__ import annotations

from pondie.normalization import UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

PREREGISTERED, EXPLORATORY = "preregistered", "exploratory"
VALUES = (PREREGISTERED, EXPLORATORY, UNKNOWN)

RULES = (
    #: Decisive, and first; see the doc named above.
    Rule.of(
        EXPLORATORY,
        r"\b(?:not|never|wasn'?t|were ?n'?t)\s+(?:formally\s+)?"
        r"(?:pre[\s-]?registered|pre[\s-]?specified|planned)",
        decisive=True,
    ),
    Rule.of(
        PREREGISTERED,
        r"pre[\s-]?regist|pre[\s-]?specifi|\bconfirmatory\b|"
        r"planned (?:comparison|contrast|analys)|a[\s-]priori hypothes|hypothesis[\s-]driven",
    ),
    Rule.of(
        EXPLORATORY,
        r"\bexplorator|post[\s-]?hoc\b|\bunplanned\b|data[\s-]driven|\bexploratory\b",
    ),
)

FIELD = ClosedField("analyses.prespecification", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report
