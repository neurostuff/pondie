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
        # `planned` alone (10 values) says what "planned comparison" says; `hypothesis-led`
        # and `hypotheses` (5) say what "hypothesis-driven" says. The decisive EXPLORATORY
        # rule above still takes "not planned" and "were not preregistered" first.
        r"pre[\s-]?regist|pre[\s-]?specifi|\bconfirmatory\b|"
        r"planned (?:comparison|contrast|analys)|^\s*planned\s*$|"
        r"a[\s-]priori hypothes|hypothes[ei]s[\s-](?:driven|led|based)|"
        r"^\s*hypothes(?:es|i[sz]ed)\s*$",
    ),
    Rule.of(
        EXPLORATORY,
        r"\bexplorator|post[\s-]?hoc\b|\bunplanned\b|data[\s-]driven|\bexploratory\b",
    ),
)

FIELD = ClosedField("analyses.prespecification", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report
