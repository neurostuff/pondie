"""`Analysis.prespecification` -> preregistered, planned, exploratory or UNKNOWN.

Whether the contrast was planned before the data were seen. The targets are the
schema's own permissible values. No OTHER.

Why, with the measurements: docs/normalization-rationale.md, "prespecification".
"""

from __future__ import annotations

from pondie.normalization import UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

PREREGISTERED, PLANNED, EXPLORATORY = "preregistered", "planned", "exploratory"
VALUES = (PREREGISTERED, PLANNED, EXPLORATORY, UNKNOWN)

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
        # A record outside the paper. The decisive EXPLORATORY rule above still takes "not
        # planned" and "were not preregistered" first.
        r"pre[\s-]?regist|registered report|clinicaltrials|trial registration|"
        r"published protocol",
    ),
    Rule.of(
        PLANNED,
        # The paper's own account: `planned` alone says what "planned comparison" says,
        # `hypothesis-led` what "hypothesis-driven" says.
        r"pre[\s-]?specifi|\bconfirmatory\b|"
        r"planned (?:comparison|contrast|analys)|^\s*planned\s*$|a[\s-]priori|"
        r"hypothes[ei]s[\s-](?:driven|led|based)|^\s*hypothes(?:es|i[sz]ed)\s*$",
    ),
    Rule.of(
        EXPLORATORY,
        r"\bexplorator|post[\s-]?hoc\b|\bunplanned\b|data[\s-]driven|\bexploratory\b",
    ),
)

FIELD = ClosedField("analyses.prespecification", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report
