"""`Analysis.prespecification` -> whether the contrast was planned before the data were seen.

The field was a closed enum and is now open, so a paper writing "post-hoc" keeps the word
instead of costing the whole analysis. This maps the wording onto the two values the schema
names, which is the distinction a reader needs: an exploratory contrast searched a space the
paper does not report, and pooling it with a planned one treats the two as equal evidence.

No `OTHER`. The field asks a yes-or-no question about when the contrast was decided, so a
third answer would not be a third kind of prespecification -- it would be a statement that
the wording does not say, which is `UNKNOWN`. That is the distinction `_lexicon` draws and
the reason the two are not collapsed: a caller may default on UNKNOWN and must not on a value
asserting something outside the set.

Seeded from the vocabulary's own synonyms rather than from measured drift -- at the time of
writing no committed record holds an off-vocabulary value, because the field was closed and
`values.cast` refused them before they could be counted. `report()` lists what no rule
matched, which is how the real surface forms arrive.
"""

from __future__ import annotations

from pondie.normalization import UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

PREREGISTERED, EXPLORATORY = "preregistered", "exploratory"
VALUES = (PREREGISTERED, EXPLORATORY, UNKNOWN)

RULES = (
    #: Decisive, and first. "not pre-registered" contains "pre-registered", so the two
    #: compete and the negation has to win rather than register as an ambiguity -- the case
    #: `_lexicon.Rule.decisive` exists for, met here as it is met by "not medicated".
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


if __name__ == "__main__":
    print(FIELD.report())
