"""`Group.handedness_distribution[].category` -> the handedness a count is reported for.

14 surface forms over 265 values for three answers. Case and hyphenation, as with sex; kept
separate because the answer set differs and a shared "demographics" module would hide that.
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Rule

RIGHT, LEFT, AMBIDEXTROUS = "RIGHT", "LEFT", "AMBIDEXTROUS"
VALUES = (RIGHT, LEFT, AMBIDEXTROUS, OTHER, UNKNOWN)

#: The negations are load-bearing. "non-left-handed" is an inclusion criterion meaning right
#: or ambidextrous, and reading it as LEFT inverts the group it describes.
RULES = (
    Rule.of(AMBIDEXTROUS, r"ambidext|\bmixed[\s-]?hand"),
    Rule.of(RIGHT, r"(?<!non[\s-])(?<!not )\bright\b|non[\s-]?left[\s-]?hand"),
    Rule.of(LEFT, r"(?<!non[\s-])(?<!not )\bleft\b"),
)

FIELD = ClosedField("groups.handedness_distribution.category", RULES, VALUES)
normalize = FIELD.normalize
#: The residual, for `pondie normalize <field>`. `__init__` states the contract --
#: "Every module exposes `normalize(...)` ... and `report(...)`" -- and five of the eight
#: closed-target modules bound only the first, so the CLI verb raised for them.
report = FIELD.report


if __name__ == "__main__":
    print(FIELD.report())
