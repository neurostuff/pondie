"""`Group.sex_distribution[].category` -> the sex a count is reported for.

18 surface forms over 859 values for two answers -- `male`, `males`, `Male`, `men`. This is
case and plural folding, not vocabulary work, and it is here rather than inline so that a
query grouping by sex reads one value and not eight.

The categories a paper reports are not always two: `OTHER` holds a reported category outside
the binary, which is a value to preserve rather than a failure to classify.
"""
from __future__ import annotations

from . import OTHER, UNKNOWN
from ._lexicon import ClosedField, Rule

MALE, FEMALE = "MALE", "FEMALE"
VALUES = (MALE, FEMALE, OTHER, UNKNOWN)

RULES = (
    Rule.of(FEMALE, r"^\s*(?:fe[\s-]?male[s]?|women|woman|girls?|\bF\b)\s*$"),
    Rule.of(MALE, r"^\s*(?:male[s]?|men|man|boys?|\bM\b)\s*$"),
    Rule.of(OTHER, r"non[\s-]?binary|\bother\b|transgender|intersex"),
)

FIELD = ClosedField("groups.sex_distribution.category", RULES, VALUES)
normalize = FIELD.normalize


if __name__ == "__main__":
    print(FIELD.report())
