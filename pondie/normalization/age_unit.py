"""`Group.age_unit` -> the unit the group's age summaries are in.

A closed target whose answers are the schema's own `AgeUnit` values. `annotate` writes the
result to `age_unit`'s `normalized` sidecar, beside the source's wording.

`gestational_weeks` is kept apart from `weeks` because the two count from different zeros.
No OTHER: a unit outside the four would not be a fifth kind of age, it would be a wording
this could not read, which is UNKNOWN.

Why, with the measurements: docs/normalization-rationale.md, "age_unit".
"""

from __future__ import annotations

from pondie.normalization import UNKNOWN, _sidecar
from pondie.normalization._lexicon import ClosedField, Rule
from pondie.normalization._records import NOT_REPORTED, value_of

YEARS, MONTHS, WEEKS, GESTATIONAL = "years", "months", "weeks", "gestational_weeks"
VALUES = (YEARS, MONTHS, WEEKS, GESTATIONAL, UNKNOWN)

PATH = "groups.age_unit"
VOCABULARY = "AgeUnit"

#: Order matters; see the doc named above.
RULES = (
    Rule.of(
        GESTATIONAL,
        r"gestation|\bGA\b|post[\s-]?menstrual|corrected age|\bPMA\b",
        decisive=True,
    ),
    # `y` on its own is 18 values here, and in an age unit it is not ambiguous.
    Rule.of(YEARS, r"\byears?\b|\byrs?\b|\bya?o\b|\bage in years\b|^\s*y\s*$"),
    Rule.of(MONTHS, r"\bmonths?\b|\bmos?\b|\bmo\.\b"),
    Rule.of(WEEKS, r"\bweeks?\b|\bwks?\b"),
)

FIELD = ClosedField(PATH, RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report


def annotate(record: dict) -> dict[str, int]:
    """Write an `AgeUnit` entry on every group's `age_unit`. Returns a tally of what changed.

    A group with no age summary at all gets none: the unit of nothing is not unmapped, it
    is absent, and an unmapped entry there would say the paper gave an age in a unit this
    could not read. A group with an age and no stated unit is that unmapped entry, written
    on the `not_reported` wrapper.
    """
    mapping_tool = _sidecar.tool("age_unit")
    tally = {"set": 0, "skipped": 0, "unmatched": 0}
    for group in record.get("groups") or []:
        if not isinstance(group, dict):
            continue
        wrapper = group.get("age_unit")
        # `is not None` was not enough: a slot the paper did not report reads as the
        # NOT_REPORTED sentinel, which is not None, so 711 of the 3951 groups this filled
        # across the corpus had no age at all and were being told their absent age was in
        # an unreadable unit -- the one thing the docstring above says not to do.
        has_age = any(
            value_of(group.get(slot)) not in (None, NOT_REPORTED)
            for slot in ("age_mean", "age_median", "age_minimum", "age_maximum")
        )
        if not isinstance(wrapper, dict) or not has_age:
            if isinstance(wrapper, dict):
                _sidecar.replace(wrapper, mapping_tool, [])
            tally["skipped"] += 1
            continue
        decision = normalize(value_of(wrapper))
        _sidecar.replace(
            wrapper, mapping_tool, [_sidecar.from_decision(decision, VOCABULARY, mapping_tool)]
        )
        tally["set"] += 1
        if decision.reason in ("unmatched", "empty"):
            tally["unmatched"] += 1
    return tally
