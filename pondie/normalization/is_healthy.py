"""`Group.is_healthy`, derived from `medical_condition` rather than asked for.

The slot was `model_extracted` until it was measured. Asked directly, a model answers with
the source's wording: across 1,817 records 168 groups came back True beside a real
diagnosis -- "healthy male smokers" with nicotine dependence, "obese subjects" with obesity
-- and 132 of those were `value_source: reported`, because the addiction and obesity
literature says "healthy" to mean free of comorbidity. Rewriting the slot description to say
otherwise moved 2 of 5 test cases and left `is_healthy=True` beside
`medical_condition=[obesity]` untouched.

So the question is removed instead of rephrased. `deterministic` in the storage schema means
the generator drops the slot from the extraction schema, and this fills it afterwards. The
flag then cannot contradict the field it summarises, which is the invariant
`schema-tutorial.md` already declared and nothing enforced.

Three states, and the third matters: unset is not False. A group whose `medical_condition`
was never read is an unread cohort, not a sick one.
"""

from __future__ import annotations

import re

from pondie.normalization._records import strings_at, value_of
from pondie.normalization.medical_condition import NO_CONDITION, triage

#: Values that assert health rather than name a condition. `triage` catches the negations
#: ("no psychiatric history") through its own NEGATION pattern; these are the positive
#: assertions of wellness, which read as conditions to it -- "healthy controls" has a head.
HEALTHY_ASSERTION = re.compile(
    r"^\W*(?:"
    r"healthy(?:\s+(?:control|controls|volunteer|volunteers|participant|participants|"
    r"subject|subjects|adult|adults|individual|individuals|comparison))?"
    r"|(?:normal|unaffected|non-?clinical|typically\s+developing)"
    r"(?:\s+(?:control|controls|volunteer|volunteers|participant|participants|"
    r"subject|subjects))?"
    r"|control|controls|comparison(?:\s+group)?|none"
    r")\W*$",
    re.I,
)

#: `medical_condition` states that mean nobody established whether a condition was present.
#: An absent key is the same thing said by omission.
UNREAD = {"not_reported", "not_applicable", "unknown", "None", "none"}


def is_condition(value: object) -> bool:
    """Whether one `medical_condition` entry names a condition the cohort has."""
    text = str(value or "").strip()
    if not text or HEALTHY_ASSERTION.match(text):
        return False
    return triage(text).kind not in {"empty", NO_CONDITION}


def derive(group: dict) -> bool | None:
    """True if free of conditions, False if any is named, None if nobody looked."""
    if not isinstance(group, dict):
        return None
    ev = group.get("medical_condition")
    if not isinstance(ev, dict):
        return None
    if str(ev.get("extraction_status")) in UNREAD:
        return None
    if any(is_condition(v) for v in strings_at(group, "medical_condition")):
        return False
    return True


def apply(record: dict) -> dict[str, int]:
    """Set `is_healthy` on every group in a record. Returns a tally of what changed.

    Writes the derivation into the same `ExtractedValue` shape the rest of the record uses,
    marked `derived` so a reader can tell it apart from anything a model said. Any previous
    value is replaced: the point is that the two cannot disagree.
    """
    tally = {"set": 0, "unset": 0, "agreed": 0, "overruled": 0}
    for group in (record.get("groups") or []):
        if not isinstance(group, dict):
            continue
        before = value_of(group.get("is_healthy"))
        after = derive(group)
        if after is None:
            group.pop("is_healthy", None)
            tally["unset"] += 1
            continue
        group["is_healthy"] = {
            "value": after,
            "extraction_status": "extracted",
            # `generated`, not `derived`: `ValueSource` offers `reported` and `generated`
            # and nothing else, and the enum's own gloss for `generated` is "Created by the
            # extraction system", which is exactly this. `derived` would have been a
            # validation error on every group this touched.
            "value_source": "generated",
            "evidence": {"status": "not_applicable"},
        }
        tally["set"] += 1
        if before is not None:
            tally["agreed" if before == after else "overruled"] += 1
    return tally
