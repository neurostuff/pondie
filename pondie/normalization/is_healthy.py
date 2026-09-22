"""`Group.is_healthy`, derived from `medical_condition` rather than asked for.

Three states, and the third matters: unset is not False. A group whose
`medical_condition` was never read is an unread cohort, not a sick one.

Why it is derived and not asked, with the measurements: docs/normalization-rationale.md, "is_healthy".
"""

from __future__ import annotations

from pondie.normalization._records import strings_at, value_of
from pondie.vocabularies.phrases import NOT_READ, triage

#: `medical_condition` states that mean nobody established whether a condition was present.
#: An absent key is the same thing said by omission.
UNREAD = {"not_reported", "not_applicable", "unknown", "None", "none"}


def is_condition(value: object) -> bool:
    """Whether one `medical_condition` entry names a condition the cohort has."""
    return bool(triage(value).heads)


def derive(group: dict) -> bool | None:
    """True if free of conditions, False if any is named, None if nobody looked."""
    if not isinstance(group, dict):
        return None
    ev = group.get("medical_condition")
    if not isinstance(ev, dict):
        return None
    if str(ev.get("extraction_status")) in UNREAD:
        return None
    triaged = [triage(v) for v in strings_at(group, "medical_condition")]
    if any(t.heads for t in triaged):
        return False
    if triaged and all(t.kind == NOT_READ for t in triaged):
        return None
    return True


def apply(record: dict) -> dict[str, int]:
    """Set `is_healthy` on every group in a record. Returns a tally of what changed.

    Written as an `ExtractedValue` marked `derived`, replacing any previous value.
    """
    tally = {"set": 0, "unset": 0, "agreed": 0, "overruled": 0}
    for group in record.get("groups") or []:
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
