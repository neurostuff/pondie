"""`Group.is_healthy`, derived from `medical_condition` rather than asked for.

Three states, and the third matters: unset is not False. A group whose
`medical_condition` was never read is an unread cohort, not a sick one.

Why it is derived and not asked, with the measurements: docs/normalization-rationale.md, "is_healthy".
"""

from __future__ import annotations

from pondie.formats.values import STATUSES
from pondie.normalization._records import strings_at, value_of
from pondie.vocabularies.phrases import NOT_READ, triage

#: The one status that means somebody established what the cohort had. Anything else is an
#: unread cohort, including a status this does not recognise.
#:
#: This was `UNREAD`, a set of five naming `not_applicable` (an *evidence* status) and
#: `unknown`/`None`/`none` (not statuses at all) -- a second, looser definition of what a
#: status is, and one that answered "read" for any junk outside its five. Asking for the
#: canonical status instead is both narrower and safer, and it changes no record here: over
#: the 1817-record corpus the slot carries `extracted` 3771 times, `not_reported` 289 times
#: and nothing else.
READ = STATUSES[0]


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
    if str(ev.get("extraction_status")) != READ:
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
