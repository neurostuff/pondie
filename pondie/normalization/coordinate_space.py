"""`Analysis.coordinate_space` -> MNI, TAL, OTHER or UNKNOWN.

MNI and TAL are the two spaces a coordinate transform can move between; OTHER is a
third space it must refuse, and UNKNOWN is no information. `resolve` answers from the
most authoritative source that has an answer.

Why, with the measurements: docs/normalization-rationale.md, "coordinate_space".
"""

from __future__ import annotations

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Decision, Rule
from pondie.normalization._records import value_of

MNI, TAL = "MNI", "TAL"
VALUES = (MNI, TAL, OTHER, UNKNOWN)

RULES = (
    Rule.of(
        MNI,
        r"\bmni|\bnmi\b|\bicbm|montreal\s+neurolog|"
        r"international\s+consortium\s+for\s+brain\s+mapping|\bcolin\s*27",
    ),
    Rule.of(TAL, r"\btal\b|t[ao]l[ai]+r[ai]+ch|tournoux"),
    Rule.of(
        OTHER,
        r"^\s*other\s*$|\bsurface\b|\bfsaverage|\bfsLR\b|\bnative\b|"
        r"\bdartel\b|\bsuit\b|\bfmrib58|custom(?:i[sz]ed)?\b|in-house|"
        r"\w+[\s-]specific\b|\bspm\s?\d+\b",
    ),
)

FIELD = ClosedField("analyses.coordinate_space", RULES, VALUES)
normalize = FIELD.normalize
report = FIELD.report


def resolve(analysis: dict, record: dict, points_by_key: dict | None = None) -> Decision:
    """The analysis's space, from the most authoritative source that answers."""
    own = normalize(value_of(analysis.get("coordinate_space")))
    if own:
        return own

    wanted = {str(t) for t in (value_of(analysis.get("tables"), True) or [])}
    seen = {
        normalize(value_of(t.get("coordinate_space"))).value
        for t in (record.get("tables") or [])
        if isinstance(t, dict) and str(value_of(t.get("local_id"))) in wanted
    }
    seen.discard(UNKNOWN)
    if len(seen) == 1:
        return Decision(seen.pop(), "tables agree")
    if len(seen) > 1:
        return Decision(UNKNOWN, "tables disagree")

    key = str(value_of(analysis.get("source_table_analysis")) or "")
    # Normalized before they are compared, as the tables are. Stage 1 writes "MNI" for one
    # sentence and "MNI152" for the next, and a set of the raw tokens reads two spellings of
    # one space as a conflict.
    parsed = [normalize(p.get("space")) for p in ((points_by_key or {}).get(key) or [])]
    spaces = {d.value for d in parsed if d}
    if len(spaces) == 1:
        return Decision(spaces.pop(), "parsed coordinates")
    if len(spaces) > 1:
        return Decision(UNKNOWN, "point spaces disagree")
    # A token no rule matched decided nothing, so it cannot be reported as though the parse
    # answered. Carrying the text is what the missing rule gets written from.
    unmatched = next((d.text for d in parsed if d.reason == "unmatched"), "")
    if unmatched:
        return Decision(UNKNOWN, "unmatched", unmatched)
    return Decision(UNKNOWN, own.reason, own.text)
