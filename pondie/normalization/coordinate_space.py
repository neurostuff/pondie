"""`Analysis.coordinate_space` -> MNI, TAL, OTHER or None.

MNI and TAL are the two spaces a coordinate transform can move between; OTHER is a
stated third space it must refuse, and None is no space stated (or one naming both).
The alias table is `study_schema.spaces`, shared with ingestion and neurostore, so a
space pondie reads and the point neurostore stores agree. `resolve` answers from the
most authoritative source that has an answer.

Why, with the measurements: docs/normalization-rationale.md, "coordinate_space".
"""

from __future__ import annotations

from study_schema.spaces import MNI, OTHER, RULES, SPACES, TAL, normalize_space

from pondie.normalization._lexicon import Decision, field_report
from pondie.formats import parse_keys
from pondie.normalization._records import value_of

VALUES = SPACES
__all__ = ["MNI", "OTHER", "TAL", "VALUES", "normalize", "report", "resolve"]

_OTHER_RULE = dict(RULES)[OTHER]


def normalize(text: object) -> Decision:
    """`study_schema.spaces.normalize_space`, with the reason the query funnel prints.

    An OTHER no rule named is `unmatched` and carries its text: it is still OTHER, but it
    is what a missing rule gets written from.
    """
    raw = text if isinstance(text, str) else ""
    if not raw.strip():
        return Decision(None, "empty", raw)
    space = normalize_space(raw)
    if space is None:
        hits = [s for s, pattern in RULES if s != OTHER and pattern.search(raw)]
        return Decision(None, "matches MNI and TAL" if len(hits) > 1 else "not stated", raw)
    if space == OTHER and not _OTHER_RULE.search(raw):
        return Decision(OTHER, "unmatched", raw)
    return Decision(space, "lexical", raw)


def report(patterns: tuple[str, ...] | None = None) -> str:
    return field_report("analyses.coordinate_space", normalize, VALUES, patterns)


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
    seen.discard(None)
    if len(seen) == 1:
        return Decision(seen.pop(), "tables agree")
    if len(seen) > 1:
        return Decision(None, "tables disagree")

    key = str(parse_keys.canonical(value_of(analysis.get("source_table_analysis"))) or "")
    # Normalized before they are compared, as the tables are. Stage 1 writes "MNI" for one
    # sentence and "MNI152" for the next, and a set of the raw tokens reads two spellings of
    # one space as a conflict.
    parsed = [normalize(p.get("space")) for p in ((points_by_key or {}).get(key) or [])]
    spaces = {d.value for d in parsed if d}
    if len(spaces) == 1:
        # An OTHER no rule named is not reported as though the parse answered: carrying
        # its text is what the missing rule gets written from.
        unmatched = next((d for d in parsed if d.reason == "unmatched"), None)
        return unmatched or Decision(spaces.pop(), "parsed coordinates")
    if len(spaces) > 1:
        return Decision(None, "point spaces disagree")
    return Decision(None, own.reason, own.text)
