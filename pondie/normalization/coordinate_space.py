"""`Analysis.coordinate_space` -> MNI, TAL, OTHER or UNKNOWN.

The record keeps the source's own words -- "Montreal Neurological Institute (MNI) standard
space", "modified Talairach stereotaxic space" -- for the same reason `Measure.source_label`
does. This maps them onto the four values a query and a coordinate transform need.

More than a spelling exercise, because this field decides whether coordinates are moved: a
wrong answer displaces foci by 5-10mm, so `OTHER` (a third space, refuse to transform) and
`UNKNOWN` (no information, a caller may default) must not be collapsed.

Resolution follows the schema's own precedence -- the analysis's field is authoritative over a
table's -- and then falls back on the spaces stage 1 read off the coordinates themselves. That
fallback is not decoration: it answers 11% of analyses, where the model left the field blank.
`Table.coordinate_space` sits between the two and is empty in every table measured, so the
middle step never fires on this corpus and is kept for the schema's sake rather than its yield.
"""
from __future__ import annotations

from . import OTHER, UNKNOWN
from ._lexicon import ClosedField, Decision, Rule
from ._records import value_of

MNI, TAL = "MNI", "TAL"
VALUES = (MNI, TAL, OTHER, UNKNOWN)

#: `\bmni` and not `\bmni\b`: "MNI152" is one token and a trailing boundary misses it.
RULES = (
    Rule.of(MNI, r"\bmni|montreal\s+neurolog"),
    Rule.of(TAL, r"\btal\b|talairach|tournoux"),
    Rule.of(OTHER, r"^\s*other\s*$|\bsurface\b|\bfsaverage\b|\bfsLR\b|\bnative\b"),
)

FIELD = ClosedField("analyses.coordinate_space", RULES, VALUES)
normalize = FIELD.normalize


def resolve(analysis: dict, record: dict, points_by_key: dict | None = None) -> Decision:
    """The analysis's space, from the most authoritative source that answers."""
    own = normalize(value_of(analysis.get("coordinate_space")))
    if own:
        return own

    wanted = {str(t) for t in (value_of(analysis.get("tables"), True) or [])}
    seen = {normalize(value_of(t.get("coordinate_space"))).value
            for t in (record.get("tables") or [])
            if isinstance(t, dict) and str(value_of(t.get("local_id"))) in wanted}
    seen.discard(UNKNOWN)
    if len(seen) == 1:
        return Decision(seen.pop(), "tables agree")
    if len(seen) > 1:
        return Decision(UNKNOWN, "tables disagree")

    key = str(value_of(analysis.get("source_table_analysis")) or "")
    spaces = {str(p.get("space") or "").upper()
              for p in ((points_by_key or {}).get(key) or [])}
    spaces.discard("")
    if len(spaces) == 1:
        return Decision(normalize(spaces.pop()).value, "parsed coordinates")
    if len(spaces) > 1:
        return Decision(UNKNOWN, "point spaces disagree")
    return Decision(UNKNOWN, own.reason, own.text)


if __name__ == "__main__":
    print(FIELD.report())
