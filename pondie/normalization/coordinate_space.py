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

from pondie.normalization import OTHER, UNKNOWN
from pondie.normalization._lexicon import ClosedField, Decision, Rule
from pondie.normalization._records import value_of

MNI, TAL = "MNI", "TAL"
VALUES = (MNI, TAL, OTHER, UNKNOWN)

#: No trailing boundary on the space names: "MNI152", "ICBM152" and "fsaverage6" are each one
#: token, and `\bmni\b` misses every one of them. `\bfsaverage\b` did, on this corpus.
#:
#: Each name is spelled out as well as abbreviated, because a paper writes either. ICBM and
#: Colin27 reach MNI rather than OTHER on the same grounds: the MNI152 template *is* the
#: ICBM-152 average and Colin27 is the MNI single-subject brain, so coordinates read off
#: them are already MNI and need no transform. `NMI` and `Talaraich` are transpositions
#: measured in the corpus, not hypothetical ones.
#:
#: The OTHER rule's second half names templates a study built for itself -- DARTEL, SUIT,
#: FMRIB58, an SPM release's own, anything "<something>-specific". Those are third spaces and
#: a transform must refuse them. It deliberately does not include the bare word "template":
#: "template image space" names no template, and not knowing which space a paper used is
#: UNKNOWN, not a third space.
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
#: The residual, for `pondie normalize <field>`. `__init__` states the contract --
#: "Every module exposes `normalize(...)` ... and `report(...)`" -- and five of the eight
#: closed-target modules bound only the first, so the CLI verb raised for them.
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


if __name__ == "__main__":
    print(FIELD.report())
