#!/usr/bin/env python3
"""Resolve an analysis's coordinate space to MNI, TAL, OTHER or UNKNOWN.

`Analysis.coordinate_space` keeps the source's own words -- "Montreal Neurological Institute
(MNI) standard space", "modified Talairach stereotaxic space" -- for the same reason
`Measure.source_label` does. This maps those words onto the four values a query and a
transform need, without touching what the record stores.

    docs/normalization-pipelines.md places this among the field-shape pipelines. It is the
    closed-target shape, so it is lexical rules and not an encoder: sixteen surface forms
    over two real spaces is a case where a rule is auditable and a similarity is not, and a
    wrong answer here moves foci 5-10mm rather than mislabelling a row.

Two decisions the rules encode:

  * A form naming BOTH spaces is not OTHER. `OTHER` asserts a third space; "MNI/TAL" asserts
    we cannot tell which of the two, which is a different claim. It falls through to the
    coordinates, which are more specific than the sentence, and is UNKNOWN only if they do
    not settle it either.
  * Nothing is bucketed silently. A form no rule matches is `UNKNOWN` with `reason=unmatched`
    and is reported, so a new spelling surfaces instead of disappearing into OTHER.

Resolution order follows the schema: the analysis's own field wins, then the tables behind it,
then the spaces stage 1 read off the coordinates themselves.

    python normalize_space.py 'data/runs/*/records/*.extraction.json'
"""
from __future__ import annotations
import argparse, glob as globlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

from schema_utils import value_of  # noqa: E402


#: `\bmni` and not `\bmni\b`: "MNI152" is one token and a trailing boundary misses it.
MNI = re.compile(r"\bmni|montreal\s+neurolog", re.I)
TAL = re.compile(r"\btal\b|talairach|tournoux", re.I)
#: Written as a space in its own right, not as a failure to name one.
OTHER = re.compile(r"^\s*other\s*$|\bsurface\b|\bfsaverage\b|\bfsLR\b|\bnative\b", re.I)

SPACES = ("MNI", "TAL", "OTHER", "UNKNOWN")


def classify(text) -> tuple[str, str]:
    """(space, reason) for one surface form."""
    s = text if isinstance(text, str) else ""
    if not s.strip():
        return "UNKNOWN", "empty"
    mni, tal = bool(MNI.search(s)), bool(TAL.search(s))
    if mni and tal:
        return "UNKNOWN", "names both spaces"
    if mni:
        return "MNI", "lexical"
    if tal:
        return "TAL", "lexical"
    if OTHER.search(s):
        return "OTHER", "lexical"
    return "UNKNOWN", "unmatched"


def resolve(analysis: dict, record: dict, stage1: dict | None = None) -> dict:
    """Space for one analysis, with where the answer came from."""
    space, reason = classify(value_of(analysis.get("coordinate_space")))
    if space != "UNKNOWN":
        return {"space": space, "source": "analysis", "reason": reason,
                "text": value_of(analysis.get("coordinate_space"))}

    wanted = {str(t) for t in (value_of(analysis.get("tables"), True) or [])}
    seen = {classify(value_of(t.get("coordinate_space")))[0]
            for t in (record.get("tables") or [])
            if isinstance(t, dict) and str(value_of(t.get("local_id"))) in wanted}
    seen.discard("UNKNOWN")
    if len(seen) == 1:
        return {"space": seen.pop(), "source": "table", "reason": "tables agree", "text": None}
    if len(seen) > 1:
        return {"space": "UNKNOWN", "source": "table", "reason": "tables disagree", "text": None}

    key = str(value_of(analysis.get("source_table_analysis")) or "")
    points = (stage1 or {}).get(key) or []
    counts = Counter(str(p.get("space") or "").upper() for p in points)
    counts.pop("", None)
    if len(counts) == 1:
        return {"space": classify(next(iter(counts)))[0], "source": "points",
                "reason": "parsed coordinates", "text": None}
    if len(counts) > 1:
        return {"space": "UNKNOWN", "source": "points", "reason": "point spaces disagree",
                "text": None}
    return {"space": "UNKNOWN", "source": "none", "reason": reason,
            "text": value_of(analysis.get("coordinate_space"))}


def stage1_points(record_path: Path) -> dict:
    """parse key -> points, from the stage-1 parse beside this record's corpus."""
    study = record_path.name.split(".")[0]
    for root in record_path.parent.parent.glob("texts"):
        path = root / study / "stage1" / "analyses.json"
        if path.is_file():
            import sys
            import parse_tables
            parsed = json.loads(path.read_text()).get("analyses") or []
            return dict(zip(parse_tables.parse_keys(parsed), parsed))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="*", default=["data/runs/*/records/*.extraction.json"])
    ap.add_argument("--out", type=Path, default=Path("data/eval/space-normalized.json"))
    args = ap.parse_args()

    paths = sorted({p for pattern in args.records for p in globlib.glob(pattern)})
    paths = [Path(p) for p in paths if not p.endswith(".raw.json")]

    space = Counter(); source = Counter(); forms = defaultdict(Counter); unmatched = Counter()
    out = {}
    for path in paths:
        try:
            body = json.loads(path.read_text())
        except Exception:
            continue
        body = body.get("study") or body
        if not isinstance(body, dict):
            continue
        keyed = stage1_points(path)
        for analysis in (body.get("analyses") or []):
            if not isinstance(analysis, dict):
                continue
            got = resolve(analysis, body, {k: (v.get("points") or []) for k, v in keyed.items()})
            space[got["space"]] += 1
            source[f"{got['source']}: {got['reason']}"] += 1
            if got["text"]:
                forms[got["space"]][str(got["text"])] += 1
            if got["reason"] == "unmatched":
                unmatched[str(got["text"])] += 1
            out[f"{path.name.split('.')[0]}|{value_of(analysis.get('local_id'))}"] = got["space"]

    total = sum(space.values())
    print(f"{len(paths)} records, {total} analyses\n")
    for s in SPACES:
        print(f"  {s:8s} {space[s]:5d}  ({space[s]/max(1,total):4.0%})")
    print("\n  resolved by:")
    for k, n in source.most_common():
        print(f"     {n:5d}  {k}")
    print("\n  surface forms folded into each space:")
    for s in ("MNI", "TAL", "OTHER"):
        if forms[s]:
            print(f"     {s}: " + ", ".join(f"{k!r}×{v}" for k, v in forms[s].most_common(6)))
    if unmatched:
        print(f"\n  {sum(unmatched.values())} value(s) no rule matched -- these need a rule, "
              f"not a bucket:")
        for k, n in unmatched.most_common(10):
            print(f"     {n:4d}  {k!r}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {args.out} ({len(out)} analyses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
