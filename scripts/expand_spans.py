"""Grow every evidence span to the sentence that contains it.

A span is whatever the locator happened to return: the extractor's quote can be a fragment
("grey matter volume"), and the repair pass writes back only what it resolved. A fragment is
a worse warrant than the sentence it came from -- a reader cannot tell what was asserted
about it, and the quote index then lists it as if it were a passage.

Deterministic and post-hoc: it needs no model and no re-extraction, so it runs over finished
records rather than inside the loop.

    python expand_spans.py --records reports/repair --corpus corpus --out reports/repair.full
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def units_for(text: str):
    """Sentence units with offsets, from pondie's own splitter."""
    from pondie.extraction.evidence.retrieval import sentence_units

    return sentence_units(text)


def grow(span: dict, units, text: str, stats: Counter) -> dict:
    """This span widened to the sentences it overlaps, or unchanged if it cannot be placed."""
    start, end = span.get("start_char"), span.get("end_char")
    if not isinstance(start, int) or not isinstance(end, int):
        stats["no offsets"] += 1
        return span
    touched = [u for u in units if u.start < end and u.end > start]
    if not touched:
        stats["no sentence found"] += 1
        return span
    lo, hi = min(u.start for u in touched), max(u.end for u in touched)
    if lo == start and hi == end:
        stats["already whole"] += 1
        return span
    # The substring, not the units joined: a unit's own text may be normalized, and a span
    # must remain a slice of the document or `spans.verify` rejects it.
    stats[f"grown to {len(touched)} sentence{'s' if len(touched) > 1 else ''}"] += 1
    return {**span, "start_char": lo, "end_char": hi, "text": text[lo:hi]}


def walk(node, units, text: str, stats: Counter) -> None:
    if isinstance(node, dict):
        evidence = node.get("evidence")
        if isinstance(evidence, dict):
            for group in evidence.get("sets") or []:
                group["spans"] = [grow(s, units, text, stats)
                                  for s in (group.get("spans") or [])]
        for value in node.values():
            walk(value, units, text, stats)
    elif isinstance(node, list):
        for value in node:
            walk(value, units, text, stats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suffix", default=".repaired.json")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from pondie.extraction.record import spans as span_tools

    totals, papers, bad = Counter(), 0, 0
    for path in sorted(args.records.glob(f"*{args.suffix}")):
        pmid = path.name.split(".")[0]
        source = args.corpus / pmid / "processed/local/text.tables.txt"
        if not source.is_file():
            continue
        text = source.read_text(errors="replace")
        record = json.loads(path.read_text())
        stats = Counter()
        walk(record, units_for(text), text, stats)

        # Every widened span must still verify against the document it came from, or the
        # record is worse than before it was touched.
        def check(node):
            nonlocal bad
            if isinstance(node, dict):
                ev = node.get("evidence")
                if isinstance(ev, dict):
                    for g in ev.get("sets") or []:
                        for sp in g.get("spans") or []:
                            try:
                                span_tools.verify(text, sp)
                            except Exception:
                                bad += 1
                for v in node.values():
                    check(v)
            elif isinstance(node, list):
                for v in node:
                    check(v)

        check(record)
        (args.out / path.name).write_text(json.dumps(record, indent=1))
        totals += stats
        papers += 1

    print(f"{papers} records -> {args.out}")
    for key, n in totals.most_common():
        print(f"  {n:>6}  {key}")
    print(f"  spans failing verification after widening: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
