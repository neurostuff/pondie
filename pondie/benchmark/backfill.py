"""Tie each reference analysis to the stage-1 table entry it came from.

The direction benchmark scores a candidate cell only when its analysis, its cell and its
`Cell.term` all align with the reference's. The first of those is a fuzzy entity match today,
and it need not be: both sides descend from the same stage-1 parse, and a candidate records
which entry it came from in `Analysis.source_table_analysis`. Every analysis in both shipped
candidate sets carries it. The reference carries it on none of 102, because it was built
before the slot existed -- so the join that could be exact is guessed instead, and 28 gold
cells on the current run are dropped for a term mismatch, 16 of them holding the right answer.

This writes the missing half. It does not re-extract: the gold direction tables are keyed to
the reference's own `local_id`s, so re-running the pipeline over these papers would produce
new ids and orphan the gold. Only `source_table_analysis` is added, and only where the join
is unambiguous.

The join is the analysis's name against the stage-1 entry's, verbatim after whitespace and
case folding, because the extractor is told to keep the parse's name in `name.value` and does.
It fires only when that name occurs once in the whole parse.

A name occurring several times was first resolved by table, using `stage1/table-map.json`,
which raised the yield from 54 of 100 to 78. It was wrong half the time. Checked against the
`source_table_analysis` the shipped candidate recorded during its own extraction -- an
independent answer to the same question -- the unique-name joins agree 54 of 54 and the
tie-broken ones 12 of 24. `JzsUUQbDr2bm` is why: four supplementary tables report the same
contrasts for FA, AD, RD and MD, so "sz > nc" names four different parse entries and the
table a reference analysis cites does not single one out. A provenance field exists to be
exact, and 78 with twelve wrong is worse than 54 with none.

So 46 of 100 are left alone: the recurring names, and the 22 with no entry of that name at
all -- an analysis read from prose has no table entry to point at.

    python -m pondie.benchmark.backfill --dry-run
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from pondie import paths
from pondie.benchmark import REFERENCE

SLOT = "source_table_analysis"


def _read(node: Any) -> Any:
    """An `ExtractedValue`'s value, or the node when it is a plain string."""
    if isinstance(node, Mapping) and "extraction_status" in node:
        return node.get("value")
    return node


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def entry_id(entries: list[Mapping[str, Any]], position: int) -> str:
    """`<table>#<n>`, the form a candidate's `source_table_analysis` already takes.

    `n` counts within the table and not across the parse: `t3#1` is the first entry of
    table 3, which on `ngDTY5BgJUuX` is the parse's third. Numbering globally produced
    `t3#3` -- an id no candidate writes, so the join it was meant to enable would have
    matched nothing while reporting 78 successes.
    """
    table = entries[position].get("table_id") or "prose"
    nth = sum(1 for e in entries[: position + 1] if (e.get("table_id") or "prose") == table)
    return f"{table}#{nth}"


def _by_name(entries: Iterable[Mapping[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for position, entry in enumerate(entries):
        out[_norm(entry.get("name"))].append(position)
    return dict(out)


def resolve(analysis: Mapping[str, Any], by_name: Mapping[str, list[int]]) -> int | None:
    """The one stage-1 entry this analysis came from, or None when that is not decidable.

    One route only. Narrowing a repeated name by table was measured against the candidate's
    own recorded provenance and agreed half the time, so it is not a route.
    """
    name = _norm(_read(analysis.get("name")))
    if not name:
        # `7HPLh5nJzmP5`'s parse carries unnamed `prose` entries, which an unnamed analysis
        # would match all of.
        return None
    hits = by_name.get(name, [])
    return hits[0] if len(hits) == 1 else None


def backfill(reference: Path, corpus: Path, write: bool) -> dict[str, int]:
    tally = {"analyses": 0, "joined": 0, "already": 0, "no_entry": 0, "repeated_name": 0}
    for path in sorted(reference.glob("*.extraction.json")):
        study = path.name.split(".")[0]
        stage1 = corpus / study / "stage1" / "analyses.json"
        if not stage1.is_file():
            continue
        entries = json.loads(stage1.read_text(encoding="utf-8")).get("analyses") or []
        by_name = _by_name(entries)

        record = json.loads(path.read_text(encoding="utf-8"))
        touched = False
        for analysis in record.get("analyses") or []:
            tally["analyses"] += 1
            if analysis.get(SLOT):
                tally["already"] += 1
                continue
            found = resolve(analysis, by_name)
            if found is None:
                key = _norm(_read(analysis.get("name")))
                tally["repeated_name" if by_name.get(key) else "no_entry"] += 1
                continue
            # An `ExtractedString`, because that is the slot's range and a bare string is
            # invisible: `flatten` yields wrappers, so the first attempt wrote 54 values the
            # scorer could not see and the alignment did not move. `generated` with no
            # evidence is the honest pairing -- this was derived by joining the parse after
            # the fact, and there is no sentence in the paper that says `t2#1`.
            analysis[SLOT] = {
                "extraction_status": "extracted",
                "value": entry_id(entries, found),
                "value_source": "generated",
                "evidence": {"status": "not_found"},
            }
            tally["joined"] += 1
            touched = True
        if touched and write:
            path.write_text(
                json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    return tally


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--corpus", type=Path, default=paths.DATA / "corpus-rev")
    parser.add_argument("--dry-run", action="store_true", help="report the join, write nothing")
    args = parser.parse_args(argv)

    tally = backfill(args.reference, args.corpus, write=not args.dry_run)
    print(f"  analyses           {tally['analyses']:4}")
    print(f"    already tied     {tally['already']:4}")
    print(f"    joined{'' if not args.dry_run else ' (dry run)':11} {tally['joined']:4}")
    print(f"    no stage-1 entry {tally['no_entry']:4}")
    print(f"    name repeats     {tally['repeated_name']:4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
