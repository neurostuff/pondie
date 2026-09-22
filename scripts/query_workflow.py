#!/usr/bin/env python3
"""The whole funnel as a query: screen the papers, select the contrasts, pool the foci.

    python scripts/query_workflow.py --records '<dir>/*/*.extraction.json' \
        --bench <neurometabench>/data --stage1 <corpus>

`query_metaanalyses.py` scores the first gate and `query_analysis_selection.py` the second.
Neither says what a meta-analyst gets by running both, which is the only output that
matters: a set of coordinates. This runs them in series -- the published inclusion criteria
over the record, then the published contrast over the analyses of the papers that passed --
joins each selected analysis to its parsed row group through
`Analysis.source_table_analysis`, and scores the pooled coordinates against the map the
meta-analysis published.

BOTH GATES RUN STRICT AND PERMISSIVE, so there are four pipelines per map. *Strict* drops
what the record cannot answer and *permissive* keeps it; the four cells separate a record
that contradicts a criterion from a record that is silent about it, at the stage where
each does its damage. Strict/strict is the query a `WHERE` clause writes.

The foci column is the one `docs/meta-analysis-queries.md` recorded as missing -- "the
query has no foci column yet, and it should" -- because it needs the stage-1 parse beside
the records. `--stage1` is that parse; without it the funnel stops at the analysis.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from query_contrasts import KEYS, UNANSWERABLE, judge_analyses  # noqa: E402
from query_foci import (  # noqa: E402
    TOLERANCE,
    foci_of,
    gold_maps,
    gold_spaces,
    match,
    papers,
    parse_entries,
    spaces_of,
)
from query_metaanalyses import QUERIES  # noqa: E402


def load_records(pattern: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        body = json.loads(Path(path).read_text())
        out[Path(path).parent.name][Path(path).name.split(".")[0]] = body.get("study") or body
    return out


def screen(records: Mapping[str, dict], predicates) -> dict[str, dict[str, bool | None]]:
    """pmid -> predicate name -> True / False / cannot say."""
    out: dict[str, dict[str, bool | None]] = {}
    for pmid, body in records.items():
        verdicts: dict[str, bool | None] = {}
        for name, predicate in predicates:
            try:
                verdicts[name] = predicate(body)
            except Exception:
                verdicts[name] = UNANSWERABLE
        out[pmid] = verdicts
    return out


def passed(verdicts: Mapping[str, bool | None], permissive: bool) -> bool:
    return all(v is True or (permissive and v is UNANSWERABLE) for v in verdicts.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--stage1", type=Path,
                    help="the corpus, for <pmid>/stage1/analyses.json -- without it the "
                         "funnel stops at the analysis and reports no foci")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE,
                    help="mm within which two foci are the same focus (default 2)")
    args = ap.parse_args()

    records = load_records(args.records)

    included: dict[str, set[str]] = defaultdict(set)
    with (args.bench / "included_studies.csv").open() as fh:
        for line in csv.DictReader(fh):
            included[line["meta_pmid"]].add(line["study_pmid"])

    modes = [("strict", False), ("permissive", True)]

    for meta_pmid, (project, predicates) in QUERIES.items():
        here = records.get(project) or {}
        if not here:
            continue
        verdicts = screen(here, predicates)
        screened = {name: {p for p in here if passed(verdicts[p], permissive)}
                    for name, permissive in modes}
        want = included[meta_pmid] & set(here)

        print("=" * 98)
        print(f"{project}  (meta {meta_pmid})   {len(here)} records, "
              f"{len(want)} of its {len(included[meta_pmid])} included papers among them")
        for name, _ in modes:
            precision, recall = papers(screened[name], want)
            print(f"  screening, {name:11} selects {len(screened[name]):4d}   "
                  f"precision {precision:5.1%}   recall {recall:5.1%}")

        maps, dropped = gold_maps(args.bench, project)
        if not maps:
            print("  the benchmark has no analysis-level gold for this one -- "
                  "the funnel stops at screening")
            continue
        if dropped:
            print(f"  {dropped} merged gold studies dropped: their foci belong to "
                  f"several papers at once and cannot be attributed to one")

        cache = {pmid: parse_entries(args.stage1, pmid) for pmid in here}
        published = gold_spaces(args.bench, project)

        for (key_project, key), spec in KEYS.items():
            if key_project != project or key not in maps:
                continue
            gold_here = {p: c for p, c in maps[key].items() if p in here}
            missing = set(maps[key]) - set(here)

            ceiling: dict[str, list] = {}
            unparsed = 0
            for pmid in gold_here:
                entries = cache.get(pmid)
                if entries is None:
                    unparsed += 1
                    continue
                found = [f for a in here[pmid].get("analyses") or []
                         if isinstance(a, Mapping)
                         for f in (foci_of(a, here[pmid], entries) or [])]
                ceiling[pmid] = sorted(set(found))

            print(f"\n  {key}   {len(gold_here)} gold papers with a record"
                  f"   ({len(missing)} of its {len(maps[key])} have none)"
                  f"   {sum(len(c) for c in gold_here.values())} gold foci")
            # A space the record and the benchmark disagree about moves the paper's foci
            # by 5-10mm and no tolerance this side of a smoothing kernel recovers them.
            # Most of the benchmark's studysets hold MNI for every study, including the
            # ones published in Talairach, so the disagreement is usually the record
            # reading the paper's own word and the benchmark having already converted.
            disagreed = sum(
                1 for pmid in gold_here
                if cache.get(pmid) and published.get(pmid)
                and not (spaces_of(here[pmid], cache[pmid]) & published[pmid])
            )
            print(f"  ceiling -- every analysis of every gold paper: foci recall "
                  f"{match(ceiling, gold_here, args.tolerance)[1]:5.1%}"
                  f"   ({unparsed} of them have no stage-1 parse, and the record and the "
                  f"benchmark name different spaces on {disagreed})")
            print(f"  {'screen / select':22} {'anlys':>6} {'nofoci':>7}   "
                  f"{'paper P':>7} {'paper R':>7}   {'foci P':>7} {'foci R':>7} {'foci F1':>7}")

            for screen_name, _ in modes:
                for select_name, select_permissive in modes:
                    chosen: dict[str, list] = {}
                    analyses = nofoci = 0
                    for pmid in sorted(screened[screen_name]):
                        judged = judge_analyses(here[pmid], spec)
                        picked = {a for a, v in judged.items()
                                  if v is True or (select_permissive and v is UNANSWERABLE)}
                        if not picked:
                            continue
                        analyses += len(picked)
                        found: list[tuple] = []
                        entries = cache.get(pmid)
                        for analysis in here[pmid].get("analyses") or []:
                            if not isinstance(analysis, Mapping):
                                continue
                            if str(analysis.get("local_id")) not in picked:
                                continue
                            reached = (foci_of(analysis, here[pmid], entries)
                                       if entries is not None else None)
                            if not reached:
                                # Selected and contributes nothing: the slot is empty, it
                                # names a row group the parse does not have, or the parse
                                # does not exist. This is the channel the annotation
                                # comparison in docs/meta-analysis-queries.md found
                                # second-largest, seen from the query side.
                                nofoci += 1
                                continue
                            found += reached
                        chosen[pmid] = sorted(set(found))
                    precision, recall = papers(set(chosen), set(gold_here))
                    fp, fr, ff = match(chosen, gold_here, args.tolerance)
                    print(f"  {screen_name + ' / ' + select_name:22} {analyses:6d} "
                          f"{nofoci:7d}   {precision:7.1%} {recall:7.1%}   "
                          f"{fp:7.1%} {fr:7.1%} {ff:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
