#!/usr/bin/env python3
"""Three ways to select papers for a meta-analysis, on one denominator.

    python scripts/compare_screening_to_queries.py --records '<dir>/*/*.extraction.json' \
        --bench <neurometabench>/data --decisions decisions.csv

The three are a language model reading the full article, the same model reading the
extraction record, and a deterministic query written from the meta-analysis's own published
criteria. They have been measured on different denominators before now -- `compare_arms.py`
scores over the papers all arms screened in common, which omits gold lost upstream, so its
PTSD recall reads 0.941 where end-to-end it is 16/22 -- so everything here is scored over
**the papers all three can see**: the intersection of the screened set and the extracted
records, with recall against the gold inside it.

The query is run twice, because its two failure modes are not the same thing and a screener
has no equivalent of either. *Strict* excludes a paper the record cannot answer for and
*permissive* admits it; the gap between them is the record's silence, not the query's
judgement.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

from query_metaanalyses import QUERIES, UNANSWERABLE  # noqa: E402

INCLUDED = "included_fulltext"


def score(selected: set[str], gold: set[str], pool: set[str]) -> tuple[float, float, float]:
    """Precision, recall, F1 over one pool."""
    selected &= pool
    want = gold & pool
    hit = len(selected & want)
    precision = hit / len(selected) if selected else 0.0
    recall = hit / len(want) if want else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--decisions", required=True, type=Path)
    args = ap.parse_args()

    gold: dict[str, set[str]] = defaultdict(set)
    with (args.bench / "included_studies.csv").open() as fh:
        for row in csv.DictReader(fh):
            gold[row["meta_pmid"]].add(row["study_pmid"])

    decisions: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    with args.decisions.open() as fh:
        for row in csv.DictReader(fh):
            decisions[(row["project"], row["arm"])][row["pmid"]] = row["decision"]

    records: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(args.records)):
        body = json.loads(Path(path).read_text())
        records[Path(path).parent.name][Path(path).name.split(".")[0]] = body.get("study") or body

    arms = ["full text", "record + evidence", "record, no evidence"]
    print(f"{'project':24} {'selector':26} {'n':>5} {'prec':>7} {'recall':>7} {'F1':>7}")
    totals: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for meta_pmid, (project, predicates) in QUERIES.items():
        here = records.get(project) or {}
        screened = decisions.get((project, "full text")) or {}
        if not here or not screened:
            continue
        # The pool every selector can see: extracted AND reached full-text screening.
        pool = set(here) & set(screened)
        want = gold[meta_pmid] & pool
        print("-" * 84)
        print(f"{project:24} {'(pool)':26} {len(pool):5d}   gold in pool {len(want)}")

        for arm in arms:
            table = decisions.get((project, arm)) or {}
            if not table:
                continue
            selected = {p for p, d in table.items() if d == INCLUDED}
            p, r, f = score(selected, gold[meta_pmid], pool)
            totals[arm].append((p, r, f))
            print(f"{'':24} {'autonima: ' + arm:26} {len(selected & pool):5d} "
                  f"{p:7.1%} {r:7.1%} {f:7.3f}")

        verdicts = {}
        for name, predicate in predicates:
            verdicts[name] = {}
            for pmid in pool:
                try:
                    verdicts[name][pmid] = predicate(here[pmid])
                except Exception:  # noqa: BLE001
                    verdicts[name][pmid] = UNANSWERABLE
        for label, admit_unknown in (("query, strict", False), ("query, permissive", True)):
            selected = {
                pmid for pmid in pool
                if all(
                    verdicts[name][pmid] is True
                    or (admit_unknown and verdicts[name][pmid] is UNANSWERABLE)
                    for name, _p in predicates
                )
            }
            p, r, f = score(selected, gold[meta_pmid], pool)
            totals[label].append((p, r, f))
            print(f"{'':24} {label:26} {len(selected):5d} {p:7.1%} {r:7.1%} {f:7.3f}")

        # A composition, because the two selectors fail differently. The query is precise
        # about EXCLUSION -- a paper whose record contradicts a stated criterion -- and
        # silent where the record is. So let it veto, and let the screener decide the rest.
        # The point is not only accuracy: every paper the veto removes is a model call the
        # screener never makes.
        vetoed = {
            pmid for pmid in pool
            if any(verdicts[name][pmid] is False for name, _p in predicates)
        }
        table = decisions.get((project, "full text")) or {}
        screened_in = {p for p, d in table.items() if d == INCLUDED}
        for arm in ("full text", "record + evidence"):
            table = decisions.get((project, arm)) or {}
            if not table:
                continue
            screened_in = {p for p, d in table.items() if d == INCLUDED}
            selected = screened_in - vetoed
            p_, r_, f_ = score(selected, gold[meta_pmid], pool)
            label = f"query veto + {arm}"
            totals[label].append((p_, r_, f_))
            print(f"{'':24} {label:26} {len(selected & pool):5d} {p_:7.1%} {r_:7.1%} "
                  f"{f_:7.3f}   veto removes {len(vetoed)} of {len(pool)} "
                  f"({len(vetoed & (gold[meta_pmid] & pool))} gold)")

    print("=" * 84)
    print(f"{'mean over the five projects':24} {'selector':26} {'':5} "
          f"{'prec':>7} {'recall':>7} {'F1':>7}")
    for name in arms + ["query, strict", "query, permissive",
                        "query veto + full text", "query veto + record + evidence"]:
        rows = totals.get(name) or []
        if not rows:
            continue
        n = len(rows)
        print(f"{'':24} {name:26} {'':5} {sum(x[0] for x in rows)/n:7.1%} "
              f"{sum(x[1] for x in rows)/n:7.1%} {sum(x[2] for x in rows)/n:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
