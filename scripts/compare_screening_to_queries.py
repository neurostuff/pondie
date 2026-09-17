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
    print(f"{'project':30} {'selector':32} {'n':>5} {'prec':>7} {'recall':>7} {'F1':>7}")
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
        print(f"{project:30} {'(pool)':32} {len(pool):5d}   gold in pool {len(want)}")

        for arm in arms:
            table = decisions.get((project, arm)) or {}
            if not table:
                continue
            selected = {p for p, d in table.items() if d == INCLUDED}
            p, r, f = score(selected, gold[meta_pmid], pool)
            totals[arm].append((p, r, f))
            print(f"{'':30} {'autonima: ' + arm:32} {len(selected & pool):5d} "
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
            print(f"{'':30} {label:32} {len(selected):5d} {p:7.1%} {r:7.1%} {f:7.3f}")

        # A PIPELINE, not a fourth selector: the arm decides, then the query removes what
        # the record contradicts. `selected = arm_included - vetoed`, so it is a subset of
        # the arm and can only ever drop a paper, never add one the arm excluded.
        #
        # The veto fires on False ALONE -- the record positively contradicts a stated
        # criterion -- and never on "cannot say", which is left to the screener. That is
        # the reason to compose them rather than pick one: the query's confident exclusions
        # and its silences are different things, and only the first is worth overriding a
        # model with.
        #
        # Ordered this way the arm's model pass still happens on every paper. The saving --
        # a paper the screener never reads -- needs the veto to run first, which is the
        # deployment and not the measurement.
        vetoed = {
            pmid for pmid in pool
            if any(verdicts[name][pmid] is False for name, _p in predicates)
        }
        for arm in ("full text", "record + evidence"):
            table = decisions.get((project, arm)) or {}
            if not table:
                continue
            selected = {p for p, d in table.items() if d == INCLUDED} - vetoed
            p_, r_, f_ = score(selected, gold[meta_pmid], pool)
            # Not "query veto + <arm>": `record + evidence` has a plus in its own name, so
            # the composition operator and the arm name were the same symbol and the label
            # read three ways.
            label = f"{arm} then query veto"
            totals[label].append((p_, r_, f_))
            print(f"{'':30} {label:32} {len(selected & pool):5d} {p_:7.1%} {r_:7.1%} "
                  f"{f_:7.3f}   veto drops {len(vetoed)} of {len(pool)} "
                  f"({len(vetoed & (gold[meta_pmid] & pool))} gold)")

    print("=" * 84)
    print(f"{'mean over the five projects':30} {'selector':32} {'':5} "
          f"{'prec':>7} {'recall':>7} {'F1':>7}")
    for name in arms + ["query, strict", "query, permissive",
                        "full text then query veto",
                        "record + evidence then query veto"]:
        rows = totals.get(name) or []
        if not rows:
            continue
        n = len(rows)
        print(f"{'':30} {name:32} {'':5} {sum(x[0] for x in rows)/n:7.1%} "
              f"{sum(x[1] for x in rows)/n:7.1%} {sum(x[2] for x in rows)/n:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
