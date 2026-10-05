"""Draw a 55-paper pool for a benchmark meta-analysis: 25 gold, 30 negatives, seed 0.

    python make_pool.py META_PMID AUTONIMA_PROJECT OUT_PREFIX

As the dementia and substance-use pools were drawn (JOURNAL E7): gold is sampled from the
benchmark's included papers that autonima screened at full text; the negatives are 15 that
autonima's full-text screener included but the benchmark did not (the hard ones) and 15
sampled from the rest it screened. Writes OUT_PREFIX.gold.pmids, .neg.pmids, .all.pmids.
"""
import csv
import json
import random
import sys
from pathlib import Path

BENCH = Path("/data/james/pondie-vs-fulltext/repos/neurometabench/data")
AUTONIMA = Path("/data/james/pondie-vs-fulltext/repos/autonima-results/projects")


def main() -> int:
    meta, project, out = sys.argv[1:4]
    gold = {r["study_pmid"] for r in csv.DictReader(open(BENCH / "included_studies.csv"))
            if r["meta_pmid"] == meta}
    screened = json.loads((sorted((AUTONIMA / project).glob("v*/outputs/fulltext_screening_results.json"))[-1])
                          .read_text())["screening_results"]
    decided = {str(r["study_id"]): r["decision"] for r in screened}
    rng = random.Random(0)
    gold_pool = sorted(gold & set(decided))
    hard = sorted(p for p, d in decided.items() if d == "included_fulltext" and p not in gold)
    rest = sorted(p for p, d in decided.items() if d != "included_fulltext" and p not in gold)
    chosen_gold = rng.sample(gold_pool, 25)
    chosen_neg = rng.sample(hard, min(15, len(hard)))
    chosen_neg += rng.sample(rest, 30 - len(chosen_neg))
    for name, pmids in (("gold", chosen_gold), ("neg", chosen_neg), ("all", chosen_gold + chosen_neg)):
        Path(f"{out}.{name}.pmids").write_text("".join(f"{p}\n" for p in pmids))
    print(f"gold {len(gold)} (screened {len(gold_pool)}), hard negatives {len(hard)}, "
          f"other screened {len(rest)} -> 25 gold, {len(chosen_neg)} negatives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
