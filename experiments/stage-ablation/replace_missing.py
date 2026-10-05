"""Replace pool papers with no text by papers from the same stratum, seed 1.

    python replace_missing.py META_PMID AUTONIMA_PROJECT PREFIX CORPUS_DIR

Writes PREFIX.candidates.txt (to build), and with --finalize CANDIDATE_CORPUS rewrites
PREFIX.{gold,neg,all,cli}.pmids from the drawn papers that have text plus the first
candidates per stratum that do. Strata as in make_pool.py: gold; autonima-included; rest.
"""
import csv
import json
import random
import sys
from pathlib import Path

from make_pool import AUTONIMA, BENCH


def strata(meta: str, project: str):
    gold = {r["study_pmid"] for r in csv.DictReader(open(BENCH / "included_studies.csv"))
            if r["meta_pmid"] == meta}
    f = sorted((AUTONIMA / project).glob("v*/outputs/fulltext_screening_results.json"))[-1]
    decided = {str(r["study_id"]): r["decision"] for r in json.loads(f.read_text())["screening_results"]}
    def stratum(p):
        return "gold" if p in gold else "hard" if decided.get(p) == "included_fulltext" else "rest"
    return gold, decided, stratum


def main() -> int:
    meta, project, prefix, corpus = sys.argv[1:5]
    gold, decided, stratum = strata(meta, project)
    drawn = Path(f"{prefix}.all.pmids").read_text().split()
    missing = [p for p in drawn if not (Path(corpus) / p).is_dir()]
    need = {"gold": 0, "hard": 0, "rest": 0}
    for p in missing:
        need[stratum(p)] += 1
    pools = {k: sorted(p for p in decided if stratum(p) == k and p not in drawn) for k in need}
    rng = random.Random(1)
    cand = {k: rng.sample(pools[k], min(len(pools[k]), need[k] * 4)) for k in need}
    if "--finalize" not in sys.argv:
        Path(f"{prefix}.candidates.json").write_text(json.dumps(cand))
        Path(f"{prefix}.candidates.txt").write_text(" ".join(p for v in cand.values() for p in v))
        print("missing", missing, "need", need)
        return 0
    built = Path(sys.argv[sys.argv.index("--finalize") + 1])
    cand = json.loads(Path(f"{prefix}.candidates.json").read_text())
    chosen = {k: [p for p in v if (built / p).is_dir()][:need[k]] for k, v in cand.items()}
    have = [p for p in drawn if p not in missing]
    golds = [p for p in have if stratum(p) == "gold"] + chosen["gold"]
    negs = [p for p in have if stratum(p) != "gold"] + chosen["hard"] + chosen["rest"]
    for name, pmids in (("gold", golds), ("neg", negs), ("all", golds + negs)):
        Path(f"{prefix}.{name}.pmids").write_text("".join(f"{p}\n" for p in pmids))
    Path(f"{prefix}.cli.pmids").write_text("".join(f"{p}\t{p}\tcorpus\n" for p in golds + negs))
    print("chosen", chosen, "->", len(golds), "gold,", len(negs), "negatives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
