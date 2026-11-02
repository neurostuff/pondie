"""Data for an evidence reranker: (claim about a record slot's value, paper unit) -> does the unit state it.

    python rerank_data.py bench OUT_DIR [--per-paper 8]                 # val 481: queries + every unit of their papers
    python rerank_data.py train OUT_DIR [--per-paper 16] [--dev 300]    # v5 train split: labeled pairs + a dev set
    (run from vista/ on beast; v6 records, which hold every v5 paper)

Claims are `evjudge_build.bare_claim` (the value alone in a sentence, the form the decision models judged best) and
unit texts `evjudge_build.doc` (a table row with its table's first row and the line before the table).
bench: OUT/bench.jsonl {qid, sid, slot, collection, claim, gold: [unit]}, OUT/units.jsonl {sid, docs, sections}
  (sections from the paper parse's `sections`, ns-pond `coordinate_space.sectionize`; null when its text differs).
train: OUT/train.jsonl {query, doc, label} rows: per slot one gold unit (label 1), 5 hard negatives (the top BM25
  units for the claim in the same paper that are neither gold nor next to a gold unit) and 2 random ones (label 0);
  OUT/dev.jsonl {query, positive: [doc], documents: [doc]} for `CrossEncoderRerankingEvaluator`, from --dev train
  papers left out of train.jsonl: documents are the claim's BM25 top 50 of the paper.
"""
import argparse
import bisect
import collections
import hashlib
import json
import math
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
import evidence as E  # noqa: E402
from evjudge_build import M, P, bare_claim, doc  # noqa: E402

DATA = Path("/data/alejandro/projects/ns-pond/data")
TOK = re.compile(r"\w+")
N_HARD, N_RAND, DEV_TOP = 5, 2, 50


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.tf = [collections.Counter(TOK.findall(d.lower())) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = collections.Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q):
        ws = [w for w in set(TOK.findall(q.lower())) if w in self.idf]
        out = []
        for t, ln in zip(self.tf, self.len):
            s = 0.0
            for w in ws:
                f = t.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


def slots_of(sid):
    rec, text, U, comp, ev = E.paper(sid, P)
    g = collections.defaultdict(set)
    for k, i in E.pairs(ev):
        g[k].add(i)
    return text, U, comp, g


def sections(sid, text, U):
    try:
        pp = json.load(open(DATA / sid / "parse/parsed_paper.json"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if pp.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest():
        return None
    secs = sorted((s["start_char"], s["kind"]) for s in pp["sections"])
    starts = [a for a, _ in secs]
    return [secs[bisect.bisect_right(starts, a) - 1][1] for a, _ in U]


def bench_paper(args):
    sid, k = args
    text, U, comp, g = slots_of(sid)
    rng = random.Random(f"bench-{sid}")
    qs = []
    for slot in rng.sample(sorted(g), min(k, len(g))):
        qs.append({"sid": sid, "slot": slot, "collection": slot.lstrip(".").split("[")[0].split(".")[0],
                   "claim": bare_claim(comp, slot), "gold": sorted(g[slot])})
    return qs, {"sid": sid, "docs": [doc(text, U, i) for i in range(1, len(U) + 1)], "sections": sections(sid, text, U)}


def train_paper(args):
    sid, k, dev = args
    text, U, comp, g = slots_of(sid)
    docs = [doc(text, U, i) for i in range(1, len(U) + 1)]
    bm = BM25(docs)
    rng = random.Random(f"train-{sid}")
    rows = []
    for slot in rng.sample(sorted(g), min(k, len(g))):
        q, gold = bare_claim(comp, slot), g[slot]
        near = {j for i in gold for j in (i - 1, i, i + 1)}
        sc = bm.scores(q)
        ranked = sorted(range(1, len(U) + 1), key=lambda i: -sc[i - 1])
        if dev:
            top = ranked[:DEV_TOP]
            rows.append({"query": q, "positive": [docs[i - 1] for i in sorted(gold)],
                         "documents": [docs[i - 1] for i in top if i not in gold]})
            continue
        free = [i for i in ranked if i not in near]
        if len(free) < N_HARD + N_RAND:
            continue
        hard = free[:N_HARD]
        rand = rng.sample(free[N_HARD:], N_RAND)
        rows.append({"query": q, "doc": docs[rng.choice(sorted(gold)) - 1], "label": 1.0})
        rows += [{"query": q, "doc": docs[i - 1], "label": 0.0} for i in hard + rand]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("bench", "train"))
    ap.add_argument("out")
    ap.add_argument("--per-paper", type=int, default=0)
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "bench":
        sids = open(M + "splits-v6/val.txt").read().split()
        qid = 0
        with ProcessPoolExecutor(a.workers) as ex, open(out / "bench.jsonl", "w") as fb, open(out / "units.jsonl", "w") as fu:
            for qs, units in ex.map(bench_paper, ((s, a.per_paper or 8) for s in sids), chunksize=4):
                for q in qs:
                    fb.write(json.dumps({"qid": qid, **q}, ensure_ascii=False) + "\n"); qid += 1
                fu.write(json.dumps(units, ensure_ascii=False) + "\n")
        print(json.dumps({"papers": len(sids), "queries": qid}))
        return
    sids = open(M + "splits/train.txt").read().split()       # v5 train: the split ev1-full and r3-27b-full used
    dev = set(random.Random(5).sample(sids, a.dev))
    n = collections.Counter()
    with ProcessPoolExecutor(a.workers) as ex, open(out / "train.jsonl", "w") as ft, open(out / "dev.jsonl", "w") as fd:
        jobs = [(s, a.per_paper or 16, False) for s in sids if s not in dev] + [(s, 4, True) for s in sorted(dev)]
        for rows in ex.map(train_paper, jobs, chunksize=8):
            for r in rows:
                if "positive" in r:
                    fd.write(json.dumps(r, ensure_ascii=False) + "\n"); n["dev_queries"] += 1
                else:
                    ft.write(json.dumps(r, ensure_ascii=False) + "\n"); n["train_rows"] += 1; n["train_pos"] += r["label"] > 0
    print(json.dumps({"train_papers": len(sids) - len(dev), "dev_papers": len(dev), **n}))


if __name__ == "__main__":
    main()
