"""A decision model as an evidence reranker: elephant-1 judges each query's top-k units from a first-stage reranker
(`rerank_eval.py` scores) on a fixed subset of the val benchmark.

    python rerank_decision.py BENCH_DIR SCORES_DIR [--first NAME|auto] [--n 1000] [--k 20]   (envs-aarch64/elephant)

The question is the pilot's (`vista/evjudge_decision.py`): the unit alone as the state, "Does the text state the
following claim, all of it? Claim: <bare claim>", P(yes) as the score; one engine call per unit, with every subset
query that has the unit in its top k as a question. --first auto takes the first stage with the best all-units
recall@20 in SCORES_DIR. Output in SCORES_DIR: elephant@<first>.decisions.jsonl {qid, unit, p} (appended, so a rerun
resumes), and elephant@<first>.metrics.json: the first stage alone and reranked by elephant-1 (top k ordered by P(yes),
the rest after them in first-stage order), on the same subset queries.
"""
import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank_eval import metrics  # noqa: E402

HUB = Path(os.environ.get("DECISION_HUB", "/scratch/07723/jdk3232/astra-distill/hf-decision/hub"))


def question(claim):
    return f"Does the text state the following claim, all of it? Claim: {claim}"


def engine():
    from qev.engine import DecisionEngine
    from qev.schema import SystemOneRequest
    snap = next((HUB / "models--baselabs--elephant-1" / "snapshots").iterdir())
    eng = DecisionEngine(str(snap), device="cuda:0")

    def decide(state, qs):
        resp, _ = eng.decide(SystemOneRequest(state=state, questions={k: {"type": "noul", "instructions": v}
                                                                       for k, v in qs.items()}))
        return {k: a.noul for k, a in resp.answers.items()}
    return decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("scores")
    ap.add_argument("--first", default="auto")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    bd, sd = Path(a.bench), Path(a.scores)
    first = a.first
    if first == "auto":
        ms = {p.name[:-len(".metrics.json")]: json.load(open(p)) for p in sd.glob("*.metrics.json") if "@" not in p.name}
        first = max(ms, key=lambda n: ms[n]["all"]["recall@20"])
        print("first stage:", first, ms[first]["all"]["recall@20"], flush=True)
    bench = [json.loads(l) for l in open(bd / "bench.jsonl")]
    units = {u["sid"]: u for u in map(json.loads, open(bd / "units.jsonl"))}
    sub = sorted(random.Random(0).sample(range(len(bench)), a.n))
    bench = [bench[i] for i in sub]
    fs = {}
    for line in open(sd / f"{first}.scores.jsonl"):
        r = json.loads(line)
        fs[r["qid"]] = r["scores"]
    top = {q["qid"]: sorted(range(len(fs[q["qid"]])), key=lambda i: -fs[q["qid"]][i])[:a.k] for q in bench}
    name = f"elephant@{first}"
    out = sd / f"{name}.decisions.jsonl"
    done = {}
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done[r["qid"], r["unit"]] = r["p"]
    todo = collections.defaultdict(dict)          # (sid, unit index) -> {qid: claim}
    for q in bench:
        for i in top[q["qid"]]:
            if (q["qid"], i) not in done:
                todo[q["sid"], i][q["qid"]] = q["claim"]
    print(f"{name}: {len(bench)} queries, top {a.k}; {len(done)} judged, {sum(map(len, todo.values()))} to judge "
          f"in {len(todo)} calls", flush=True)
    if todo:
        decide = engine()
        t0 = time.time()
        with open(out, "a") as f:
            for n, ((sid, i), qs) in enumerate(todo.items(), 1):
                p = decide(units[sid]["docs"][i], {str(qid): question(c) for qid, c in qs.items()})
                for qid in qs:
                    done[qid, i] = p[str(qid)]
                    f.write(json.dumps({"qid": qid, "unit": i, "p": round(p[str(qid)], 5)}) + "\n")
                if n % 500 == 0:
                    f.flush()
                    print(f"  {n}/{len(todo)} calls, {time.time() - t0:.0f}s", flush=True)
    base = {q["qid"]: fs[q["qid"]] for q in bench}
    rer = {}
    for q in bench:
        s = fs[q["qid"]]
        rank = {i: r for r, i in enumerate(sorted(range(len(s)), key=lambda i: -s[i]))}
        rer[q["qid"]] = [(1e6 + done[q["qid"], i]) if (q["qid"], i) in done else -rank[i] for i in range(len(s))]
    m = {"first": first, "k": a.k, "subset": a.n, "first_stage": metrics(bench, units, base), "elephant": metrics(bench, units, rer)}
    json.dump(m, open(sd / f"{name}.metrics.json", "w"), indent=1)
    for k in ("first_stage", "elephant"):
        print("RESULT", k, json.dumps({x: m[k]["all"][x] for x in ("hit@1", "hit@3", "hit@5", "hit@10", "recall@10", "mrr")}), flush=True)


if __name__ == "__main__":
    main()
