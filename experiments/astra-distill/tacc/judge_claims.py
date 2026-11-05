"""elephant-1 as a value checker: P(yes) that a text states a claim, for prepared (state, claim) rows.

    python judge_claims.py ROWS.jsonl OUT.jsonl   (envs-aarch64/elephant, via rerank.sbatch ENV=elephant)

ROWS: {id, state, claim}. Rows sharing a state are asked in one engine call (at most MAX_QS questions), one question
per row, in the wording of `rerank_decision.py`. OUT: {id, p} (appended, so a rerun resumes).
"""
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank_decision import engine, question  # noqa: E402

MAX_QS = 32


def main():
    rows_path, out = sys.argv[1:3]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    by_state = collections.defaultdict(dict)
    for r in map(json.loads, open(rows_path)):
        if r["id"] not in done:
            by_state[r["state"]][r["id"]] = r["claim"]
    calls = [(s, dict(list(qs.items())[i:i + MAX_QS])) for s, qs in by_state.items() for i in range(0, len(qs), MAX_QS)]
    print(f"{len(done)} judged, {sum(map(len, by_state.values()))} to judge in {len(calls)} calls", flush=True)
    if not calls:
        return
    decide = engine()
    t0 = time.time()
    with open(out, "a") as f:
        for n, (state, qs) in enumerate(calls, 1):
            p = decide(state, {str(i): question(c) for i, c in qs.items()})
            for i in qs:
                f.write(json.dumps({"id": i, "p": round(p[str(i)], 5)}) + "\n")
            if n % 500 == 0:
                f.flush()
                print(f"  {n}/{len(calls)} calls, {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(calls)} calls in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
