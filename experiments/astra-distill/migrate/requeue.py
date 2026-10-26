"""Re-run the aligner on queued spans only, one process per paper: python requeue.py QUEUE OLD NEW OUT_JSONL [workers]."""
import collections
import json
import pathlib
import random
import sys
from concurrent.futures import ProcessPoolExecutor

from align import Aligner, char_similarity


def one(args):
    sid, its, old_dir, new_dir = args
    old = (pathlib.Path(old_dir) / sid / "processed/pubget/text.txt").read_text()
    al = Aligner((pathlib.Path(new_dir) / sid / "processed/pubget/text.txt").read_text(), len(old))
    out = []
    for t, osc in its:
        r = al.place(t, osc)
        m = r[2] if r else "unresolved"
        sim = char_similarity(t, al.new[r[0]:r[1]]) if r else None
        if r and m in ("gapped", "fuzzy") and sim < 0.9:
            m = "still_low_" + m
        out.append({"sid": sid, "text": t, "old_start_char": osc, "method": m, "similarity": sim,
                    "start_char": r[0] if r else None, "end_char": r[1] if r else None,
                    "placed": al.new[r[0]:r[1]] if r else None})
    return out


def main():
    q, old_dir, new_dir, outp = sys.argv[1:5]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    by = collections.defaultdict(dict)
    for ln in open(q):
        x = json.loads(ln)
        if x["kind"] == "span":
            by[x["sid"]].setdefault(x["text"], x.get("old_start_char"))
    print(f"{sum(map(len, by.values()))} distinct spans in {len(by)} papers", flush=True)
    C, ex = collections.Counter(), collections.defaultdict(list)
    with open(outp, "w") as fo, ProcessPoolExecutor(workers) as pool:
        for res in pool.map(one, ((s, list(d.items()), old_dir, new_dir) for s, d in by.items())):
            for r in res:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                C[r["method"]] += 1
                if len(ex[r["method"]]) < 200:
                    ex[r["method"]].append(r)
    print(dict(C))
    random.seed(1)
    for m in ("wildcard", "wildcard_b"):
        for e in random.sample(ex[m], min(8, len(ex[m]))):
            print(m, "|", repr(e["text"][:110]), "\n   ->", repr(e["placed"][:150]))
    for e in random.sample(ex["unresolved"], min(10, len(ex["unresolved"]))):
        print("UNRES |", e["sid"], repr(e["text"][:120]))


if __name__ == "__main__":
    main()
