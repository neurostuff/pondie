"""Orphan coordinate sets (points, no analysis) by the role the migrated record gives them.
    python orphan_roles.py TRAIN_DIR PARSE_DIR SPLIT_FILE"""
import collections, json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
TRAIN, PARSE, SPLIT = sys.argv[1:4]
V = lambda x: x.get("value") if isinstance(x, dict) else x


def one(sid):
    tr = json.load(open(f"{TRAIN}/{sid}.extraction.json"))
    p = json.load(open(f"{PARSE}/{sid}/coordinate_parse.json"))
    pointed = {a["key"]: a for a in p["analyses"] if a.get("points")}
    used = {V(a.get("source_table_analysis")) for a in tr.get("analyses") or []}
    sets = {c["local_id"]: c for c in tr.get("coordinate_sets") or []}
    out = []
    for k, a in pointed.items():
        if k in used:
            continue
        c = sets.get(k)
        role = V(c.get("role")) if c else "no_set"
        out.append((sid, k, a.get("origin"), role, (a.get("name") or "")[:90]))
    return out


if __name__ == "__main__":
    sids = [l.strip() for l in open(SPLIT) if l.strip()]
    rows = []
    with ProcessPoolExecutor(16) as pool:
        for r in pool.map(one, sids, chunksize=50):
            rows += r
    C = collections.Counter((r[3], r[2]) for r in rows)
    print(len(rows)); [print(f"{n:6d}  role={k[0]}  origin={k[1]}") for k, n in C.most_common()]
    random.seed(3)
    for role in sorted({r[3] for r in rows}, key=str):
        sub = [r for r in rows if r[3] == role]
        print("== examples role", role)
        for r in random.sample(sub, min(6, len(sub))):
            print("  ", r[0], r[2], r[4])
    json.dump(rows, open(os.path.join(os.path.dirname(TRAIN), "..", "audit", "orphan_sets_train.json"), "w"))
