"""Coordinate sets with points but no analysis in the migrated record, and Astra's unlinked analyses (no key) pruned.
    python orphans.py RAW_DIR TRAIN_DIR PARSE_DIR SPLITS_DIR OUT_JSON"""
import collections, json, os, sys
from concurrent.futures import ProcessPoolExecutor
RAW, TRAIN, PARSE, SPL, OUT = sys.argv[1:6]
V = lambda x: x.get("value") if isinstance(x, dict) else x


def one(f):
    sid = f[:-16]
    tr = json.load(open(f"{TRAIN}/{f}"))
    raw = json.load(open(f"{RAW}/{f}"))
    p = json.load(open(f"{PARSE}/{sid}/coordinate_parse.json"))
    pointed = {a["key"]: a for a in p["analyses"] if a.get("points")}
    used = {V(a.get("source_table_analysis")) for a in tr.get("analyses") or []}
    orphan = [k for k in pointed if k not in used]
    unlinked = [a for a in raw.get("analyses") or [] if not V(a.get("source_table_analysis"))]
    kept_ids = {a["local_id"] for a in tr.get("analyses") or []}
    return sid, dict(sets_pointed=len(pointed), orphan_sets=len(orphan),
                     orphan_text=sum(pointed[k].get("origin") == "text" for k in orphan),
                     orphan_table=sum(pointed[k].get("origin") == "table" for k in orphan),
                     orphan_points=sum(len(pointed[k]["points"]) for k in orphan),
                     raw_unlinked=len(unlinked), raw_unlinked_dropped=sum(a["local_id"] not in kept_ids for a in unlinked),
                     ex=[(k, pointed[k].get("origin"), (pointed[k].get("name") or "")[:80]) for k in orphan[:3]])


if __name__ == "__main__":
    split = {}
    for n in ("train", "val", "test"):
        for l in open(f"{SPL}/{n}.txt"):
            split[l.strip()] = n
    T = collections.defaultdict(collections.Counter); ex = []
    with ProcessPoolExecutor(16) as pool:
        for sid, c in pool.map(one, sorted(os.listdir(TRAIN)), chunksize=50):
            s = split.get(sid, "excluded")
            for k, v in c.items():
                if k != "ex":
                    T[s][k] += v
            T[s]["papers"] += 1
            T[s]["papers_with_orphan"] += bool(c["orphan_sets"])
            if c["ex"] and len(ex) < 12 and s == "train":
                ex.append((sid, c["ex"]))
    json.dump({"by_split": T, "examples": ex}, open(OUT, "w"), indent=1)
    for s, c in T.items():
        print(s, dict(c))
    for e in ex:
        print(e)
