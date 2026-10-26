"""How CoordinateSet.analysis and Analysis.source_table_analysis relate in the migrated (unpruned) records.
    python links.py RUN_DIR SPLITS_DIR"""
import collections, json, os, sys
from concurrent.futures import ProcessPoolExecutor
RUN, SPL = sys.argv[1:3]
V = lambda x: x.get("value") if isinstance(x, dict) else x


def one(f):
    sid = f[:-16]
    r = json.load(open(f"{RUN}/records/{f}"))
    p = json.load(open(f"{RUN}/parse/{sid}/coordinate_parse.json"))
    pointed = {a["key"] for a in p["analyses"] if a.get("points")}
    an = {a["local_id"]: a for a in r.get("analyses") or []}
    by_sta = collections.defaultdict(list)
    for a in an.values():
        by_sta[V(a.get("source_table_analysis"))].append(a["local_id"])
    C = collections.Counter()
    for cs in r.get("coordinate_sets") or []:
        k = cs["local_id"]
        if k not in pointed:
            continue
        role = V(cs.get("role")); x = cs.get("analysis")
        fwd = by_sta.get(k, [])
        if fwd:
            C["set_has_forward"] += 1
            C["  ..and backlink " + ("same" if x in fwd else "none" if not x else "other")] += 1
            continue
        C[f"set_no_forward role={role}"] += 1
        if not x:
            C[f"  role={role}: no backlink"] += 1
        elif x not in an:
            C[f"  role={role}: backlink to missing analysis"] += 1
        else:
            s = V(an[x].get("source_table_analysis"))
            kind = "analysis sta empty" if not s else "analysis sta = pointed set" if s in pointed else "analysis sta = point-less key"
            C[f"  role={role}: backlink, {kind}"] += 1
    # analyses with empty sta named by >1 pointed set
    back = collections.Counter(cs.get("analysis") for cs in r.get("coordinate_sets") or []
                               if cs["local_id"] in pointed and cs.get("analysis") in an
                               and not V(an[cs["analysis"]].get("source_table_analysis")))
    C["analyses_backlinked_only"] += len(back); C["  ..by >1 pointed set"] += sum(v > 1 for v in back.values())
    return sid, C


if __name__ == "__main__":
    split = {l.strip(): n for n in ("train", "val", "test") for l in open(f"{SPL}/{n}.txt")}
    files = [f for f in os.listdir(f"{RUN}/records") if f[:-16] in split]
    T = collections.defaultdict(collections.Counter); P = collections.defaultdict(set)
    with ProcessPoolExecutor(16) as pool:
        for sid, c in pool.map(one, files, chunksize=50):
            T[split[sid]].update(c)
            if c["analyses_backlinked_only"]:
                P[split[sid]].add(sid)
    for s in ("train", "val", "test"):
        print("==", s, "papers with backlink-only analyses:", len(P[s]))
        for k, v in sorted(T[s].items()):
            print(f"   {v:7d}  {k}")
