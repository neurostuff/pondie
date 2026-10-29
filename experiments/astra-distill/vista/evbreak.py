"""    python evbreak.py VAL_PRED.jsonl   (run from vista/ on beast; v6 records and splits)

FP/FN breakdown of ev1 evidence predictions: by slot outcome, set size, and distance of wrong units to gold"""
import collections, json, statistics, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")
import evidence as E, tasks as T
M = "/data/james/agents/astra-distill/migrate/"
P = T.Paths(records=M + "train-v6/records", splits=M + "splits-v6")
preds = {}
for l in open(sys.argv[1]):
    d = json.loads(l); preds[d["sid"]] = d.get("prediction")

def one(s):
    _, _, U, _, ev = E.paper(s, P)
    pr = preds.get(s)
    g, p = collections.defaultdict(set), collections.defaultdict(set)
    for k, i in E.pairs(ev): g[k].add(i)
    for k, i in E.pairs(pr if isinstance(pr, (dict, list)) else {}): p[k].add(i)
    C = collections.Counter(); dist = collections.Counter(); top = collections.defaultdict(collections.Counter)
    sizes = []
    for k in g.keys() | p.keys():
        gs, ps = g.get(k, set()), p.get(k, set())
        t = k.split(".")[1].split("[")[0]
        if not ps:
            C["slot_missed"] += 1; C["fn_missed_slot"] += len(gs); top[t]["fn_missed_slot"] += len(gs); continue
        if not gs:
            C["slot_spurious"] += 1; C["fp_spurious_slot"] += len(ps); top[t]["fp_spurious_slot"] += len(ps); continue
        C["slot_both"] += 1
        sizes.append((len(ps), len(gs)))
        C["size_more" if len(ps) > len(gs) else "size_fewer" if len(ps) < len(gs) else "size_same"] += 1
        C["exact" if ps == gs else "overlap" if ps & gs else "disjoint"] += 1
        C["fp_shared_slot"] += len(ps - gs); C["fn_shared_slot"] += len(gs - ps)
        top[t]["fp_shared_slot"] += len(ps - gs); top[t]["fn_shared_slot"] += len(gs - ps)
        for i in ps - gs:
            d = min(abs(i - j) for j in gs); dist["1" if d == 1 else "2-3" if d <= 3 else "4-10" if d <= 10 else ">10"] += 1
    C["gold_units"] = sum(map(len, g.values())); C["pred_units"] = sum(map(len, p.values()))
    C["gold_slots"] = len(g); C["pred_slots"] = len(p)
    return C, dist, top, sizes

if __name__ == "__main__":
    sids = (open(M + "splits-v6/val.txt").read().split())
    C, D, TOP, S = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter), []
    with ProcessPoolExecutor(16) as ex:
        for c, d, top, sz in ex.map(one, sids, chunksize=8):
            C.update(c); D.update(d); S += sz
            for k, v in top.items(): TOP[k].update(v)
    print(json.dumps(C, indent=0)); print("fp distance to nearest gold unit (shared slots):", dict(D))
    print("mean units per shared slot: pred", round(statistics.mean(a for a, b in S), 3), "gold", round(statistics.mean(b for a, b in S), 3))
    for k, v in sorted(TOP.items(), key=lambda kv: -sum(kv[1].values()))[:10]: print(k, dict(v))
