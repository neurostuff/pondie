"""python dirflip.py PREDS.json SPLIT OUT_FLIPS.json   (run from vista/ with the vista venv)

Direction agreement of effect cells under the scorer's own pairing (linked translation when the model is
equivalent, strict otherwise): for every (term, level) both sides have, compare directions."""
import collections, json, sys
from concurrent.futures import ProcessPoolExecutor
import score as S
import tasks as T

STATS = []
_orig = S.score_key

def wrapped(sid, gold, key, pobj, trans):
    G, P = S.rows_of(gold.call2.get(key)), S.rows_of(pobj)
    for p, g in S.pair_analyses(P, G):
        if p is None or g is None:
            continue
        eq = S.model_equiv(sid, gold, p, g, trans)
        cells = S.linked_cells(sid, p, g, trans) if eq else S.strict_cells(sid, p, g)
        gd = {(c["term"], S.norm_level(c["level"])): c["direction"] for c in g["cells"]}
        pd = {(c["term"], S.norm_level(c["level"])): c["direction"] for c in cells}
        shared = [k for k in gd if k in pd]
        STATS.append(dict(sid=sid, key=key, name=g["name"], n_gold=len(gd),
                          pairs=[(gd[k], pd[k]) for k in shared]))
    return _orig(sid, gold, key, pobj, trans)

S.score_key = wrapped

def work(args):
    sid, rec, paths = args
    S.Ctx.paths = T.Paths(**paths); S.Ctx.ix = None
    STATS.clear()
    try:
        S.score_record(sid, rec)
    except Exception as e:
        return sid, None, repr(e)[:200]
    return sid, list(STATS), None

def main():
    preds, split = sys.argv[1], sys.argv[2]
    paths = {k: str(v) for k, v in T.DEFAULTS.items()}
    S.Ctx.paths = T.Paths(**paths)
    sids = S.read_split([split])
    recs = json.load(open(preds))["record"]
    rows, errs = [], 0
    with ProcessPoolExecutor(16) as ex:
        for sid, st, err in ex.map(work, [(s, recs.get(s), paths) for s in sids], chunksize=4):
            if err: errs += 1; continue
            rows += st
    vocab = collections.Counter(d for r in rows for pair in r["pairs"] for d in pair)
    POL = {"positive": 1, "negative": -1, "increase": 1, "decrease": -1, "greater": 1, "less": -1}
    C = collections.Counter(); flips = []
    for r in rows:
        opp = same = other = 0
        for gd, pd in r["pairs"]:
            if gd == pd: same += 1
            elif gd in POL and pd in POL and POL[gd] == -POL[pd]: opp += 1
            else: other += 1
        C["cells_shared"] += len(r["pairs"]); C["same"] += same; C["opposite"] += opp; C["other"] += other
        C["analyses"] += 1
        directed = sum(1 for gd, pd in r["pairs"] if gd in POL and pd in POL)
        if opp: C["analyses_any_opposite"] += 1; flips.append((r["sid"], r["key"], r["name"], opp, directed))
        if directed and opp == directed: C["analyses_all_directed_opposite"] += 1
        if directed: C["analyses_with_directed_shared"] += 1
    papers = {f[0] for f in flips}
    print(json.dumps({"errors": errs, "vocab": vocab.most_common(10), **C, "papers_with_any_opposite": len(papers),
                      "papers": len(sids)}, indent=1))
    json.dump(flips, open(sys.argv[3], "w"))

if __name__ == "__main__":
    main()
