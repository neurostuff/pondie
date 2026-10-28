"""flipdump.py: per-analysis dump of direction-reversed cells (same pairing as dirflip.py). Writes flips-adj/d_<sid>.txt"""
import json, re, sys, os
import score as S
import tasks as T
OUT = "/data/james/agents/astra-distill/vista-out/v5/runs/r3-27b-final/flips-adj"
RUN = "/data/james/agents/astra-distill/vista-out/v5/runs/r3-27b-final"
POL = {"positive": 1, "negative": -1, "increase": 1, "decrease": -1, "greater": 1, "less": -1}
flips = json.load(open(RUN + "/direction-flips.json"))
want = {(f[0], f[1], f[2]) for f in flips}
sids = sorted({f[0] for f in flips})
listing = {}
for l in open("/data/james/agents/astra-distill/vista-out/v5/val.jsonl"):
    d = json.loads(l)
    if d["sid"] in sids:
        p = d["prompt"]; i = p.find("Coordinate sets:")
        listing[d["sid"]] = p[i:]
def parse_entry(sid, key):
    L = listing.get(sid, ""); i = L.find("Coordinate set: " + key)
    if i < 0: return "(not found)"
    j = L.find("Coordinate set:", i + 10)
    blk = L[i:j if j > 0 else None]
    return re.sub(r"\n(-?\d+(\.\d+)? +-?\d+(\.\d+)? +-?\d+(\.\d+)?\n?)+", "\n[coords omitted]\n", blk + "\n")[:1800]
def tdump(row):
    s = []
    for t in row["terms"]:
        s.append("   term %s | %s | type=%s var=%s | levels=%s" % (t["term"], t["name"], t["type"], t["variation_level"],
                 [(lv.get("level"), lv.get("name") or lv.get("link") or lv.get("entity"), {k: v for k, v in lv.items() if k not in ("level",)}) for lv in t["levels"]]))
    return "\n".join(s)
_orig = S.score_key
paths = {k: str(v) for k, v in T.DEFAULTS.items()}
S.Ctx.paths = T.Paths(**paths); S.Ctx.ix = None
recs = json.load(open(RUN + "/preds.json"))["record"]
bysid = {}
def wrapped(sid, gold, key, pobj, trans):
    G, P = S.rows_of(gold.call2.get(key)), S.rows_of(pobj)
    for p, g in S.pair_analyses(P, G):
        if p is None or g is None or (sid, key, g["name"]) not in want: continue
        eq = S.model_equiv(sid, gold, p, g, trans)
        cells = S.linked_cells(sid, p, g, trans) if eq else S.strict_cells(sid, p, g)
        gd = {(c["term"], S.norm_level(c["level"])): c["direction"] for c in g["cells"]}
        pd = {(c["term"], S.norm_level(c["level"])): c["direction"] for c in cells}
        rev = [(k, gd[k], pd[k]) for k in gd if k in pd and gd[k] in POL and pd[k] in POL and POL[gd[k]] == -POL[pd[k]]]
        o = ["=" * 100, "SID %s KEY %s\nGOLD NAME: %s\nPRED NAME: %s  (model_equiv=%s)" % (sid, key, g["name"], p["name"], eq),
             "PARSE ENTRY:\n" + parse_entry(sid, key),
             "REVERSED CELLS (translated pred term, level): gold vs pred"]
        o += ["   %s | %s : gold=%s pred=%s" % (k[0], k[1], a, b) for k, a, b in rev]
        o += ["GOLD cells: %s" % [(c["term"], c["level"], c["direction"]) for c in g["cells"]],
              "PRED cells (translated): %s" % [(c["term"], c["level"], c["direction"]) for c in cells],
              "PRED cells (raw): %s" % [(c["term"], c["level"], c["direction"]) for c in p["cells"]],
              "GOLD kind=%s  PRED kind=%s" % (g["kind"], p["kind"]),
              "GOLD definition/interp: %s | %s" % (g["a"].get("definition"), g["a"].get("interpretations")),
              "PRED definition/interp: %s | %s" % (p["a"].get("definition"), p["a"].get("interpretations")),
              "GOLD terms:\n" + tdump(g), "PRED terms:\n" + tdump(p), ""]
        bysid.setdefault(sid, []).append("\n".join(o))
    return _orig(sid, gold, key, pobj, trans)
S.score_key = wrapped
os.makedirs(OUT, exist_ok=True)
for sid in sids:
    S.score_record(sid, recs.get(sid))
    open("%s/d_%s.txt" % (OUT, sid), "w").write("\n".join(bysid.get(sid, ["NONE"])))
    print(sid, len(bysid.get(sid, [])))
