"""Does ev1 cite evidence for values that are wrong? Val records with some values replaced by wrong ones.

    python ev_perturb.py build CODE_DIR BENCH_DIR OUT_DIR [--k 8]     # OUT_DIR/bodies.jsonl, OUT_DIR/perturbed.jsonl
    python ev_perturb.py score CODE_DIR OUT_DIR BASE_PRED PERT_PRED

Only slots with gold evidence are changed, at most k per paper, so the rest of the record stays true:
numbers (ints, floats, lists of them) to a value that does not occur in the paper's text, booleans flipped, and
enum-like strings (a field with <= 15 distinct short values across val) swapped to another value of the same field.
Free text is left alone. score: for the changed slots, the share cited and the overlap with the gold units, with the
true value (BASE_PRED, the ordinary val run) and with the wrong one (PERT_PRED); the unchanged slots of the same
papers as a control.
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

field = lambda p: re.sub(r"\[\d+\]", "[]", p)  # noqa: E731
STEP = re.compile(r"\.([^.\[]+)|\[(\d+)\]")


def get(x, path):
    for k, i in STEP.findall(path):
        x = x[k] if k else x[int(i)]
    return x


def put(x, path, v):
    steps = STEP.findall(path)
    for k, i in steps[:-1]:
        x = x[k] if k else x[int(i)]
    k, i = steps[-1]
    if k:
        x[k] = v
    else:
        x[int(i)] = v


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def wrong_num(v, text, rng):
    for _ in range(50):
        if isinstance(v, int):
            w = v + rng.choice([-1, 1]) * max(2, round(abs(v) * rng.uniform(0.3, 0.8)))
            if v >= 0 and w < 0:
                w = v + max(2, round(abs(v) * 0.5))
        else:
            w = round(v * rng.choice([0.5, 0.6, 1.5, 1.8, 2.5]), 3)
        s = repr(w)
        if w != v and not re.search(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", text):
            return w
    return None


def near(a, b):
    """near-synonym enum values (years/year, fMRI/fmri), not a wrong value"""
    n = lambda x: re.sub(r"[^a-z0-9]", "", x.lower()).rstrip("s")  # noqa: E731
    return n(a) == n(b) or n(a) in n(b) or n(b) in n(a)


def groups(ev):
    d = collections.defaultdict(set)
    for s, i in E.pairs(ev if isinstance(ev, (dict, list)) else {}):
        d[s].add(i)
    return d


def build(a):
    sids = sorted({json.loads(l)["sid"] for l in open(Path(a.bench) / "bench.jsonl")})
    papers = {s: E.paper(s, P) for s in sids}
    seen = collections.defaultdict(collections.Counter)            # enum-like string fields
    for sid, (rec, text, U, comp, ev) in papers.items():
        for s in groups(ev):
            v = get(comp, s)
            if isinstance(v, str):
                seen[field(s)][v] += 1
    enums = {f: sorted(c) for f, c in seen.items() if 2 <= len(c) <= 15 and sum(c.values()) >= 20 and max(map(len, c)) <= 40}
    rng = random.Random(11)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    kinds = collections.Counter()
    with open(out / "bodies.jsonl", "w") as fb, open(out / "perturbed.jsonl", "w") as fp:
        for sid, (rec, text, U, comp, ev) in papers.items():
            g = groups(ev)
            cand = sorted(g); rng.shuffle(cand)
            changed = []
            for s in cand:
                if len(changed) >= a.k:
                    break
                v = get(comp, s)
                if isinstance(v, bool):
                    w, kind = (not v), "bool"
                elif isnum(v):
                    w, kind = wrong_num(v, text, rng), "number"
                elif isinstance(v, list) and v and all(isnum(x) for x in v):
                    w = [wrong_num(x, text, rng) for x in v]
                    w, kind = (None if None in w else w), "number"
                elif isinstance(v, str) and field(s) in enums:
                    others = [x for x in enums[field(s)] if not near(x, v)]
                    w, kind = (rng.choice(others) if others else None), "enum"
                else:
                    continue
                if w is None:
                    continue
                put(comp, s, w)
                changed.append({"sid": sid, "slot": s, "field": field(s), "kind": kind, "true": v, "wrong": w,
                                "gold": sorted(g[s])})
            kinds.update(c["kind"] for c in changed)
            for c in changed:
                fp.write(json.dumps(c, ensure_ascii=False) + "\n")
            prompt = (f"{E.INSTR}\n\nText:\n{E.tagged(text, U)}\n\nRecord:\n"
                      f"{json.dumps(comp, ensure_ascii=False, separators=(',', ':'))}")
            fb.write(json.dumps({"sid": sid, "n_units": len(U), "n_input_tokens": T.ntok(prompt) + T.ntok(E.TEMPLATE) + 40,
                                 "prompt": prompt, "template": E.TEMPLATE, "schema": {"type": "object"}},
                                ensure_ascii=False) + "\n")
    print(json.dumps({"papers": len(papers), "changed": sum(kinds.values()), "by_kind": kinds, "enum_fields": len(enums)}))


def score(a):
    rows = [json.loads(l) for l in open(Path(a.out) / "perturbed.jsonl")]
    load = lambda f: {r["sid"]: groups(r["prediction"]) for r in map(json.loads, open(f)) if r.get("finish_reason")}  # noqa: E731
    base, pert = load(a.base), load(a.pert)
    changed = {(r["sid"], r["slot"]) for r in rows}

    def tally(items, pred):
        c = collections.Counter()
        for sid, slot, gold in items:
            p = pred.get(sid, {}).get(slot, set())
            c["n"] += 1; c["cited"] += bool(p); c["hits_gold"] += bool(p & gold)
            c["tp"] += len(p & gold); c["fp"] += len(p - gold); c["fn"] += len(gold - p)
        return c

    def show(name, c):
        P_ = c["tp"] / max(1, c["tp"] + c["fp"]); R_ = c["tp"] / max(1, c["tp"] + c["fn"])
        print(f"{name:34s} n={c['n']:5d} cited {c['cited'] / c['n']:.3f}  cites a gold unit {c['hits_gold'] / c['n']:.3f}  "
              f"P {P_:.3f} R {R_:.3f}")

    sids = {r["sid"] for r in rows} & set(pert) & set(base)
    by = collections.defaultdict(list)
    for r in rows:
        if r["sid"] in sids:
            by[r["kind"]].append((r["sid"], r["slot"], set(r["gold"])))
    allc = [x for v in by.values() for x in v]
    print(f"papers {len(sids)}")
    show("changed slots, true value", tally(allc, base))
    show("changed slots, WRONG value", tally(allc, pert))
    for k, v in sorted(by.items()):
        show(f"  {k}: true", tally(v, base)); show(f"  {k}: WRONG", tally(v, pert))
    ctrl = []
    for sid in sids:
        rec, text, U, comp, ev = E.paper(sid, P)
        for s, gs in groups(ev).items():
            if (sid, s) not in changed:
                ctrl.append((sid, s, gs))
    show("unchanged slots, base run", tally(ctrl, base))
    show("unchanged slots, perturbed run", tally(ctrl, pert))
    # wrong values still cited: same units as with the true value?
    same = diff = 0
    for sid, slot, gold in allc:
        p, b = pert.get(sid, {}).get(slot, set()), base.get(sid, {}).get(slot, set())
        if p:
            same += p == b; diff += p != b
    print(f"wrong values still cited: {same + diff}; same units as for the true value {same}, different {diff}")
    by_f = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["sid"] in sids:
            by_f[r["field"]]["n"] += 1
            by_f[r["field"]]["cited"] += bool(pert.get(r["sid"], {}).get(r["slot"]))
    print("fields with >= 25 changed slots, share still cited when wrong:")
    for f, c in sorted(by_f.items(), key=lambda x: x[1]["cited"] / x[1]["n"]):
        if c["n"] >= 25:
            print(f"  {f[:60]:60s} n={c['n']:4d} {c['cited'] / c['n']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("code")
    a0, rest = ap.parse_known_args()
    sys.path.insert(0, a0.code)
    import evidence as E  # noqa: E402
    import tasks as T  # noqa: E402
    M = "/data/james/agents/astra-distill/migrate/"
    P = T.Paths(records=M + "train-v6/records", splits=M + "splits-v6")
    if a0.cmd == "build":
        bp = argparse.ArgumentParser(); bp.add_argument("bench"); bp.add_argument("out"); bp.add_argument("--k", type=int, default=8)
        build(bp.parse_args(rest))
    else:
        sp = argparse.ArgumentParser(); sp.add_argument("out"); sp.add_argument("base"); sp.add_argument("pert")
        score(sp.parse_args(rest))
