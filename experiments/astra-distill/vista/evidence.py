"""Evidence by unit index: the paper text with its units tagged [S1] [S2] ..., and the compact record (no evidence) in;
the record's evidence as unit numbers out.

    python evidence.py forms --split train.txt --n 300                    # output-form token comparison
    python evidence.py build --split train.txt --out PREFIX [--bodies]    # PREFIX.jsonl (train rows) [+ PREFIX.bodies.jsonl]
    python evidence.py score PREDS.json --split val.txt [--sids FILE] --out score.json   # PREDS: {"record": ...} or eval_record's val-pred.jsonl
    python evidence.py compare score_A.json score_B.json --split val.txt --out diff.json   # paired paper bootstrap of B - A

Units (`units`): a line holding a tab (a table row) or of at most 200 characters (a heading, a caption line) is one
unit; longer lines are cut into sentences. Astra's spans fit them: 98.2% lie in one unit and cover it (evfeas.py).

Target ("shape" form): the compact record's own structure, keeping only what has evidence. A slot becomes the sorted
unit numbers its spans touch (the union over evidence sets; 0.02% of slots have more than one); a dict keeps the
slots and children with evidence, plus `local_id`; a list keeps one item per compact item, `{}` for one without
evidence, trailing `{}` dropped, so positions line up with the compact record the model was given.
"""
import argparse
import bisect
import collections
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compact as K  # noqa: E402
import tasks as T  # noqa: E402

ABBR = r"(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bFigs)(?<!\bvs)(?<!\bcf)(?<!\bapprox)(?<!\bNo)(?<!\bEq)(?<!\bRef)"
CUT = re.compile(ABBR + r"[.!?](?=\s+[A-Z0-9(\[])")
SHORT = 200
INSTR = ("Below is a neuroimaging paper cut into numbered units ([S1], [S2], ...: sentences, table rows, captions) and "
         "the record extracted from it. For every value in the record that the paper supports, give the numbers of the "
         "units that state it. Return the record's structure with each supported value replaced by its unit numbers "
         "(a list of integers); keep local_id, keep one entry per list item ({} when nothing in it is supported), and "
         "leave out values the paper does not state.")
TEMPLATE = '{"<record field>": ["integer"], "<record list>": [{"local_id": "string", "<field>": ["integer"]}]}'


def units(text):
    """[(start, end)] of every unit, in text order"""
    out, pos = [], 0
    for line in text.split("\n"):
        s, e = pos, pos + len(line)
        pos = e + 1
        if not line.strip():
            continue
        if "\t" in line or len(line) <= SHORT:
            out.append((s, e))
            continue
        a = s
        for m in CUT.finditer(line):
            out.append((a, s + m.end()))
            a = s + m.end()
        if a < e:
            out.append((a, e))
    return [(a + len(text[a:b]) - len(text[a:b].lstrip()), b) for a, b in out if text[a:b].strip()]


def tagged(text, U):
    return "\n".join(f"[S{i + 1}] {text[a:b]}" for i, (a, b) in enumerate(U))


class Mapper:
    def __init__(self, U):
        self.U, self.starts = U, [a for a, _ in U]

    def __call__(self, ev):
        """unit numbers (1-based) an evidence object's spans touch"""
        idx = set()
        for st in (ev or {}).get("sets") or []:
            for sp in st.get("spans") or []:
                a, b = sp["start_char"], sp["end_char"]
                i = max(0, bisect.bisect_right(self.starts, a) - 1)
                while i < len(self.U) and self.U[i][0] < b:
                    if self.U[i][1] > a:
                        idx.add(i + 1)
                    i += 1
        return sorted(idx)


_M = K._MISSING


def _ev(x, excluded, mp, item_excluded=frozenset()):
    """(compact value, evidence shape) of x, in step with compact._compact; `item_excluded`: fields left out of each
    item of a top-level list (compact.to_compact's EXCLUDED)"""
    if K.is_wrapped(x):
        c = K._unwrap(x)
        if c is _M:
            return _M, _M
        idx = mp(x.get("evidence"))
        return c, (idx or _M)
    if isinstance(x, dict):
        cout, eout = {}, {}
        for k, v in x.items():
            if k in K.DROP_KEYS or k in excluded:
                continue
            c, e = _ev(v, frozenset(), mp)
            c = K._clean(c)
            if c is not _M:
                cout[k] = c
                if e is not _M:
                    eout[k] = e
        if eout and isinstance(cout.get("local_id"), str):
            eout = {"local_id": cout["local_id"], **eout}
        return cout, (eout or _M)
    if isinstance(x, list):
        cout, eout = [], []
        for v in x:
            c, e = _ev(v, item_excluded, mp)
            c = K._clean(c)
            if c is not _M:
                cout.append(c)
                eout.append(e if e is not _M else {})
        while eout and eout[-1] == {}:
            eout.pop()
        return cout, (eout or _M)
    return x, _M


def ev_target(record, mp):
    """(compact record, evidence shape); the compact record equals compact.to_compact(record)"""
    cout, eout = {}, {}
    for k, v in record.items():
        if k in K.DROP_KEYS:
            continue
        ex = K.EXCLUDED[K._COLLECTION_DEF[k]] if k in K._COLLECTION_DEF and isinstance(v, list) else frozenset()
        c, e = _ev(v, frozenset(), mp, ex)
        c = K._clean(c)
        if c is not _M and c != []:
            cout[k] = c
            if e is not _M:
                eout[k] = e
    return cout, eout


def pairs(ev, path=""):
    """(slot path, unit) for every unit an evidence shape cites; list items by position"""
    if isinstance(ev, dict):
        for k, v in ev.items():
            if k != "local_id":
                yield from pairs(v, f"{path}.{k}")
    elif isinstance(ev, list):
        if ev and all(isinstance(i, int) for i in ev):
            for i in ev:
                yield path, i
        else:
            for n, v in enumerate(ev):
                yield from pairs(v, f"{path}[{n}]")


def paper(sid, P):
    rec = json.load(open(P.records / f"{sid}.extraction.json"))
    text = T.load_text(sid, P)
    U = units(text)
    comp, ev = ev_target(rec, Mapper(U))
    assert comp == K.to_compact(rec), sid
    return rec, text, U, comp, ev


def row(sid, P):
    rec, text, U, comp, ev = paper(sid, P)
    prompt = (f"{INSTR}\n\nText:\n{tagged(text, U)}\n\nRecord:\n"
              f"{json.dumps(comp, ensure_ascii=False, separators=(',', ':'))}")
    target = json.dumps(ev, separators=(",", ":"))
    n_in = T.ntok(prompt) + T.ntok(TEMPLATE) + 40
    n_t = T.ntok(target)
    return {"sid": sid, "prompt": prompt, "template": TEMPLATE, "target_json": target, "n_tokens": n_in + n_t + 1,
            "n_input_tokens": n_in, "n_target_tokens": n_t, "n_units": len(U)}


def _row(args):
    return row(*args)


def read_split(a, P):
    f = Path(a.split)
    sids = (f if f.exists() else P.splits / a.split).read_text().split()
    if a.split.endswith("test.txt") or Path(a.split).name == "test.txt":
        raise SystemExit("test split: not built here")
    return sids


def cmd_forms(a, P):
    sids = random.Random(7).sample(read_split(a, P), a.n)
    shape, path, rec_in = [], [], []
    for s in sids:
        rec, text, U, comp, ev = paper(s, P)
        shape.append(T.ntok(json.dumps(ev, separators=(",", ":"))))
        flat = collections.defaultdict(list)
        for p, i in pairs(ev):
            flat[p].append(i)
        path.append(T.ntok(json.dumps(flat, separators=(",", ":"))))
        rec_in.append(T.ntok(json.dumps(comp, ensure_ascii=False, separators=(",", ":"))))
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    print(json.dumps({f: {p: q(v, p) for p in (0.5, 0.9, 0.99)} for f, v in
                      (("shape", shape), ("path", path), ("record_in", rec_in))}, indent=1))


def cmd_build(a, P):
    sids = read_split(a, P)
    out = Path(a.out + ".jsonl")
    bodies = open(a.out + ".bodies.jsonl", "w") if a.bodies else None
    n = 0
    with ProcessPoolExecutor(a.workers) as ex, open(out, "w") as f:
        for r in ex.map(_row, ((s, P) for s in sids), chunksize=8):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if bodies:
                bodies.write(json.dumps({"sid": r["sid"], "n_input_tokens": r["n_input_tokens"], "prompt": r["prompt"],
                                         "template": r["template"], "schema": {"type": "object"}}, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({"rows": n, "out": str(out)}))


def score_one(gold, pred):
    """micro counts over (slot path, unit) pairs, overall and by top-level collection, and slot-level exact match"""
    g, p = set(pairs(gold)), set(pairs(pred if isinstance(pred, (dict, list)) else {}))
    C = collections.Counter(tp=len(g & p), fp=len(p - g), fn=len(g - p))
    gs, ps = collections.defaultdict(set), collections.defaultdict(set)
    for s, i in g:
        gs[s].add(i)
    for s, i in p:
        ps[s].add(i)
    C["slots_gold"] = len(gs)
    C["slots_exact"] = sum(ps.get(s) == v for s, v in gs.items())
    C["slots_any"] = sum(bool(ps.get(s, set()) & v) for s, v in gs.items())
    by = collections.defaultdict(collections.Counter)
    for s, i in g | p:
        top = s.split(".")[1].split("[")[0] if s.count(".") else s
        by[top]["tp" if (s, i) in g and (s, i) in p else "fp" if (s, i) in p else "fn"] += 1
    return C, by


def f1(c):
    pr = c["tp"] / max(1, c["tp"] + c["fp"]); rc = c["tp"] / max(1, c["tp"] + c["fn"])
    return round(pr, 4), round(rc, 4), round(2 * pr * rc / max(1e-9, pr + rc), 4)


def load_preds(path):
    """{sid: evidence shape | None} from {"record": {...}} or eval_record.py's jsonl rows (sid, prediction)"""
    if path.endswith(".jsonl"):
        return {r["sid"]: r["prediction"] for r in map(json.loads, open(path)) if r.get("finish_reason")}
    return json.load(open(path))["record"]


def cmd_score(a, P):
    preds = load_preds(a.files[0])
    sids = read_split(a, P)
    if a.sids:
        keep = set(Path(a.sids).read_text().split())
        sids = [s for s in sids if s in keep]
    T_, by, per, boot = collections.Counter(), collections.defaultdict(collections.Counter), {}, []
    for s in sids:
        _, _, _, _, ev = paper(s, P)
        c, b = score_one(ev, preds.get(s))
        T_.update(c); per[s] = dict(c)
        for k, v in b.items():
            by[k].update(v)
    rng = random.Random(0)
    for _ in range(a.boot):
        smp = [per[rng.choice(sids)] for _ in sids]
        boot.append(f1(collections.Counter({k: sum(x[k] for x in smp) for k in ("tp", "fp", "fn")}))[2])
    boot.sort()
    pr, rc, f = f1(T_)
    res = {"papers": len(sids), "missing_predictions": sum(preds.get(s) is None for s in sids),
           "precision": pr, "recall": rc, "f1": f, "f1_ci95": [boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]] if boot else None,
           "slot_exact": round(T_["slots_exact"] / max(1, T_["slots_gold"]), 4),
           "slot_any_overlap": round(T_["slots_any"] / max(1, T_["slots_gold"]), 4),
           "by_collection": {k: dict(zip(("precision", "recall", "f1"), f1(v)), n_gold=v["tp"] + v["fn"]) for k, v in sorted(by.items())},
           "per_paper": per}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_paper"}, indent=1))


def cmd_compare(a, P):
    """B - A on the papers both score files hold: micro precision / recall / F1 and slot match, paired paper bootstrap"""
    A, B = (json.load(open(f))["per_paper"] for f in a.files[:2])
    sids = sorted(set(A) & set(B))

    def stats(per, ss):
        c = collections.Counter()
        for s in ss:
            c.update(per[s])
        pr, rc, f = f1(c)
        return {"precision": pr, "recall": rc, "f1": f, "slot_exact": c["slots_exact"] / max(1, c["slots_gold"]),
                "slot_any_overlap": c["slots_any"] / max(1, c["slots_gold"])}
    a_, b_ = stats(A, sids), stats(B, sids)
    rng = random.Random(0); boot = collections.defaultdict(list)
    for _ in range(a.boot):
        smp = [rng.choice(sids) for _ in sids]
        sa, sb = stats(A, smp), stats(B, smp)
        for k in sa:
            boot[k].append(sb[k] - sa[k])
    res = {"papers": len(sids), "only_a": len(set(A) - set(B)), "only_b": len(set(B) - set(A)), "metrics": {}}
    for k in a_:
        v = sorted(boot[k])
        res["metrics"][k] = {"a": round(a_[k], 4), "b": round(b_[k], 4), "diff": round(b_[k] - a_[k], 4),
                             "ci95": [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v))], 4)]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("forms", "build", "score", "compare"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="")
    ap.add_argument("--bodies", action="store_true")
    ap.add_argument("--sids", default="")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    for k, v in T.DEFAULTS.items():
        ap.add_argument(f"--{k}", default=str(v))
    a = ap.parse_args()
    P = T.Paths(**{k: getattr(a, k) for k in T.DEFAULTS})
    {"forms": cmd_forms, "build": cmd_build, "score": cmd_score, "compare": cmd_compare}[a.cmd](a, P)


if __name__ == "__main__":
    main()
