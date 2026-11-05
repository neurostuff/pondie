"""Can elephant-1 tell a wrong value from a true one? The wrong-value test's swapped slots (ev_perturb.py), each as a
true claim and a wrong claim (evjudge_build.bare_claim on the record with the true / the wrong value).

    python ev_verify.py bench  CODE PERTURB_DIR BENCH_DIR OUT                 # OUT/bench/{bench,units}.jsonl for rerank_eval
    python ev_verify.py rows-a CODE PERTURB_DIR BASE_PRED PERT_PRED OUT       # OUT/rows-a.jsonl
    python ev_verify.py rows-b CODE OUT SCORES.jsonl [--k 10]                 # OUT/rows-b.jsonl
    python ev_verify.py rows-c CODE PERTURB_DIR OUT [--only FIELD ...] [--tag T] [--empty] [--name F]   # OUT/rows-c.jsonl
    python ev_verify.py score  OUT                                            # OUT/rows-*.jsonl + OUT/judged-*.jsonl

A, ev1's citations: the true claim against the units ev1 cited for the true value (ordinary val run), the wrong claim
against the units it cited for the wrong one (perturbed run); state "alone" (the units, evjudge_build.doc) or "head"
(each unit under its markdown heading path, "Section: Methods > Participants").
B, search: each claim's top k units by a reranker (rerank_eval scores over every unit of the paper); each unit judged
alone with its heading ("unit"), and the top 3 joined ("top3"). A claim's score is the max over its states.
Score: AUC true vs wrong, and the share of wrong values rejected at the cut-off that accepts 90% / 95% of true values
(cut-off from the first half of the papers, applied to the second half).
C, plain words: test A's "alone" states with claims that say what the value means (`PLAIN`), naming the analysis or
term ("named") or not ("plain"); --empty asks the same claims against no text (N). Scored by ev_verify_c.py.
"""
import argparse
import bisect
import collections
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)


def headings(text):
    """[(start_char, 'A > B')] for each markdown heading, as a path of the headings above it"""
    out, stack = [], []
    for m in HEAD.finditer(text):
        lvl = len(m.group(1))
        stack = [s for s in stack if s[0] < lvl] + [(lvl, m.group(2))]
        out.append((m.start(), " > ".join(t for _, t in stack)))
    return out


def head_of(hs, a):
    i = bisect.bisect_right([s for s, _ in hs], a) - 1
    return hs[i][1] if i >= 0 else None


def state(text, U, units, hs=None):
    """units (1-based) joined; with hs, a heading line before each run of units under the same heading"""
    parts, last = [], object()
    for i in sorted(units):
        if hs is not None:
            h = head_of(hs, U[i - 1][0])
            if h != last:
                parts.append(f"Section: {h or '(none)'}")
                last = h
        parts.append(doc(text, U, i))
    return "\n".join(parts)


def load_paper(sid):
    rec, text, U, comp, ev = E.paper(sid, P)
    return text, U, comp


def claims(rows):
    """(sid, slot) -> {"true": claim, "wrong": claim}"""
    out, by = {}, collections.defaultdict(list)
    for r in rows:
        by[r["sid"]].append(r)
    for sid, rs in by.items():
        text, U, comp = load_paper(sid)
        for r in rs:
            w = copy.deepcopy(comp)
            put(w, r["slot"], r["wrong"])
            out[sid, r["slot"]] = {"true": bare_claim(comp, r["slot"]), "wrong": bare_claim(w, r["slot"])}
    return out


def cmd_bench(a):
    rows = [json.loads(l) for l in open(Path(a.perturb) / "perturbed.jsonl")]
    cl = claims(rows)
    out = Path(a.out) / "bench"; out.mkdir(parents=True, exist_ok=True)
    qid = 0
    with open(out / "bench.jsonl", "w") as f:
        for r in rows:
            for v in ("true", "wrong"):
                f.write(json.dumps({"qid": qid, "sid": r["sid"], "slot": r["slot"], "variant": v, "kind": r["kind"],
                                    "collection": r["slot"].lstrip(".").split("[")[0].split(".")[0],
                                    "claim": cl[r["sid"], r["slot"]][v], "gold": r["gold"]}, ensure_ascii=False) + "\n")
                qid += 1
    (out / "units.jsonl").unlink(missing_ok=True)
    (out / "units.jsonl").symlink_to(Path(a.bench).resolve() / "units.jsonl")
    print(json.dumps({"queries": qid}))


def groups(pred):
    d = collections.defaultdict(set)
    for s, i in E.pairs(pred if isinstance(pred, (dict, list)) else {}):
        d[s].add(i)
    return d


def cmd_rows_a(a):
    rows = [json.loads(l) for l in open(Path(a.perturb) / "perturbed.jsonl")]
    cl = claims(rows)
    load = lambda f: {r["sid"]: groups(r["prediction"]) for r in map(json.loads, open(f))}  # noqa: E731
    base, pert = load(a.base), load(a.pert)
    n = collections.Counter()
    with open(Path(a.out) / "rows-a.jsonl", "w") as f:
        by = collections.defaultdict(list)
        for r in rows:
            by[r["sid"]].append(r)
        for sid, rs in by.items():
            text, U, comp = load_paper(sid)
            hs = headings(text)
            for r in rs:
                for v, pred in (("true", base), ("wrong", pert)):
                    us = sorted(u for u in pred.get(sid, {}).get(r["slot"], set()) if 1 <= u <= len(U))
                    if not us:
                        n[f"uncited_{v}"] += 1
                        continue
                    for form, h in (("alone", None), ("head", hs)):
                        f.write(json.dumps({"id": f"A|{form}|{v}|{sid}|{r['slot']}", "state": state(text, U, us, h),
                                            "claim": cl[sid, r["slot"]][v]}, ensure_ascii=False) + "\n")
                        n["rows"] += 1
    print(json.dumps(n))


def cmd_rows_b(a):
    bench = [json.loads(l) for l in open(Path(a.out) / "bench" / "bench.jsonl")]
    sc = {r["qid"]: r["scores"] for r in map(json.loads, open(a.scores))}
    by = collections.defaultdict(list)
    for q in bench:
        by[q["sid"]].append(q)
    n = collections.Counter()
    with open(Path(a.out) / "rows-b.jsonl", "w") as f:
        for sid, qs in by.items():
            text, U, comp = load_paper(sid)
            hs = headings(text)
            for q in qs:
                s = sc[q["qid"]]
                top = [i + 1 for i in sorted(range(len(s)), key=lambda i: -s[i])[:a.k]]
                key = f"{q['variant']}|{sid}|{q['slot']}"
                for u in top:
                    f.write(json.dumps({"id": f"B|unit|{key}|{u}", "state": state(text, U, [u], hs), "claim": q["claim"]},
                                       ensure_ascii=False) + "\n")
                f.write(json.dumps({"id": f"B|top3|{key}", "state": state(text, U, top[:3], hs), "claim": q["claim"]},
                                   ensure_ascii=False) + "\n")
                n["claims"] += 1; n["rows"] += len(top) + 1
                n["gold_in_top"] += bool(set(top) & set(q["gold"]))
    print(json.dumps(n))


#: C, claims in plain words for the five weakest categorical fields; {o} is the owner ("the analysis "X"", "the
#: model term "age"") in the "named" style and a generic one in the "plain" style, {l} a cell's level
PLAIN = {
    "analyses.effect.cells.direction": {
        "positive": "In {o}, {l} is on the higher side of the effect (greater activation, connectivity or value).",
        "negative": "In {o}, {l} is on the lower side of the effect (less activation, connectivity or value).",
        "undirected": "In {o}, the effect involving {l} is reported without a direction (for example an F test).",
        "held": "In {o}, {l} is held constant: the comparison is made within {l}, not between {l} and something else."},
    "model_estimations.terms.type": {
        "categorical": "{O} is a categorical variable (groups or conditions).",
        "continuous": "{O} is a continuous variable (a numeric measure such as a score or age)."},
    "model_estimations.terms.variation_level": {
        "between_subject": "{O} differs between participants (each participant has one value or belongs to one group).",
        "within_subject": "{O} varies within each participant (each participant contributes data at several of its values).",
        "mixed": "{O} varies both between and within participants.",
        "between datasets": "{O} varies between datasets.",
        "between studies": "{O} varies between studies, as in a meta-analysis.",
        "across experiments": "{O} varies across experiments, as in a meta-analysis.",
        "across published experiments": "{O} varies across published experiments, as in a meta-analysis."},
    "analyses.effect.kind": {
        "contrast": "{O} directly compares two conditions or groups (a contrast).",
        "simple_effect": "{O} tests a simple effect: one factor's effect within one level of another factor.",
        "interaction": "{O} tests an interaction between factors.",
        "omnibus": "{O} is an omnibus test across several groups or conditions (such as an ANOVA F test).",
        "cross_subject_regression": "{O} relates a measure to a variable across participants (a correlation or regression).",
        "parametric_modulation": "{O} tests a parametric modulation: brain response scaling with a trial-by-trial variable."},
    "coordinate_sets.role": {
        "result": "{O} are findings of an analysis, such as peaks of an effect.",
        "localization": "{O} only give where a region is, and are not findings of an analysis.",
        "anchor": "{O} are a seed or region of interest chosen in advance, which an analysis starts from."},
}
#: test C's first wording for coordinate_sets.role ("are results of the study's own analysis") asked more than a
#: results table states (median P(yes) of true claims .04); rows-c --only coordinate_sets.role --tag 2 reruns it
GENERIC = {"analyses": "the analysis", "model_estimations.terms": "a model term", "coordinate_sets": "the coordinates"}


def owner(comp, slot, style):
    """(owner phrase, cell level or None) for a slot of a PLAIN field"""
    st = steps(slot)
    top, i = st[0][0], st[1][1]
    item = comp[top][i]
    lvl = None
    if top == "analyses" and len(st) > 4 and st[3][0] == "cells":
        lvl = say(item["effect"]["cells"][st[4][1]].get("level"))
    if style == "plain":
        if top == "model_estimations":
            return GENERIC["model_estimations.terms"], lvl
        return GENERIC[top], lvl
    if top == "model_estimations":
        return f'the model term "{item["terms"][st[3][1]].get("name")}"', lvl
    if top == "coordinate_sets":
        a = next((x for x in comp.get("analyses", []) if x.get("local_id") == item.get("analysis")), None)
        return (f'the coordinates for the analysis "{a["name"]}"' if a and a.get("name") else "the coordinates"), lvl
    return f'the analysis "{item.get("name")}"', lvl


def plain_claim(comp, slot, value, style):
    o, lvl = owner(comp, slot, style)
    t = PLAIN[re.sub(r"\[\d+\]", "", slot).lstrip(".")][value]
    return t.format(o=o, O=o[0].upper() + o[1:], l=lvl or "a group or condition")


NO_TEXT = "(No text is given.)"


def cmd_rows_c(a):
    """test A's states (form alone) with PLAIN claims in both styles: C|{plain|named}{tag}|{true|wrong}|sid|slot;
    --empty asks them against no text instead (N|...), to see how much elephant-1 knows without the paper"""
    only = set(a.only) if a.only else set(PLAIN)
    rows = {(r["sid"], r["slot"]): r for r in map(json.loads, open(Path(a.perturb) / "perturbed.jsonl"))
            if re.sub(r"\[\d+\]", "", r["slot"]).lstrip(".") in only}
    st = {}
    for r in map(json.loads, open(Path(a.out) / "rows-a.jsonl")):
        t, form, var, sid, slot = r["id"].split("|")
        if form == "alone" and (sid, slot) in rows:
            st[var, sid, slot] = r["state"]
    n = collections.Counter()
    t = "N" if a.empty else "C"
    with open(Path(a.out) / a.name, "w") as f:
        for sid in sorted({s for s, _ in rows}):
            text, U, comp = load_paper(sid)
            for (s, slot), r in rows.items():
                if s != sid:
                    continue
                for var in ("true", "wrong"):
                    if (var, sid, slot) not in st:
                        continue
                    v = value_at(comp, slot) if var == "true" else r["wrong"]
                    if v not in PLAIN[re.sub(r"\[\d+\]", "", slot).lstrip(".")]:
                        n["unknown_value"] += 1
                        continue
                    for style in ("plain", "named"):
                        f.write(json.dumps({"id": f"{t}|{style}{a.tag}|{var}|{sid}|{slot}",
                                            "state": NO_TEXT if a.empty else st[var, sid, slot],
                                            "claim": plain_claim(comp, slot, v, style)}, ensure_ascii=False) + "\n")
                        n["rows"] += 1
    print(json.dumps(n))


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    r, i, rp = 0.0, 0, 0.0
    while i < len(s):                       # average ranks over ties
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        rp += ((i + j + 1) / 2) * sum(l for _, l in s[i:j])
        i = j
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def half(sid):
    return int(hashlib.md5(sid.encode()).hexdigest(), 16) % 2


def cmd_score(a):
    out = Path(a.out)
    p = {}
    for f in out.glob("judged-*.jsonl"):
        for r in map(json.loads, open(f)):
            p[r["id"]] = r["p"]
    kinds = {(r["sid"], r["slot"]): r["kind"] for r in map(json.loads, open(out / "bench" / "bench.jsonl"))}
    # per (test, form, variant, sid, slot): the claim's score (max over its states)
    S = collections.defaultdict(float)
    for i, v in p.items():
        t, form, var, sid, slot = i.split("|")[:5]
        S[t, form, var, sid, slot] = max(S[t, form, var, sid, slot], v)
    for (t, form, var, sid, slot), v in list(S.items()):   # B both: max of a claim's units and its top 3 joined
        if t == "B":
            S["B", "both", var, sid, slot] = max(S["B", "both", var, sid, slot], v)
    tests = sorted({(k[0], k[1]) for k in S})
    print(f"{'test':10s} {'kind':7s} {'n':>5s}  AUC    reject wrong @ accept true 90% / 95%   (cut-off from half 0, on half 1)")
    for t, form in tests:
        for kind in (None, "enum", "number"):
            keys = {(sid, slot) for (t_, f_, var, sid, slot) in S if (t_, f_) == (t, form)
                    and (kind is None or kinds[sid, slot] == kind)}
            both = [k for k in keys if (t, form, "true", *k) in S and (t, form, "wrong", *k) in S]
            tr = {h: [S[t, form, "true", *k] for k in both if half(k[0]) == h] for h in (0, 1)}
            wr = {h: [S[t, form, "wrong", *k] for k in both if half(k[0]) == h] for h in (0, 1)}
            res = []
            for q in (0.10, 0.05):
                cut = sorted(tr[0])[int(q * len(tr[0]))] if tr[0] else 0
                acc = sum(x >= cut for x in tr[1]) / max(1, len(tr[1]))
                rej = sum(x < cut for x in wr[1]) / max(1, len(wr[1]))
                res.append(f"{rej:.3f} (true kept {acc:.3f}, cut {cut:.3g})")
            print(f"{t + ':' + form:10s} {kind or 'all':7s} {len(both):5d}  {auc(tr[0] + tr[1], wr[0] + wr[1]):.3f}  " + "   ".join(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("code")
    a0, rest = ap.parse_known_args()
    if a0.cmd == "score":
        cmd_score(argparse.Namespace(out=a0.code)); sys.exit()
    sys.path.insert(0, a0.code)
    import evidence as E  # noqa: E402
    import tasks as T  # noqa: E402
    from evjudge_build import bare_claim, doc, say, steps, value_at  # noqa: E402
    from ev_perturb import put  # noqa: E402
    M = "/data/james/agents/astra-distill/migrate/"
    P = T.Paths(records=M + "train-v6/records", splits=M + "splits-v6")
    sp = argparse.ArgumentParser()
    if a0.cmd == "bench":
        for x in ("perturb", "bench", "out"):
            sp.add_argument(x)
        cmd_bench(sp.parse_args(rest))
    elif a0.cmd == "rows-a":
        for x in ("perturb", "base", "pert", "out"):
            sp.add_argument(x)
        cmd_rows_a(sp.parse_args(rest))
    elif a0.cmd == "rows-c":
        sp.add_argument("perturb"); sp.add_argument("out"); sp.add_argument("--name", default="rows-c.jsonl")
        sp.add_argument("--only", nargs="*"); sp.add_argument("--tag", default="")
        sp.add_argument("--empty", action="store_true")
        cmd_rows_c(sp.parse_args(rest))
    elif a0.cmd == "rows-b":
        sp.add_argument("out"); sp.add_argument("scores"); sp.add_argument("--k", type=int, default=10)
        cmd_rows_b(sp.parse_args(rest))
