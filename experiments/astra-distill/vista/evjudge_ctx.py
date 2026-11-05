"""Context variants of the evidence pilot (`evjudge_build.py` rows, with their "units"): does more of the paper help a
judge tell a unit that states a slot's value from one that does not?

    python evjudge_ctx.py PILOT.jsonl OUT.jsonl   (run from vista/ on beast; v6 records)
    python evjudge_ctx.py score D3CTX.jsonl D3.jsonl

Rows of OUT, judged by `evjudge_decision.py d3ctx`:
  win   one per pilot row: the judged units marked >>> … <<< within 3 units either side (a table row also gets its
        table's first row and the line before the table); asks whether the marked text states the bare claim
  full  one per slot (60 far + 300 pos/neg slots): the whole paper with candidate units tagged [S1] … [Sk] — far: the
        model's far units and Astra's; pos/neg: Astra's units and the random ones — and a choice of the tag that
        states the bare claim, or none. "cands": [{"tag", "unit", "label"}]; label 1 Astra (pos), 0 random (neg),
        the verdict for a far unit, None for Astra's units of a far slot
"""
import collections
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, ".")
import evidence as E  # noqa: E402
from evjudge_build import P, doc  # noqa: E402

WINDOW = 3


def window(text, U, units):
    def t(j):
        a, b = U[j - 1]
        return text[a:b].strip()
    parts = []
    for i in units:
        head = doc(text, U, i).split("\n")[:-1]   # a table row's header lines
        lines = [f">>> {t(j)} <<<" if j in units else t(j)
                 for j in range(max(1, i - WINDOW), min(len(U), i + WINDOW) + 1)]
        parts.append("\n".join(head + lines))
    return "\n...\n".join(parts)


def tagged(text, U, tags):
    """text with "[Sk] " before each candidate unit"""
    out, at = [], 0
    for i in sorted(tags):
        a = U[i - 1][0]
        out += [text[at:a], f"[{tags[i]}] "]
        at = a
    return "".join(out + [text[at:]])


def score(ctx, unit_scores):
    """win: `evjudge_run.py score` beside the unit-only d3 scores. full: each candidate's choice probability, scored
    like a judge's (AUCs, the cut-off with the best balanced accuracy on pos/neg candidates), and the choice's top tag"""
    import evjudge_run as R
    rows = [json.loads(line) for line in open(ctx)]
    print("errors:", collections.Counter((r["set"], r["error"]) for r in rows if "error" in r))
    win = [{**r, "set": r["src"]} for r in rows if r["set"] == "win" and "scores" in r]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.writelines(json.dumps(r) + "\n" for r in win)
    print("== window (unit with 3 units either side) vs unit alone ==")
    R.score([unit_scores, f.name])
    os.unlink(f.name)
    full = [r for r in rows if r["set"] == "full" and "probs" in r]
    cands = [{**c, "src": r["src"], "p": r["probs"][c["tag"]]} for r in full for c in r["cands"]]
    pos = [c["p"] for c in cands if c["src"] == "pn" and c["label"] == 1]
    neg = [c["p"] for c in cands if c["src"] == "pn" and c["label"] == 0]
    fs = [c["p"] for c in cands if c["src"] == "far" and c["label"] == 1]
    fn = [c["p"] for c in cands if c["src"] == "far" and c["label"] == 0]
    fa = [c["p"] for c in cands if c["src"] == "far" and c["label"] is None]

    def acc(xs, cut, want):
        return round(sum((x > cut) == want for x in xs) / max(1, len(xs)), 3)
    cut = max(sorted(set(pos + neg)), key=lambda c: acc(pos, c, True) + acc(neg, c, False))
    print(f"\n== full paper, choice of tag ({len(full)} slots) ==")
    print(f"candidates: pos {len(pos)} neg {len(neg)} far states {len(fs)} far not {len(fn)} far-slot Astra {len(fa)}")
    print(f"AUC far {R.auc(fs, fn)}  pos/neg {R.auc(pos, neg)}  cut {cut:.2f}: pos acc {acc(pos, cut, True)} neg acc "
          f"{acc(neg, cut, False)} far states {acc(fs, cut, True)} far not {acc(fn, cut, False)} "
          f"far-slot Astra {acc(fa, cut, True)}")
    for src in ("pn", "far"):
        rs = [r for r in full if r["src"] == src]
        top = collections.Counter()
        for r in rs:
            t = max(r["probs"], key=r["probs"].get)
            lab = {c["tag"]: c["label"] for c in r["cands"]}.get(t, "none")
            top[{1: "states/Astra", 0: "wrong/random", 0.5: "partly", None: "Astra", "none": "none"}[lab]] += 1
        pn = sorted(r["probs"]["none"] for r in rs)
        print(f"{src}: top choice {dict(top)}; P(none) median {pn[len(pn) // 2]:.2f}")


def main():
    if sys.argv[1] == "score":
        return score(*sys.argv[2:4])
    pilot, out = sys.argv[1:3]
    rows = [json.loads(line) for line in open(pilot)]
    papers = {}

    def paper(s):
        if s not in papers:
            _, text, U, _, ev = E.paper(s, P)
            papers[s] = text, U, ev
        return papers[s]

    res = []
    for r in rows:
        text, U, _ = paper(r["sid"])
        res.append({"set": "win", "src": r["set"], "id": r["id"], "sid": r["sid"], "slot": r["slot"],
                    "label": r["label"], "bare": r["bare"], "state": window(text, U, r["units"])})
    rng = random.Random(5)
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["set"] if r["set"] == "far" else "pn", r["id"]][r["set"]] = r
    for (kind, n), g in sorted(by.items()):
        r = g.get("far") or g["pos"]
        text, U, ev = paper(r["sid"])
        if kind == "far":
            gold = sorted({i for k, i in E.pairs(ev) if k == r["slot"]})
            labels = {**{i: None for i in gold}, **{i: r["label"] for i in r["units"]}}
        else:
            labels = {**{i: 0 for i in g["neg"]["units"]}, **{i: 1 for i in g["pos"]["units"]}}
        order = sorted(labels)
        rng.shuffle(order)
        tags = {i: f"S{k}" for k, i in enumerate(order, 1)}
        res.append({"set": "full", "src": kind, "id": n, "sid": r["sid"], "slot": r["slot"], "bare": r["bare"],
                    "cands": [{"tag": tags[i], "unit": i, "label": labels[i], "text": doc(text, U, i)} for i in order],
                    "state": tagged(text, U, tags)})
    with open(out, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    full = [len(r["state"]) for r in res if r["set"] == "full"]
    print(collections.Counter((r["set"], r["src"]) for r in res), "full chars median", sorted(full)[len(full) // 2],
          "max", max(full), "papers", len(papers))


if __name__ == "__main__":
    main()
