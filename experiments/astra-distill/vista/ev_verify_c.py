"""Score tests C and N (claims in plain words, ev_verify.py rows-c) against test A's bare claims on the same slots.
    python ev_verify_c.py JUDGED_A JUDGED_C [JUDGED ...]
Styles: bare (A:alone), plain / named (C), plain+2 / named+2 (C with the roles reworded by rows-c --tag 2), no-text
plain / named (N, against no text).
Per field and over the five: AUC / wrong rejected at the cut-off keeping 90% of true (cut-off from paper half 0, on half 1),
with the share of true kept on half 1 in brackets.
"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ev_verify import auc, half  # noqa: E402

p = collections.defaultdict(dict)
for f in sys.argv[1:]:
    for r in map(json.loads, open(f)):
        t, form, var, sid, slot = r["id"].split("|")
        if t == "A" and form != "alone":
            continue
        st = {"A": "bare", "C": form, "N": f"no-text {form}"}[t]
        p[st][var, sid, slot] = r["p"]
for f in ("plain", "named"):  # the C claims with the reworded roles in place of the first wording
    if f"{f}2" in p:
        p[f"{f}+2"] = {**p[f], **p[f"{f}2"]}
field = lambda sl: re.sub(r"\[\d+\]", "", sl).lstrip(".")  # noqa: E731
slots = {(s, sl) for st in p if st != "bare" for (v, s, sl) in p[st]}
groups = collections.defaultdict(set)
for k in slots:
    groups[field(k[1])].add(k)
groups = dict(sorted(groups.items(), key=lambda x: -len(x[1])))
groups["all five"] = slots
styles = [s for s in ("bare", "plain", "named", "plain+2", "named+2", "no-text plain", "no-text named") if s in p]


def cell(P, ks):
    ks = [k for k in ks if ("true", *k) in P and ("wrong", *k) in P]
    if not ks:
        return "", 0
    tr0 = sorted(P["true", *k] for k in ks if half(k[0]) == 0)
    cut = tr0[int(0.10 * len(tr0))] if tr0 else 0
    h1 = [k for k in ks if half(k[0]) == 1]
    rej = sum(P["wrong", *k] < cut for k in h1) / max(1, len(h1))
    kept = sum(P["true", *k] >= cut for k in h1) / max(1, len(h1))
    return f"{auc([P['true', *k] for k in ks], [P['wrong', *k] for k in ks]):.3f} / {rej:.2f} ({kept:.2f})", len(ks)


print(f"{'field':40s} {'n':>4s}  " + "  ".join(f"{s:>20s}" for s in styles))
for g, ks in groups.items():
    cells = [cell(p[s], ks) for s in styles]
    print(f"{g:40s} {max(n for _, n in cells):4d}  " + "  ".join(f"{c:>20s}" for c, _ in cells))
