"""Test A breakdown: categorical values by field, and numbers the paper writes as words.
    python ev_verify_breakdown.py VERIFY_DATA_DIR JUDGED_A [--form alone] [--examples 3]
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ev_verify import auc  # noqa: E402

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n):
    """English words for 0 <= n < 1,000,000, hyphenated and unhyphenated forms ("fifty-five", "fifty five")"""
    if n < 20:
        return [ONES[n]]
    if n < 100:
        t, o = divmod(n, 10)
        return [TENS[t]] if o == 0 else [f"{TENS[t]}-{ONES[o]}", f"{TENS[t]} {ONES[o]}"]
    if n < 1000:
        h, r = divmod(n, 100)
        if r == 0:
            return [f"{ONES[h]} hundred"]
        return [f"{ONES[h]} hundred{sep}{w}" for w in words(r) for sep in (" and ", " ")]
    k, r = divmod(n, 1000)
    if r == 0:
        return [f"{w} thousand" for w in words(k)]
    return [f"{a} thousand{sep}{b}" for a in words(k) for b in words(r) for sep in (" ", " and ")]


def nums(s):
    return [x.replace(",", "") for x in re.findall(r"\d[\d,]*(?:\.\d+)?", s)]


def digit_in(x, text):
    """x as a number in text (allowing thousands separators), not as part of a longer number"""
    pat = re.escape(x)
    if "." not in x and len(x) > 3:
        i = int(x)
        pat = f"(?:{re.escape(x)}|{i:,}|{i:,}".replace(",", "[,  ]?") + ")"
    return re.search(rf"(?<![\d.]){pat}(?![\d]|\.\d)", text) is not None


def word_in(x, text):
    if "." in x or int(x) >= 1_000_000:
        return False
    t = text.lower()
    return any(re.search(rf"(?<![a-z-]){re.escape(w)}(?![a-z-])", t) for w in words(int(x)))


def field(slot):
    return re.sub(r"\[\d+\]", "", slot).lstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("judged")
    ap.add_argument("--form", default="alone"); ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    d = Path(a.data)
    bench = {(r["sid"], r["slot"], r["variant"]): r for r in map(json.loads, open(d / "bench" / "bench.jsonl"))}
    state = {}
    for r in map(json.loads, open(d / "rows-a.jsonl")):
        t, form, var, sid, slot = r["id"].split("|")
        if form == a.form:
            state[var, sid, slot] = r["state"]
    p = {}
    for r in map(json.loads, open(a.judged)):
        t, form, var, sid, slot = r["id"].split("|")
        if form == a.form:
            p[var, sid, slot] = r["p"]
    keys = [(sid, slot) for (var, sid, slot) in p if var == "true" and ("wrong", sid, slot) in p]
    kind = {k: bench[k[0], k[1], "true"]["kind"] for k in keys}

    def cut(ks, q=0.10):
        tr = sorted(p["true", *k] for k in ks)
        return tr[int(q * len(tr))]

    # categorical, by field: one cut-off for all categorical values (keeps 90% of true values overall)
    en = [k for k in keys if kind[k] == "enum"]
    c = cut(en)
    by = collections.defaultdict(list)
    for k in en:
        by[field(k[1])].append(k)
    print(f"CATEGORICAL by field (form {a.form}; one cut-off {c:.3g}, keeping 90% of all true categorical values)")
    print(f"{'field':45s} {'n':>5s}  AUC   true kept  wrong rejected")
    for f, ks in sorted(by.items(), key=lambda x: -len(x[1])):
        tk = sum(p["true", *k] >= c for k in ks) / len(ks)
        wr = sum(p["wrong", *k] < c for k in ks) / len(ks)
        au = auc([p["true", *k] for k in ks], [p["wrong", *k] for k in ks])
        print(f"{f:45s} {len(ks):5d}  {au:.3f}  {tk:.3f}      {wr:.3f}")
    rnd = random.Random(0)
    print("\nEXAMPLES of wrong categorical values accepted (p >= cut-off), by field:")
    for f, ks in sorted(by.items(), key=lambda x: -len(x[1]))[:8]:
        miss = [k for k in ks if p["wrong", *k] >= c]
        for k in rnd.sample(miss, min(a.examples, len(miss))):
            st = state["wrong", *k].replace("\n", " / ")
            print(f"- [{f}] {k[0]}\n    true:  {bench[k[0], k[1], 'true']['claim']}  p={p['true', *k]:.3f}"
                  f"\n    wrong: {bench[k[0], k[1], 'wrong']['claim']}  p={p['wrong', *k]:.3f}"
                  f"\n    cited for wrong: {st[:400]}{'…' if len(st) > 400 else ''}")

    # numbers: is the true value in its cited units as digits, as words, or neither?
    nu = [k for k in keys if kind[k] == "number"]
    c = cut(nu)
    cls = collections.defaultdict(list)
    for k in nu:
        xs = [x for x in nums(bench[k[0], k[1], "true"]["claim"])]
        st = state["true", *k]
        if not xs:
            cls["no number in claim"].append(k)
        elif all(digit_in(x, st) for x in xs):
            cls["digits"].append(k)
        elif all(digit_in(x, st) or word_in(x, st) for x in xs):
            cls["words"].append(k)
        else:
            cls["neither"].append(k)
    print(f"\nNUMBERS: how the true value appears in the units ev1 cited for it (form {a.form}; cut-off {c:.3g}, "
          "keeping 90% of all true numbers)")
    print(f"{'true value in cited text':22s} {'n':>5s}  true kept  wrong rejected  AUC   median p(true)")
    for name in ("digits", "words", "neither", "no number in claim"):
        ks = cls.get(name, [])
        if not ks:
            continue
        tp = sorted(p["true", *k] for k in ks)
        print(f"{name:22s} {len(ks):5d}  {sum(x >= c for x in tp) / len(ks):.3f}      "
              f"{sum(p['wrong', *k] < c for k in ks) / len(ks):.3f}           "
              f"{auc(tp, [p['wrong', *k] for k in ks]):.3f}  {tp[len(tp) // 2]:.3f}")
    for name in ("words", "neither"):
        print(f"\nEXAMPLES, true number as {name}:")
        for k in sorted(cls.get(name, []), key=lambda k: p["true", *k])[:: max(1, len(cls.get(name, [])) // 8)][:8]:
            st = state["true", *k].replace("\n", " / ")
            print(f"- {k[0]} {field(k[1])}\n    true:  {bench[k[0], k[1], 'true']['claim']}  p={p['true', *k]:.3f}"
                  f"   wrong: {bench[k[0], k[1], 'wrong']['claim']}  p={p['wrong', *k]:.3f}"
                  f"\n    cited: {st[:300]}{'…' if len(st) > 300 else ''}")


if __name__ == "__main__":
    main()
