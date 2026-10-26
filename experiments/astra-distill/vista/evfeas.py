"""Feasibility of evidence by unit index: the paper text tagged [S1] [S2] ..., the evidence-free record in, and for
every slot with evidence the indices of the units its spans fall in, out.

    python evfeas.py --split train.txt --n 600 --out evfeas.json [--records DIR --splits DIR]

Units: a line that holds a tab (a table row) or is short (a heading, a caption line) is one unit; other lines are cut
into sentences. Measured per paper: unit count, tokens of the tagged text, of the compact record and of the output
(`{"<slot path>": [[i, j], ...]}`, one inner list per evidence set); per span: how many units it touches and how much
of those units' text it covers; per slot path: that it resolves back to exactly one slot of the record.
"""
import argparse
import bisect
import collections
import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compact as K  # noqa: E402
import tasks as T  # noqa: E402

# a sentence ends at . ! ? followed by space and an upper-case letter or a digit, unless the word before is an
# abbreviation the papers use mid-sentence
ABBR = r"(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)(?<!\bFig)(?<!\bFigs)(?<!\bvs)(?<!\bcf)(?<!\bapprox)(?<!\bNo)(?<!\bEq)(?<!\bRef)"
CUT = re.compile(ABBR + r"[.!?](?=\s+[A-Z0-9(\[])")
SHORT = 200


def units(text):
    """[(start, end)] covering every non-blank stretch of the text, in order."""
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
            b = s + m.end()
            out.append((a, b))
            a = b
        if a < e:
            out.append((a, e))
    return [(a + len(text[a:b]) - len(text[a:b].lstrip()), b) for a, b in out if text[a:b].strip()]


def slots(x, path=""):
    """(path, evidence) for every wrapped slot whose evidence has spans; list items by local_id where they have one"""
    if isinstance(x, dict):
        ev = x.get("evidence") if "extraction_status" in x else None
        if isinstance(ev, dict) and ev.get("sets"):
            yield path, ev
        for k, v in x.items():
            if k != "evidence":
                yield from slots(v, f"{path}.{k}" if path else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            tag = v["local_id"] if isinstance(v, dict) and isinstance(v.get("local_id"), str) else i
            yield from slots(v, f"{path}[{tag}]")


def resolve(rec, path):
    """the object a slot path names, or None"""
    x = rec
    for part in re.findall(r"[^.\[\]]+|\[[^\]]+\]", path):
        if part.startswith("["):
            tag = part[1:-1]
            if not isinstance(x, list):
                return None
            hit = [v for v in x if isinstance(v, dict) and v.get("local_id") == tag]
            if not hit and tag.isdigit() and int(tag) < len(x):
                hit = [x[int(tag)]]
            if len(hit) != 1:
                return None
            x = hit[0]
        else:
            if not isinstance(x, dict) or part not in x:
                return None
            x = x[part]
    return x


def one(sid, P):
    rec = json.load(open(P.records / f"{sid}.extraction.json"))
    text = T.load_text(sid, P)
    U = units(text)
    starts = [a for a, _ in U]
    tagged = "".join(f"[S{i + 1}] {text[a:b]}\n" for i, (a, b) in enumerate(U))
    out, span_units, cover, outside, n_paths, bad_paths = {}, [], [], 0, 0, 0
    for path, ev in slots(rec):
        n_paths += 1
        if resolve(rec, path) is None:
            bad_paths += 1
        sets = []
        for st in ev["sets"]:
            idx = set()
            for sp in st.get("spans") or []:
                a, b = sp["start_char"], sp["end_char"]
                i = max(0, bisect.bisect_right(starts, a) - 1)
                hit = []
                while i < len(U) and U[i][0] < b:
                    if U[i][1] > a:
                        hit.append(i)
                    i += 1
                if not hit:
                    outside += 1
                    continue
                span_units.append(len(hit))
                tot = sum(U[j][1] - U[j][0] for j in hit)
                cover.append(min(1.0, (b - a) / max(1, tot)))
                idx.update(hit)
            if idx:
                sets.append(sorted(j + 1 for j in idx))
        if sets:
            out[path] = sets
    out_s = json.dumps(out, separators=(",", ":"))
    rec_s = json.dumps(K.to_compact(rec), ensure_ascii=False)
    return dict(sid=sid, units=len(U), tok_text=T.ntok(text), tok_tagged=T.ntok(tagged),
                tok_record=T.ntok(rec_s), tok_out=T.ntok(out_s), slots=len(out), paths=n_paths,
                bad_paths=bad_paths, spans_outside=outside, span_units=span_units, cover=cover,
                sets_per_slot=[len(v) for v in out.values()], units_per_slot=[len({i for s in v for i in s}) for v in out.values()])


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    for k, v in T.DEFAULTS.items():
        ap.add_argument(f"--{k}", default=str(v))
    a = ap.parse_args()
    P = T.Paths(**{k: getattr(a, k) for k in T.DEFAULTS})
    f = Path(a.split)
    sids = (f if f.exists() else P.splits / a.split).read_text().split()
    sids = random.Random(a.seed).sample(sids, min(a.n, len(sids)))
    rows = [one(s, P) for s in sids]
    budget = [r["tok_tagged"] + r["tok_record"] + r["tok_out"] for r in rows]
    su = [x for r in rows for x in r["span_units"]]
    cv = [x for r in rows for x in r["cover"]]
    summary = {
        "papers": len(rows),
        "units_per_paper": {p: q([r["units"] for r in rows], p) for p in (0.5, 0.9, 0.99)},
        "marker_overhead_tokens": {p: q([r["tok_tagged"] - r["tok_text"] for r in rows], p) for p in (0.5, 0.9, 0.99)},
        "tok_tagged_text": {p: q([r["tok_tagged"] for r in rows], p) for p in (0.5, 0.9, 0.99)},
        "tok_record_in": {p: q([r["tok_record"] for r in rows], p) for p in (0.5, 0.9, 0.99)},
        "tok_output": {p: q([r["tok_out"] for r in rows], p) for p in (0.5, 0.9, 0.99)},
        "tok_total": {p: q(budget, p) for p in (0.5, 0.9, 0.99, 1.0)},
        "over_49152": sum(b > 49152 for b in budget),
        "slots_per_paper": {p: q([r["slots"] for r in rows], p) for p in (0.5, 0.9)},
        "slot_paths": sum(r["paths"] for r in rows), "unresolved_paths": sum(r["bad_paths"] for r in rows),
        "spans": len(su), "spans_outside_units": sum(r["spans_outside"] for r in rows),
        "span_in_one_unit": sum(x == 1 for x in su) / max(1, len(su)),
        "span_units": {p: q(su, p) for p in (0.5, 0.9, 0.99)},
        "span_cover_of_units": {p: q(cv, p) for p in (0.1, 0.25, 0.5)},
        "units_per_slot": {p: q([x for r in rows for x in r["units_per_slot"]], p) for p in (0.5, 0.9, 0.99)},
        "slots_with_several_sets": sum(x > 1 for r in rows for x in r["sets_per_slot"]) / max(1, sum(r["slots"] for r in rows)),
    }
    json.dump({"summary": summary, "papers": [{k: v for k, v in r.items() if k not in ("span_units", "cover")} for r in rows]},
              open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
