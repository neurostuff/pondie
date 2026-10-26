"""Manual decisions for what migrate.py could not place: review packets, an offset helper, a checker.

    python manual.py packets RUN OUT --old DIR --new DIR [--per 20]
        RUN/queue.jsonl -> OUT/batches/<kind>-NNN.json, one item per queued span (distinct (sid, text) only) or
        coordinate set, with the context a reviewer needs.
    python manual.py locate --new DIR SID "TEXT IN THE NEW FILE" [--near CHAR]
        offsets of TEXT in the new text (every occurrence, nearest first). Reviewers copy text from the new file and
        take offsets from here, never count characters by hand.
    python manual.py check OUT --new DIR
        every decision in OUT/decisions/*.jsonl: offsets in range, the span text, its similarity to the old text;
        coordinate keys exist in the new parse. Writes OUT/manual.jsonl (the --manual input of migrate.py),
        expanding each span decision to every queued path with the same (sid, text).

Decision lines (JSONL, one per item, in OUT/decisions/<batch>.jsonl):
  {"batch": ..., "id": ..., "kind": "span", "decision": "place", "start_char": S, "end_char": E, "note": "..."}
  {"batch": ..., "id": ..., "kind": "span", "decision": "absent", "note": "what the new text has instead"}
  {"batch": ..., "id": ..., "kind": "coordinate_set", "decision": "map", "new_key": "analysis#...", "note": ...}
  {"batch": ..., "id": ..., "kind": "coordinate_set", "decision": "no_match", "note": "..."}
"""
import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from align import char_similarity  # noqa: E402
import coords as CO  # noqa: E402

CTX = 900
_LATEX = __import__("re").compile(r"\\documentclass.*?\\end\{document\}", __import__("re").S)


def _text(d, sid):
    return (pathlib.Path(d) / sid / "processed/pubget/text.txt").read_text()


def packets(run, out, old_dir, new_dir, per):
    run, out = pathlib.Path(run), pathlib.Path(out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "decisions").mkdir(exist_ok=True)
    q = [json.loads(l) for l in open(run / "queue.jsonl")]
    items = collections.defaultdict(list)
    seen = {}
    for x in q:
        if x["kind"] == "span":
            k = (x["sid"], x["text"])
            if k in seen:
                seen[k]["paths"].append(x["item"])
                continue
            old_t = _text(old_dir, x["sid"])
            new_t = _text(new_dir, x["sid"])
            o = x.get("old_start_char") or 0
            exp = int(o / max(1, len(old_t)) * len(new_t))
            c = x.get("candidate")
            centre = c["start_char"] if c else exp
            lo = max(0, centre - CTX)
            it = {"id": f"{x['sid']}|span|{len(items['span'])}", "kind": "span", "sid": x["sid"],
                  "paths": [x["item"]], "old_text": x["text"],
                  "old_context": old_t[max(0, o - 300): o + len(x["text"]) + 300],
                  "candidate": c, "new_window_start": lo, "new_window": new_t[lo: centre + len(x["text"]) + CTX]}
            seen[k] = it
            items["span"].append(it)
        elif x["kind"] == "coordinate_set":
            cp = json.load(open(pathlib.Path(new_dir) / x["sid"] / "parse/coordinate_parse.json"))["analyses"]
            st1p = pathlib.Path(old_dir) / x["sid"] / "stage1/analyses.json"
            st1 = json.load(open(st1p)).get("analyses") or [] if st1p.exists() else []
            old_e = dict(zip(CO.positional_keys(st1), st1)).get(CO.canon_key(x["item"]))
            cand_keys = {c["key"] for c in x["match"].get("candidates") or []}
            parse = [{"key": a["key"], "name": a.get("name"), "origin": a.get("origin"),
                      "table_id": a.get("table_id"), "role": a.get("role"),
                      "points": [p.get("coordinates") for p in a.get("points") or []][:40],
                      "candidate": a["key"] in cand_keys} for a in cp]
            items["coordinate_set"].append({
                "id": f"{x['sid']}|set|{x['item']}", "kind": "coordinate_set", "sid": x["sid"],
                "old_key": x["item"], "status": x["match"]["status"], "old_role": x.get("old_role"),
                "old_description": x.get("description"), "old_table_id": x.get("old_table_id"),
                "old_name": (old_e or {}).get("name"), "old_caption": (old_e or {}).get("table_caption"),
                "old_points": [p.get("coordinates") for p in (old_e or {}).get("points") or []],
                "collides_with": x["match"].get("with"), "new_parse": parse})
        else:
            items[x["kind"]].append({"id": f"{x['sid']}|{x['kind']}|{x['item']}", **x})
    for kind, its in items.items():
        for b in range(0, len(its), per):
            name = f"{kind}-{b // per:03d}"
            json.dump({"batch": name, "items": its[b:b + per]},
                      open(out / "batches" / f"{name}.json", "w"), ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in items.items()})


def locate(new_dir, sid, text, near):
    t = _text(new_dir, sid)
    hits, i = [], t.find(text)
    while i != -1:
        hits.append(i)
        i = t.find(text, i + 1)
    if near is not None:
        hits.sort(key=lambda h: abs(h - near))
    print(json.dumps([{"start_char": h, "end_char": h + len(text)} for h in hits[:10]]))
    if not hits:
        print("NOT FOUND: copy the text exactly from the new file", file=sys.stderr)
        sys.exit(1)


def check(out, new_dir):
    out = pathlib.Path(out)
    items = {}
    for b in sorted((out / "batches").glob("*.json")):
        for it in json.load(open(b))["items"]:
            items[it["id"]] = it
    dec = {}
    for f in sorted((out / "decisions").glob("*.jsonl")):
        for ln in open(f):
            if ln.strip():
                d = json.loads(ln)
                if d.get("decision") == "span":   # an early batch wrote the kind as the decision
                    d["decision"] = "place"
                dec[d["id"]] = d
    C, problems, manual = collections.Counter(), [], []
    for iid, it in items.items():
        d = dec.get(iid)
        if d is None:
            C["missing"] += 1
            continue
        C[f"{it['kind']}:{d['decision']}"] += 1
        if it["kind"] == "span" and d["decision"] == "place":
            t = _text(new_dir, it["sid"])
            rng = d.get("ranges") or [[d["start_char"], d["end_char"]]]
            if not all(0 <= s < e <= len(t) for s, e in rng):
                problems.append((iid, "offsets out of range"))
                continue
            s, e = rng[0]
            sim = char_similarity(it["old_text"], _LATEX.sub(" ", " ".join(t[a:b] for a, b in rng)))
            if sim < 0.5:
                problems.append((iid, f"low similarity {sim:.2f}: {t[s:e][:120]!r}"))
            for path in it["paths"]:
                manual.append({"sid": it["sid"], "kind": "span", "item": path, "decision": "place",
                               "start_char": s, "end_char": e, "ranges": rng, "similarity": round(sim, 3),
                               "reviewer": d.get("reviewer", "agent"), "note": d.get("note")})
        elif it["kind"] == "coordinate_set" and d["decision"] == "map":
            keys = {a["key"] for a in it["new_parse"]}
            if d["new_key"] not in keys:
                problems.append((iid, f"unknown key {d['new_key']}"))
                continue
            manual.append({"sid": it["sid"], "kind": "coordinate_set", "item": it["old_key"], "decision": "map",
                           "new_key": d["new_key"], "reviewer": d.get("reviewer", "agent"), "note": d.get("note")})
        elif it["kind"] == "span" and d["decision"] == "absent":
            for path in it["paths"]:
                manual.append({"sid": it["sid"], "kind": "span", "item": path, "decision": "absent",
                               "reviewer": d.get("reviewer", "agent"), "note": d.get("note")})
        elif it["kind"] == "coordinate_set" and d["decision"] == "no_match":
            manual.append({"sid": it["sid"], "kind": "coordinate_set", "item": it["old_key"], "decision": "no_match",
                           "reviewer": d.get("reviewer", "agent"), "note": d.get("note")})
        else:
            problems.append((iid, f"{d['decision']}: {d.get('note')}"))
    with open(out / "manual.jsonl", "w") as fo:
        for m in manual:
            fo.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(dict(C))
    for p in problems:
        print("  ", p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packets")
    p.add_argument("run")
    p.add_argument("out")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--per", type=int, default=20)
    p = sub.add_parser("locate")
    p.add_argument("--new", required=True)
    p.add_argument("sid")
    p.add_argument("text")
    p.add_argument("--near", type=int)
    p = sub.add_parser("check")
    p.add_argument("out")
    p.add_argument("--new", required=True)
    a = ap.parse_args()
    if a.cmd == "packets":
        packets(a.run, a.out, a.old, a.new, a.per)
    elif a.cmd == "locate":
        locate(a.new, a.sid, a.text, a.near)
    else:
        check(a.out, a.new)


if __name__ == "__main__":
    main()
