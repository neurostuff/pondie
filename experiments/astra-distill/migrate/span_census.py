"""How many evidence spans each alignment pass places, on a set of papers.

    python span_census.py PAPERS_DIR RECORDS_DIR OUT_JSONL
PAPERS_DIR/<sid>/processed/pubget/text.txt is the new text; RECORDS_DIR/<sid>.extraction.json the old record.
OUT_JSONL gets one line per unresolved span (sid, field path, old offsets, text).
"""
import collections
import json
import pathlib
import sys

from align import Aligner, canonical, char_similarity


def spans(x, path=""):
    if isinstance(x, dict):
        ev = x.get("evidence")
        if isinstance(ev, dict):
            for si, s in enumerate(ev.get("sets") or []):
                for pi, sp in enumerate(s.get("spans") or []):
                    yield f"{path}/evidence/sets/{si}/spans/{pi}", sp
        for k, v in x.items():
            if k != "evidence":
                yield from spans(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from spans(v, f"{path}/{i}")


def main(papers, records, out):
    C = collections.Counter()
    Q = collections.defaultdict(list)
    bad = []
    per = []
    with open(out, "w") as fo:
        for p in sorted(pathlib.Path(papers).iterdir()):
            rec = pathlib.Path(records) / f"{p.name}.extraction.json"
            if not rec.exists():
                continue
            d = json.load(open(rec))
            secs = d.get("extraction_metadata", {}).get("paper_sections") or []
            old_len = max([s.get("end_char", 0) for s in secs] or [1])
            al = Aligner((p / "processed/pubget/text.txt").read_text(), old_len)
            cache, n, ok = {}, 0, 0
            for path, sp in spans(d):
                t = sp.get("text") or ""
                key = (t, sp.get("start_char"))
                if key not in cache:
                    cache[key] = al.place(t, sp.get("start_char"))
                r = cache[key]
                n += 1
                if r is None:
                    C["unresolved"] += 1
                    fo.write(json.dumps({"sid": p.name, "path": path, "start_char": sp.get("start_char"),
                                         "end_char": sp.get("end_char"), "text": t}) + "\n")
                else:
                    ok += 1
                    C[r[2]] += 1
                    if r[2] in ("fuzzy", "gapped"):
                        cov = char_similarity(t, al.new[r[0]:r[1]])
                        Q[r[2]].append(cov)
                        if cov < 0.9 and len(bad) < 15:
                            bad.append((p.name, r[2], round(cov, 2), t[:90], al.new[r[0]:r[1]][:90]))
            per.append(ok / max(1, n))
    tot = sum(C.values())
    print({k: f"{v} ({v / tot:.4f})" for k, v in C.most_common()})
    for m, q in Q.items():
        q.sort()
        print(m, "char similarity of placed region: min %.2f p05 %.2f p50 %.2f; <0.9: %d of %d" % (
            q[0], q[len(q) // 20], q[len(q) // 2], sum(x < 0.9 for x in q), len(q)))
    for b in bad:
        print("  LOW", b)
    per.sort()
    print(f"papers {len(per)}; fully resolved {sum(x == 1 for x in per)}; per-paper min {per[0]:.3f} "
          f"p10 {per[len(per) // 10]:.3f} median {per[len(per) // 2]:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
