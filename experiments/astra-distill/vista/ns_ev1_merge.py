"""Put ev1's evidence back into the ns records: each cited unit becomes a span of the value's evidence.

    python ns_ev1_merge.py SET_DIR PRED_DIR OUT_DIR [--workers 8] [--limit N]

SET_DIR is the `collect_ns.py` set the bodies were built from (`ns_ev1_bodies.py`); PRED_DIR holds eval_record's
`*/pred.jsonl` for it (a later file's usable prediction replaces an earlier unusable one, so a redo run can sit next
to the shards). Units are found with `evidence.ev_target`, the same walk that built the training targets, so a slot
path in a prediction names the same value it named in training. A cited value gets
`{"status": "present", "sets": [{"source": "retriever", "spans": [...]}]}`, one span per cited unit (its text and
character offsets in `corpus/<sid>/processed/pubget/text.txt`); the schema has no field naming the evidence model, so
OUT_DIR/README.txt does. Values ev1 cited nothing for keep `not_found`. Citations of a unit past the paper's end or of
a path the record does not have are dropped and counted. Each merged record is checked: `ev_target` on it gives back
exactly the kept citations, and it is validated against the extraction schema. Out: OUT_DIR/records/<sid>.extraction.json
(every record of the set, unchanged where ev1 has no prediction), OUT_DIR/manifest.tsv, OUT_DIR/summary.json.
A rerun resumes: records already in the manifest are kept.
"""
import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as E  # noqa: E402
import tasks as T  # noqa: E402

COLS = ["sid", "status", "evidence", "values", "cited_values", "spans", "dropped_out_of_range", "dropped_no_slot",
        "roundtrip", "schema_errors"]
README = """Records of {set_dir} with evidence from ev1-full (NuExtract3 / Qwen3.5-4B, LoRA fine-tuned on Astra's
sentence-level evidence; pondie experiments/astra-distill, VISTA-JOURNAL H40), predictions in {pred_dir}.
Evidence sets written by this merge carry source "retriever": the model cited numbered units of the paper text
(evidence.units) and each cited unit is one span. Values without a citation keep evidence status not_found.
"""


def load_preds(pred_dir):
    """{sid: its pred.jsonl line}, kept as text so the set's predictions fit in memory"""
    preds = {}
    for f in sorted(Path(pred_dir).glob("*/pred.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if isinstance(r.get("prediction"), dict) or r["sid"] not in preds:
                preds[r["sid"]] = line if isinstance(r.get("prediction"), dict) else None
    return preds


def merge(args):
    sid, line, set_dir, out_dir, status = args
    pred = json.loads(line)["prediction"] if line else None
    row = dict.fromkeys(COLS, 0)
    row.update(sid=sid, status=status, evidence="none", roundtrip="")
    rec = json.load(open(set_dir / "records" / f"{sid}.extraction.json"))
    if isinstance(pred, dict):
        text = (set_dir / "corpus" / sid / "processed/pubget/text.txt").read_text()
        U = E.units(text)
        slots = []                                   # slot n -> that value's evidence object (in rec)
        _, ev = E.ev_target(rec, lambda e: slots.append(e) or [len(slots)])
        where = {path: slots[n - 1] for path, n in E.pairs(ev)}
        cited = collections.defaultdict(set)
        for path, u in E.pairs(pred):
            if path not in where:
                row["dropped_no_slot"] += 1
            elif not 1 <= u <= len(U):
                row["dropped_out_of_range"] += 1
            else:
                cited[path].add(u)
        for path, us in cited.items():
            spans = [{"text": text[a:b], "start_char": a, "end_char": b} for a, b in (U[u - 1] for u in sorted(us))]
            e = where[path]
            e.clear()
            e.update(status="present", sets=[{"source": "retriever", "spans": spans}])
        row.update(evidence="ev1", values=len(where), cited_values=len(cited), spans=sum(map(len, cited.values())))
        _, back = E.ev_target(rec, E.Mapper(U))
        got = {(p, u) for p, u in E.pairs(back)}
        row["roundtrip"] = "ok" if got == {(p, u) for p, us in cited.items() for u in us} else "MISMATCH"
    from jsonschema import Draft202012Validator
    row["schema_errors"] = sum(1 for _ in Draft202012Validator(T.index(str(T.Paths().schema)).schema).iter_errors(rec))
    (out_dir / "records" / f"{sid}.extraction.json").write_text(json.dumps(rec))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir"); ap.add_argument("pred_dir"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    d, out = Path(a.set_dir), Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text(README.format(set_dir=d.resolve(), pred_dir=Path(a.pred_dir).resolve()))
    lines = open(d / "manifest.tsv").read().splitlines()
    head = lines[0].split("\t")
    meta = [dict(zip(head, l.split("\t"))) for l in lines[1:]]
    meta = [m for m in meta if (d / "records" / f"{m['sid']}.extraction.json").exists()]
    meta = meta[: a.limit] if a.limit else meta
    fm_path = out / "manifest.tsv"
    done = {}
    if fm_path.exists():
        for l in fm_path.read_text().splitlines()[1:]:
            if len(l.split("\t")) == len(COLS):     # a torn last line is merged again
                done[l.split("\t")[0]] = l
    with open(fm_path, "w") as fm:                  # rewritten without a torn last line
        fm.write("\t".join(COLS) + "\n" + "".join(l + "\n" for l in done.values()))
    todo = [m for m in meta if m["sid"] not in done]
    print(f"{len(meta)} records, {len(done)} already merged, {len(todo)} to merge", flush=True)
    preds = load_preds(a.pred_dir)
    with ProcessPoolExecutor(a.workers) as ex, open(fm_path, "a") as fm:
        for c in range(0, len(todo), 2000):          # a batch at a time: the queue holds 2,000 predictions, not all
            jobs = [(m["sid"], preds.get(m["sid"]), d, out, m["status"]) for m in todo[c:c + 2000]]
            for row in ex.map(merge, jobs, chunksize=16):
                fm.write("\t".join(str(row[k]) for k in COLS) + "\n")
            fm.flush()
            print(f"  {min(c + 2000, len(todo))}/{len(todo)}", flush=True)
    tot = collections.Counter()
    for l in fm_path.read_text().splitlines()[1:]:
        row = dict(zip(COLS, l.split("\t")))
        tot["records"] += 1; tot[f"evidence_{row['evidence']}"] += 1; tot[f"roundtrip_{row['roundtrip'] or 'na'}"] += 1
        tot[f"schema_{'invalid' if int(row['schema_errors']) else 'valid'}_was_{row['status']}"] += 1
        for k in ("values", "cited_values", "spans", "dropped_out_of_range", "dropped_no_slot"):
            tot[k] += int(row[k])
    (out / "summary.json").write_text(json.dumps(dict(tot), indent=1))
    print(json.dumps(dict(tot)))


if __name__ == "__main__":
    main()
