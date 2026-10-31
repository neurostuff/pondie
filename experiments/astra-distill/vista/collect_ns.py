"""Collect the ns extraction's predictions into one set of records, and check them.

    python collect_ns.py OUT --runs RUNS --shards DIR [DIR ...] [--drop TSV] [--data DIR] [--workers 12]

RUNS holds ns-27b-e-sNNN/pred.jsonl; each shards DIR holds sNNN.bodies.jsonl (`shard_ns.py`). A paper run in more
than one shard keeps the prediction of the latest shard (a later set re-runs papers whose parse or text changed).
Papers whose base_study_id is in a --drop list (a TSV with that column) and that no later shard re-ran are left
out: their parse has no analyses any more.

Each kept prediction goes through `compact.from_compact` and is validated against the extraction schema
(OUT/records/<sid>.extraction.json, written whether valid or not). Its parse keys (coordinate sets' local_id,
analyses' source_table_analysis) are checked against the paper's current `parse/coordinate_parse.json`. OUT/corpus/
<sid>/{parse, processed/pubget/text.txt} link to the paper's parse and the text the prompt was built from, which is
the layout `consistency.py` reads. Writes OUT/manifest.tsv (one row per paper) and OUT/summary.json.
"""
import argparse
import collections
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compact as K  # noqa: E402
import tasks as T  # noqa: E402

HEAD = re.compile(r'^\{"sid": "([^"]+)", "article_id": "([^"]+)", "text_source": "([^"]+)"')
COLS = ("sid", "article_id", "shard", "text_source", "status", "n_analyses", "n_sets", "keys_not_in_parse", "note")


def heads(shard_dirs):
    """{shard: {sid: (article_id, text_source)}} from the bodies' first fields"""
    out = {}
    for d in shard_dirs:
        for f in sorted(Path(d).glob("s[0-9][0-9][0-9].bodies.jsonl")):
            rows = {}
            for line in open(f):
                m = HEAD.match(line)
                if not m:
                    raise ValueError(f"{f}: a body without sid, article_id, text_source first")
                rows[m[1]] = (m[2], m[3])
            out[f.name.split(".")[0]] = rows
    return out


def check(job):
    """one paper: restore, validate, compare parse keys, write the record and the corpus links -> manifest row"""
    sid, article_id, shard, source, pred, out, data = job
    from jsonschema import Draft202012Validator
    ix = T.index(str(T.Paths().schema))
    row = dict(sid=sid, article_id=article_id, shard=shard, text_source=source, status="", n_analyses="",
               n_sets="", keys_not_in_parse="", note="")
    corpus = Path(out) / "corpus" / sid
    (corpus / "processed" / "pubget").mkdir(parents=True, exist_ok=True)
    for link, target in ((corpus / "parse", Path(data) / sid / "parse"),
                         (corpus / "processed" / "pubget" / "text.txt",
                          Path(data) / sid / "processed" / source / "text.txt")):
        if not link.is_symlink():
            link.symlink_to(target)
    if pred["prediction"] is None:
        row.update(status="no_prediction", note=(pred.get("error") or pred.get("finish_reason") or "")[:120])
        return row
    try:
        rec = K.from_compact(pred["prediction"], ix, sid)
    except Exception as e:  # noqa: BLE001
        row.update(status="restore_failed", note=f"{type(e).__name__}: {e}"[:120])
        return row
    errs = list(Draft202012Validator(ix.schema).iter_errors(rec))
    row.update(status="invalid" if errs else "valid",
               note="; ".join(f"{'/'.join(map(str, e.absolute_path))[:60]}: {e.message[:80]}" for e in errs[:2]))
    (Path(out) / "records" / f"{sid}.extraction.json").write_text(json.dumps(rec))
    row.update(n_analyses=len(rec.get("analyses") or []), n_sets=len(rec.get("coordinate_sets") or []))
    parse = Path(data) / sid / "parse" / "coordinate_parse.json"
    if not parse.is_file():
        row.update(keys_not_in_parse="parse_gone")
        return row
    keys = {a["key"] for a in json.load(open(parse))["analyses"]}
    used = {k for k in [s.get("local_id") for s in rec.get("coordinate_sets") or [] if isinstance(s, dict)] +
            [a.get("source_table_analysis") for a in rec.get("analyses") or [] if isinstance(a, dict)]
            if isinstance(k, str)}   # other shapes are schema errors, reported above
    row.update(keys_not_in_parse=len(used - keys))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--data", default=str(T.DEFAULTS["text"]))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    meta = heads(a.shards)
    tot = collections.Counter()
    chosen = {}   # sid -> (shard, pred)
    for shard in sorted(meta):
        preds = {}
        for line in open(Path(a.runs) / f"ns-27b-e-{shard}" / "pred.jsonl"):
            p = json.loads(line)
            preds[p["sid"]] = p
        missing, extra = set(meta[shard]) - set(preds), set(preds) - set(meta[shard])
        if missing or extra:
            raise ValueError(f"{shard}: {len(missing)} bodies without a prediction, {len(extra)} predictions without a body")
        for sid, p in preds.items():
            tot["replaced"] += sid in chosen
            chosen[sid] = (shard, p)
    drop = set()
    for f in a.drop:
        lines = open(f).read().splitlines()
        col = lines[0].split("\t").index("base_study_id")
        drop |= {line.split("\t")[col] for line in lines[1:]}
    for sid in sorted(drop):
        if sid in chosen and chosen[sid][0] < "s085":
            del chosen[sid]
            tot["dropped_stale"] += 1
    jobs = [(sid, meta[shard][sid][0], shard, meta[shard][sid][1], p, str(out), a.data)
            for sid, (shard, p) in sorted(chosen.items())]
    with ProcessPoolExecutor(a.workers) as ex, open(out / "manifest.tsv", "w") as f:
        f.write("\t".join(COLS) + "\n")
        for row in ex.map(check, jobs, chunksize=64):
            f.write("\t".join(str(row[c]) for c in COLS) + "\n")
            tot[row["status"]] += 1
            k = row["keys_not_in_parse"]
            tot["parse_gone" if k == "parse_gone" else "records_with_keys_not_in_parse"] += bool(k)
            tot["analyses"] += row["n_analyses"] or 0
    tot["papers"] = len(jobs)
    (out / "summary.json").write_text(json.dumps(dict(tot), indent=1))
    print(json.dumps(dict(tot)))


if __name__ == "__main__":
    main()
