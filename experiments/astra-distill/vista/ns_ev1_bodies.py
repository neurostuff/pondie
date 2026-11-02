"""Evidence request bodies (ev1, `evidence.py`) for model-extracted ns records, in eval_record shards.

    python ns_ev1_bodies.py SET_DIR OUT_DIR [--max-input 52000] [--per-shard 2000] [--workers 16] [--limit N] [--only SIDS]

SET_DIR is a `collect_ns.py` set (`manifest.tsv`, `records/<sid>.extraction.json`, `corpus/<sid>/processed/pubget/
text.txt`, the text the record was extracted from). Each record with a prediction gets the ev1 prompt that
`evidence.row` builds for a gold record: the text cut into numbered units, then `to_compact(record)`. Records invalid
against the schema are included (the manifest status goes along). Bodies over --max-input tokens are left out
(OUT_DIR/over-max.txt); the rest go, longest first, to the shard with the least input that has room, as OUT_DIR/
sNNN.bodies.jsonl. Papers that fail are listed in OUT_DIR/errors.jsonl. --only takes a file whose lines start
with the sids to build (an over-max.txt works).
"""
import argparse
import heapq
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compact as K  # noqa: E402
import evidence as E  # noqa: E402
import tasks as T  # noqa: E402


def body(args):
    sid, meta, d = args
    try:
        rec = json.load(open(d / "records" / f"{sid}.extraction.json"))
        text = (d / "corpus" / sid / "processed/pubget/text.txt").read_text()
        U = E.units(text)
        comp = K.to_compact(rec)
        prompt = (f"{E.INSTR}\n\nText:\n{E.tagged(text, U)}\n\nRecord:\n"
                  f"{json.dumps(comp, ensure_ascii=False, separators=(',', ':'))}")
        n_in = T.ntok(prompt) + T.ntok(E.TEMPLATE) + 40
    except Exception as e:  # noqa: BLE001
        return None, {"sid": sid, "error": f"{type(e).__name__}: {e}"[:300]}
    return {"sid": sid, "article_id": meta["article_id"], "status": meta["status"], "n_units": len(U),
            "n_input_tokens": n_in, "prompt": prompt, "template": E.TEMPLATE, "schema": {"type": "object"}}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir"); ap.add_argument("out")
    ap.add_argument("--max-input", type=int, default=52000)
    ap.add_argument("--per-shard", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only")
    a = ap.parse_args()
    d, out = Path(a.set_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = open(d / "manifest.tsv").read().splitlines()
    head = lines[0].split("\t")
    meta = [dict(zip(head, l.split("\t"))) for l in lines[1:]]
    meta = [m for m in meta if m["status"] != "no_prediction"]
    if a.only:
        keep = {l.split("\t")[0] for l in open(a.only).read().splitlines() if l}
        meta = [m for m in meta if m["sid"] in keep]
    meta = meta[: a.limit] if a.limit else meta
    rows, over = [], []
    with ProcessPoolExecutor(a.workers) as ex, open(out / "errors.jsonl", "w") as fe:
        for r, err in ex.map(body, ((m["sid"], m, d) for m in meta), chunksize=16):
            if err:
                fe.write(json.dumps(err) + "\n")
            elif r["n_input_tokens"] > a.max_input:
                over.append(f"{r['sid']}\t{r['n_input_tokens']}")
            else:
                rows.append(r)
    (out / "over-max.txt").write_text("".join(x + "\n" for x in over))
    rows.sort(key=lambda r: -r["n_input_tokens"])
    n_sh = -(-len(rows) // a.per_shard)
    heap = [(0, i, 0) for i in range(n_sh)]            # (input tokens so far, shard, papers)
    files = [open(out / f"s{i:03d}.bodies.jsonl", "w") for i in range(n_sh)]
    for r in rows:
        full = []
        while True:
            tok, i, n = heapq.heappop(heap)
            if n < a.per_shard:
                break
            full.append((tok, i, n))
        files[i].write(json.dumps(r, ensure_ascii=False) + "\n")
        heapq.heappush(heap, (tok + r["n_input_tokens"], i, n + 1))
        for x in full:
            heapq.heappush(heap, x)
    for f in files:
        f.close()
    print(json.dumps({"bodies": len(rows), "shards": n_sh, "over_max": len(over), "errors": sum(1 for _ in open(out / "errors.jsonl")),
                      "input_tokens": sum(r["n_input_tokens"] for r in rows)}))


if __name__ == "__main__":
    main()
