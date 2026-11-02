"""Score evidence rerankers on the val benchmark (`vista/rerank_data.py bench`): each query's claim against every unit
of its paper; how high do Astra's units rank?

    python rerank_eval.py BENCH_DIR OUT_DIR MODEL [MODEL ...] [--max-length 512] [--batch 256]
    MODEL: bm25 | a Hub id or local path of a CrossEncoder (HF_HOME cache, offline) | name=path

Per model: OUT/<name>.scores.jsonl {qid, scores} and OUT/<name>.metrics.json. Metrics per query, averaged:
hit@k (some gold unit in the top k), recall@k (share of the gold units in the top k), mrr (1 / rank of the first gold
unit). "all": every unit is a candidate; "mr": units of the methods and results sections first, then the rest (a
section prior; papers without sections are ranked as in "all"). Also by collection (hit@10, recall@10).
"""
import argparse
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

KS = (1, 3, 5, 10, 20, 50)
TOK = re.compile(r"\w+")
# Qwen3-Reranker's prompt (its model card); the seq-cls conversions do not add it themselves
QWEN_Q = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct '
          'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
          '<Instruct>: Given a claim about a neuroimaging study, judge whether this passage of the paper states the '
          'claim\n<Query>: {}\n')
QWEN_D = "<Document>: {}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def formatted(path, claim, doc):
    if "qwen3-reranker" in path.lower():
        return QWEN_Q.format(claim), QWEN_D.format(doc)
    return claim, doc


def bm25_scores(docs, queries, k1=1.2, b=0.75):
    tf = [collections.Counter(TOK.findall(d.lower())) for d in docs]
    ln = [sum(t.values()) for t in tf]
    avg = sum(ln) / max(1, len(ln))
    df = collections.Counter(w for t in tf for w in t)
    idf = {w: math.log(1 + (len(docs) - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    out = []
    for q in queries:
        ws = [w for w in set(TOK.findall(q.lower())) if w in idf]
        out.append([sum(idf[w] * t[w] * (k1 + 1) / (t[w] + k1 * (1 - b + b * n / avg)) for w in ws if w in t)
                    for t, n in zip(tf, ln)])
    return out


def metrics(bench, units, scores):
    agg = {m: collections.defaultdict(list) for m in ("all", "mr")}
    by_col = collections.defaultdict(lambda: collections.defaultdict(list))
    for q in bench:
        s, u = scores[q["qid"]], units[q["sid"]]
        sec = u["sections"]
        for mode in ("all", "mr"):
            key = (lambda i: -s[i]) if mode == "all" or sec is None else \
                  (lambda i: (sec[i] not in ("methods", "results"), -s[i]))
            order = sorted(range(len(s)), key=key)
            rank = {i + 1: r + 1 for r, i in enumerate(order)}
            gr = sorted(rank[g] for g in q["gold"])
            a = agg[mode]
            a["mrr"].append(1 / gr[0])
            for k in KS:
                a[f"hit@{k}"].append(float(gr[0] <= k))
                a[f"recall@{k}"].append(sum(r <= k for r in gr) / len(gr))
            if mode == "all":
                c = by_col[q["collection"]]
                c["hit@10"].append(float(gr[0] <= 10)); c["recall@10"].append(sum(r <= 10 for r in gr) / len(gr))
    mean = lambda v: round(sum(v) / len(v), 4)  # noqa: E731
    return {"queries": len(bench), **{m: {k: mean(v) for k, v in a.items()} for m, a in agg.items()},
            "by_collection": {c: {"n": len(v["hit@10"]), **{k: mean(x) for k, x in v.items()}}
                              for c, v in sorted(by_col.items(), key=lambda kv: -len(kv[1]["hit@10"]))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench"); ap.add_argument("out"); ap.add_argument("models", nargs="+")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    bd, out = Path(a.bench), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = [json.loads(l) for l in open(bd / "bench.jsonl")]
    units = {u["sid"]: u for u in map(json.loads, open(bd / "units.jsonl"))}
    by_paper = collections.defaultdict(list)
    for q in bench:
        by_paper[q["sid"]].append(q)
    for spec in a.models:
        name, path = spec.split("=", 1) if "=" in spec else (spec.replace("/", "__"), spec)
        t0 = time.time()
        scores = {}
        if path == "bm25":
            for sid, qs in by_paper.items():
                for q, s in zip(qs, bm25_scores(units[sid]["docs"], [q["claim"] for q in qs])):
                    scores[q["qid"]] = s
        else:
            import torch
            from sentence_transformers import CrossEncoder
            qwen = "qwen3-reranker" in path.lower()
            model = CrossEncoder(path, max_length=a.max_length + (128 if qwen else 0), device="cuda",
                                 model_kwargs={"torch_dtype": torch.bfloat16}, trust_remote_code=True)
            if qwen:      # formatted() renders the prompt; a repo's chat template would re-wrap the pair (4B ships one)
                model[0].modality_config.pop("message", None)
                model.tokenizer.chat_template = None
            pairs = [(q["qid"], i, *formatted(path, q["claim"], d)) for q in bench for i, d in enumerate(units[q["sid"]]["docs"])]
            pairs.sort(key=lambda p: len(p[2]) + len(p[3]))          # similar lengths per batch: less padding
            for q in bench:
                scores[q["qid"]] = [0.0] * len(units[q["sid"]]["docs"])
            CH = 100_000
            for c in range(0, len(pairs), CH):
                chunk = pairs[c:c + CH]
                with torch.inference_mode():
                    ys = model.predict([(p[2], p[3]) for p in chunk], batch_size=a.batch, show_progress_bar=False,
                                       convert_to_numpy=True)
                for p, y in zip(chunk, ys):
                    scores[p[0]][p[1]] = float(y)
                print(f"  {name}: {c + len(chunk)}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
            del model
            torch.cuda.empty_cache()
        el = time.time() - t0
        with open(out / f"{name}.scores.jsonl", "w") as f:
            for qid in sorted(scores):
                f.write(json.dumps({"qid": qid, "scores": [round(x, 5) for x in scores[qid]]}) + "\n")
        m = {"model": path, "seconds": round(el), **metrics(bench, units, scores)}
        json.dump(m, open(out / f"{name}.metrics.json", "w"), indent=1)
        print("RESULT", name, json.dumps({"s": m["seconds"], "all": {k: m["all"][k] for k in ("hit@1", "hit@5", "hit@10", "recall@10", "recall@20", "mrr")},
                                          "mr": {k: m["mr"][k] for k in ("hit@10", "recall@10", "mrr")}}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
