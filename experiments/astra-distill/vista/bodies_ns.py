"""Request bodies for ns-pond articles with no gold record: the paper prompt of `tasks.prompt_of` from the article's own
text, coordinate parse and stage-1 captions, as `make_bodies.py` builds them for val.

    python bodies_ns.py ARTICLES.tsv --out bodies.jsonl [--limit N] [--workers 12]

ARTICLES.tsv: the ns-pond staging list (article_id, base_study_id, dir, texts, coordinate_parse, stage1, ...; header
row). The text is `<dir>/processed/<source>/text.txt` for the first source in SOURCES the article has: pubget is what
training saw. Rows: {"sid" (base_study_id), "article_id", "text_source", "n_input_tokens", "n_keys", "n_captioned",
"prompt", "template", "schema"}; the schema is a placeholder (JSON mode). Articles that fail are listed in
<out>.errors.jsonl.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1  # noqa: E402
import tasks as T  # noqa: E402

SOURCES = ("pubget", "pmc", "elsevier", "ace", "pdf")


def body(args):
    row, tmpl = args
    d = Path(row["dir"])
    try:
        src = next(s for s in SOURCES if (d / "processed" / s / "text.txt").exists())
    except StopIteration:
        return None, {"sid": row["base_study_id"], "error": "no text"}
    try:
        p = json.load(open(row["coordinate_parse"]))
        caps = stage1.captions(row["base_study_id"], d.parent)
        parse = {e["key"]: dict(e, **caps.get(e["key"], {})) for e in p["analyses"]}
        prompt = T.prompt_of((d / "processed" / src / "text.txt").read_text(), parse)
        n_in = T.ntok(T.user_turn(prompt, tmpl))
    except Exception as e:  # noqa: BLE001
        return None, {"sid": row["base_study_id"], "error": f"{type(e).__name__}: {e}"[:300]}
    return {"sid": row["base_study_id"], "article_id": row["article_id"], "text_source": src, "n_input_tokens": n_in,
            "n_keys": len(parse), "n_captioned": sum(bool(e.get("caption") or e.get("footer")) for e in parse.values()),
            "prompt": prompt, "template": tmpl, "schema": {"type": "object"}}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    lines = open(a.tsv).read().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[1:]]
    rows = rows[: a.limit] if a.limit else rows
    tmpl = T.template_str(T.record_shape(T.index(str(T.Paths().schema)), "template"))
    n = bad = 0
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "w") as f, open(a.out + ".errors.jsonl", "w") as fe:
        for d, err in ex.map(body, ((r, tmpl) for r in rows), chunksize=8):
            if err:
                fe.write(json.dumps(err) + "\n"); bad += 1
                continue
            f.write(json.dumps(d, ensure_ascii=False) + "\n"); n += 1
    print(json.dumps({"bodies": n, "errors": bad, "out": a.out}))


if __name__ == "__main__":
    main()
