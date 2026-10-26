"""Per-paper request bodies for a split: {"sid", "n_input_tokens", "prompt", "template", "schema"}, the schema being the
guided schema of `tasks.record_shape` with the paper's parse keys enumerated.

    python make_bodies.py --split val.txt --out vista-out/v5/val.bodies.jsonl [--check-against val.bodies.v1.jsonl]
--check-against: the old file; prompt, template and n_input_tokens must be identical (only the schema may differ).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", action="append", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--check-against", default="")
a = ap.parse_args()
P = T.Paths()
ix = T.index(str(P.schema))
old = {}
if a.check_against:
    old = {d["sid"]: d for d in map(json.loads, open(a.check_against))}
tmpl = T.template_str(T.record_shape(ix, "template"))
n = bad = 0
with open(a.out, "w") as f:
    for sp in a.split:
        for sid in (P.splits / sp).read_text().split():
            parse = T.load_parse(sid, P)
            prompt = T.prompt_of(T.load_text(sid, P), parse)
            n_in = T.ntok(T.user_turn(prompt, tmpl))
            d = {"sid": sid, "n_input_tokens": n_in, "prompt": prompt, "template": tmpl,
                 "schema": T.record_shape(ix, "schema", list(parse))}
            if old:
                o = old[sid]
                bad += (o["prompt"], o["template"], o["n_input_tokens"]) != (prompt, tmpl, n_in)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
print(json.dumps({"bodies": n, "differ_from_old_in_prompt_template_or_tokens": bad}))
