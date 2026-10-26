"""Inference driver: one guided call per paper.

The prompt is tasks.prompt_of(text, parse) inside NuExtract3's single-turn format (tasks.user_turn); the constraint
is tasks.record_shape(ix, "schema", parse_keys): one JSON schema for the whole record, in which
`coordinate_sets[].local_id` and `analyses[].source_table_analysis` are enums of the listing's keys. Ids the output
creates are free strings; whether they resolve is read after decoding (score.py `reference_resolution`). One request
per paper (no turns, so nothing carries between requests); papers run in parallel. Use the raw completions endpoint
with the rendered prompt, or the chat endpoint with `chat_template_kwargs={"template": ..., "enable_thinking": False}`.
Allow max_tokens above the longest gold target (see tasks.py stats). Not exercised here: no server was available.

    python infer.py run --split val.txt --out preds.json --url http://localhost:PORT/v1/chat/completions --model NAME
    python infer.py prompt SID                                   # print the request body for one paper
"""
import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tasks as T  # noqa: E402


def request_body(sid, P, model="nuextract3", max_tokens=24000):
    ix = T.index(str(P.schema))
    parse = T.load_parse(sid, P)
    return {"model": model, "temperature": 0.0, "max_completion_tokens": max_tokens,
            "messages": [{"role": "user", "content": T.prompt_of(T.load_text(sid, P), parse)}],
            "chat_template_kwargs": {"template": T.template_str(T.record_shape(ix, "template")), "enable_thinking": False},
            "structured_outputs": {"json": T.record_shape(ix, "schema", list(parse))}}


def ask(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    j = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    ch = j["choices"][0]
    try:
        return json.loads(ch["message"]["content"]), ch.get("finish_reason")
    except (TypeError, ValueError):
        return None, ch.get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "prompt"))
    ap.add_argument("sid", nargs="?")
    ap.add_argument("--split", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--url", default="")
    ap.add_argument("--model", default="nuextract3")
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--workers", type=int, default=16)
    for k, v in T.DEFAULTS.items():
        ap.add_argument(f"--{k}", default=str(v))
    a = ap.parse_args()
    P = T.Paths(**{k: getattr(a, k) for k in T.DEFAULTS})
    if a.cmd == "prompt":
        b = request_body(a.sid, P, a.model, a.max_tokens)
        print(json.dumps({k: (v if k != "structured_outputs" else "<schema>") for k, v in b.items()})[:3000])
        return
    sids = []
    for sp in a.split:
        f = Path(sp)
        sids += (f if f.exists() else P.splits / sp).read_text().split()
    sids = sids[: a.limit] if a.limit else sids
    fin = {}

    def one(sid):
        rec, why = ask(a.url, request_body(sid, P, a.model, a.max_tokens))
        fin[sid] = why
        return sid, rec
    with ThreadPoolExecutor(a.workers) as ex:
        res = dict(ex.map(one, sids))
    Path(a.out).write_text(json.dumps({"record": res, "finish": fin}))


if __name__ == "__main__":
    main()
