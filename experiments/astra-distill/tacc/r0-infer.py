"""R0 driver: one guided chat request per val paper against a local vLLM. stdlib only.
  python r0-infer.py BODIES.jsonl OUT.jsonl URL MODEL KIND(nu|qwen) WORKERS [LIMIT]
OUT rows: {sid, prediction, finish_reason, tokens, error, wall_s, prompt_tokens}"""
import json, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
bodies, out, url, model, kind, workers = sys.argv[1:7]; workers = int(workers)
limit = int(sys.argv[7]) if len(sys.argv) > 7 else 0
rows = [json.loads(l) for l in open(bodies)]
rows = rows[:limit] if limit else rows
import os
have = set()
if os.path.exists(out):  # resume: keep finished rows, drop failed ones
    def _ok(l):  # a line half-written when the time limit killed the last job is dropped and redone
        try:
            return bool(json.loads(l).get("finish_reason"))
        except ValueError:
            return False
    keep = [l for l in open(out) if l.strip() and _ok(l)]
    have = {json.loads(l)["sid"] for l in keep}; open(out, "w").write("".join(keep))
rows = [r for r in rows if r["sid"] not in have]; print("resume: skipping", len(have), "todo", len(rows), flush=True)
lock = threading.Lock(); fout = open(out, "a"); done = [0]; T0 = time.time()
agg = dict(tokens=0, prompt=0)
def body(r):
    content = r["prompt"]; kw = {"enable_thinking": False}
    if kind == "nu":
        kw["template"] = r["template"]; room = 49152 - r["n_input_tokens"] - 64
    else:
        content += "\n\nReturn the record as JSON that follows this template (values show the type or the allowed choices):\n" + r["template"]
        room = 49152 - r["n_input_tokens"] - 3000
    return {"model": model, "temperature": 0.0, "max_tokens": max(256, min(20000, room)),
            "messages": [{"role": "user", "content": content}], "chat_template_kwargs": kw,
            "structured_outputs": {"json": r["schema"]}}
def one(r):
    t = time.time(); res = dict(sid=r["sid"], prediction=None, finish_reason=None, tokens=None, error=None, prompt_tokens=None)
    try:
        q = urllib.request.Request(url, json.dumps(body(r)).encode(), {"Content-Type": "application/json"})
        j = json.load(urllib.request.urlopen(q, timeout=7200)); ch = j["choices"][0]
        res.update(finish_reason=ch.get("finish_reason"), tokens=j["usage"]["completion_tokens"], prompt_tokens=j["usage"]["prompt_tokens"])
        txt = ch["message"].get("content")
        try: res["prediction"] = json.loads(txt)
        except Exception as e: res["error"] = "invalid_json: " + repr(e)[:150]; res["raw_tail"] = (txt or "")[-300:]
    except urllib.error.HTTPError as e:
        res["error"] = f"http {e.code}: " + e.read().decode()[:400]
    except Exception as e:
        res["error"] = "exc: " + repr(e)[:300]
    res["wall_s"] = round(time.time() - t, 1)
    with lock:
        fout.write(json.dumps(res) + "\n"); fout.flush(); done[0] += 1
        agg["tokens"] += res["tokens"] or 0; agg["prompt"] += res["prompt_tokens"] or 0
        if done[0] <= 5 or done[0] % 20 == 0:
            el = time.time() - T0
            print(f"[{el:.0f}s] done {done[0]}/{len(rows)} out_tok {agg['tokens']} ({agg['tokens']/el:.0f}/s) prompt_tok {agg['prompt']} last: {res['finish_reason']} tok={res['tokens']} err={res['error']}", flush=True)
with ThreadPoolExecutor(workers) as ex: list(ex.map(one, rows))
el = time.time() - T0
rs = [json.loads(l) for l in open(out)]
print("R0SUMMARY", json.dumps(dict(n=len(rs), elapsed_s=round(el), out_tokens=agg["tokens"], prompt_tokens=agg["prompt"],
      out_tok_s=round(agg["tokens"]/el, 1), length_trunc=sum(r["finish_reason"] == "length" for r in rs),
      invalid_json=sum(bool(r["error"] and r["error"].startswith("invalid_json")) for r in rs),
      http_errors=sum(bool(r["error"] and r["error"].startswith(("http", "exc"))) for r in rs),
      no_prediction=sum(r["prediction"] is None for r in rs))))
