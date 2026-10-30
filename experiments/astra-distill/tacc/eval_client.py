"""Client half of Vista's eval_record.py for a vLLM server that is already up (beast): same request body for the
NuExtract3 model (chat template kwargs, JSON mode or guided schema, max_tokens room), same output rows, resumable.
    python eval_client.py BODIES OUT --port 8395 --workers 8 [--guided] [--max-len 49152] [--max-tokens 20000]"""
import argparse, collections, json, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("bodies"); ap.add_argument("out")
ap.add_argument("--port", type=int, default=8395); ap.add_argument("--name", default="ev1")
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--guided", action="store_true")
ap.add_argument("--max-len", type=int, default=49152); ap.add_argument("--max-tokens", type=int, default=20000)
a = ap.parse_args()
url = f"http://127.0.0.1:{a.port}/v1/chat/completions"

rows = [json.loads(l) for l in open(a.bodies)]
rows.sort(key=lambda r: -r["n_input_tokens"])      # longest first: the tail is short papers
outp = Path(a.out); have = set()
if outp.exists():   # resume: keep finished rows, drop failed ones
    keep = [l for l in open(outp) if l.strip() and json.loads(l).get("finish_reason")]
    have = {json.loads(l)["sid"] for l in keep}; outp.write_text("".join(keep))
rows = [r for r in rows if r["sid"] not in have]; print(f"resume: have {len(have)}, todo {len(rows)}", flush=True)
lock = threading.Lock(); fout = open(outp, "a"); done = [0]; agg = dict(tok=0); T1 = time.time()


def body(r):
    room = a.max_len - r["n_input_tokens"] - 64
    return {"model": a.name, "temperature": 0.0, "max_tokens": max(256, min(a.max_tokens, room)),
            "messages": [{"role": "user", "content": r["prompt"]}],
            "chat_template_kwargs": {"enable_thinking": False, "template": r["template"]},
            **({"structured_outputs": {"json": r["schema"]}} if a.guided else {"response_format": {"type": "json_object"}})}


def one(r):
    t = time.time(); res = dict(sid=r["sid"], prediction=None, finish_reason=None, tokens=None, error=None, prompt_tokens=None)
    try:
        q = urllib.request.Request(url, json.dumps(body(r)).encode(), {"Content-Type": "application/json"})
        j = json.load(urllib.request.urlopen(q, timeout=7200)); ch = j["choices"][0]
        res.update(finish_reason=ch.get("finish_reason"), tokens=j["usage"]["completion_tokens"], prompt_tokens=j["usage"]["prompt_tokens"])
        txt = ch["message"].get("content")
        try:
            res["prediction"] = json.loads(txt)
        except Exception as e:
            res["error"] = "invalid_json: " + repr(e)[:150]; res["raw_tail"] = (txt or "")[-300:]
    except urllib.error.HTTPError as e:
        res["error"] = f"http {e.code}: " + e.read().decode()[:400]
    except Exception as e:
        res["error"] = "exc: " + repr(e)[:300]
    res["wall_s"] = round(time.time() - t, 1)
    with lock:
        fout.write(json.dumps(res) + "\n"); fout.flush(); done[0] += 1; agg["tok"] += res["tokens"] or 0
        if done[0] <= 5 or done[0] % 20 == 0:
            el = time.time() - T1
            print(f"[{el:.0f}s] done {done[0]}/{len(rows)} out_tok {agg['tok']} ({agg['tok']/el:.0f}/s) last {res['finish_reason']} tok={res['tokens']} err={res['error']}", flush=True)
    return res


with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(one, rows))
rs = [json.loads(l) for l in open(outp)]
print("EVALSUMMARY", json.dumps(dict(n=len(rs), elapsed_s=round(time.time() - T1), out_tokens=agg["tok"],
      out_tok_s=round(agg["tok"] / max(1, time.time() - T1), 1), finish_reasons=dict(collections.Counter(r["finish_reason"] for r in rs)),
      length_trunc=sum(r["finish_reason"] == "length" for r in rs),
      invalid_json=sum(bool(r["error"] and r["error"].startswith("invalid_json")) for r in rs),
      http_errors=sum(bool(r["error"] and r["error"].startswith(("http", "exc"))) for r in rs),
      no_prediction=sum(r["prediction"] is None for r in rs))), flush=True)
