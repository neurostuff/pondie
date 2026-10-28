"""Merge (if an adapter), serve with vLLM, run every val paper with its guided schema (greedy), write predictions
incrementally with resume. Scoring happens on beast. Run with the serve env's python.
  python eval_record.py CONFIG.json
Config: run, model (nu|q27), adapter (dir | null: serve the base), n (0: all val), workers, max_tokens, max_len, gpu_mem,
        bodies (val.bodies.jsonl: sid, prompt, template, schema, n_input_tokens, built by vista/ on beast), out_name,
        sids (file of val sids: evaluate only those), tp (vLLM tensor-parallel size; GPUs from CUDA_VISIBLE_DEVICES),
        vllm_args (extra `vllm serve` flags, e.g. ["--disable-custom-all-reduce"] for tp over PCIe on LS6).
Envs: $ASTRA_ENVS (default <W>/envs-aarch64, Vista; LS6 sets <W>/envs).
Output: runs/<run>/val-pred.jsonl (or <out_name>) rows {sid, prediction, finish_reason, tokens, error, wall_s, prompt_tokens}."""
import json, os, subprocess, sys, time, threading, urllib.request, urllib.error, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

W = Path(os.environ["ASTRA_W"]); E = Path(os.environ.get("ASTRA_ENVS", W / "envs-aarch64"))
D = dict(run=None, model="nu", adapter=None, n=0, workers=24, max_tokens=20000, max_len=49152, gpu_mem=0.80, port=8391, tp=1,
         bodies=str(W / "data/v5/val.bodies.jsonl"), out_name="val-pred.jsonl", guided=True, keep_merged=False, merge_device="cpu")
c = dict(D); c.update(json.load(open(sys.argv[1]))); c.setdefault("run", Path(sys.argv[1]).stem)
run = W / "runs" / c["run"]; run.mkdir(parents=True, exist_ok=True)
vis = os.environ.get("CUDA_VISIBLE_DEVICES")
if vis is not None and len([g for g in vis.split(",") if g]) < c["tp"]:   # before the merge, not after it
    sys.exit(f"tp {c['tp']} needs {c['tp']} GPUs; CUDA_VISIBLE_DEVICES={vis!r}")
log = W / "logs"; tag = os.environ.get("SLURM_JOB_ID", "local")
hub = W / "hf/hub"
base = sorted((hub / f"models--{'numind--NuExtract3' if c['model'] == 'nu' else 'Qwen--Qwen3.8-27B'}" / "snapshots").glob("*"))[0]
name = "nuextract3" if c["model"] == "nu" else "q27"
T0 = time.time()
model_dir = base
if c["adapter"]:
    ad = Path(c["adapter"]); ad = ad if ad.is_absolute() else run.parent / ad
    if not (ad / "adapter_config.json").exists() and not (ad / "config.json").exists():
        sys.exit(f"adapter not saved yet: {ad}")
merged = None
if c["adapter"] and not (ad / "adapter_config.json").exists():
    model_dir = ad   # a full fine-tune: served as is
elif c["adapter"]:
    merged = Path(os.environ["ASTRA_SCRATCH"]) / "merged" / c["run"]
    if not (merged / ".ok").exists():
        print(f"MERGE {ad} -> {merged}", flush=True)
        shutil.rmtree(merged, ignore_errors=True)
        # GH200: the HBM is a NUMA node too, so page cache from writing the merged shards can fill the GPU (r3-ck50,
        # 2026-10-26: 39 of 95 GiB free at vLLM start). Prefer CPU memory (node 0) for the merge and its writes.
        numa = ["numactl", "--preferred=0"] if shutil.which("numactl") else []
        r = subprocess.run(numa + [str(E / "train/bin/python"), "-I", str(W / "code/merge_adapter.py"), c["model"], str(ad), str(merged), c["merge_device"]])
        if r.returncode:
            sys.exit(f"merge failed rc={r.returncode}")
    model_dir = merged
# drop the page cache the merge left on the weight files (on GH200 it can sit in the HBM node and shrink vLLM's share)
for f in Path(model_dir).iterdir():
    if f.is_file() and f.suffix == ".safetensors":
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; numactl -H | grep -E 'size|free'", shell=True)
print(f"[{time.time()-T0:.0f}s] serving {model_dir}", flush=True)
vl = log / f"vllm-{c['run']}-{tag}.log"
srv = subprocess.Popen([str(E / "serve/bin/vllm"), "serve", str(model_dir), "--served-model-name", name, "--host", "127.0.0.1",
                        "--port", str(c["port"]), "--max-model-len", str(c["max_len"]), "--gpu-memory-utilization", str(c["gpu_mem"]),
                        "--max-num-seqs", str(c["workers"]), "--language-model-only", "--tensor-parallel-size", str(c["tp"]),
                        *c.get("vllm_args", [])],
                       stdout=open(vl, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                       env=dict(os.environ, PATH=f"{E}/serve/bin:" + os.environ["PATH"]))
url = f"http://127.0.0.1:{c['port']}/v1/chat/completions"


def stop():
    if srv.poll() is None:
        srv.terminate()
        try:
            srv.wait(60)
        except Exception:
            srv.kill()


try:
    up = False
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{c['port']}/v1/models", timeout=3); up = True; break
        except Exception:
            if srv.poll() is not None:
                break
            time.sleep(10)
    if not up:
        print(open(vl).read()[-3000:]); sys.exit("vllm did not start")
    print(f"[{time.time()-T0:.0f}s] vllm up", flush=True)
    for l in open(vl):
        if "KV cache" in l or "Maximum concurrency" in l:
            print("  ", l.strip()[-140:])
    rows = [json.loads(l) for l in open(c["bodies"])]
    rows.sort(key=lambda r: -r["n_input_tokens"])      # longest first: the tail is short papers
    if c.get("sids"):   # a fixed subset (one sid per line), same papers for every checkpoint
        keep = {l.strip() for l in open(c["sids"]) if l.strip()}; rows = [r for r in rows if r["sid"] in keep]
    if c["n"]:
        rows = rows[-c["n"]:] if c.get("shortest") else rows[:c["n"]]
    outp = run / c["out_name"]; have = set()
    if outp.exists():   # resume: keep finished rows, drop failed ones
        keep = [l for l in open(outp) if l.strip() and json.loads(l).get("finish_reason")]
        have = {json.loads(l)["sid"] for l in keep}; outp.write_text("".join(keep))
    rows = [r for r in rows if r["sid"] not in have]; print(f"resume: have {len(have)}, todo {len(rows)}", flush=True)
    lock = threading.Lock(); fout = open(outp, "a"); done = [0]; agg = dict(tok=0); T1 = time.time()

    def body(r):
        content = r["prompt"]; kw = {"enable_thinking": False}
        if c["model"] == "nu":
            kw["template"] = r["template"]; room = c["max_len"] - r["n_input_tokens"] - 64
        else:   # same user turn as training (train_record.render)
            content += "\n\nReturn the record as JSON that follows this template (values show the type or the allowed choices):\n" + r["template"]
            room = c["max_len"] - r["n_input_tokens"] - 3000
        return {"model": name, "temperature": 0.0, "max_tokens": max(256, min(c["max_tokens"], room)),
                "messages": [{"role": "user", "content": content}], "chat_template_kwargs": kw,
                **({"structured_outputs": {"json": r["schema"]}} if c.get("guided", True)
                   else {"response_format": {"type": "json_object"}})}

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

    with ThreadPoolExecutor(c["workers"]) as ex:
        list(ex.map(one, rows))
    rs = [json.loads(l) for l in open(outp)]
    print("EVALSUMMARY", json.dumps(dict(run=c["run"], n=len(rs), elapsed_s=round(time.time() - T1), out_tokens=agg["tok"],
          out_tok_s=round(agg["tok"] / max(1, time.time() - T1), 1), finish_reasons=dict(__import__("collections").Counter(r["finish_reason"] for r in rs)), length_trunc=sum(r["finish_reason"] == "length" for r in rs),
          invalid_json=sum(bool(r["error"] and r["error"].startswith("invalid_json")) for r in rs),
          http_errors=sum(bool(r["error"] and r["error"].startswith(("http", "exc"))) for r in rs),
          no_prediction=sum(r["prediction"] is None for r in rs), wall_h=round((time.time() - T0) / 3600, 2))), flush=True)
    if merged and not c["keep_merged"]:
        shutil.rmtree(merged, ignore_errors=True)
finally:
    stop()
