"""LoRA / QLoRA / full-parameter SFT on the whole-record rows (data v5, one call per paper). One sequence per step, gradient
checkpointing, loss on the target tokens only, token-sum normalised per optimizer step. Resumable.

  python train_record.py CONFIG.json

Everything tunable is in the config (read at start); see DEFAULTS below. Exit codes: 0 finished (or soft stop that
the next chained job will continue, "soft_stop_exit"), 75 soft stop before the epoch ended.
Outputs in <runs>/<run>/: adapter-25/50/75/final (peft; mode full: a bf16 model directory vLLM serves as is), resume.pt (latest optimizer state), train-log.jsonl, DONE.
"""
import json, math, os, random, subprocess, sys, time, shutil, re
from pathlib import Path

DEFAULTS = dict(
    run=None, model="nu",            # nu: numind/NuExtract3 (bf16 LoRA) | q27: Qwen3.8-27B (QLoRA nf4)
    mode=None,                       # bf16 | nf4 (LoRA; default by model) | full (all weights: fp32 master + bf16 autocast, 8-bit AdamW)
    data="/work/07723/jdk3232/vista/astra-distill/data/v5/train.jsonl",
    runs="/work/07723/jdk3232/vista/astra-distill/runs",
    n_papers=0,                      # 0: all; else first N of the fixed shuffle (nested subsets)
    sids=None,                       # file of sids (one per line): keep only those rows, before the shuffle
    subset_seed=20261023, seed=0, max_len=49152,
    accum=16, lr=None, warmup=0.03, epochs=1, weight_decay=0.01, clip=1.0,
    rank=64, alpha=128, dropout=0.05,
    ckpt_fracs=[0.25, 0.5, 0.75],
    max_steps=0,                     # smoke: stop after N optimizer steps (no cosine rescale)
    force_longest=False,             # smoke: swap the first selected row for the longest one under max_len
    soft_stop_margin_s=900,          # save + exit this long before the Slurm time limit
    autocast=True, loss_chunk=512,
    pad_to=256,                      # right-pad each sequence to a multiple of this (causal: no effect on earlier tokens; stabilises kernel shapes)
)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
IM_END = "<|im_end|>"
Q27_SUFFIX = "\n\nReturn the record as JSON that follows this template (values show the type or the allowed choices):\n"
GB = 2 ** 30


def snap(name):
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    return sorted((hub / f"models--{name}" / "snapshots").glob("*"))[0]


def load_cfg(path):
    c = dict(DEFAULTS); c.update(json.load(open(path)))
    c.setdefault("run", Path(path).stem)
    c["mode"] = c["mode"] or ("bf16" if c["model"] == "nu" else "nf4")
    c["lr"] = c["lr"] or (2e-4 if c["model"] == "nu" else 1e-4)
    return c


def deadline():
    """epoch seconds at which the Slurm job ends (None outside Slurm)"""
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        return None
    try:
        out = subprocess.run(["scontrol", "show", "job", "-o", jid], capture_output=True, text=True, timeout=30).stdout
        run = re.search(r"RunTime=(?:(\d+)-)?(\d+):(\d+):(\d+)", out); lim = re.search(r"TimeLimit=(?:(\d+)-)?(\d+):(\d+):(\d+)", out)
        sec = lambda m: ((int(m[1] or 0) * 24 + int(m[2])) * 60 + int(m[3])) * 60 + int(m[4])  # noqa: E731
        return time.time() + sec(lim) - sec(run)
    except Exception as e:
        print("deadline unknown:", e, flush=True)
        return None


def select_rows(c):
    """rows of the run in training order: first n_papers of a fixed shuffle (nested 1k < 3k < 10k < full), then the
    papers over max_len dropped. Returns (rows, n_selected, n_dropped)."""
    rows = []
    with open(c["data"]) as f:
        for l in f:
            r = json.loads(l)
            rows.append({k: r[k] for k in ("sid", "prompt", "template", "target_json", "n_tokens")})
    if c["sids"]:
        keep = {l.strip() for l in open(c["sids"]) if l.strip()}
        rows = [r for r in rows if r["sid"] in keep]
        print(f"DATA sids {c['sids']}: {len(rows)} of {len(keep)} sids found in {c['data']}", flush=True)
    idx = list(range(len(rows)))
    random.Random(c["subset_seed"]).shuffle(idx)
    if c["n_papers"]:
        idx = idx[: c["n_papers"]]
    if c["force_longest"]:
        ok = [i for i in range(len(rows)) if rows[i]["n_tokens"] <= c["max_len"]]
        idx = [max(ok, key=lambda i: rows[i]["n_tokens"])] + idx[1:]
    sel = [rows[i] for i in idx]
    keep = [r for r in sel if r["n_tokens"] <= c["max_len"]]
    return keep, len(sel), len(sel) - len(keep)


def render(tok, r, model):
    """-> (head_ids, target_ids): the prompt as the serving side renders it, then target_json + EOS"""
    if model == "nu":   # NuExtract3 single-turn wire format (= apply_chat_template(template=, enable_thinking=False))
        head = (f"<|im_start|>user\n【task】structured\n【template_start】{r['template']}【template_end】\n【document_start】\n"
                f"{r['prompt']}\n【document_end】{IM_END}\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    else:               # Qwen: chat template with the record template in the user turn, as the R0 / serving request does
        head = tok.apply_chat_template([{"role": "user", "content": r["prompt"] + Q27_SUFFIX + r["template"]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
    h = tok(head, add_special_tokens=False)["input_ids"]
    t = tok(r["target_json"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return h, t


def main():
    c = load_cfg(sys.argv[1])
    out = Path(c["runs"]) / c["run"]; out.mkdir(parents=True, exist_ok=True)
    (out / "train-config.json").write_text(json.dumps(c, indent=1))
    if (out / "DONE").exists():
        print("already DONE", flush=True); return 0
    dl = deadline()
    t_start = time.time()
    import torch
    from torch.utils.checkpoint import checkpoint
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    import transformers, peft
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    print(f"versions torch {torch.__version__} cuda {torch.version.cuda} transformers {transformers.__version__} peft {peft.__version__}", flush=True)
    random.seed(c["seed"]); torch.manual_seed(c["seed"])
    mpath = str(snap("numind--NuExtract3" if c["model"] == "nu" else "Qwen--Qwen3.8-27B"))
    tok = AutoTokenizer.from_pretrained(mpath)
    assert tok.eos_token == IM_END, tok.eos_token

    rows, n_sel, n_drop = select_rows(c)
    print(f"DATA selected {n_sel} papers (n_papers={c['n_papers']}), dropped {n_drop} over max_len={c['max_len']}, kept {len(rows)}; "
          f"total tokens (n_tokens) {sum(r['n_tokens'] for r in rows):,}", flush=True)
    A = c["accum"]; total_steps = math.ceil(len(rows) / A) * int(c["epochs"])
    order = [rows[i % len(rows)] for i in range(total_steps * A)] if c["epochs"] > 1 else rows
    if c["max_steps"]:
        total_steps = min(total_steps, c["max_steps"])
    ck_steps = {max(1, round(f * total_steps)): f for f in c["ckpt_fracs"]}
    warm = max(1, round(c["warmup"] * total_steps))
    lr_at = lambda s: c["lr"] * ((s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))  # noqa: E731
    print(f"PLAN optimizer steps {total_steps}, {A} papers each, warmup {warm}, lr {c['lr']} cosine, checkpoints at {sorted(ck_steps)}", flush=True)

    kw = dict(dtype=torch.bfloat16, device_map={"": 0})
    if c["mode"] == "full":   # fp32 master weights: at lr ~1e-5 a bf16 weight would round most updates away
        kw["dtype"] = torch.float32
    if c["mode"] == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(mpath, **kw)
    for holder in (model, getattr(model, "model", None)):
        for attr in ("visual", "vision_tower", "vision_model"):
            if holder is not None and getattr(holder, attr, None) is not None:
                setattr(holder, attr, None)
    print(f"LOAD {time.time()-t0:.0f}s mode {c['mode']} mem {torch.cuda.memory_allocated()/GB:.1f} GiB", flush=True)
    full = c["mode"] == "full"
    for p in model.parameters():
        p.requires_grad_(full)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads(); model.config.use_cache = False
    if not full:
        model = get_peft_model(model, LoraConfig(r=c["rank"], lora_alpha=c["alpha"], lora_dropout=c["dropout"], bias="none",
                                                 task_type="CAUSAL_LM", target_modules=TARGETS))
    model.train()   # trap 2: gradient checkpointing only acts in train mode
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    print(f"{'FULL' if full else 'LORA'} r={c['rank']} alpha={c['alpha']} dropout={c['dropout']}: {sum(p.numel() for p in params)/1e6:.1f}M params in "
          f"{len(params)} tensors, dtypes {sorted({str(p.dtype) for p in params})}", flush=True)
    inner = model if full else model.base_model.model
    core = inner.model.language_model if hasattr(inner.model, "language_model") else inner.model
    W = inner.lm_head.weight
    if full:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=c["lr"], weight_decay=c["weight_decay"])
    else:
        opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=c["weight_decay"])

    def save_weights(d):
        if not full:
            model.save_pretrained(d); return
        # a bf16 copy for vLLM; the base snapshot's other files linked as in merge_adapter.py
        model.save_pretrained(d, state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()},
                              safe_serialization=True, max_shard_size="10GB")
        for f in Path(mpath).iterdir():
            t = d / f.name
            if f.name == "config.json":
                shutil.copy(f.resolve(), t)
            elif not (t.exists() or f.name.startswith(("model.safetensors", "model-0")) or f.suffix in (".png", ".svg")):
                t.symlink_to(f.resolve())

    step = 0
    rp = out / "resume.pt"
    if rp.exists():
        st = torch.load(rp, map_location="cuda")
        with torch.no_grad():
            for n, p in named:
                p.copy_(st["params"][n])
        opt.load_state_dict(st["opt"]); step = st["step"]
        print(f"RESUME from step {step}/{total_steps} (paper position {step*A})", flush=True); del st

    def ce_chunk(h, lab, w):
        return torch.nn.functional.cross_entropy((h @ w.t()).float(), lab, ignore_index=-100, reduction="sum")

    def loss_sum(h, lab):   # token-sum; chunked + checkpointed so the 248k-vocab logits never all exist at once
        tot = 0.0
        for i in range(0, h.size(0), c["loss_chunk"]):
            tot = tot + checkpoint(ce_chunk, h[i:i + c["loss_chunk"]], lab[i:i + c["loss_chunk"]], W, use_reentrant=False)
        return tot

    def save_ckpt(tag=None):
        if tag:
            save_weights(out / f"adapter-{tag}")
        tmp = out / "resume.pt.tmp"
        torch.save({"params": {n: p.detach() for n, p in named}, "opt": opt.state_dict(), "step": step}, tmp)
        os.replace(tmp, rp)
        print(f"CKPT step {step} tag {tag}", flush=True)

    logf = open(out / "train-log.jsonl", "a")
    torch.cuda.reset_peak_memory_stats()
    t_run = time.time(); tok_run = 0; skipped = 0; stopped = False
    while step < total_steps:
        group = order[step * A:(step + 1) * A]
        seqs = []
        for r in group:
            h, t = render(tok, r, c["model"])
            if len(h) + len(t) > c["max_len"]:
                skipped += 1; print(f"  skip {r['sid']}: {len(h)+len(t)} tokens over max_len after rendering", flush=True); continue
            seqs.append((r["sid"], h, t))
        norm = max(1, sum(len(t) for _, _, t in seqs))   # token-sum over the whole optimizer step's targets
        for g in params:
            g.grad = None
        ts = time.time(); lsum = 0.0; ntok = 0
        per = []
        for sid, h, t in seqs:
            tq = time.time()
            seq = h + t
            if c["pad_to"]:
                seq = seq + [tok.pad_token_id] * (-len(seq) % c["pad_to"])
            ids = torch.tensor([seq], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=c["autocast"]):
                hid = core(input_ids=ids).last_hidden_state[0, len(h) - 1:len(h) + len(t) - 1]
                l = loss_sum(hid, ids[0, len(h):len(h) + len(t)])
            (l / norm).backward()
            lsum += l.item(); ntok += len(h) + len(t)
            torch.cuda.synchronize(); per.append((len(h) + len(t), round(time.time() - tq, 1)))
        gn = torch.nn.utils.clip_grad_norm_(params, c["clip"]).item()
        bad = not math.isfinite(lsum) or not math.isfinite(gn)
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        if bad:
            print(f"  step {step+1}: non-finite loss/grad ({lsum}, {gn}); step skipped", flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - ts; step += 1; tok_run += ntok
        rec = dict(step=step, of=total_steps, papers=len(seqs), loss_per_tok=round(lsum / norm, 4), target_tokens=norm, grad_norm=round(gn, 3),
                   lr=lr_at(step - 1), step_s=round(dt, 1), tok_s=round(ntok / dt), run_tok_s=round(tok_run / (time.time() - t_run)),
                   peak_gib=round(torch.cuda.max_memory_allocated() / GB, 1), max_seq=max(len(h) + len(t) for _, h, t in seqs) if seqs else 0,
                   elapsed_h=round((time.time() - t_start) / 3600, 3),
                   per_seq=per, alloc_retries=torch.cuda.memory_stats().get("num_alloc_retries", 0), reserved_gib=round(torch.cuda.memory_reserved() / GB, 1))
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step <= 10 or step % 10 == 0 or step in ck_steps or step == total_steps:
            print("STEP", json.dumps(rec), flush=True)
        if step in ck_steps and step < total_steps:
            save_ckpt(f"{int(round(ck_steps[step]*100))}")
        done = step >= total_steps
        if not done and dl and time.time() > dl - c["soft_stop_margin_s"]:
            save_ckpt(); stopped = True
            print(f"SOFT STOP at step {step}/{total_steps}: Slurm limit near; resume by rerunning this config", flush=True)
            break
    if skipped:
        print(f"skipped {skipped} papers after rendering", flush=True)
    if stopped:
        return 75
    save_weights(out / "adapter-final")
    (out / "DONE").write_text(json.dumps(dict(steps=step, papers=len(rows), dropped=n_drop, elapsed_h=(time.time() - t_start) / 3600)))
    print(f"DONE {step} steps, {len(rows)} papers, dropped {n_drop}; run tok/s {tok_run/(time.time()-t_run):.0f}; "
          f"peak {torch.cuda.max_memory_allocated()/GB:.1f} GiB; wall {(time.time()-t_start)/3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
