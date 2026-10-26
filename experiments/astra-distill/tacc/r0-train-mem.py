"""Throughput/memory smoke for LoRA/QLoRA on a Qwen3.5-hybrid model (NuExtract3, Qwen3.8-27B). One sequence per step,
gradient checkpointing, token-sum loss (sum of per-token CE over the last --tgt tokens, as a short target after a long prompt).
LoRA on attention, MLP and the linear-attention mixers, as nx/train_prose.py. Usage:
  python vista-train-smoke.py --model PATH --mode bf16|nf4 --lens 16384,32768 --steps 5
"""
import argparse, json, time, sys, traceback
import torch
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--mode", default="bf16", choices=("bf16", "nf4"))
ap.add_argument("--lens", default="16384"); ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--tgt", type=int, default=8192); ap.add_argument("--val", required=True); ap.add_argument("--rank", type=int, default=16)
a = ap.parse_args()
GB = 2 ** 30
t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
kw = dict(dtype=torch.bfloat16, device_map={"": 0})
if a.mode == "nf4":
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
for holder in (model, getattr(model, "model", None)):
    for attr in ("visual", "vision_tower", "vision_model"):
        if holder is not None and getattr(holder, attr, None) is not None:
            setattr(holder, attr, None)
torch.cuda.synchronize()
print(f"LOAD {time.time()-t0:.0f}s  mem alloc {torch.cuda.memory_allocated()/GB:.1f} GiB  mode {a.mode}", flush=True)
for p in model.parameters(): p.requires_grad_(False)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads(); model.config.use_cache = False
T = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
     "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
                                         task_type="CAUSAL_LM", target_modules=T))
model.train()  # gradient checkpointing only acts in train mode (from_pretrained returns eval)
params = [p for n, p in model.named_parameters() if p.requires_grad]
print(f"LORA params {sum(p.numel() for p in params)/1e6:.1f}M in {len(params)} tensors; dtypes {sorted({str(p.dtype) for p in params})}", flush=True)
inner = model.base_model.model
core = inner.model.language_model if hasattr(inner.model, "language_model") else inner.model
print("fast paths: fla", __import__("importlib").util.find_spec("fla") is not None,
      "causal_conv1d", __import__("importlib").util.find_spec("causal_conv1d") is not None, flush=True)
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
W = inner.lm_head.weight

def ce_chunk(h, lab, w):
    return torch.nn.functional.cross_entropy((h @ w.t()).float(), lab, ignore_index=-100, reduction="sum")

def loss_fn(h, lab):  # token-sum, chunked + checkpointed so logits for 248k vocab never all exist at once
    tot = 0.0
    for i in range(0, h.size(0), 512):
        tot = tot + checkpoint(ce_chunk, h[i:i+512], lab[i:i+512], W, use_reentrant=False)
    return tot

# real val paper: NuExtract3 single-turn format + gold target; input padded (document text repeated from its start) to reach L
IM_END = "<|im_end|>"
rows = [json.loads(l) for l in open(a.val)]
rows.sort(key=lambda r: -r["n_tokens"])
row = next(r for r in rows if r["n_target_tokens"] >= 6000)  # longest val paper with a 6k+ token target
def ids_of(s): return tok(s, add_special_tokens=False)["input_ids"]
pre = ids_of(f"<|im_start|>user\n【task】structured\n【template_start】{row['template']}【template_end】\n【document_start】\n")
doc = ids_of(row["prompt"]); post = ids_of(f"\n【document_end】{IM_END}\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
tgt = ids_of(row["target_json"] + IM_END)
print(f"PAPER {row['sid']} prompt {len(pre)+len(doc)+len(post)} target {len(tgt)} tok (dataset n_tokens {row['n_tokens']})", flush=True)
res = []
for L in [int(x) for x in a.lens.split(",")]:
    tg = tgt[:a.tgt]; room = L - len(pre) - len(post) - len(tg)
    d = (doc * (room // len(doc) + 1))[:room]  # truncate or repeat the document to fit L exactly
    seq = pre + d + post + tg; assert len(seq) == L, (len(seq), L)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ids = torch.tensor([seq], device="cuda"); times = []; losses = []; nt = len(tg)
    try:
        for s_ in range(a.steps):
            torch.cuda.synchronize(); t = time.time()
            h = core(input_ids=ids).last_hidden_state[0, -nt - 1:-1]
            lab = ids[0, -nt:]
            loss = loss_fn(h, lab); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); times.append(time.time() - t); losses.append(loss.item())
            print(f"  L={L} loss_tokens={nt} step {s_+1}/{a.steps} loss_sum {losses[-1]:.1f} ({losses[-1]/nt:.3f}/tok) {times[-1]:.1f}s "
                  f"{L/times[-1]:,.0f} tok/s peak {torch.cuda.max_memory_allocated()/GB:.1f} GiB", flush=True)
        steady = times[1:] or times
        r = dict(L=L, ok=True, loss_tokens=nt, steady_tok_s=L / (sum(steady) / len(steady)), first_step_s=times[0],
                 peak_alloc_gib=torch.cuda.max_memory_allocated() / GB, peak_reserved_gib=torch.cuda.max_memory_reserved() / GB)
    except torch.OutOfMemoryError as e:
        r = dict(L=L, ok=False, error="OOM: " + str(e)[:200], peak_alloc_gib=torch.cuda.max_memory_allocated() / GB)
        opt.zero_grad(set_to_none=True); del e
    except Exception as e:
        traceback.print_exc(); r = dict(L=L, ok=False, error=repr(e)[:300])
    print("RESULT", json.dumps(r), flush=True); res.append(r)
    if r["ok"]: break  # the first length that fits is enough
print("SUMMARY", json.dumps(dict(model=a.model, mode=a.mode, results=res)))
