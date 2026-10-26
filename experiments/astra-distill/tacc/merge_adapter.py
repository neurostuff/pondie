"""Merge a peft adapter into the bf16 base and write a vLLM-loadable directory.
  python merge_adapter.py MODEL(nu|q27) ADAPTER_DIR OUT_DIR [cpu|cuda]
The base is always loaded in bf16 (also for an adapter trained as QLoRA nf4). Non-weight files of the base snapshot
are symlinked, config.json is copied from the base."""
import json, os, sys, shutil, time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText
model, adapter, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); dev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
hub = Path(os.environ["HF_HOME"]) / "hub"
base = sorted((hub / f"models--{'numind--NuExtract3' if model == 'nu' else 'Qwen--Qwen3.8-27B'}" / "snapshots").glob("*"))[0]
t = time.time()
m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map={"": dev if dev == "cpu" else 0})
for holder in (m, getattr(m, "model", None)):
    for attr in ("visual", "vision_tower", "vision_model"):
        if holder is not None and getattr(holder, attr, None) is not None:
            setattr(holder, attr, None)   # as in training, so the adapter's module names match
m = PeftModel.from_pretrained(m, str(adapter)); print(f"loaded {time.time()-t:.0f}s", flush=True)
m = m.merge_and_unload(); print(f"merged {time.time()-t:.0f}s", flush=True)
out.mkdir(parents=True, exist_ok=True)
m.save_pretrained(out, safe_serialization=True, max_shard_size="10GB"); print(f"saved {time.time()-t:.0f}s", flush=True)
for f in base.iterdir():
    d = out / f.name
    if f.name == "config.json":
        shutil.copy(f.resolve(), d)
    elif d.exists() or f.name.startswith("model.safetensors") or f.name.startswith("model-0") or f.suffix in (".png", ".svg"):
        continue
    else:
        d.symlink_to(f.resolve())
# the merged weight names must be the base's (vLLM loads by name)
def names(d):
    ix = d / "model.safetensors.index.json"
    if ix.exists():
        return set(json.load(open(ix))["weight_map"])
    from safetensors import safe_open
    with safe_open(str(d / "model.safetensors"), "pt") as f:
        return set(f.keys())
a, b = names(base), names(out)
print(f"weights: base {len(a)} merged {len(b)} only-in-base {len(a-b)} only-in-merged {len(b-a)}", flush=True)
print("  e.g. only-in-base", sorted(a - b)[:3], "only-in-merged", sorted(b - a)[:3], flush=True)
bad = {x for x in (b - a) if "lora" in x}
assert not bad, bad
(out / ".ok").write_text("ok"); print("MERGE OK", flush=True)
