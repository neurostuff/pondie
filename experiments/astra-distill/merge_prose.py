"""GENERATED from /tmp/nspond/merge_v21.py by sed: checkpoint and output are arguments.

Fold a prose fine-tune into the base weights, including the parts LoRA cannot reach.

`direct.pt` holds A_log, dt_bias, the convolutions and the norms -- the Mamba
parameters trained directly because LoRA has no handle on them. Merging the
adapter alone would produce a model that loads cleanly, runs, and is wrong, so
the direct tensors go in FIRST and the merge is verified against the
adapter path before anything is published.

CPU and bfloat16: the cards are 7.66 GiB and the model is 8.8 GB.
"""
import json, shutil, sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

sys.path.insert(0, "/work/07723/jdk3232/ls6/astra-distill/code/scans")
from ckpt_io import DirectWeights

BASE = "numind/NuExtract3"
CKPT = sys.argv[1]
OUT = Path(sys.argv[2])

print("loading base in bf16 on CPU ...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="cpu")

# direct.pt's keys are the PEFT-wrapped names, so the wrapper goes on first
# and the direct tensors are written through it. Applying them to the bare
# base model raises on every key.
print("loading adapter ...", flush=True)
peft = PeftModel.from_pretrained(model, CKPT, dtype=torch.bfloat16)
lora = sum(1 for k, _ in peft.named_parameters() if "lora_" in k)
print("  %d lora tensors" % lora, flush=True)
if lora == 0:
    raise SystemExit("no LoRA tensors loaded -- wrong module tree")

print("applying direct.pt ...", flush=True)
n = DirectWeights(peft).apply(CKPT)
print("  %d directly-trained tensors written" % n, flush=True)
if n == 0:
    raise SystemExit("direct.pt applied nothing -- refusing to publish a half-merged model")

print("merging ...", flush=True)
merged = peft.merge_and_unload()
merged.config.architectures = ["Qwen3_5ForConditionalGeneration"]

OUT.mkdir(parents=True, exist_ok=True)
print("saving to %s ..." % OUT, flush=True)
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
try:
    AutoProcessor.from_pretrained(BASE).save_pretrained(OUT)
except Exception as exc:
    print("  (no processor: %s)" % exc, flush=True)

print("\nwrote:", flush=True)
for f in sorted(OUT.iterdir()):
    print("  %-40s %8.1f MB" % (f.name, f.stat().st_size / 1e6), flush=True)
print("done", flush=True)
