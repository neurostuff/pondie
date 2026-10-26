# Lonestar6 smoke test, astra-distill (2026-10-14, allocation IBN26027, queue gpu-a100-dev)

Result: the stack works. NuExtract3 + the effects adapter merge, serve under vLLM with guided JSON on ONE A100 40 GB and reproduce beast's
outputs on 40 of 41 analyses; QLoRA training of the beast recipe runs on one A100. Third job passed all stages except one bug in my slice
(missing results/ dir, fixed in the template; the effects raw output and the comparison were written before it).

## Compute nodes have outbound internet
`curl -sI https://huggingface.co` -> 200, pypi.org 200, download.pytorch.org 403 on the bare host (index paths work: torch wheels installed there).
NuExtract3 (8.8 GB) downloaded from the job in about 1 minute. Login nodes also have internet.
Node: A100-PCIE-40GB x3, driver 570.195.03 (CUDA 12.8), 250 GB RAM, 288 GB /tmp. `module avail cuda` lists 11.3-12.8; none was needed.
TACC sets OMP_NUM_THREADS=1, so `nproc` prints 1; the job has all 128 cores.

## Stack that works (versions = beast's, except where noted)
| env ($WORK/astra-distill/envs/) | python | torch | key packages |
|---|---|---|---|
| serve (beast /home/james/vllm-venv) | 3.12.15 | 2.13.0+cu129 | vllm 0.28.0+cu129, transformers 5.16.1, xgrammar 0.2.3, flashinfer-python 0.6.16.post3, triton 3.7.1, numpy 2.3.5 |
| train (beast /home/james/venv-train) | 3.10.22 | 2.11.0+cu128 | transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2, trl 1.13.0, accelerate 1.15.0, liger-kernel 0.8.3, triton 3.6.0, numpy 2.2.6, flash-linear-attention 0.5.2 |
| score (client only; beast scoring venv has torch 2.11, transformers 5.19.0, peft 0.21.2, accelerate 1.15.0) | 3.12.15 | none | transformers 5.19.0 (tokenizer only), numpy |
Beast's train venv also has causal-conv1d 1.7.0 (not installed on LS6: needs nvcc to build). transformers falls back to a PyTorch reference
conv, "correct but much slower", so LS6 training throughput below is if anything pessimistic. uv 0.13.1, Python 3.10/3.12 via `uv python install`
(no python module newer than 3.9.7 on LS6).

Two traps found:
1. PyPI `vllm==0.28.0` is a CUDA 13 build (`ImportError: libcudart.so.13`). Beast's is `+cu129`: install with
   `vllm==0.28.0+cu129 --extra-index-url https://wheels.vllm.ai/0.28.0/cu129 --index-strategy unsafe-best-match --torch-backend cu129`.
2. flashinfer's sampler JIT-compiles with nvcc, absent on the node (`RuntimeError: Could not find nvcc`): set `VLLM_USE_FLASHINFER_SAMPLER=0`
   (all requests are temperature 0). Prefix caching off, `--language-model-only`, max-model-len 8192, max-num-seqs 32, as nx/serve.sh but TP1.

## Reproduction of beast's effects outputs
Slice: 4 test papers (22fv2dADxHqR, 2EKk476BvHTo, 2P4aTGgYCdAX, 2Jr4Ux5Y8geq), 41 analyses; reference = beast
nx/runs/effects-merged-ctx2-s0/raw.jsonl (oracle-root effects, merged-ctx2-s0, vLLM TP2). Adapter merged on CPU with merge_prose.py (direct.pt applied,
~3.5 min), served by vLLM 0.28.0 on 1 GPU (startup 4-5 min incl. compile; KV cache 23 GiB), same prompts, template and guided JSON (`effects_nx.py infer`).
- 41/41 requests ok, 0 truncated, 5.4 s wall for all 41.
- Exact match (kind plus ordered cells): 40/41 = 97.6 %. Kind match 40/41, cell set match 40/41.
- One difference, 2EKk476BvHTo|ana_text_fa: beast `omnibus` with direction `undirected` on three cells (CON, PLP, PRP); LS6 `contrast` with direction null on the same three cells.
  A near-tie decision; TP2 vs TP1 and different kernels/versions change float order. I did not read the paper text for this one (smoke test); it is one sample.
Not run: coord_scorer.py / q_eff.sh (the full pipeline scorer needs the 100-paper root); the effects client itself is what was compared. The scoring
tail of effects_nx (agreement with Astra on the slice) crashed on the missing results/ dir, so no agreement number for the slice.

## Training (nx/train_prose.py, QLoRA nf4, lr 1e-4, slow-lr-div 1, accum 4, 1 card, no trackio, seed 0)
Data: first 48 pairs of e2e2/data/merged-ctx2-ft.jsonl (train papers; the file beast trained on). 8 optimizer steps (4 pairs each).
- Loads 4-bit base, 29.6M LoRA + 0.96M direct params (same as beast's log). Canary with longest pair (4,299 tokens): peak 5,672 MiB; peak over the 8 steps 5,908 MiB.
- Loss per step (mean of the step's last pair): 0.055, 0.115, 0.023, 0.015, 0.048, 0.056, 0.028, 0.049 (logged mean 0.027 at step 8); the adapter starts from scratch, so these are about 0.05-0.1 from step 1 only because the base already writes JSON well. Not a convergence claim.
- Step time: step 1-2 27 s each (Triton/liger warm-up, long-pair canary), steps 3-8 about 4.3 s each (8.4k tokens per step).
  Steady state about 1,950 tokens/s on ONE A100 (cumulative figure incl. warm-up: 799 tok/s at step 8). About 0.23 steps/s.
- Beast comparison: train-merged-ctx2-s0.log reports 1,365 tok/s total on 2 cards (7.66 GiB cards, DDP, accum 4 each), 717 steps in 2.30 h. One LS6 A100 is about 1.4x beast's two cards
  (extrapolation from 6 steps, one seed; the data slice and the warm-up are not comparable to a full run).
- train_prose.py has no token-sum option; run unmodified (per-pair mean loss), as beast did. A copy with one added log line (peak mem, time per step) was used: tacc/patch_train.py.
- Output adapter: $WORK/astra-distill/runs/smoke-train/final (not useful, 8 steps).

## SUs
3 jobs, 3 SU/node-hour, minimum charge 15 min:
| job | elapsed | outcome | SU (computed) |
|---|---|---|---|
| 3511355 | 11:27 | env, weights, venvs, merge ok; serve failed (CUDA 13 vllm wheel) | 0.75 (15-min minimum) |
| 3511373 | 06:03 | serve failed (flashinfer nvcc) | 0.75 (15-min minimum) |
| 3511379 | 11:31 | all stages; effects scoring tail crashed on missing results dir | 0.58 |
Total about 2.1 SU of the 6 SU budget. `sacct` here has no SU field and `taccinfo` still showed 1500 available right after the runs; the figure is computed
from elapsed time, so recheck `/usr/local/etc/taccinfo` later. (AllocTRES shows billing=128, which I did not interpret.)
Cheaper next time: only the stage that failed needs to rerun, since the template is idempotent (envs, weights and merged model are reused). The failing-stage markers made each failure obvious.

## Paths on Lonestar6 (WORK=/work/07723/jdk3232/ls6)
- $WORK/astra-distill/{envs,models,code,data,bin,python,cache,hf,logs,runs}; 19 GB envs, 20 GB caches, 8.8 GB hf, 8.6 GB models/merged-ctx2-s0 (kept; delete when not needed).
- models/adapter-ctx2-s0 (effects adapter from beast e2e2/ckpt-merged-ctx2-s0/final), models/merged-ctx2-s0, hf/ (HF_HOME with numind/NuExtract3).
- data/slice: layout of a beast root (work/, data/corpus, data/records, ref-raw.jsonl, train-slice.jsonl, nx/runs/effects-ls6/raw.jsonl).
- code/: nx/effects_nx.py, nx/train_prose.py, effects_score.py, merge_prose.py (path to ckpt_io patched), scans/ckpt_io.py, tacc/*.
- logs/: smoke-<jobid>.out (stage log), vllm-*.log, venv-*.log, merge-*.log, train-*.log, effects-*.log. Copy of the last stage log locally: tacc/logs/smoke-3511379.out.
- uv at $WORK/astra-distill/bin/uv; use UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, HF_HOME under $WORK (set in the template). Files were installed in $WORK (not $SCRATCH as the TACC recipe suggests) so nothing is purged.

## Local files (experiments/astra-distill/tacc/)
ls6-smoke.sbatch (template, stage markers `=== STAGE x OK|FAIL`), compare_effects.py (exact-match vs reference raw.jsonl), patch_train.py (adds per-step peak mem/time line), logs/smoke-3511379.out.
Slice and code were staged beast -> local scratchpad -> ls6 (rsync). The template's last edit (mkdir -p $ROOT/results) was not re-run.

## Recommendation
Move 4B effects serving and QLoRA runs to LS6 gpu-a100: one A100 per run, three independent runs per node (CUDA_VISIBLE_DEVICES=0/1/2), 1 SU per GPU-hour.
Next smoke step if wanted: causal-conv1d (needs nvcc via `module load cuda/12.8`) to see whether training gets faster, and a full-test-set (669 analyses) effects run to put a tighter number on reproduction.
