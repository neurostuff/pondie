# Vista smoke test (GH200, aarch64), 2026-10-23

Saved from the tacc-runner's hand-back, which could not write files. Allocation IBN26027, queue gh-dev, six jobs,
**1.5 SU**. Template: `vista-smoke.sbatch` (`STAGES="env t4b t27q t27b serve4b serve27 serve27b"`), with
`vista-train-smoke.py`, `vista-client.py` and logs in `logs/`. Copy the code to `/work/07723/jdk3232/vista/astra-distill/code/`
before submitting; `$WORK` does not expand in an rsync destination.

## Node and versions

- **Node:** c642-071, GH200 with 95 GiB HBM, driver 590.48.01 (CUDA 13.1), 212 GB host RAM, 72 cores.
- **Network:** compute nodes reach huggingface.co, pypi.org and download.pytorch.org.
- **Toolchain:** uv 0.13.5, Python 3.12.15 from `uv python install`. Environments build in 2.5 min cold, under 40 s cached.

| env (`$WORK/astra-distill/envs-aarch64/`) | torch | other |
|---|---|---|
| train | 2.13.0+cu129 | transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2 (nf4 works on sm90), trl 1.13.0, accelerate 1.15.0, flash-linear-attention 0.5.2, triton 3.7.1; no causal-conv1d |
| serve | 2.13.0+cu129 | vllm 0.28.0+cu129 from wheels.vllm.ai, transformers 5.16.1, xgrammar 0.2.3, flashinfer 0.6.16.post3; FlashAttention 3 |

## Training throughput

One sequence per step, gradient checkpointing, LoRA r=16 on all linear layers including the linear-attention mixers,
token-sum loss on the last 1,024 tokens (chunked). The loss values mean nothing: the text was repeated.

| model | mode | seq len | tok/s | peak alloc GiB |
|---|---|---:|---:|---:|
| NuExtract3 4B | bf16 LoRA | 16,384 | 7,223 | 16.5 |
| NuExtract3 4B | bf16 LoRA | 32,768 | 6,955 | 24.8 |
| Qwen3.8-27B | QLoRA nf4 | 8,192 | 1,900 | 27.8 |
| Qwen3.8-27B | QLoRA nf4 | 16,384 | 1,911 | 37.8 |
| Qwen3.8-27B | bf16 LoRA | 8,192 | 1,890 | 61.5 |

QLoRA and bf16 LoRA run at the same speed, so the 27B is compute-bound, and bf16 LoRA fits at least 8k. Likely
fits but not run: 4B at 64k, 27B QLoRA at 32k.

## Serving (vLLM, max-model-len 32768, gpu-memory-utilization 0.80)

| model | startup | KV cache | guided JSON | single-stream decode |
|---|---|---|---|---|
| NuExtract3 | 150 s | 66.3 GiB | valid | 244 tok/s |
| Qwen3.8-27B bf16 | 130 s | 22.7 GiB (9.9× at 32k) | JSON printed, validity line cut off | 57 tok/s |
| Qwen3.8-27B-FP8 | 241 s | 45.4 GiB (19.7× at 32k) | valid | 85 tok/s |

## Traps (fixed in the template)

1. Vista's default `CC=nvc` breaks Triton's JIT: `export CC=gcc CXX=g++`.
2. `model.train()` is needed after `from_pretrained`, or gradient checkpointing does nothing (OOM at 93 GiB).
3. flash-linear-attention 0.5.2 refuses triton 3.4–3.7.0 on Hopper, so use torch 2.13 + triton 3.7.1. Beast's
   torch 2.11 has the same latent problem, which only matters on Hopper GPUs.
4. vLLM at `gpu-memory-utilization 0.90` failed right after a training process in the same job: use 0.80, or serve in
   its own job.
5. Serving the FP8 27B needs `VLLM_USE_DEEP_GEMM=0` (its JIT needs nvcc).
6. Login node needs `RAYON_NUM_THREADS=2` for uv, and `HF_HUB_DISABLE_XET=1 hf download --max-workers 4` for
   downloads (`$WORK/astra-distill/bin/dl2.sh`).
7. Install the `+cu129` vllm build: PyPI's 0.28.0 targets CUDA 13, and wasn't tested here.

## Weights on Vista (`$WORK/astra-distill/hf/hub/`)

| model | size |
|---|---:|
| `numind/NuExtract3` | 8.8 GB |
| `Qwen/Qwen3.8-27B` | 52 GiB |
| `Qwen/Qwen3.8-27B-FP8` | 29 GiB |

## Not done

- The real `nx/train_prose.py` recipe.
- causal-conv1d (no usable nvcc).
- Batched vLLM throughput.
- Serving a merged adapter.
