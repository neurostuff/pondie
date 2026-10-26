# R0 (Vista), 2026-10-23: partial. NuExtract3 zero-shot and the 27B 48k memory check done; Qwen3.8-27B-FP8 zero-shot is queued

Allocation IBN26027. **SUs used: about 0.81 (0.56 + 0.25)** (one node, 1 SU/h; gh-dev and gh both 1). Budget 15.

| job | queue | what | elapsed | SU |
|---|---|---|---|---|
| 1095811 | gh-dev | NuExtract3 bf16, zero-shot, 481 val papers | 33:43 | 0.56 |
| 1095900 | gh-dev | 27B QLoRA memory check at 49,152 | 3:59 (min charge 15 min) | 0.25 |
| 1095812 | gh, -t 6:00:00 | Qwen3.8-27B-FP8 zero-shot, 481 val papers | PENDING (queue full; no start estimate) | 0 so far |


## Setup (identical to the smoke report: serve env vllm 0.28.0+cu129, torch 2.13.0+cu129, transformers 5.16.1, xgrammar 0.2.3, FA3; train env torch 2.13.0+cu129, transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2, triton 3.7.1; node c642-001, GH200)
- Serve flags: `--max-model-len 49152 --gpu-memory-utilization 0.80 --max-num-seqs 24 --language-model-only`, `CC=gcc`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_USE_DEEP_GEMM=0`, own job. Startup 150 s, KV cache 66.0 GiB (2.01M tokens, 41x at 49k).
- Requests: chat completions, greedy (temperature 0), thinking off, `structured_outputs.json` = `record_shape(ix,"schema",parse_keys)` per paper; the NuExtract3 template goes in via `chat_template_kwargs.template` as `infer.py` does. 24 requests in flight. max_tokens = min(20000, 49152 - n_input_tokens - 64), so the 20,000 cap applied to every paper except those whose input leaves less room (input up to 32.5k tokens).
- Request bodies were built on beast (`tasks.py`; checked equal to `infer.request_body` on a paper) into `vista-out/v5/val.bodies.jsonl` (prompt, template, schema, n_input_tokens), because Vista has no paper text. Only val files were copied (`val.jsonl`, `val.bodies.jsonl`); no train or test data.
- For Qwen (job 1095812) the template is not part of its chat template, so the driver appends it to the prompt ("Return the record as JSON that follows this template ...") and keeps the guided schema. max_tokens leaves 3,000 tokens of margin for that.

## 1a. NuExtract3 zero-shot (bf16), val 481 papers
- Inference wall 1,864 s (31 min) plus 150 s load. Input 7,459,340 tokens, output 4,096,794 tokens: **2,199 output tok/s** aggregate at 24 concurrent (about 4,000 input tok/s average included).
- Guided-decoding failures (HTTP errors): 0. Truncations (`finish_reason=length` at the 20,000 cap): **18**. Invalid JSON: 18 (the same 18, so they cannot be parsed; they score as null records). Median output 7,807 tokens (gold target median 4,483; the model writes about 1.7x the gold, 4.1M versus 2.4M total).
- Files: Vista `/work/07723/jdk3232/vista/astra-distill/out/r0-nu-val.jsonl` (`{sid, prediction, finish_reason, tokens, error, wall_s, prompt_tokens}`); beast `/data/james/agents/astra-distill/vista-out/v5/r0/{r0-nu-val.jsonl, r0-nu-preds.json, score-nu.json, score-nu.txt}`; log `tacc/logs/r0-nu-1095811.out`.

### Score (beast `score.py run`, val, 95% paper bootstrap CI)
| metric | value |
|---|---|
| **cell F1 linked (primary)** | **0.0533 [0.0420, 0.0667]** |
| cell F1 local | 0.0736 [0.0594, 0.0896] |
| cell F1 strict | 0.0487 [0.0378, 0.0619] |
| exact strict / local / linked | 0.0213 [0.0131, 0.0321] / 0.0299 [0.0201, 0.0426] / 0.0230 [0.0144, 0.0336] |
| kind accuracy | 0.2690 [0.2382, 0.3028] |
| model slot: linked / paired / equivalent / id exact | 0.2284 [0.2006, 0.2598] / 0.2239 [0.1964, 0.2547] / 0.0086 [0.0031, 0.0161] / 0.0000 |
| compared entities F1 linked / strict | 0.1718 [0.1522, 0.1914] / 0.1706 [0.1513, 0.1898] |
| compared dir F1 linked / strict | 0.1002 [0.0867, 0.1145] / 0.0991 [0.0856, 0.1135] |
| entity F1 all | 0.4843 [0.4696, 0.4988] |
| valid_record = guided_valid | 0.9626 [0.9439, 0.9792] |
| reference_resolution | 0.9026 [0.8933, 0.9117] |
| listing_keys_share | 1.0000 |
| n predicted / gold analyses | 1,771 [1684, 1860] / 2,439 |
| term F1 linked / local / strict | 0.1247 [0.1064, 0.1452] / 0.1555 [0.1344, 0.1774] / 0.1118 [0.0941, 0.1314] |
| role F1 linked / local / strict | 0.0960 [0.0805, 0.1142] / 0.1221 [0.1036, 0.1421] / 0.0875 [0.0721, 0.1052] |
| slots F1 linked / strict | 0.2628 [0.2462, 0.2798] / 0.2556 [0.2396, 0.2718] |

Entity F1 per class (CI in `score-nu.txt`): acquisitions 0.808, groups 0.802, devices 0.777, preprocessings 0.759, tasks 0.720, conditions 0.625, inference_settings 0.611, external_datasets 0.585, timepoints 0.323, regions 0.312, assessments 0.292, measures 0.254, arms 0.190. Slot F1 linked: acquisitions 0.461, groups 0.423, tasks 0.391, inference_settings 0.374, measure 0.185, assessments 0.110, defines_regions 0.112, regions 0.035, seed/target regions 0. The full 81-line table with every CI (accuracy of details_type, outcome, spatial_scope, statistic_family, and others) is in `score-nu.txt`.

Read: the zero-shot model gets the entities in part but the effect cells almost never (linked cell F1 0.05); it also writes about 1.7x more than the gold.

First outputs were read early (paper 4 of the run): real, well-formed records.

## 3. 27B QLoRA at 49,152 tokens: fits
Job 1095900, `tacc/r0-train-mem.py`: Qwen3.8-27B nf4 (double quant, bf16 compute), LoRA r=16 alpha 32 on all linear layers including the linear-attention mixers, gradient checkpointing (non-reentrant), `model.train()`, one sequence per step, token-sum loss chunked. Sequence = the longest val paper with a 6k+ token target (`JocaEbuxNsyk`: prompt 32,558 + target 12,707 tokens in the dataset), in NuExtract3 single-turn format; the target was cut to its first 8,192 tokens (loss on those) and the document text was repeated to reach exactly 49,152 tokens. 3 steps:

| step | time | tok/s | peak alloc |
|---|---|---|---|
| 1 | 64.9 s (warm-up) | 757 | 76.9 GiB |
| 2 | 28.1 s | 1,750 | 77.8 GiB |
| 3 | 28.2 s | 1,746 | 77.8 GiB |

**Peak allocated 77.8 GiB, reserved 83.1 GiB, of 95 GiB; steady 1,747 tok/s** (matches the 1,900 tok/s at 8–16k). Load 77 s. Loss sum 1,792 to 1,522 over three steps on the same sequence (0.22 to 0.19 per token). 40,960 not needed. Headroom is about 12 GiB reserved, so 49k is the ceiling for this layout; a 4k-token loss region would save a little. Note the actual val maximum is 45,265 tokens, train 48k cap.

## 1b. Qwen3.8-27B-FP8: pending
Job 1095812 (gh, `-t 6:00:00`, `r0-serve.sbatch`, `MODEL=qwen WORKERS=24`) has been PENDING since submission, priority 1,581, no start time. It writes `/work/07723/jdk3232/vista/astra-distill/out/r0-qwen-val.jsonl` incrementally; if it hits the limit, resubmit the same command (the driver skips finished sids and retries failed ones). After it ends: copy the file through the local scratchpad to beast `vista-out/v5/r0/`, build the preds file (`{"record": {sid: prediction}, "finish": {...}}`, see how `r0-nu-preds.json` was made) and run
`cd /data/james/agents/astra-distill/code/vista; ../../vista-venv/bin/python score.py run ../../vista-out/v5/r0/r0-qwen-preds.json --split val.txt --out ../../vista-out/v5/r0/score-qwen.json`.
Expected cost: about 481 x 4.5k output tokens at roughly 600-900 tok/s batched, so 1.5-2.5 h, 2-3 SU.

## Files
- Local: `tacc/r0-infer.py`, `tacc/r0-serve.sbatch` (gh; the gh-dev copy is the same file with `-p gh-dev`), `tacc/r0-train-mem.py`, `tacc/logs/r0-nu-1095811.out`, `tacc/logs/r0-train-1095900.out`.
- Vista: `$WORK/astra-distill/{code,data/v5,out,logs}`; logs `vllm-r0-nu-1095811.log`, `r0-train-1095900.out`.
