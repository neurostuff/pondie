# Serving the proposer

`repair` proposes with NuExtract 3. It can run the model in-process (`recall.NuExtract`) or
talk to a vLLM server (`recall_server.NuExtractServer`). `Settings.proposer_url` points at
the server and is tried first; an address nothing answers falls back to the in-process model
and says so on stderr, so a run never fails for want of a server -- but it also never
silently produces a different record without telling you.

## Why serve it

Not the speed, though there is some. A served model decodes under a grammar, and that ends a
failure the in-process path cannot address: on 16962339 the free-running decoder repeats a
*completed* `tasks` object until `max_tokens` -- 2,048 tokens and 27.5 s to produce something
unparseable, which the report then shows as a pass with nothing to add. Under a schema the
same call answers in 70 tokens and 2.4 s.

A server also keeps ~5 GB of weights out of every worker, which is what lets several sweeps
share one card.

## Starting it

```bash
CUDA_VISIBLE_DEVICES=3 PATH=/path/to/vllm-venv/bin:$PATH \
vllm serve numind/NuExtract3-W4A16 \
  --served-model-name nu --port 8311 \
  --max-model-len 16384 --max-num-seqs 16 \
  --gpu-memory-utilization 0.97 \
  --language-model-only \
  --no-enable-prefix-caching
```

Every flag is load-bearing on an 8 GB card:

* **`--language-model-only`** is mandatory. Without it the server dies at startup with
  `Available KV cache memory: -0.01 GiB`: the W4A16 checkpoint keeps its vision tower in
  bfloat16, and dropping it takes the weights from 5.11 to 4.48 GiB and frees a further
  512 MB of encoder-cache profiling allocation. The obvious alternative,
  `--limit-mm-per-prompt`, crashes vLLM 0.28.0 outright.
* **`PATH`** must include the venv's `bin`, because flashinfer's JIT shells out to `ninja`.
* **`--no-enable-prefix-caching`** because it buys nothing here: the chat template puts the
  per-class template *before* the document, so the dozen calls of a sweep share about
  fifteen tokens of prefix, and prefix caching is prefix-only.

Startup is ~35 s warm and ~2.5 min cold; the first start after clearing the compile cache is
~5 min. `Application startup complete` in the log means ready -- `NuExtractServer.reachable`
returning true is a real signal, not an optimistic one.

## Stopping it

**Send `SIGTERM`. Never `kill -9` a server that may be compiling.**

vLLM caches its inductor-compiled graph under `~/.cache/vllm/torch_compile_cache/`. Killing
the process partway through compilation leaves the graph cached but its Triton cubins
missing, and every subsequent start reloads that broken cache and dies on the first request:

```
RuntimeError: ('Cubin file saved by TritonBundler not found at %s', '.../....cubin')
Failed to reload cubin file statically launchable autotuner
  triton_red_fused__to_copy_add_fused_add_rms_norm_marlin_gemm_2
...
buf6 = torch.ops._C.marlin_gemm.default(...)
RuntimeError: torch_call_dispatcher("aten::empty", ...) failed at torch/csrc/stable/ops.h:631
EngineDeadError
```

The message names `marlin_gemm` and `aten::empty`, so it reads as a quantisation-kernel bug
or an allocation failure. It is neither. The kernel is fine and memory is fine -- the cubin
was never loaded, and the error surfaces two frames after the real fault. Diagnosing this by
the last line costs hours; the `TritonBundler` warning above it is the actual cause.

Recovery:

```bash
pkill -TERM -f NuExtract3-W4A16          # or kill the pid; give it time to exit
rm -rf ~/.cache/vllm/torch_compile_cache
```

Then start again and accept the ~5 min recompile. `--enforce-eager` also avoids the crash,
because eager mode never touches that cache -- but it is treating the symptom, and it costs
roughly 45% of the throughput (18 calls: 58.2 s compiled against 104.6 s eager).

Two smaller traps, both of which cost time here:

* `pkill -f "vllm serve"` matches *the shell running that very command*, killing your own ssh
  session and leaving the server up. Match on something the kill command does not itself
  contain, e.g. `pgrep -f "NuExtract3-W4A1[6]"`.
* Killing only the API server leaves `VLLM::EngineCore` holding the card. Check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and, if the port
  frees but the memory does not, kill the EngineCore pid too. A half-dead server that still
  holds port 8311 makes the next start fail with `Address already in use`, which then looks
  like a startup bug rather than a cleanup one.

## The schema, and why `strict` is on

`json_schema_for` projects a NuExtract template into a JSON schema, and **every leaf admits
`null`**. That is what keeps it honest. A grammar with no way to say "nothing here" does not
decline: on 16759342, a paper that declares no arms and whose every group the extractor left
empty, a non-nullable schema filled `Group.arm` with 'smoking' and 'non-smoking'.

Nullability is the variable, not `strict`. Measured on that paper: nullable answers `null`
under `strict: true` with every property required, and non-nullable invents the same two
values under `strict: true` and `strict: false` alike. So `strict` is on -- a strict grammar
over a nullable schema both guarantees parseable output and permits an empty answer.

Note the enum branch carries `None` inside the `enum` rather than in a type union, because a
closed `enum` cannot be widened by one. A converter that forgets this loses nullability for
exactly the slots most likely to be inapplicable.
