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
CUDA_VISIBLE_DEVICES=3 VLLM_DISABLE_COMPILE_CACHE=1 PATH=/path/to/vllm-venv/bin:$PATH \
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

## Never reuse the compile cache

**`VLLM_DISABLE_COMPILE_CACHE=1` is not optional here.** Without it the server starts, reports
a healthy KV cache, and then dies on its very first request:

```
buf6 = torch.ops._C.marlin_gemm.default(..., s72, 18432, 2560, True, False, True, False)
RuntimeError: torch_call_dispatcher("aten::empty", ...) failed at torch/csrc/stable/ops.h:631
EngineDeadError
```

The frame above it is always the same generated file under
`~/.cache/vllm/torch_compile_cache/torch_aot_compile/<hash>/inductor_cache/`. A start that
*compiles* is fine; a start that *reloads* that cache is not. Measured four times with
identical flags, and it separates cleanly:

| compiled how | KV at startup | 18-call sweep |
|---|---|---|
| reloaded cache | 2.31 GiB | died on call 1 |
| reloaded cache | 2.46 GiB | died on call 1 |
| fresh, after `rm -rf` of the cache | 1.45 GiB | 57.9 s, no deaths |
| fresh, `VLLM_DISABLE_COMPILE_CACHE=1`, cache still on disk | 1.45 GiB | 58.0 s, no deaths |

The last row is the control: the cache is present and merely unused, so it is reuse that
breaks and not the directory existing.

Three things this is *not*, each of which cost time to rule out:

* **Not memory.** The two runs that worked had the least KV of the four.
* **Not `strict: true`.** Constrained and free-form decoding die and survive together; the
  measured difference between them is under half a percent.
* **Not an sm_86 quantisation-kernel bug.** The kernel named in the error is fine — it is
  reached through a compiled graph that failed to load, and the error surfaces two frames
  past the fault. Diagnosing this from the last line leads nowhere.

A missing-cubin warning (`Cubin file saved by TritonBundler not found`) sometimes appears
above it and sometimes does not. It is one way the reload fails, not the reason.

Cost: every start compiles, so ~5-7 minutes instead of ~35 seconds. That is the price of the
compiled path, and it is worth paying — `--enforce-eager` also avoids the crash, by never
touching the cache, but gives back most of the speed: 104.6 s against 58.0 s over the same
eighteen calls.

## Stopping it

Send `SIGTERM` and give it time to exit. `kill -9` is survivable now that the cache is never
reused, but it still leaves the port and the engine in a state worth checking.

Two traps, both of which cost real time here:

* `pkill -f "vllm serve"` matches *the shell running that very command*, killing your own ssh
  session and leaving the server up. A bracketed pattern (`pgrep -f "NuExtract3-W4A1[6]"`)
  protects against matching the pattern string, but **not** against a command that also
  contains the literal model name somewhere else -- a combined kill-and-relaunch one-liner
  will still kill itself. Kill and launch in separate commands.
* Killing only the API server leaves `VLLM::EngineCore` holding the card. Check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`; if the port frees
  but the memory does not, kill that pid too. A half-dead server that still holds port 8311
  makes the next start fail with `Address already in use`, which reads as a startup bug
  rather than a cleanup one.

## When the engine dies anyway

`NuExtractServer` treats both shapes of death as the same event: an HTTP 5xx carrying vLLM's
engine markers, and a refused connection from a server that is gone entirely. The second was
missed at first, so a killed server produced `URLError: Connection refused`, no recovery was
attempted, and the paper failed -- `HTTPError` subclasses `URLError`, so the clauses must
stay in that order.

On either, `recover()` runs `Settings.proposer_restart` if one is set and polls until the
server answers, up to seven minutes -- generous because a start now always compiles. Verified
against a real killed server: back and answering in 41 s.

One recovery per call. Dying once is the server; dying twice on the same request is the
request, and the study goes into `repair.POISON`: its sweep is abandoned, the report says so,
and any later attempt in that run repairs it without a proposer rather than spending another
engine on it. The deterministic guards and the grounding pass still run, so the record is
still improved.

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
