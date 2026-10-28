# Vista experiment journal: NuExtract3-4B vs Qwen3.8-27B on the migrated Astra data

Started 2026-10-23. This is the single record of this phase: question, protocol, hypotheses with their decision rules
(fixed before any training run), the run log, results, and the conclusion. Earlier work (H1–H33) is in `JOURNAL.md`;
how the data was built is in `MIGRATION.md`, and the reasons for each design choice in `TRAINING-PLAN.md`.

## Question

Trained on ≈12k Astra-extracted papers, how closely can (a) NuExtract3-4B and (b) Qwen3.8-27B reproduce Astra's
coordinate-analysis records? Which model and training size give the best agreement per GPU-hour on a GH200?

## Data (frozen)

- **Records (data v5, 2026-10-23):** 13,435 unflagged papers, migrated to study_schema `main` (`8fe87a1`).
  - Evidence spans are 100% aligned to the new text.
  - Coordinates come from the old parse, written as a CoordinateParse. Record keys and old-parse keys are kept apart.
    Implausible entries are dropped, and Brodmann-fused x values (`BA=1344` → 44) are recovered.
  - Records are pruned to coordinate-backed analyses: an entry with points, or a null result in a coordinate table
    (94). Every entity they depend on is kept.
  - Sets whose analyses carry effect cells are `result`; shared condition stimulus content is moved to the task
    (1,969 tasks).
  - Excluded as degenerate: dangling references (17), cell terms off the model chain (1), analyses without cells (1).
  - Reviews: `migrate/REVIEW-2.md` (independent, two rounds) and `migrate/review_train.py`.
  - On beast: `/data/james/agents/astra-distill/migrate/{train/records, run-old/parse}`. Text:
    `ns-pond/data/<sid>/processed/pubget/text.txt`.
- **Splits** (`migrate/splits.py`, seed 20261023): by paper. Linked papers stay together: shared pmid, pmcid, doi,
  preprint id or title, or ≥ 50% shared text (18 groups, 36 papers, none crosses a split). Stratified by number of
  coordinate analyses. Copies in `migrate/data/splits/`.

| split | papers | coordinate analyses | sha256 |
|---|---:|---:|---|
| train | 12,454 | 65,852 | `c782c7fb…4bf1` |
| val | 481 | 2,439 | `aa940a11…bea3` |
| test | 500 | 2,574 | `797e7f08…b7` |

- **Out of every split:** the 3,699 flagged papers, the 100 original Astra papers, and the excluded papers
  (unalignable 84, different document 27, abstract only 7, no new text 406, degenerate 19).

## Protocol

- **All runs queued at once (user, 2026-10-23),** because the gh backlog is days long. A run whose prerequisite shows a fatal
  problem is cancelled from the queue; hyperparameters were fixed before any result, so queueing early does not
  let results shape them.
- **Test is touched once per finalist configuration,** after every choice is made on val. Training code reads
  `train.txt` and `val.txt` only, and the test list is write-protected on beast.
- **Task: one call per paper (user, 2026-10-23).** Input: the paper text, plus the parse listing (every entry's key,
  origin, table caption and footer, and points). Output: the whole compact record (entities, coordinate sets keyed
  to the parse, analyses, models and effect cells). This replaces the two-call design in `TRAINING-PLAN.md`; that
  design's evidence (entities once, per-entry anchoring) was never tested against a single call.
- **Decoding:** greedy, thinking off. Zero-shot: guided JSON against the target schema. Fine-tuned: JSON mode, unconstrained field order (changed 2026-10-25; see the data log).
- **Loss:** token-sum, on outputs only.
- **Seeds: one per run (user, 2026-10-23).** A difference is called real when its paired paper bootstrap 95% CI
  (2,000 resamples) excludes 0 on test. Seed variance is not measured in this phase. H28 measured it on the 8 GB
  setup at ≈0.03–0.05 local cell F1 and up to 0.12 linked, so any difference smaller than that is reported as
  "within the old seed spread" even when its CI excludes 0.

## Metrics (agreement with Astra, per coordinate analysis, paper-level bootstrap)

- **Primary:** effect cell F1, linked (the "agreement in substance" scorer from H24, ported to the new format).
- **Secondary:**
  - effect cell F1, strict and local;
  - effect kind accuracy;
  - model slot (exact / equivalent / linked);
  - compared entities (groups, conditions, timepoints, arms);
  - entity F1 per class (call 1);
  - share of restored records valid against the schema;
  - share of references that resolve.
- **Cost:** GPU-hours (SUs) per run, and training tokens/s.

## Hypotheses (fixed before running)

| # | hypothesis | test | decision rule | outcome |
|---|---|---|---|---|
| H34 | With ≈12k papers, more data still helps the 4B | NuExtract3 LoRA at 1k / 3k / 10k train papers, one seed each, scored on val | supported if 10k − 1k linked cell F1 has a CI excluding 0 and exceeds H28's seed spread | untested |
| H35 | Fine-tuned NuExtract3 beats zero-shot NuExtract3 and zero-shot Qwen3.8-27B | R0 zero-shot vs R2 on test | as in Protocol | untested |
| H36 | Qwen3.8-27B QLoRA beats NuExtract3 LoRA at equal data | both on the full train split, one seed each, test | as in Protocol; also report the gap per GPU-hour | untested |
| H37 | Whole-paper input beats the 8 GB-era per-analysis text (ctx2) | the H36 winner, both inputs, val | as in Protocol | untested |
| H38 | Errors are mostly entity creation, not effect structure | from the single call's output: effect cells scored with entity-equivalence credit vs strict, plus the per-class entity F1, on test | descriptive: where the loss sits | untested |

## Data log

| date | change |
|---|---|
| 2026-10-25 00:20 | **Guided decoding blocks fine-tuned output (harness bug).** h34-1k's guided eval wrote no entity lists in 479 of 479 records (entity F1 0, references 11%). xgrammar emits properties in schema order; the schema listed entity collections before `analyses`, and the training targets put `analyses` first, so once the model wrote `analyses` the grammar could not return to them. Worse, the targets carry no single key order inside objects: Astra's field order varies across records (36 object paths, 22 with conflicting orders; 131k training objects break any single order). **Verified 00:55:** none of 479 guided records holds any blocked field (groups … measures), all 479 keep schema order, and the same adapter in JSON mode writes every collection (groups in 472, models in 472) and scores linked cell F1 0.486 instead of 0.000. **Protocol change:** fine-tuned models are evaluated with JSON-mode decoding (`response_format: json_object`, `guided: false` in the eval configs); schema validity is still scored afterwards. Zero-shot R0 keeps guided decoding, since an untrained model has no learned order. The h34-1k guided score is void; a JSON-mode re-eval runs on gh-dev (1099043). `val.bodies.v3.jsonl` (schema reordered to the dominant training order) is kept for reference. |
| 2026-10-24 23:30 | **Fixed guided schema in use on Vista** (`data/v5/val.bodies.jsonl`, sha256 `cf4b96e7…`; old one kept as `val.bodies.v1.jsonl`). Verified: 0 of 12,935 train+val gold records rejected; fuzz of 30k mutated train records gives 0 that pass the guided schema but fail extraction; NuExtract3 R0 predictions unchanged; the 78 invalid Qwen records are now rejected. Every fine-tuned eval reads this file. |
| 2026-10-24 | **Harness fixes from R0b's first 176 papers.** (1) The scorer crashed on analyses with a null coordinate-set key (121 of 176 Qwen papers silently unscored); they now score as false positives. (2) The guided schema did not require the extraction schema's required fields (Qwen omitted `model_estimation` 184 times, `details` 40, `measure` 19) and mediation did not round-trip: 44% of Qwen records invalid. The schema is being fixed, `val.bodies.jsonl` rebuilt, and **R0a and R0b rerun with it**, so H35's floor uses the same schema as the fine-tuned evals. |
| 2026-10-23 | One-call harness on v5 (`vista/`, beast `vista-out/v5/`): self-test 1.000 on gold, perturbations behave as expected; all gold targets valid under the guided schema. Lengths (whole sequence, NuExtract3 tokens): train median 19.2k, p90 28.6k, max 86k, 254M per epoch; val max 45k. **Training context 48k** (drops 43 of 12,454 train papers, no val papers). |
| 2026-10-23 | **Data v5 passes the independent final check** (`migrate/REVIEW-2.md`, "Final check (v5)"): rules 1–6, null results (94, 5 read against the text), 0 implausible points, role evidence, stimulus move (1,591 kept tasks, 5 read), leakage (no test twin in train or val). Residual, minor: 8% of analyses state a coordinate space different from the parse's; for 36% of quotes printed twice, the occurrence chosen is not the best context match (text verbatim either way). |
| 2026-10-23 | Harness (`vista/`): round trip exact on all records; gold validates against the guided schemas; scorer 1.000 on gold vs gold, and responds to perturbations as expected. **Call-2 unit is the parse key** (12% of keys hold 2–57 Astra analyses; the target is the list, paired by name). Call 2 packed per paper as one multi-turn sequence, with table caption and footer added (H14). Entries with points but no Astra analysis get an empty target (abstain). Keys with more than 20 analyses skipped. |
| 2026-10-23 | Out of extraction for this phase (user): `Task.paradigm`, `Task.variant_qualifiers` (and `stable_id`, `Table.title`/`description`, never Astra's). |
| 2026-10-23 | data v2–v5 before any training. v3 fixed REVIEW-2's second round (analyses on point-less entries, implausible coordinates, role evidence). v4 kept null results on point-less table entries and moved shared stimulus content to the task. v5 filtered description coordinates and recovered Brodmann-fused x. Splits redrawn each time; v5 is final (above). |
| 2026-10-23 | splits v1 (12,425 / 481 / 500) superseded before any training: REVIEW-2 found key collisions, a pruning leak and 30 near-duplicate pairs. Fixed, rebuilt, re-split (v2, above). |

## Run log

| run | date | model | method | data | seed | SUs | val primary | notes |
|---|---|---|---|---|---|---|---|---|
| smoke | 2026-10-23 | both | env + a few steps | synthetic | — | 1.5 | — | `tacc/REPORT-vista-smoke.md`: aarch64 stack works (torch 2.13 cu129, bnb nf4, vLLM 0.28 cu129). Training tok/s: 4B bf16 LoRA 7,223 (16k) / 6,955 (32k); 27B QLoRA 1,900 (8k–16k), same speed as 27B bf16 LoRA (compute-bound). Serve: 4B 244, 27B bf16 57, 27B FP8 85 tok/s single-stream |
| R0a | 2026-10-23 | NuExtract3 | zero-shot, guided JSON per paper | val 481 | — | 0.56 | **0.053** [0.042, 0.067] | entity F1 0.48; model slot 0.23; kind 0.27; 96% valid records; 90% references resolve; output 1.7× gold length, 18 hit the 20k cap; 2,199 out tok/s at 24 in flight (`tacc/REPORT-R0.md`) |
| R0b | 2026-10-24 | Qwen3.8-27B-FP8 | zero-shot, old guided schema | val 481 | — | 1.8 | **0.086** [vs NuExtract3 +0.033, CI +0.012..+0.054] | gh-dev 1096039 (1:45) + 1096040 (2 min, resume check). 478 stop / 3 length; 592 out tok/s. Kind 0.43, outcome 0.57, model slot 0.36, entity F1 0.63 (groups 0.89, regions 0.47). Only 57% records schema-valid (omitted required fields): to be rerun with the fixed schema |
| h34-1k | 2026-10-24 | NuExtract3 | bf16 LoRA r=64, 1 epoch | 998 train papers (2 over 49k dropped) | 0 | 0.78 | **0.486** [vs zero-shot +0.432, CI +0.386..+0.476] (JSON mode) | trained: 63 steps, 7,409 tok/s, peak 35.1 GiB, loss/token 0.25 → 0.21–0.25; eval moved to gh-dev (1098973), fixed schema | JSON-mode eval (gh-dev 1099043, 0.34 h): 473 stop / 8 length; model slot 0.65, kind 0.68, outcome 0.79, entity F1 0.78 (groups 0.92, tasks 0.87, regions 0.61), 92% valid records, 98% references resolve. The guided eval (1098973) is void: no entity lists, see the data log.
| R0a-v2 | 2026-10-25 | NuExtract3 | zero-shot, fixed guided schema | val 481 | — | 0.55 | **0.049** | 460 stop / 21 length; kind 0.28; model slot 0.22; entity F1 0.49; 96% valid |
| R0b-v2 | 2026-10-25 | Qwen3.8-27B-FP8 | zero-shot, fixed guided schema | val 481 | — | 1.77 | **0.079** (+0.029 [+0.009, +0.051] over R0a-v2) | 471 stop / 10 length; kind 0.43; model slot 0.35; entity F1 0.62; valid records 57% → **98%** with the fixed schema |
| h34-10k | 2026-10-25 | NuExtract3 | bf16 LoRA r=64, 1 epoch | 10k | 0 | 7.34 + 0.35 | **0.646** (+0.161 [+0.110, +0.208] over 1k) | eval gh-dev 1099455: 480 stop / 1 length. Exact (linked) 0.58, kind 0.84, outcome 0.91, model slot 0.82, entity F1 0.87 (regions 0.78), 99% valid records |
| h34-3k | 2026-10-25 | NuExtract3 | bf16 LoRA r=64, 1 epoch | 3k | 0 | 0.84 + resume | | **failed at 0.84 h**: Triton kernel cache defaulted to `~/.triton/cache` on /home1 (NFS), shared by 3 concurrent jobs on 3 nodes; one read a kernel file another was replacing (OSError 116 stale file handle). Our env.sh moved HF and uv caches but not Triton's. Fix: per-job caches in node /tmp (`env.sh`). Resumes from its 25% checkpoint on gh-dev (1099456 → eval 1099457) |
| r2-4b-full | 2026-10-25 | NuExtract3 | bf16 LoRA r=64, 1 epoch | 12.4k | 0 | ≈9.5 | | trained (DONE); eval moved to gh-dev |
| r3-27b-full | 2026-10-25 | Qwen3.8-27B | QLoRA nf4 r=64, 1 epoch | 12.4k | 0 | running | | chunk a (1095923) started ≈04:00; b and c resume, then eval |
| mem | 2026-10-23 | Qwen3.8-27B | QLoRA r=16, 3 steps at 49,152 | 1 paper | — | 0.25 | — | fits: 77.8 GiB alloc / 83.1 reserved of 95; 1,747 tok/s |
| queue | 2026-10-23 | both | all runs queued (user): h34-1k 1095915, h34-3k 1095917, h34-10k 1095919, r2-4b-full 1095921, r3-27b-full 1095923→1095924→1095972 (resuming chunks), each with an afterok val eval | train v5 | 0 | ≈70 expected / ≤105 worst (+0.67 smoke) | | gh-dev smoke passed: 4B r=64 at 49k 34 GiB, 7.2k tok/s; 27B QLoRA r=64 at 49k with an 11.6k-token target 74 GiB allocated / 78 reserved, 2.05k tok/s (kept r=64). Padding to multiples of 256 removed kernel re-specialisation stalls. Jobs listed in `tacc/QUEUE.md`. |

## Budget (from the smoke test)

≈250M training tokens per full epoch (12.5k papers × ≈20k tokens, inputs included):
- 4B bf16 LoRA at ≈7k tok/s: ≈10 GPU-h (SU) per epoch.
- 27B QLoRA at ≈1.9k tok/s: ≈37 SU per epoch.

With one seed per run (user), the full-data runs cost ≈10 SU (4B) and ≈37 SU (27B) per epoch. With the learning
curve (≈15 SU) and evaluation, the plan fits well within 500 SUs.

## Results

- **H35 floor (fixed schema):** zero-shot NuExtract3 0.049, zero-shot Qwen3.8-27B 0.079 (+0.029 [+0.009, +0.051]); with the fixed schema 98% of Qwen's records are valid (was 57%). Fine-tuned NuExtract3 on 1k papers (0.486) beats zero-shot Qwen by +0.407 [+0.360, +0.452]. **H35 supported on val** (test pending).
- **H34 so far (val, NuExtract3, JSON mode):** 1k 0.486 → 10k **0.646** linked cell F1 (+0.161 [+0.110, +0.208]); model slot 0.65 → 0.82, regions 0.61 → 0.78, valid records 92% → 99%. More Astra data still helps the 4B a lot. 3k and full pending.
- **H34, first point (1k papers, NuExtract3, JSON mode):** linked cell F1 **0.486** vs 0.053 zero-shot (+0.432 [+0.386, +0.476]); model slot 0.65 vs 0.23; entity F1 0.78 vs 0.48. 1,000 papers already brings the end-to-end pipeline to roughly the level of the 8 GB pipeline (≈0.48 linked, H28), from one call per paper.
- **R0 full (481 val papers, old guided schema):** Qwen3.8-27B zero-shot 0.086 vs NuExtract3 0.053 linked cell F1 (+0.033 [+0.012, +0.054]); Qwen is better on kind (+0.16), outcome (+0.13), model slot (+0.13) and entity F1 (+0.14, with groups +0.09 and regions +0.15), but only 57% of its records are schema-valid vs 96%. Both are far from usable zero-shot. Reruns with the fixed schema to follow.
- **R0 interim (176 shared papers, old guided schema; superseded by the reruns):** Qwen3.8-27B zero-shot beats NuExtract3 zero-shot on content: linked cell F1 0.101 vs 0.059 (+0.041 [+0.003, +0.080]), kind 0.42 vs 0.27, model slot 0.33 vs 0.20, entity F1 0.64 vs 0.49. But only 56% of its records are schema-valid, against 96%.
- **R0a (H35 floor):** zero-shot NuExtract3 reaches linked cell F1 0.053 on val: entities partly (0.48), effect structure almost never.

## Conclusion

Not yet known.

### 2026-10-25: adjudication of h34-10k vs Astra (val, 25 papers, seed 7)

Full write-up: `vista/ADJUDICATION-h34-10k.md`. The sample is stratified half below and half above linked F1 0.6, and
papers where the two records agree exactly were skipped. Of 45 disagreements judged against the paper text: Astra
right 23, model right 4, both acceptable 16, unclear 2. Among decided cases the model is right in 15%. The model's
typical errors:
- picking the first-level GLM where the table comes from the group-level test;
- getting the comparison wrong for the contrast that defines the table;
- dropping context cells;
- merging analyses.

Astra's errors: a few direction and linking mistakes. All 16 "both acceptable" items are penalised by `score.py`.
Naming or detail accounts for 10, splitting an analysis into several for 2, a contrast term written as separate cells
for 2, a model chain for 1, and a factor written as a group rather than a condition for 1. Rescoring those as matches
raises the sample's F1 from 0.46 to 0.67. That is a rough estimate, not a measurement over the whole val set. Action:
add an equivalence-tolerant F1 next to the strict one before comparing methods further.

### 2026-10-25: audit of the suspected migration artifacts

The adjudication flagged six papers' gold as possible migration artifacts:
- directions missing;
- `model_family` missing although `model_type` is present;
- duplicate analyses on one key;
- repeated nuisance terms;
- per-ROI names.

None of them comes from the migration. Each is in Astra's raw output.
- **Raw vs migrated record** (`migrate/rawdiff.py`, all 16,716 training records, entities matched by local_id, evidence
  removed). The only content changes are the intended ones:
  - `source_table_analysis` keys;
  - rebuilt coordinate sets;
  - `result_reporting`;
  - condition `stimulus_content` moved to the task;
  - 1,064 timepoints and 518 arms pruned.

  Effect cells, directions, models, terms, groups and names are identical to Astra's.
- **Compaction:** no `extraction_status` wrapper leaks into the 12,935 train and val targets.
- **Key remapping:** no old key maps to two new keys. 17 new keys in about 12 papers (1 in val, LxrP75jESY2h) each take
  analyses from two old keys. In every case the two entries are distinct tables with identical coordinates, for example
  GM vs WM, or r = 0.20 vs 0.25. study_schema's content key gives them the same key, so they share one key by design.
  This is negligible and left as is.
- **The duplicate in ds2mKHx39PGb** comes from a single old key: Astra wrote it twice.

Teacher-side gaps that are faithful copies of Astra:
- 3,102 of 148,698 cells (2.1%) have no direction. 942 contrast analyses have such a cell, and 293 of those have a
  name with `<`, `>` or `vs`.
- 2,238 of 52,402 models (4.3%) have no `model_family`, and 2,147 of those do have a `model_type`.

Fixing these would mean curating Astra's labels, not repairing the migration.

### 2026-10-25: h34-3k and r2-4b-full scored (val, JSON mode, 481 papers)

Linked cell F1 by number of training papers (NuExtract3 4B):

| papers | linked cell F1 |
|---:|---:|
| 1k | 0.486 |
| 3k | 0.621 |
| 10k | 0.646 |
| full, 12.4k | 0.686 |

Paired differences:
- 3k vs 1k: +0.135 [+0.096, +0.175]
- 10k vs 3k: +0.026 [−0.014, +0.063]
- full vs 10k: +0.040 [+0.008, +0.072]

Other metrics for the full run:
- entity F1 0.871
- model linked 0.831
- kind 0.838
- valid records 99.6%

Gains flatten after 3k, but the full run still beats 10k. Outputs: beast `vista-out/v5/runs/{h34-3k,r2-4b-full}/`.
The 27B run (r3) was at step 140 of 776.

### 2026-10-25: checkpoint evals for r3-27b-full

`r3-27b-full` saves adapters at 25, 50 and 75% (steps 194, 388 and 582). Each checkpoint is scored on a fixed random
subset of val: `data/v5/val-sub160.txt`, 160 papers, seed 1. The full val run takes too long for gh-dev's 2 h limit.
- `eval_record.py` gains two things: a `sids` option, and an exit before serving when the adapter is not saved yet.
- Jobs on gh-dev, with start times set to just after each save:
  - ck25: 1100098
  - ck50: 1100099
  - ck75: 1102765, `--dependency=after:1095924+60` (60 min after chunk b starts; step 582 is about 16 steps of
    ~140 s past the resume at 566)
  - Chunk b (1095924) lowered from 24 h to 11 h on 2026-10-27 01:20 to fit backfill: it was eligible for 18 h at
    low fairshare. The rest of the run is 210 steps × ~145 s ≈ 8.5 h; `train_record.py` reads the new limit for its
    soft stop, and chunk c exits at once on `DONE` if b finishes.
- References on the same 160 papers (linked cell F1): 4B 10k 0.629, 4B full 0.666.

### 2026-10-25: r3-27b at 25% (step 194, about 3.1k papers seen), val-sub160

Run in JSON mode on all 160 papers, in 37 min, with no truncations.

Paired comparison against the 4B full run on the same papers:

| metric | 4B full | 27B ck25 | difference [95% CI] |
|---|---:|---:|---|
| linked cell F1 | 0.666 | 0.665 | −0.001 [−0.055, +0.056] |
| kind | 0.816 | 0.806 | −0.011 [−0.076, +0.064] |
| model | 0.828 | 0.826 | −0.002 [−0.056, +0.053] |
| groups F1 | 0.948 | 0.979 | +0.032 [+0.003, +0.067] |
| regions F1 | 0.789 | 0.835 | +0.046 [−0.014, +0.115] |
| schema-valid | 0.994 | 0.975 | 4 of 160 invalid, but references and keys still resolve |

- **Against the 4B at 10k:** cell F1 +0.036 [−0.037, +0.110].
- **The 13-paper preview was misleading.** It showed the 27B well ahead on kind and model, and that lead disappeared at 160
  papers.
- **Training loss** has fallen from about 0.17 to 0.138 by step 274.
- **Next:** ck50 (job 1100099) tests whether the 27B keeps improving past the 4B.

### H35 (2026-10-25): full-parameter fine-tuning vs LoRA, NuExtract3 4B, 3k papers

**Question.** Does full-parameter SFT beat LoRA r64 on the same data?

**Design.**
- The comparison is h34-3k (LoRA r64, lr 2e-4; val linked cell F1 0.621) against h35-full-3k: the same 3,000 papers
  (`n_papers` 3000, `subset_seed` 20261023), one epoch, token-sum loss and 49k context.
- h35-full-3k trains every weight:
  - fp32 master weights under bf16 autocast, because at lr 1e-5 a bf16 weight would round most updates away;
  - bitsandbytes AdamW8bit;
  - lr 1e-5 cosine with 3% warmup;
  - one seed.
- It saves a bf16 model directory that vLLM serves directly.
- The test is the paired val comparison, in JSON mode.

**Code.**
- `train_record.py` gains `mode: full`; the LoRA path is unchanged.
- `eval_record.py` serves a full model directory without merging.
- New `chain.sbatch`: on a soft stop it resubmits itself on gh-dev, and when training is done it submits the eval. The gh
  queue has 600 jobs pending and r3 waited 33 h to start.

**Jobs.**
- Smoke test 1100657: 3 steps on the longest paper, then a save and a 3-paper eval.
- Run 1100658, after the smoke test passes.

**Decision rule.**
- If the CI of full minus LoRA is entirely above 0, run full fine-tuning on all the data.
- Otherwise keep LoRA.

**Change (2026-10-25).** `chain.sbatch` no longer resubmits itself or submits the eval: Vista doesn't allow `sbatch`
from a compute node. Continuation chunks (afterany) and the eval are submitted from the login node as gh-dev slots
free up. TACC's docs (Vista guide, Good Conduct) state no rule against production work on gh-dev, and the user said
to keep the run there for now, mindful of the policy.

**Smoke test 1100657 (2026-10-25).**
- **Mechanics: passed.** The fp32 master weights load in 50 s at 15.7 GiB, and peak memory is 61.1 GiB at 49k tokens.
  The bf16 save is 9.7 GB, with the tied lm_head saved separately. vLLM serves it.
- **Output: all 3 eval papers looped to the 20k-token limit**, repeating fields with coherent record text, and gave no
  valid JSON. The cause is 3 steps at full lr with no warmup. LoRA's smoke test stays coherent because its update starts
  at zero.
- **My error:** I judged the smoke test by its exit code and didn't read its outputs before the 3k run started.
- **Guard added:** a checkpoint at 50% (step 94), scored on val-sub160 (`configs/h35-full-3k-ck50-eval.json`) as soon as
  it is saved.

**Early loss, same batches in the same order** (h34-3k LoRA vs h35 full):

| step | LoRA | full |
|---:|---:|---:|
| 5 | 0.337 | 0.327 |
| 10 | 0.253 | 0.248 |
| 20 | 0.244 | 0.238 |
| 30 | 0.255 | 0.246 |

**Chunk 2:** 1101010, afterany.

### 2026-10-26: overnight

- **h35-full-3k finished.** It ran in two chunks: a soft stop at step 143, then the rest in 0.6 h. Peak memory was
  71.5 GiB. No adapter-50 was saved: chunk 1 had passed step 94 before the `ckpt_fracs` change could be read.

  Training loss per token, mean over 30-step windows, same batches. The h34-3k log repeats some steps from its failed
  first attempt, so its means are approximate.

  | steps | LoRA | full |
  |---|---:|---:|
  | 1–30 | 0.281 | 0.277 |
  | 61–90 | 0.206 | 0.193 |
  | 121–150 | 0.194 | 0.177 |
  | 181–187 | 0.186 | 0.173 |

  The eval was not submitted, because chain.sbatch no longer submits it. Submitted now: 1101539, full val.
- **r3-ck50 eval 1100099 failed.** vLLM found only 39.3 of 95 GiB free at startup. The ~56 GiB missing matches the
  merged 27B shards written just before.
  - Cause, a hypothesis not yet verified: on GH200 the HBM is a NUMA node, so the page cache from the merge's writes
    landed in GPU memory.
  - Fix: `eval_record.py` runs the merge under `numactl --preferred=0`, and logs GPU and NUMA free memory before vLLM
    starts.
  - Resubmitted as 1101538. That job reuses the merged directory already on scratch, so the fix isn't exercised until
    the next merge.
- **r3-27b-a** is at step 519 of 776 and will hit its 24 h limit at about step 570. Chunk b then has to queue on gh.

### 2026-10-26: r3-27b at 50% (step 388, about 6.2k papers), val-sub160

Rerun 1101538 ran all 160 papers in 31 min, with no truncations and no invalid JSON.

| metric | 4B full | 27B ck25 | 27B ck50 | ck50 − ck25 | ck50 − 4B full |
|---|---:|---:|---:|---|---|
| linked cell F1 | 0.666 | 0.665 | **0.745** | +0.080 [+0.032, +0.134] | +0.079 [+0.025, +0.138] |
| strict cell F1 | 0.594 | 0.596 | **0.690** | +0.094 [+0.039, +0.155] | +0.095 [+0.032, +0.163] |
| model | 0.828 | 0.826 | **0.898** | +0.072 [+0.029, +0.116] | +0.069 [+0.026, +0.115] |
| kind | 0.816 | 0.806 | **0.860** | +0.054 [+0.011, +0.099] | +0.044 [−0.013, +0.108] |
| regions F1 | 0.789 | 0.835 | 0.759 | −0.076 [−0.161, +0.007] | −0.030 [−0.109, +0.044] |
| schema-valid | 0.994 | 0.975 | 0.994 | | |

- **Steady improvement:** the 27B keeps improving between checkpoints and now beats the 4B full run.
- **Regions** dropped from ck25 to ck50, though not significantly; watch it at ck75.

**GH200 NUMA layout (compute node, from the new logging).** Node 0 is CPU memory: 120.6 GB, 111.9 GB free. Node 1 is
the HBM: 97.3 GB, all free at job start. The HBM is a NUMA node, which is consistent with the page-cache hypothesis,
but neither job merged, so the hypothesis itself is still untested.

### 2026-10-26: adjudication of r3-27b-ck50 vs Astra (val-sub160, 25 papers, seed 11)

Full write-up: `vista/ADJUDICATION-r3-ck50.md`.

- **Agent's tally, 46 items:** Astra right 13, model right 5, both acceptable 26, unclear 2.
- **My check of the 5 "model right" items** against the text and records: 4 hold and 1 flips.
  - f5Zu2kabHbnj flips to Astra. The key's only point (45 −52 32) sits in the Words > Partial row of Table 3, while the
    coordinate parser named the key "Initial > Final", a contrast with no peaks. Astra's "Words > Partial" analysis is
    the right one for this point. This is also a parser naming error.
  - 6teXKZNVzJ62 holds: the gold lacks it. Astra linked it from the coordinate set's side, and a pruning bug dropped it.
    See "pruning bug, one-sided analysis link" below.
  - BSGoTkY8SLJN: Astra never wrote the Cox regressions, and its DMN analysis name contradicts its own cells.
  - 32WaBSTPCZYm: the model's UCS expectancy term matches the text.
- **Corrected tally:** Astra right 14, model right 4, both acceptable 26, unclear 2. The model is right in 4 of 18
  decided items (22%); for h34-10k the figure was 3 of 27 (11%).
- **The both-acceptable items are 57% of disagreements here, against 36% for h34-10k.** 18 of them are naming or detail.
  The agent checked each quote against its offset in code; I did not re-judge these items.
- **Where the 27B is wrong:**
  - the direction or comparison of a contrast the table defines (flipped AD vs elderly; group × time instead of time
    within a group);
  - directions asserted for null correlations or for an omnibus interaction;
  - covariates or analyses the paper doesn't support;
  - dropped held cells and SVC analyses;
  - the first-level model chosen where the table comes from the group level.

### 2026-10-26: orphan coordinate sets (points, no analysis) in the training targets

The user's rule: an analysis that reports coordinates belongs on its coordinate set, so Astra leaving one unlinked is an
Astra error. Scripts: `migrate/orphans.py`, `migrate/orphan_roles.py`; output on beast in `migrate/audit/`.

- **Scale (train):** 7,742 of 60,200 coordinate sets with points have no analysis, in 3,173 of 12,454 papers. Val has
  296 sets in 115 papers; test has 277 in 122.
- **All 128,445 of Astra's raw analyses with no key were pruned**, as the rule requires.
- **By the role Astra gave the set:**

  | role | sets | notes |
  |---|---:|---|
  | anchor | 3,995 | ROIs and seeds; correctly no analysis |
  | localization / reference | 164 | correctly no analysis |
  | not in Astra's coordinate sets | 1,324 | mixed; 431 look like citations, ROIs or networks |
  | **result** | **2,259** | in 1,264 papers; 324 named as repeats of a table result ("also reported in Table 2"), the rest look like results Astra left unlinked |

- **Effect on training and scoring:** about 1,900 result sets in roughly 10% of training papers teach the model to leave
  a listed result without an analysis. Val gold has the same gap, so a model that links them correctly is scored as
  wrong. The 6teXKZNVzJ62 adjudication item is one case.

### 2026-10-26: pruning bug, one-sided analysis link (fixed in data v6, `migrate/train-v6`)

**Correction.** The earlier entries say Astra left the 6teXKZNVzJ62 analysis unlinked. That is wrong. Astra linked it from
the coordinate set's side (`CoordinateSet.analysis = ana_s224`). The schema says this join is "the same join
`Analysis.source_table_analysis` makes from the other side, here for sets read from the text as well".

**Bug.**
- `prune.py` kept an analysis only through its own `source_table_analysis`.
- It then cleared the set's `analysis` link because the analysis had been "removed".

**Sets with points and no forward-linked analysis, train** (`migrate/links.py`):

| what the set links to | sets | status |
|---|---:|---|
| an analysis kept on another key | 1,581 | a text repeat or second set; Astra's form is one analysis that both sets name; fine |
| **an analysis with no key of its own** | **619** | **the bug: 652 analyses dropped in 431 papers** |
| nothing | 59 | real Astra omissions |

**Fix.** A keyless analysis named by a pointed set is kept and takes that set's key: generated, evidence `not_found`,
the first set in record order when several sets name it. `prune.py` now also reports every cleared set link. After the
fix no pointed set loses its link; 73 point-less ones do, as intended. `review_train.py` invariants on v6: all zero.

**Impact of v6 vs v5:**

| split | papers changed | analyses added | other entities restored |
|---|---:|---:|---|
| train | 431 (3.5%) | 652 (+1.0%) | 402 models, 196 regions, 217 inference settings, 64 measures, 38 groups, 33 assessments, 11 tasks |
| val | 11 | 17 | |
| test | 22 | 28 | the paper list is unchanged; only these targets change |

- **48 papers** that previously had no coordinate analyses become usable. None is in a split.
- **Rescoring every run on v6 val gold** changes linked cell F1 by at most 0.003 and model by at most 0.005, for every
  run. The bug did not distort any comparison.
- **Effect on training:** for about 1% of analyses, the old targets taught the model to leave a listed text result
  without an analysis. The likely effect on the trained models is small, and it has not been measured.
- **Use of v6:** the next data build only. Runs in flight stay on v5.

### 2026-10-26: data v6 built (beast `vista-out/v6/`, splits `migrate/splits-v6/`)

- **Records:** `migrate/train-v6`, pruning with both link directions.
- **Splits:** `splits.py --extend splits`.
  - val and test are unchanged; their hashes equal v5's.
  - train gains 35 papers: 12,489, sha256 `364c6c07…`. The other 13 of the 48 newly usable papers are on the flagged
    list; none is linked to a val or test paper.
- **Targets** (`tasks.py build --records train-v6/records --splits splits-v6`):
  - train.jsonl: 12,489 rows, sha256 `2663ded6…`; target tokens 63.42M → 63.86M (+0.7%).
  - val.jsonl: 481 rows, sha256 `8ec299cd…`.
  - Prompts are identical to v5 for every v5 paper. Only the targets of 431 train and 11 val papers change.
  - `val.bodies.jsonl` is unchanged: the prompts and parse keys are the same.
- **Checks:** the compact round trip on 16,764 records is all valid. `review_train` invariants are all zero.
  `verify_schema gold` was not rerun: it has no `--records` option. It only matters for guided decoding.
- **Staging to Vista:** not done. The ssh master to Vista had closed.

**Target size if the dropped fields were added** (300 train records, characters relative to the compact target):
- `value_source`: +42%
- `unreported_reason`: +3%
- evidence: +602%, of which the quote text alone is +416%; analyses +165%, models +134%, groups +80%

### 2026-10-26: feasibility of evidence by unit index (`vista/evfeas.py`; 2,000 v6 train papers, seed 7)

**Design.** A second model takes the paper text with units tagged `[S1] …` and the record without evidence. It returns
`{slot path: [unit indices]}`. Units are sentences in prose; each table row, caption or other short line is one unit.

**Token budget** (NuExtract3 tokenizer):

| | median | p90 | p99 |
|---|---:|---:|---:|
| tagged text (markers add 1.5k / 2.3k / 3.4k) | 11.5k | 18.0k | 26.2k |
| record in | 5.3k | 9.7k | 17.2k |
| output | 5.8k | 11.5k | 23.9k |
| **total** | **23.1k** | **37.3k** | **61.5k** |

The maximum is 108.6k. 48 of 2,000 papers (2.4%) are over 49,152 tokens.

**How Astra's spans map to units:**
- 1,004,444 spans, none outside a unit.
- 98.2% fall within one unit; p99 is 2 units.
- A span covers ≥ 99% of its units' text at p10, so Astra's evidence is already whole sentences and rows.
- Units per slot: median 1, p90 3, p99 7. Evidence sets per slot are effectively always 1 (0.02% have more).
- 608,855 slot paths, all resolving back to exactly one slot.

**Reading:** sentence-index evidence loses almost nothing relative to Astra's spans. The output is the cost: it is
about as long as the record, because every path is spelled out. A shorter path form, or the record's own shape, would
cut it.

### 2026-10-26: H35 result, full-parameter vs LoRA (NuExtract3 4B, 3k papers, val 481, JSON mode)

Eval 1101539: 481 of 481 stopped normally, with no loops and no invalid JSON. The smoke-test looping came from 3
unwarmed steps.

Paired differences, full minus LoRA (h34-3k):

| metric | LoRA | full | difference [95% CI] |
|---|---:|---:|---|
| linked cell F1 | 0.621 | 0.633 | +0.013 [−0.019, +0.045] |
| local cell F1 | 0.581 | 0.606 | +0.025 [−0.003, +0.056] |
| model | 0.786 | 0.800 | +0.013 [−0.010, +0.039] |
| kind | 0.797 | 0.814 | +0.017 [−0.007, +0.041] |
| entity F1, all | 0.832 | 0.843 | **+0.011 [+0.003, +0.019]** |
| groups | 0.939 | 0.951 | **+0.012 [+0.002, +0.023]** |
| regions | 0.722 | 0.746 | +0.024 [−0.004, +0.055] |

**Conclusion (H35): not supported by the pre-set rule.** The CI on cell F1 includes 0, so LoRA stays. Full fine-tuning
is ahead on every metric, by about +0.01 to +0.02, but only the entity gains are significant. The gap is much smaller
than the 27B's gain over the 4B (+0.08). Cost: 2 chunks, about 2.3 h of training, the same order as LoRA. One seed.

**Other status:**
- r3-27b-a soft-stopped at step 566 of 776 (exit 75). Chunk b, 1095924, is pending on gh by priority.
- Data v6 is staged on Vista at `data/v6/`; the hashes match beast.

### H39 (2026-10-26): evidence by unit index, a second model (NuExtract3 4B LoRA, 1k papers)

**Question.** Given the tagged text and the evidence-free record, can a fine-tuned NuExtract3 return Astra's evidence
as unit numbers?

**Data.** ev1, built by `vista/evidence.py` from v6 records with v6 splits.
- train: 12,489 rows, sha256 `40d7eccf…`. 84 are over 49k tokens and are dropped by the trainer.
- val: 481 rows, sha256 `fe88b864…`, plus `val.bodies.jsonl`.
- Input: the instruction, the tagged text, and the compact record. The compact record passes an assertion that it
  equals `to_compact(record)`.
- Output: the record's own structure, keeping only slots with evidence, each slot holding its unit numbers.

**Output-form comparison** (300 train papers, output tokens, median / p90 / p99):

| form | median | p90 | p99 |
|---|---:|---:|---:|
| record shape | 2.9k | 5.8k | 11.3k |
| spelled-out slot paths | 4.4k | 8.9k | 17.9k |

The record shape was chosen. Full sequences: median 19.6k, p90 30.6k, p99 46.0k.

**Metric.** Micro precision / recall / F1 over (slot path, unit) pairs, against Astra's units, with paper bootstrap.
Also reported: exact slot match, any-overlap slot match, and the same broken down by collection. Scorer check on
val-sub160: gold as the prediction scores 1.0, an empty prediction 0.0.

**Bug found while building.** `Table.title` / `Table.description` are excluded per item, and my mirror first applied
the exclusion to the list. The assertion caught it on Zum6LhmfVvPj; fixed before any data was used.

**Run.** `ev1-1k`: 1,000 papers from the fixed shuffle, LoRA r64 at lr 2e-4, as h34-1k.
- Train: job 1102134, gh-dev.
- Eval: job 1102135 (afterok), JSON mode, on val-sub160 (`configs/ev1-1k-eval.json`).

**Result** (2026-10-26). Training: 63 steps on 995 papers (5 over 49k dropped) in 0.90 h, peak 35.5 GiB, loss per
token 0.31 → 0.06–0.10. Eval: 160 papers in 296 s of generation. 1 paper (GFpBFSeX83Wh) ran to the 20k-token limit
counting unit numbers upward past the paper's units (…4329, 4330); it has no prediction and scores 0. Scored on beast
(`evidence.py score … --records train-v6/records --splits splits-v6`, 2,000 bootstraps;
`vista-out/ev1/runs/ev1-1k/score.json`):

| metric | value |
|---|---:|
| precision | 0.718 |
| recall | 0.726 |
| F1 [95% CI] | **0.722** [0.707, 0.736] |
| slot exact match | 0.582 |
| slot any overlap | 0.867 |
| per-paper F1, median / p10 | 0.751 / 0.622 |

By collection (F1, gold pairs): hypothesis 0.98 (226), devices 0.95 (344), preprocessings 0.91 (2,367), acquisitions
0.86 (2,478), external_datasets 0.85 (102), groups 0.84 (11,936), assessments 0.84 (1,178), result_reporting 0.79
(225), tasks 0.74 (5,445), model_estimations 0.74 (16,598), inference_settings 0.73 (2,439), regions 0.71 (4,826),
measures 0.68 (1,652), design 0.67 (2,969), description 0.66 (900), coordinate_sets 0.64 (4,909), **analyses 0.63**
(23,023).

- **Units mostly found:** 87% of gold slots get at least one right unit, but only 58% get exactly the right set.
- **Weakest where the gold is largest:** analyses and coordinate_sets hold a third of the gold pairs and score lowest;
  their evidence is mostly table rows, where the right row among many similar ones has to be picked.

### 2026-10-28: r3-27b at 75% (step 582, about 9.3k papers), val-sub160

Eval 1102765 (gh-dev, 41 min) ran all 160 papers in 30 min of generation, with no truncations and no invalid JSON.
Scored on beast as ck50 (`score.py run … --split val-sub160.txt --boot 2000`; `vista-out/v5/runs/r3-27b-ck75/`).

| metric | 4B full | 27B ck50 | 27B ck75 | ck75 − ck50 | ck75 − 4B full |
|---|---:|---:|---:|---|---|
| linked cell F1 | 0.666 | 0.745 | **0.758** | +0.013 [−0.022, +0.045] | +0.092 [+0.037, +0.150] |
| strict cell F1 | 0.594 | 0.690 | 0.684 | −0.006 [−0.051, +0.033] | +0.090 [+0.034, +0.152] |
| model | 0.828 | 0.898 | 0.865 | −0.033 [−0.066, +0.001] | +0.036 [−0.006, +0.087] |
| kind | 0.816 | 0.860 | 0.840 | −0.020 [−0.053, +0.012] | +0.024 [−0.040, +0.095] |
| entity F1, all | 0.877 | 0.883 | **0.905** | +0.023 [+0.003, +0.044] | +0.029 [+0.013, +0.046] |
| regions F1 | 0.789 | 0.759 | **0.847** | +0.088 [+0.010, +0.174] | +0.058 [+0.001, +0.123] |
| schema-valid | 0.994 | 0.994 | 0.994 | | |

- **Plateau on cells:** ck50 → ck75 changes no cell metric beyond its interval; the gain over the 4B full run holds.
- **Regions recovered:** the ck25 → ck50 drop reversed, and entity F1 rose significantly.
- **Model slipped:** −0.033, at the edge of its interval; check at the final checkpoint.

### 2026-10-28: r3-27b-full training done; final eval split in three

Chunk b (1095924) finished all 776 steps at 03:45 (9.0 h, exit 0; 12,411 papers, 43 over 49k dropped; peak 77.7 GiB).
Chunk c (1095972) was cancelled, since it had nothing left to do.
- **The single 6 h full-val eval (1095973) was replaced:** it would have started with no queue age, behind 505 gh jobs.
- **Its replacement:** three gh jobs of 1:20, with shorter jobs more likely to start in gaps (backfill): 1106375-7,
  `configs/r3-27b-final-p{1,2,3}-eval.json`.
- **Splitting:** the 481 val papers are split by `n_input_tokens` into `data/v5/val-part{1,2,3}.txt` (161/160/160
  papers, about 2.49M input tokens each).
- **Separate run names:** each part has its own run name, because the merged model directory is keyed on the run.
- Same SU cost as one job; each part merges the adapter on its own node (about 10 min each).

### 2026-10-28: the 27B on Lonestar6, A100 split across 2 GPUs (`tacc/ls6-eval.sbatch`)

Load test of r3-27b-full `adapter-final` on LS6 `gpu-a100-dev`: 24 longest papers of val-sub160, JSON mode, 24 workers.
- **Setup:** the base model was downloaded to LS6 (same revision and files as Vista's) and the adapter was merged on the
  node's CPU in 15 min (kept on `$SCRATCH`).
- **First failure (3538672):** `GPUS=1,2` passed inside `--export`, which splits on commas, so the job saw one GPU.
  `eval_record.py` now checks `tp` against `CUDA_VISIBLE_DEVICES` before merging.
- **Second failure (3538683):** the model loaded (25.2 GiB per GPU), then vLLM's custom all-reduce failed on the PCIe
  A100s (`custom_all_reduce.cuh:164 'invalid argument'`).
- **Third run (3538699, `vllm_args: ["--disable-custom-all-reduce"]`):** worked. vLLM was up in 440 s; 24/24 stop,
  no invalid JSON, 1,316 s, 134 output tok/s.
- **Memory:** 9.5 GiB KV per GPU, 283k tokens in all, so 5.75 concurrent 49k-token requests (Vista GH200: 2.0M,
  41x). The cache, not the GPUs, limits the rate.
- **Speed:** on the same longest-first papers Vista reaches about 115 tok/s after 5 papers and 418 tok/s over 160, so
  LS6 A100 x2 is roughly a third of a GH200. Full val (481) would take about 5 h: too long for `gpu-a100-dev`'s 2 h,
  and `gpu-a100` had 523 jobs waiting.
