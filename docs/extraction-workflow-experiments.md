# Searching the extraction workflow space

`pondie.benchmark.scoring` gives a per-record score. This file is the design for using it to
choose a pipeline shape: what the candidate shapes are, which variables actually distinguish
them, how many runs it takes to tell them apart, and — first — what is currently wrong with
the measurement substrate, because that bounds everything else.

Metric definitions: [extraction-comparison-metrics.md](extraction-comparison-metrics.md).
Text preprocessing on this same substrate, and the evidence that §5.8's numbers no longer
reproduce:
[text-preprocessing-experiments.md](text-preprocessing-experiments.md).
The pipeline being varied: `pondie extract`, and
[baseline-run.md](baseline-run.md) for how its present shape was arrived at.

---

## 0. What the measurement can currently support: one paper

**`data/gold/xevP8UDRAVh9.extraction.json` is the only human-verified record.** Nothing else
in the corpus has been checked, and the experiment is bounded by that far more tightly than
by cost.

It is tempting to manufacture more. `corrections/<id>.corrections.json` applies to the
model's own output, and for five papers those corrections still replay cleanly, so a
"gold" record can be produced mechanically. **This does not work**, and the numbers say why:

| paper | corrections | composite | direction F1 |
|---|---:|---:|---:|
| `krv8mTKTUHSp` | 2 | 100.0% | 100% |
| `aVGe9BmFTMDR` | 35 | 99.7% | 100% |
| `6oTrCJA43Jcd` | 39 | 98.9% | 100% |
| `QQCjAAT6SwwQ` | 6 | 97.1% | 94.1% |
| `SULKxviGFurw` | 24 | 89.2% | 81.8% |
| **`xevP8UDRAVh9`** (verified) | — | **44.0%** | **0%** |

A record derived by correcting a pipeline's own output is that pipeline plus a handful of
fixes, so scoring the pipeline against it measures how many corrections someone made, not
how much of the paper was read correctly. The five derived rows sit 45–56 points above the
one verified paper, and the gap is the artefact, not a finding. They were generated, measured,
and deleted; the table is kept only so nobody repeats the idea.

The size of the effect is measurable a second way. Re-running the baseline pipeline on
`krv8mTKTUHSp` produced a fresh record scoring **94.7%** against the same derived gold that
the on-disk record scores **100%** against. Roughly five points of that 100% is the record
being gold's own parent.

This is `baseline-run.md`'s warning — "adjudicating model output measures precision only, and
nothing here measures what the extraction *missed*" — with a number on it.

### What follows for the design

1. **n = 1 paper, 12 direction cells.** No factorial design is estimable. A screening matrix
   over five factors would produce eight numbers with no error bars, and reading a rank order
   off them would be indistinguishable from reading tea leaves.
2. **The first measurement is not a comparison: it is the run-to-run spread of one
   configuration.** Until the same-config standard deviation is known, no between-config
   delta on one paper can be interpreted. This is what `sweep_extractions.py --replicates`
   exists for, and it is the only sweep worth running today.
3. **Getting more verified gold is the highest-value work available**, ahead of any pipeline
   change. Twelve to fifteen adjudicated papers turn every hypothesis below from an argument
   into a measurement.

The cheapest route to that gold: `7HPLh5nJzmP5`, `ngDTY5BgJUuX`, `TgcHKMRfrVog`,
`84rGLhCbUJTh`, `DTpwdoGbjqsq` and `YwwKWoEFwY3G` already have correction files whose paths
have gone stale (20, 15, 9, 1, 2 and 2 failed ops). Repairing those paths and then *verifying
the result against the paper* — the second half being the part that matters — is far less work
than adjudicating from scratch. But it is human reading time, and no amount of model spend
substitutes for it.

---

## 1. What the one clean paper says the problem is

Every failure on `xevP8UDRAVh9` is structural, not a reading failure. The model knew what the
paper said; it assembled it wrong.

| symptom | what actually happened |
|---|---|
| direction F1 **0%** | the sign was attached to `term_perfusion_condition`; gold puts it on `term_gray_matter_volume` and *holds* the perfusion level |
| 4 cells where gold has 12 | one cell per analysis instead of the two a held-level correlation takes |
| `Region` recall **0%** | the entities pass emitted no regions, so `Analysis.regions` had nothing to point at |
| `Task` + 2 `Condition` invented | the entities pass created a paradigm the study does not have |
| 2 of 6 analyses missing | the VBM contrasts were never enumerated |

The middle three share one cause: **the entities pass runs before the analyses pass and has
to guess what the analyses will need.** It guessed short on regions and long on tasks. The
entities-mode prompt already contains a paragraph warning about exactly the region failure
("A paper that ran any ROI, seed, mask or parcel analysis and emits no `regions` leaves the
analyses pass with nothing to point at") — and the failure happened anyway. That is evidence
that this is a *workflow ordering* problem and not a prompt-wording problem, which is
precisely what makes it worth an experiment rather than an edit.

The leading hypothesis is **H1: demand-driven ordering beats supply-driven ordering.**
Extract analyses first; let each analysis name the entities it needs; resolve those into
records afterwards. Nothing gets created that no analysis asked for, and nothing an analysis
asked for is missing.

---

## 2. The cost model, measured

Three facts, measured on this repo, that determine which shapes are affordable.

**The paper is a fifth of the prompt.**

| prompt component | chars | share |
|---|---:|---:|
| `extraction-readme.md` (conventions) | 46,982 | 32% |
| `representing-models.md` §5 (worked models) | 32,472 | 22% |
| rendered schema, entities mode | 62,075 | 42% |
| the paper itself (`xevP8UDRAVh9`) | 27,567 | 19% |

(Entities-mode prompt: 147,644 chars total. Analyses mode: 122,494, with a 33,442-char
schema.)

**Prompt caching is live on the gateway.** A second call with the same 9,302-token prefix
reported `cached_tokens=9299`. Verified directly, not assumed.

**A single class needs a tiny schema slice.** `ModelTerm` renders in 9,543 chars, `Group` in
6,249, `Region` in 2,823. All 23 entity-mode classes rendered one at a time sum to 57,523
chars — *less* than the 62,075 of the monolithic render.

Together these overturn the obvious objection to fine-grained decomposition. Splitting the
entity pass into 23 per-class calls does **not** cost 23× — the conventions, worked models
and paper text are one cached prefix, and each call adds only its own 2–9k-char slice plus
its output. The requirement is that **the prompt be reordered so the invariant prefix comes
first**: conventions → worked models → paper → *then* the per-class schema slice. The current
order puts the schema before the paper, which is exactly wrong for caching a per-class sweep
over one paper.

That reordering is a precondition for testing axis B below, and it is cheap.

---

## 3. The axes

The four shapes in the original question are not four rival pipelines; they are points on
three orthogonal axes, plus effort allocation. Treating them as axes is what makes a
screening design possible.

### Axis A — anchor and direction of travel

*What decides the set of things that exist.*

| | shape | rationale / risk |
|---|---|---|
| **A0** | **entity-first** (current): entities → analyses link to them | supply-driven. Risk is exactly the observed one: guessing the inventory wrong in both directions. |
| **A1** | **analysis-first**: analyses pass emits inline entity *stubs* (`{"kind": "Region", "label": "frontal lobe"}`); a resolve pass expands stubs into records | demand-driven. Every entity has a caller; every caller has an entity. Risk: an analysis-blind entity (a Group's demographics) has no stub to hang on, so it needs a sweep-up pass. |
| **A2** | **table-anchored analysis-first**: A1, plus the stage-1 coordinate table *rows* in the prompt, not just the caption digest | the analyses pass currently cannot see the coordinates it is describing. Highest-information version of A1. Risk: rows are long; may crowd out reasoning. |
| ~~A3~~ | ~~entity-first + prune~~ | **tested and dropped — see below** |

**Pruning unreferenced entities is not the cheap half of this hypothesis; it is actively
wrong.** The idea was that entities no analysis references are the hallucinated ones, so
dropping them is a free precision win. On the one verified paper the opposite holds. Six
candidate entities are unreferenced — two `Timepoint`s, `preproc_vbm`, `model_glm_paired`,
`term_treatment_condition` and `term_gray_matter_volume` — and **all six are correct and
present in gold**, including the term that should have carried the direction. Meanwhile all
three hallucinated entities (`task_drug_perfusion`, `condition_heroin`, `condition_placebo`)
*are* referenced, because the analyses that invented them also linked to them.

The inference runs the other way. Those six are unreferenced precisely *because* the pipeline
dropped the two VBM analyses and the gray-matter cell that would have referenced them.
**Unreferencedness is a symptom of a missing analysis, not a marker of a spurious entity** —
which makes it a good signal to act on, just not by deleting:

| | shape |
|---|---|
| **C5** | **back-pressure**: hand the model the entities nothing references and ask what analysis uses them |

That is a one-call pass aimed straight at `analysis_missed`, and it is only visible as an
option because the prune idea was checked before it was built.

### Axis B — decomposition granularity

*How much attention each item gets.*

| | shape | cost with caching |
|---|---|---|
| **B0** | one call for all entities (current) | 1 call |
| **B1** | one call per entity class | ~12–23 calls, ~1 cached prefix |
| **B2** | one call per entity instance | ~25–60 calls |
| **B3** | one call per analysis (analysis side) | ~4–20 calls |

B1 is the user's "one category at a time"; B2/B3 the "one instance at a time". The prediction
is that B has a large effect on *field* metrics (more attention per instance → fewer dropped
slots) and a small effect on *direction*, because direction is a reasoning error not an
attention error. B3 is the exception: it gives each contrast its own call, and the observed
"one cell where gold has two" failure is plausibly an attention error.

### Axis C — reconciliation

*What checks the assembly.*

| | shape | cost |
|---|---|---|
| **C0** | none (current) | 0 |
| **C2** | whole-record critic, one pass | 1 call |
| **C3** | targeted re-ask on high-stakes slots only: `Cell.term`, `Cell.direction`, `Analysis.regions` | 1 call per analysis |
| **C4** | self-consistency: *k* independent analyses passes, majority vote per cell | *k*× the analysis side |
| **C5** | back-pressure on unreferenced entities (axis A) | 1 call |

(C1, pruning, was tested and dropped — see axis A.)

C3 is the one aimed straight at the weighted metric. C4 is the standard way to buy accuracy
on a discrete label and `Cell.direction` is a five-way categorical — but it triples cost, so
it belongs in a follow-up on the winner, not in screening.

**How many judge steps?** The honest prior is *one, targeted*. A whole-record critic (C2) has
to re-derive the schema rules to say anything, which is what the 80%-scaffolding prompt
already costs; a critic asked only "does this cell's sign belong on this term, given this
sentence?" is a question with a short answer and a checkable ground. Screening should
establish whether C3 pays before anyone builds C2 or C4.

### Axis D — reasoning effort allocation

| | shape |
|---|---|
| **D0** | low everywhere (current) |
| **D1** | high everywhere |
| **D2** | graded: low for descriptive slot-filling, high for enumeration and effect/cell construction |

`baseline-run.md` found low effort sufficient and that "the corrections are conventions and
source conflicts, which more thinking about the same prompt would not fix." But that was
judged on correction counts, before direction was measured. D2 is the sharper hypothesis: the
tasks that are *structural decisions* (which analyses exist; which term carries the sign) are
the ones effort should buy, and slot-filling is not. If D1 ≈ D0 but D2 > D0, effort is worth
spending selectively — which is the actual question.

### Not an axis: the evidence pass

Stage 4 (`evidence/quote.py`) adds 4–7 calls per paper and `pondie.benchmark.scoring` does not
score evidence spans at all. It is switched **off** for every sweep run. That is a ~70%
saving on call count for zero metric effect, and it should not be confused with a finding.

---

## 4. The design, at n = 1

Full factorial is 4 × 4 × 5 × 3 = 240 configurations, and with one paper none of them are
distinguishable. The design has to be staged against how much gold exists, not against how
many ideas there are.

### Stage 0 (runnable today) — how noisy is one configuration?

Run the *same* configuration k times on `xevP8UDRAVh9` and measure the spread.

    python sweep_extractions.py --configs baseline effort_high \
        --papers xevP8UDRAVh9 --replicates 4 --jobs 4

This is the gating measurement and it decides what the next year of this work is worth
doing. Two outcomes, both decisive:

- **Spread is small** (say sd < 3 points of direction F1). Then a single paper can resolve
  large config effects, and the axes below can be screened one at a time on it — as a filter
  for what deserves gold, not as a conclusion.
- **Spread is large** (sd comparable to the config deltas being chased). Then no
  single-paper comparison means anything, every number in a screening matrix would be noise,
  and the only rational spend is on adjudicating more papers.

The same run also gives the effort axis (D) for free, since D0 vs D1 is flag-only.

### Stage 1 (needs ~12 verified papers) — screening, 8 runs

Binarise five factors and use a resolution-III fractional factorial (2^(5−2)), estimating
all five main effects in eight runs with two-factor interactions aliased onto them:

| factor | − | + |
|---|---|---|
| **A** anchor | entity-first (A0) | analysis-first (A1) |
| **B** granularity | monolithic (B0) | per-class (B1) |
| **C** reconcile | none (C0) | targeted re-ask (C3) |
| **D** effort | low (D0) | graded (D2) |
| **E** table rows | digest only | rows visible (A2's ingredient) |

Aliasing is acceptable because the aim is to *rank* factors for follow-up, not to estimate a
response surface.

### Stage 2 — follow-up, ~6 runs

Full factorial on the two factors screening ranks highest, plus self-consistency (C4, k=3)
on the best cell, plus B2/B3 if B screened positive.

### Blocking and the unit of analysis

Paper difficulty dominates everything, so **every comparison is within-paper** and the report
is a delta matrix, never a bare mean of means. At n = 1 that collapses to: replicates within
one paper, and the same-config sd as the yardstick every delta must clear.
`sweep_extractions.py` prints deltas in units of that pooled sd for exactly this reason.

For the headline the unit is the **direction cell**, not the paper — `xevP8UDRAVh9` has 12,
and pooling cells across replicates is the only way to get a usable denominator right now.
For entity and relationship metrics the unit is the entity and the edge, pooled micro.

Report effect sizes with intervals. A p-value on four runs of one paper would be theatre.

### Cost

Per paper per config with evidence off: 2 calls at B0 (~35k input tokens each), ~14 at B1
over a shared cached prefix; about 40 seconds of wall clock. At this scale **neither spend
nor wall-clock is the binding constraint — verified gold is**, which is the whole point of §0.

---

## 5. What stage 0 found

Four replicates each of `baseline` (low effort) and `effort_high` on `xevP8UDRAVh9`, evidence
pass off. Eight runs, ~12 minutes.

| metric | baseline mean | sd | range | effort_high mean | sd | Δ |
|---|---:|---:|---|---:|---:|---:|
| composite | 42.4% | 13.6 | 25.9–59.0 | 45.6% | 7.1 | +3.2 (0.3 sd) |
| direction F1 | 10.0% | 20.0 | 0.0–40.0 | 10.0% | 12.8 | +0.0 |
| entity F1 | 77.7% | 19.5 | 48.8–91.2 | 85.3% | 3.2 | +7.6 (0.5 sd) |
| relationship F1 | 47.6% | 24.9 | 10.5–64.5 | 59.6% | 5.9 | +12.0 (0.7 sd) |
| field accuracy | 85.8% | 5.3 | 81.5–93.5 | 80.8% | 1.4 | −4.9 (−1.3 sd) |

Pooled over all 48 cell-instances, both configurations are identical: tp 3, fp 9, fn 45.

### 5.1 The run-to-run spread swamps every effect worth chasing

Direction F1 has a standard deviation of **20 points on a mean of 10**; relationship F1, 25
points. The same configuration, the same paper, the same prompt. **No single-paper comparison
between configurations is interpretable**, which is stage 0's outcome (b), and it settles what
to do next: adjudicate more papers before running any screening matrix.

The spread is real pipeline variance, not scorer instability. The four baseline runs differ in
what they *contain*: 0–2 `ModelEstimation`s, 0–1 `Region`s, 0–2 `Measure`s, 0–3 `ModelTerm`s.

### 5.2 One run in four emitted an empty entity payload

`baseline#2` returned a structurally valid payload — correct top-level keys, `study` filled
with description, hypothesis and a complete `design` including both arms and both timepoints
— and **every entity list empty**. 3,149 bytes against 13,613–16,797 for its siblings. The
call finished normally (`[stop]`, 3,543 output tokens); nothing was truncated, nothing was
malformed, no validator objects to a record whose entity lists are legally empty.

This is a larger effect than any axis in §3, and it has a cheaper fix than any of them: **a
plausibility gate with retry.** A paper with coordinate tables and a Methods section that
yields zero groups, zero acquisitions and zero model estimations has failed, and that is
decidable without a model. One extra call on the ~25% of runs that trip it would remove the
dominant source of variance in the table above.

That is also the honest answer to "how many judge steps should there be": **the first one
should not be a judge at all.** A deterministic post-condition on each stage's payload costs
nothing and catches a failure no LLM critic was going to be pointed at.

### 5.3 Effort buys consistency, not accuracy

High effort left direction F1 exactly where it was — the pooled cell counts are identical —
cost 3.6× the wall clock (142s vs 38s per paper), and *lowered* field accuracy by 1.3 sd. What
it did do is cut the spread: sd falls from 13.6 to 7.1 on composite, 19.5 to 3.2 on entities,
24.9 to 5.9 on relationships.

Effort does not affect correctness here, confirming `baseline-run.md`'s claim that "low
reasoning effort held up." It does affect *reproducibility*, and a pipeline
whose entity F1 ranges over 42 points run to run has a reproducibility problem worth paying
for. If the plausibility gate in §5.2 lands, it should be re-tested — the gate and high effort
may be two fixes for the same failure, in which case the cheap one wins.

### 5.4 Analysis recall has two independent ceilings, and the first is structural

Gold has 6 analyses. Stage 1 offers **4**. Every one of the eight runs emitted **3**.

The two analyses stage 1 never sees are `analysis_vbm_heroin_gt_placebo` and
`analysis_vbm_heroin_lt_placebo`, and gold records `tables: []` for both. They are the VBM
contrast between conditions; they are reported in the text and not in a coordinate table.
**A table-anchored enumeration cannot find them at any level of effort or reasoning.**

This is the sharpest result for the strategy question, because "parse tables first, then work
outward from the analyses" is the shape the pipeline already has, and on this paper it caps
analysis recall at 67% before the first entity is extracted. `baseline-run.md` predicted the
class of failure — "a tested null result... appears in no table, so stage 1 never surfaces it"
— but the case here is not a null result. It is a fully reported contrast with a direction,
worth 4 of the 12 gold cells, invisible for no reason other than where it was printed.

Table-anchoring is still the right *spine*: it is deterministic, it is the only source that
enumerates coordinates, and 4 of 6 is a good start. What it cannot be is the only source. Any
configuration aiming at full recall needs a **text-side enumeration pass unioned with the
table parse**, and reconciling the two is a new axis the original matrix did not have:

| | shape |
|---|---|
| **A4** | **dual-anchor**: stage 1 tables ∪ a text-side analysis sweep, reconciled before the analyses pass |

The second ceiling — 4 offered, 3 emitted — is the analyses pass dropping one it was handed,
reproducibly, in all eight runs. That one is a workflow or prompt defect and is what axes B3
and C3 were aimed at.

### 5.5 The direction failure is upstream of the cells: the model itself is wrong

The targeted cell re-ask (C3, ``recheck_cells` (removed; see what-was-removed.md)`) works. Handed the baseline's
one-cell contrasts, it rewrote all three into gold's shape — `held` on the condition, the
sign on the slope — and **every cell it could ground was correct**: direction accuracy
100%, kappa 1.0, no sign flips.

Its F1 is nonetheless 42.9%, but the cells do not explain the result:

| | candidate | gold |
|---|---|---|
| perfusion term | `"cerebral perfusion"`, **continuous**, no levels | `"perfusion condition"`, **categorical**, levels `heroin-associated` / `placebo-associated` |
| gray-matter term | `"gray matter volume"`, continuous | `"gray matter volume"`, continuous |

The gray-matter term aligns and all three of its signed cells are credited. The perfusion
term does not align, because it is *not the same term* — the entities pass modelled the
treatment condition as a continuous covariate rather than as a factor whose levels the arms
carry. A `held` cell on a continuous term is not even coherent: there is no level to hold.

The cell re-ask therefore tried to repair a contrast over a model that cannot express it.
**`Effect.cells` cannot be righter than `ModelEstimation.terms`**, and the term inventory is
decided in the entities pass — before any analysis has said what it needs. That is the
strongest evidence yet for demand-driven ordering (axis A1/A2): the model structure should
be determined by the contrasts that have to be expressed in it, not guessed ahead of them.

It also predicts a ceiling: on this paper, no configuration that keeps the current entity
pass can exceed ~50% direction F1, however good its cell logic, because half the gold cells
name a term the pipeline never creates.

Two defects in `recheck_cells.py` itself, found the same way and worth fixing before it is
judged: it returns `level: null` on categorical cells (the schema requires a level exactly
when the term declares them), and it accepts terms of the wrong `type` without objecting.

### 5.6 Eight configurations, two replicates, scored on the four table analyses

Scope: `--scope tables`, so gold is the 4 analyses a publication table reported (8 cells)
and the entities they reach. Direction pooled over all 16 cell-instances per configuration.

Scored with the structural matcher (`docs/extraction-comparison-metrics.md` §2). The
name-weighted matcher these runs were first scored with put baseline at 24.0% and
`recheck_cells` at 35.7%; both were undercounts caused by a term failing to align on its
name. The ranking survived, the gaps did not.

| config | P | R | **F1** | tp | fp | fn | per-replicate | degen | analysis recall |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **recheck_high** | 100.0% | 75.0% | **85.7%** | 12 | 0 | 4 | 85.7 / 85.7 | 0 | 75% |
| recheck_cells | 83.3% | 62.5% | 71.4% | 10 | 2 | 6 | 85.7 / 57.1 | 0 | 75% |
| baseline | 66.7% | 37.5% | 48.0% | 6 | 3 | 10 | **85.7 / 0.0** | 0 | 75% |
| effort_medium | 0.0% | 0.0% | 0.0% | 0 | 6 | 16 | 0 / 0 | 1 | 75% |
| effort_high | 0.0% | 0.0% | 0.0% | 0 | 6 | 16 | 0 / 0 | 0 | 75% |
| effort_graded | 0.0% | 0.0% | 0.0% | 0 | 6 | 16 | 0 / 0 | 0 | 75% |
| table_rows | 0.0% | 0.0% | 0.0% | 0 | 6 | 16 | 0 / 0 | 0 | 75% |
| no_stage1 | 0.0% | 0.0% | 0.0% | 0 | 0 | 16 | 0 / 0 | 0 | **0%** |

Four things are solid, and one headline is not.

**The stage-1 table anchor is load-bearing.** `no_stage1` collapses: analysis recall 0%,
relationship F1 0.0%, entity F1 −46 points. Without the listing the analyses pass does not
produce recognisable analyses at all. Whatever replaces the current shape must keep it.

**Analysis recall is pinned at 75% (3 of 4) in every configuration that has stage 1** — the
same missing one every time, `analysis_placebo_negative_correlation`, whose stage-1 entry
says "no significant". The pipeline reproducibly declines to emit an analysis that found
nothing, and that is a definitional decision worth making explicitly rather than leaving to
an unprompted inference.

**Reasoning effort alone makes direction worse, not better.** medium, high and graded all
score 0% where baseline scores 24%, and all three lose field accuracy (−6.5 to −10.1 points,
2–3 sd). One `effort_medium` run was degenerate. On this evidence effort spent on the
existing passes is not merely wasted, it is harmful — it elaborates the model
(`effort_high` invented five terms where gold has two) and elaboration moves it further from
gold.

**Feeding the coordinate rows in did not help.** `table_rows` scores 0%, with entity F1 down
14.5 points and a huge spread (64.9–92.0). More context, worse output.

**What the re-ask actually buys is reliability, not a mean.** The per-replicate column is the
result. Baseline is bimodal: one run emitted six cells in gold's held-plus-slope form and
scored 85.7%, the other emitted three signed cells on the condition term and scored 0. It
knows the right shape and produces it about half the time.

| run | cells emitted |
|---|---|
| `baseline#0` | 3: `(condition, positive) × 2`, `(condition, negative)` — wrong shape, 0% |
| `baseline#1` | 6: `(condition, held)` + `(perfusion, ±)` per analysis — right shape, 85.7% |
| `recheck_cells#0` | 6, right shape — 85.7% |
| `recheck_cells#1` | 6, right shape, but one analysis swapped which term is `held` — 57.1% |
| `recheck_high#0/1` | 6, right shape, no swaps — 85.7% both |

The re-ask raises the floor: it makes the two-cell shape reliable where baseline gets it by
luck. Effort on the re-ask then removes the residual held/signed swap. The remaining gap
between `recheck_high` and `recheck_cells` is 14 points with `recheck_cells` ranging
57.1–85.7, so it is **not** separable from noise at two replicates — a much weaker claim than
the 50-point gap the name-weighted matcher appeared to show, and the right one.

**The 85.7% is a ceiling set by analysis recall, not by cell logic.** Precision is 100%; recall
is 75% because three of four analyses are found. Every remaining direction error on this paper
is the missing `analysis_placebo_negative_correlation`.

### 5.7 Replicated: the re-ask, the zero-foci rule, and demand-driven ordering

Fifteen replicates of baseline and the cell re-ask; ten each of the 2x2 over the zero-foci
rule and demand-driven ordering. One paper, four table analyses, eight gold cells, so a run
is the independent unit and the run-level tests below are the conservative ones.

**A leak was found and the affected arm re-run.** `recheck_cells.py`'s instructions
contained a worked example lifted verbatim from this paper's gold record
(`"placebo-associated perfusion"`). Every contaminated run was discarded and the arm re-run
against a neutral example; `tests/test_prompt_leakage.py` now fails the build if any
distinctive value from any record in `data/gold/` appears in the static prompt. The clean
numbers came out *stronger*, so the leak was not what produced the effect -- but that was
luck, not method.

**The cell re-ask eliminates total failure without delivering a complete contrast set.**

| | baseline | recheck_cells | test |
|---|---|---|---|
| per-run F1 | `0 86 0 86 0 0 0 0 86 86 0 0 0 86 0` | `31 43 86 71 40 86 29 29 86 86 14 29 57 50 86` | |
| any correct grounded cell | 5/15 | **15/15** | z=3.87, **p=0.0001** |
| fully correct contrast set | 5/15 | 7/15 | p=0.46, n.s. |
| run-level dominance | | P(recheck > baseline) = 0.72 | Mann-Whitney **p=0.038** |

Baseline is all-or-nothing: it either builds gold's held-plus-slope shape or emits three
signed cells and scores zero. The re-ask never returns nothing, and never returns everything.
It is a floor, not a ceiling — which is the opposite of what two replicates suggested, and
the reason fifteen were worth running.

**The zero-foci rule and demand-driven ordering interact, and neither reading survives
alone.**

| config | pooled F1 | analysis recall | runs at F1 >= 0.5 |
|---|---:|---:|---|
| **analysis_first + zero-foci** | **80.0%** | 100% | 8/10 — **and all eight scored 100%** |
| analysis_first | 64.1% | 75% | 7/10 |
| baseline | 38.1% | 85% | 4/10 |
| zero-foci alone | **12.9%** | 100% | 1/10 |

The zero-foci rule *on its own makes things worse* — recall rises to 100% and direction F1
falls to 12.9%, precision to 18%. Told to emit the null-result analysis, the entity-first
pipeline emits it over a model that cannot express it, and the extra cells are wrong. Paired
with demand-driven ordering, where the model is built to fit the contrasts, the same rule is
worth 16 points. Running the combination alone would have credited the pair for an effect one
of them subtracts from.

**Every non-degenerate run of the best configuration was perfect.** Eight of ten scored 100%
direction F1 — all 8 cells, all 4 contrasts exact. The other two were not wrong: they hit the
empty-skeleton failure of §5.2, the `satisfy` pass returning a well-formed payload with every
entity list empty (`finish=stop`, no truncation). Conditional on the entity pass producing
anything at all, the configuration is 8/8.

That closes the loop on §5.2. The deterministic post-condition with retry — recommended there
on the strength of a 25% silent-failure rate, still unbuilt — is now the *only* measured
obstacle between this pipeline and a perfect direction score on this paper.

### 5.8 The post-condition, built and measured

> **The 92.3% below no longer reproduces, and nobody knows why.** The archived records still
> score 93.6% under today's scorer — 11 of 12 runs built the categorical term — but the same
> configuration re-run in August 2026 scores 23.1% over thirteen runs. The scorer did not
> change. The prompt material was the obvious suspect and has been tested: two trees
> differing only in `extraction-readme.md`, `representing-models.md` §5 and the extraction
> schema differ by +28.8 points at p = 0.23, which does not account for it. The untested
> candidates are the uncommitted `prompt/render.py` state those runs used and model-side
> drift. See [text-preprocessing-experiments.md](text-preprocessing-experiments.md) §5.4.
> Treat the numbers in this section as a record of what the pipeline did in August 2026, not
> of what it does now.

`extract_record.py --max-attempts N` now checks each pass's payload against a condition no
schema check covers, and retries naming the fault when it fails:

- an entity pass whose every list is empty
- an analyses pass that emitted no analyses
- a `demands` pass that declared no `required_entities`
- a `satisfy` pass missing an entity the `demands` pass declared — the dangling-reference
  case, and the strongest of the four, since it is checked against a contract rather than
  against a threshold

Ten replicates, demand-driven ordering plus the zero-foci rule, with and without it:

| | pooled F1 | P | R | degenerate runs |
|---|---:|---:|---:|---:|
| analysis_first_zf | 80.0% | 80.0% | 80.0% | 2/10 |
| **+ post-condition retry** | **92.3%** | 94.7% | 90.0% | **0/10** |

Three runs recovered — two on the second attempt, one on the third — and no run exhausted
its budget. The retry is not blind resampling: it appends the failed condition to the prompt,
which is why a second draw of a prompt that had just failed succeeds.

It also caught a defect in itself. The `demands` pass sometimes declares `tbl1`/`tbl2` as
required entities, and the `satisfy` pass is forbidden to emit Tables — they are copied from
the pubget manifest. The post-condition demanded them anyway and spent whole retry budgets on
a fault no retry could clear, so declarations of a deterministic class are now exempt.

**This does not generalise to a validator-driven repair loop for free.** `record/validate.py`
emits `path: message` and `adjudicate.py` already applies `{op, path, value}` patches at
exactly that addressing, so the machinery for a full repair loop exists. But a validator
measures internal consistency, and the cheapest way to satisfy one is to delete the offending
content — set the field `not_reported`, drop the analysis whose cell will not validate. A
repair loop optimising validity would do that, score better on the validator, and lose
recall. Any such loop has to be forbidden from deleting, and scored against gold rather than
against itself.

### 5.9 Cell construction fails systematically, not erratically

Every run emitted 3 cells against gold's 12, and the pooled direction result (tp 3, fp 9,
fn 45) is identical across both configurations. Unlike the entity variance, this does not
fluctuate: the pipeline reliably builds one cell per analysis where the correlation contrasts
each take two — a held level plus a signed slope. It is a reproducible misreading of what a
`Cell` is, which makes it a good target for C3 (a targeted re-ask) and a bad target for C4
(self-consistency), since voting over a consistent error returns the error.

---

## 6. Predictions, registered before the runs

Writing these down first is what makes the sweep informative rather than a fishing trip.

1. **A (anchor) is the largest main effect**, and it moves `Region` recall, relationship F1
   and direction recall together — because they have one cause.
2. **A3 (prune alone) recovers most of the precision loss and none of the recall loss.**
   If pruning matches full analysis-first, the ordering hypothesis is wrong and the problem
   was only over-generation.
3. **B (granularity) moves field metrics and not direction.** Direction is a reasoning error.
   Exception: B3 (per-analysis) may move the cell-count failure.
4. **D1 ≈ D0; D2 > D0.** Effort bought uniformly is wasted; effort bought on structural
   decisions is not.
5. **C3 (targeted re-ask) is the best cost-per-point on the weighted metric**, because it
   attacks the single highest-weighted failure directly.

### How they fared against §5

| | prediction | verdict |
|---|---|---|
| 1 | anchor is the largest main effect | **untested** — the spread makes it unmeasurable at n=1 |
| 2 | pruning recovers the precision loss | **wrong**, and it would delete correct entities (§3, axis A) |
| 3 | granularity moves fields, not direction | untested |
| 4 | D1 ≈ D0; D2 > D0 | **half right**: D1 ≈ D0 on accuracy, but D1 ≫ D0 on *consistency* (§5.3) |
| 5 | targeted re-ask is the best cost-per-point | untested, and now better motivated by §5.5 |

Two things not predicted turned out to matter more than anything that was: the silent empty
payload (§5.2) and the structural table ceiling (§5.4). Both were found by running eight
cheap replicates of a *single* configuration rather than one run each of eight configurations
— which is the general lesson about how to spend a small budget on a noisy pipeline.

### The revised recommendation

Ordered by evidence, not by how interesting the idea is. The first three are now built and
measured; what follows them is not.

1. ~~Deterministic post-conditions with retry~~ — **done** (§5.8), 80.0% → 92.3%, degenerate
   runs 2/10 → 0/10.
2. ~~Demand-driven ordering + the zero-foci rule~~ — **done** (§5.7). Together 38.1% → 80.0%;
   note the rule *subtracts* from the entity-first ordering it was first tried on.
3. ~~Targeted re-ask on cell construction~~ — **done** (§5.7). Eliminates total failure
   (5/15 → 15/15 runs with any correct cell) without delivering a complete contrast set.
4. **Stratified gold, ~5 papers each across ~8 analysis shapes.** Now the binding
   constraint on everything else. Every failure found so far was structural rather than a
   misreading, and structural failures are shape-specific: a factorial, a mediation model
   and a seed-based connectivity analysis are three different modelling problems, and this
   work has evidence about none of them. A blended accuracy over a random sample would hide
   exactly the variation that decides whether the pipeline scales.
5. **Constrained decoding.** `response_format: {"type": "json_schema", strict: true}` is
   supported on the gateway — verified, not assumed — and the pipeline currently asks for a
   free-form `json_object`. **57% of runs need at least one shape repair** from
   `build_record` (141 repairs over 70 runs, 118 of them a bare scalar where an
   ExtractedValue belongs), and the variants it cannot repair are what crashed the scorer
   (`Cell.term` emitted as a dict) and lost the shopping list (`required_entities` filed
   inside `analyses`). A whole class of defect removed structurally rather than by asking.
   Held below stratified gold only because it would otherwise be optimised against a
   benchmark of one paper.
6. **Run-to-run disagreement as a triage signal.** Sample k=3 and route the disagreeing
   papers to a human. Weakly supported here — baseline was bimodal and disagreement tracked
   correctness, while the fixed configuration agreed 9/10 — but it is the only route that
   turns an unverifiable corpus into a sampled estimate plus a review queue.
7. **Then the screening matrix**, once (4) exists to run it against.

---

## 7. Threats to validity

- **One paper.** Everything in §5 is one paper's behaviour. The variance result generalises
  least badly — a pipeline this noisy on one paper is unlikely to be stable on others — but
  the table ceiling (§5.4) and the cell defect (§5.5) are single-paper observations and could
  be idiosyncratic. `xevP8UDRAVh9` is also the *hardest* paper in the corrections set, at 89
  corrections; it was verified because it was hard.
- **One annotator.** No inter-annotator ceiling exists, so "10% direction F1" has no scale.
  Two independent gold extractions of one paper, scored against each other with this same
  script, would supply it — and that number is the ceiling every configuration is really
  being compared against.
- **Four replicates.** Enough to show the spread is large; not enough to estimate it well. The
  sds above have wide intervals of their own.
- **Aliased interactions**, when the screening matrix eventually runs. Resolution III means a
  main effect and a two-factor interaction are indistinguishable — fine for ranking, wrong for
  a final recommendation, hence stage 2.
- **The scorer shares no code with the extractor**, which is the one thing genuinely in this
  design's favour: `pondie.benchmark.scoring` reads the schema, not the pipeline.
