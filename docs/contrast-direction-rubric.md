# Scoring contrast direction against reviewer gold

`pondie.benchmark.scoring` asks how much of a paper an extractor got right, over four
families of metric. This file scopes that down to the one question a synthesis cannot
recover from anywhere else: **for a term that both sides agree is in the contrast, did the
extractor put it on the right side?**

    pondie benchmark \
        --gold data/gold/direction/84rGLhCbUJTh.direction.json -v

Everything here is deliberately narrower than §4 of
[extraction-comparison-metrics.md](extraction-comparison-metrics.md). That section scores
direction with F1, so a dropped cell is a miss. This one does not: missing terms and
mislabelled other terms are explicitly **out of scope**, because the question being asked
is about polarity on the terms that are present, not about contrast completeness.

## Why the gold is a table and not a record

There is one verified gold *record* (`xevP8UDRAVh9`), and manufacturing more by replaying
reviewer corrections onto extractions is how a benchmark quietly starts grading an
extractor against itself. That prohibition is about whole records, and it stands.

What the contrast project produces is narrower and does not have that defect. Every
`contrast_row_{i}_{j}` answer is a reviewer looking at one named cell and choosing its
direction from the same six-value vocabulary the schema uses for `Cell.direction`. It is a
direct human judgement of exactly the field being scored, not an inference from a diff.

The gold artefact is a **direction table**: a mapping

    (paper, analysis local_id, term local_id, level) -> direction

and nothing else is gold. The record the reviewer was shown supplies the *identity* of the
terms — it is scaffolding that gives each row something to be a row of — but none of its
own field values are treated as truth. A scorer that reads any other field off that record
has stopped measuring what the human said.

The consequence worth stating plainly: **the gold table can only contain cells the deployed
record proposed.** A term the paper contains and that extraction missed was never shown to
a reviewer and is not in the table. This benchmark therefore cannot measure recall against
the paper, and no number it produces should be read as if it could.

## The pre-fill problem, and why gold has provenance tiers

The contrast task arrives with every direction radio **already selected**, from the
prediction the deployed extractor produced. A reviewer who agrees changes nothing, and a
reviewer who never read the row also changes nothing. The two are indistinguishable in the
stored answer.

Measured on the live project: **610 of 719 answers are the prediction verbatim.** Taking
answers at face value therefore builds a gold set that mostly restates the extractor's own
output. Scored against it, the record that produced the predictions returns 100% — and did,
on the first run of this scorer, on 76 of 76 signed cells, having been "corrected" on
exactly none. That number measured nothing.

The reviewers were not idle: they ticked `direction_wrong` on **35 of 126** analyses. The
signal was in the verdict checkbox while the radios stayed where the extractor left them.

Every answer carries a provenance tier, and the scorer reports the split rather than a
blended figure:

| tier | what it is | count | independent of the prediction? |
|---|---|---:|---|
| `changed` | the reviewer moved the radio | 109 | **yes** |
| `accepted` | left on the prediction, the analysis affirmatively ticked `accept` | 354 | only for a *different* extractor |
| `unflagged` | left on the prediction, the analysis flagged but **not** for its direction | 55 | as `accepted` |
| `silent` | left where the direction itself was doubted, or no verdict | 201 | no — **excluded from gold** |

`unflagged` exists because "the reviewer complained" is not the same as "the reviewer
complained about the sign". `cells_wrong` is a claim about which cells exist and
`upstream_wrong` about whether the analysis should exist at all — both out of scope here.
A reviewer who itemised what was wrong and did not name the direction engaged with the row
without disputing its sign, so those rows are gold.

`silent` is dropped on evidence rather than on principle. Of its 201 rows, **164 sit on
analyses ticked `direction_wrong` — "right terms, wrong sides"** — where the untouched
radio is precisely the sign the reviewer rejected; 25 more are `uncertain`, and 11 carry no
verdict at all. Reading those as endorsement would enter known-wrong signs as truth.

The middle row is the one to keep straight. `accepted` is an affirmative human claim that
the record says what the paper says, so it is legitimate gold **for any extractor that did
not generate those predictions**. For `claude-opus-5 / adjudicated-0.1.0`, which did, it is
circular and its scores there must be read as tautology.

Where two reviewers answered the same cell, they are weighed at the strongest tier present
rather than pooled: a reviewer who moved the radio has demonstrably read the row, and
letting an untouched answer outvote a moved one would reinstate the pre-fill bias the tiers
exist to remove.

## The alignment problem

Extractors do not agree on `local_id`, and they do not agree on wording: gold's
`term-group-baseline` is a candidate's `term_diagnosis`, and `"diagnostic group"` is
`"patient group"`. Same-ness has to be established on content.

This does not need new machinery. §1 of the comparison metrics already resolves entities by
optimal bipartite assignment over four kinds of evidence — attributes, outgoing references,
incoming references, containment — iterated to a fixed point. That is what establishes
whether a candidate `ModelTerm` *is* a gold one, and this rubric reuses it unchanged rather
than introducing a second, weaker notion of the same thing.

Two properties of that machinery matter here specifically:

- **Direction is excluded from cell alignment** (`ALIGN_EXCLUDE`). Aligning cells by their
  direction would manufacture the agreement being measured. Cells align on term and level.
- **Incoming references carry the terms.** A continuous `ModelTerm` declares no levels and
  makes no outgoing edges; matched on its name alone it falls below threshold, and every
  cell hanging off it then vanishes as *unaligned* instead of being scored. Reading the
  analyses whose cells name it recovers the match.

## The tiers

Reported as a funnel, because each tier's denominator is the tier above it and quoting any
one alone is misleading.

### Tier 0 — coverage (reported, never scored)

How many gold cells reached the point of being scorable, and where the rest went:
`analysis_unaligned`, `term_unaligned`, `cell_unaligned`, `cell_absent_in_candidate`.

Per your scoping these are **not penalties**. They are printed anyway, and the primary
metric is never quoted without them, because an extractor that emits one easy cell per
analysis and drops the rest can otherwise post a perfect score on three cells. Coverage is
what stops the headline being a lie by omission.

### Tier 1 — polarity accuracy: the headline

Denominator: cells where the term is **grounded** (its reference resolves through the
entity map to the same `ModelTerm` on both sides), gold direction is `positive` or
`negative`, and candidate direction is `positive` or `negative`.

Metric: the fraction with the same sign. Its complement is the **sign-flip rate** — the
catastrophic case, a finding reported backwards.

This is a binary decision on a near-balanced class distribution (the current reviewer set is
89 positive / 81 negative, 52.4% / 47.6%), so it has an honest chance baseline: **50% is a
coin flip, 52.4% is always guessing the majority class.** An extractor scoring 60% here is
barely doing anything. That clean baseline is the reason to keep the denominator restricted
to signed-versus-signed rather than folding in the unsigned classes, which are 68% `absent`
and would push a trivial baseline over 0.8.

Cohen's kappa is reported alongside for the same reason it is in §4, but on this restricted
two-class problem accuracy against 50% is the more legible number.

### Tier 2 — polarity retention (secondary)

Gold is signed and the cell exists in the candidate, but the candidate did not sign it:

- `sign_loss` — candidate says `undirected`, `held`, `absent` or `not_reported`.
  Conservative rather than wrong. The finding survives as uninformative.
- `sign_missing` — the candidate produced no such cell at all. Out of scope by your
  scoping; counted in Tier 0, repeated here so the two failure shapes are not conflated.

### Tier 3 — sign invention (reported, excluded from the headline)

Gold is unsigned (`absent`, `held`, `undirected`, `not_reported`) and the candidate asserts
`positive` or `negative`: a direction claimed that no test in the paper produced.

This sits outside the headline because you scoped "other terms being mislabelled" out. It
is still reported, because it is the one unsigned-class error that damages a synthesis in
the same way a flip does — it puts a signed claim into the record that nothing supports.

## Two pairing bugs, and why the level test has no threshold

A reviewer row is labelled with the level its **ModelTerm declares**; a `Cell` carries its own
`level`, which the schema permits to differ in wording (`Cell.label` exists for that case).
Both the exporter and the first version of this scorer keyed the two together as exact
strings, and so failed to pair `"schizophrenia or schizoaffective disorder"` on the row with
`"Patients with schizophrenia or schizoaffective disorder"` on the cell.

In the exporter this was the worse of the two defects. A missed lookup pre-selected `absent`,
and `absent` is not the absence of a hint — it is the positive claim that the contrast
weighted this level out. **35 of 531 live contrast rows asserted the opposite of what the
record said**, concentrated in `QQCjAAT6SwwQ` (15), `kzMj26hGWacQ` (8) and `SULKxviGFurw` (6),
three of the four weakest papers in the table above. In the scorer it silently dropped 31
signed gold cells from the denominator without counting them anywhere, which is the
lie-by-omission Tier 0 exists to prevent.

The repair is *not* a similarity threshold, and that is the part worth remembering. These
level vocabularies are full of pairs that differ by an affix and mean opposite things: `men`
is a substring of `women`, `synchronous` of `asynchronous`, and an edit-distance ratio puts
that second pair at **0.96**. Any graded matcher pairs a level with its own negation, and the
scorer then reads a flipped sign as correct — the exact error being measured, introduced by
the instrument. A first attempt here used containment plus a 0.85 ratio and matched both
pairs.

The test therefore uses whole words: the sides must be equal after normalization, or one
side's word set must be contained in
the other's. `{schizophrenia, or, schizoaffective, disorder}` is a subset of `{patients,
with, …}`; `{men}` is not a subset of `{women}`. The rule lives in both `tasks._same_level`
and `pondie.benchmark.scoring._same_level`, with a test asserting the two agree — if they drift, the
scorer pairs cells the grid never showed.

### The tier history has to be frozen before re-exporting

An answer's provenance tier is *what the reviewer did with the value in front of them*, so it
can only be judged against the pre-fill they actually saw. Re-exporting overwrites that, and
a gold rebuild afterwards would compare answers against predictions that postdate the review
and reclassify every tier — turning the 116 `changed` answers into `silent` ones and
destroying the only prediction-independent evidence in the set.

`build_direction_gold.py --freeze-history` captures the pre-fills first, and the build then
prefers them over the live values. The history for the review taken through 2026-08-26 is
`data/gold/direction-prediction-history.json` (531 rows, 389 of them `absent`, i.e. as they
stood before the exporter fix). **Freeze before exporting, never after.**

## Vocabulary drift

Twelve live answers carry `unstated`, a member deleted from every vocabulary in favour of
`not_reported` as the single encoding of silence. They are read as `not_reported`. Both are
unsigned, so this cannot move Tier 1; it is done so the confusion matrix has one column for
one concept rather than two for a rename.

## Reporting: per paper, with an interval

Two facts about this pipeline make a single blended number untrustworthy, and the report is
shaped around both.

**Variance.** Run-to-run spread on a fixed configuration swamps the difference between
configurations; roughly one run in four loses the entity pass outright and produces an
empty skeleton. §5.8 of the workflow experiments records a configuration that measured
92.3% and re-measured 23.1% with no change to the scorer. **A single run is an anecdote.**
Replicates are `k=3` by default and the report carries the spread, not just the mean.

**Shape, not paper, is the unit of difficulty.** Every failure characterised so far has been
structural rather than a misreading, and structural failures are specific to the shape of
the model — a factorial, a mediation and a seed-based connectivity analysis are three
different problems. A blended accuracy over 14 papers hides exactly the variation that
decides whether the pipeline scales, so results are broken out per paper and the
degenerate runs are reported as their own count rather than averaged in.

Confidence intervals are bootstrapped over papers, not over cells: cells within a paper are
not independent — one misread model flips every cell hanging off it at once — and an
interval over cells would be far too narrow.

## What this rubric cannot tell you

- **Recall against the paper.** The gold table holds only cells the deployed extraction
  proposed. Terms it never found are invisible here.
- **Whether the contrast set is right.** `wrong_axis` and `cells_wrong` verdicts are carried
  through to the output as flags but score nothing.
- **Recall of the reviewed set.** 19 contrast tasks are still unreviewed, and `silent`
  answers drop 256 more. The scored set is what survived both filters.

## The ceiling, measured

The workflow experiments record "no inter-annotator ceiling exists, so 10% direction F1 has
no scale" as a threat to validity. The contrast project carries two annotations per task,
so it now does exist.

**239 cells were reviewed twice.** Read naively they agree 78.2%; weighed by provenance
tier, 95.8%. But the number that scales Tier 1 is narrower — restricted to cells where
*both* reviewers chose a sign:

> **44 cells, 42 agreed: 95.5%.**

Two humans disagree about polarity on roughly one signed cell in twenty. That is the
ceiling, and it is the figure an extractor's Tier 1 accuracy should be read against rather
than against 100%.

The shape of the remaining disagreement is worth recording, because it validates the
scoping of this rubric. Of 52 disputed cells only **2** are `positive` vs `negative`. The
rest are arguments about whether a term is in the contrast at all — `absent` vs
`undirected` (10), `absent` vs `positive` (8), `absent` vs `negative` (7), `absent` vs
`held` (7). Humans agree about direction and disagree about membership, which is precisely
why membership is out of scope here and polarity is the headline.

## First run: the recommended configuration, k=3

`demand-driven --zero-foci-rule --max-attempts 3`, model `gpt-5.6-luna`, over the 14
reviewer-gold papers, 2026-08-26. Driver: `run_direction_bench.sh`.

Scored against the **complete** review — all 102 contrast tasks, 16 papers, 109 signed gold
cells. Earlier runs of this table used a partial pull and reported two perfect replicates;
they were an artefact of gold that was missing, not of an extractor that was right.

| paper | n | rep 1 | rep 2 | rep 3 |
|---|---:|---:|---:|---:|
| `JzsUUQbDr2bm` | 34 | 100% | 65% | 100% |
| `eaEGQiVtDp9e` | 8 | 100% | 100% | 100% |
| `kzMj26hGWacQ` | 8 | 75% | 75% | 75% |
| `7HPLh5nJzmP5` | 4 | 100% | 100% | 100% |
| `SULKxviGFurw` | 3 | 33% | 33% | 100% |
| `TgcHKMRfrVog` | 3 | — | 100% | 100% |
| `aVGe9BmFTMDR` | 1 | 100% | 100% | 100% |
| **Tier 1** | | **93.0%** (57) | **73.3%** (60) | **96.7%** (60) |

Mean 87.7%, sd 10.2%, against a coin-flip baseline of 50% and a human ceiling of 95.8%.
The deployed `claude-opus-5` record scores 96.7% (90) on the same gold.

| tier | rep 1 | rep 2 | rep 3 | deployed |
|---|---:|---:|---:|---:|
| `changed` — prediction-independent | 63.6% (11) | 63.6% (11) | 81.8% (11) | 80.0% (15) |
| `accepted` | 100% (46) | 73.9% (46) | 100% (46) | 100% (67) |
| `unflagged` | — | 100% (3) | 100% (3) | 100% (8) |

The tier split is what makes this readable. `accepted` is near-perfect in two runs of three,
but `changed` — the rows a reviewer moved off the prediction, and the only tier that cannot
flatter the extractor that generated it — sits at **66–83% across all three**. The headline
is carried by the easy tier; the independent tier is consistently the weakest, and no
replicate reaches the human ceiling on it.

`kzMj26hGWacQ` is stable at 75% in every replicate, which makes it a **reproducible** defect
rather than variance, and the more useful thing to debug than rep 2's collapse.

**The spread is one paper in one replicate.** Every one of the 12 errors is `JzsUUQbDr2bm`
in replicate 2, and they are not scattered: whole analyses invert together, `analysis_01`
and `analysis_02` flipping every cell at once. That is a reference-level failure — the run
chose the other group as the baseline — not a per-cell misreading, which is why an interval
over cells would be badly wrong here and the interval is bootstrapped over papers.

This is the §5.8 variance reproducing rather than a property of the configuration. Two runs
of three are perfect; the third is wrong in a single structured way.

**Coverage is the binding constraint, not accuracy.** Only 3 of the 14 papers produced any
scorable signed cell, and 34 of the 46 come from one of them. The other eleven contribute
nothing: their reviewer gold is `absent` almost throughout, or their signed cells never
resolved to a candidate cell. A mean over this set is close to a statement about
`JzsUUQbDr2bm`.

The actionable result is not 91.3%. **The benchmark has 46 usable cells from
3 papers**, and widening it — more reviewed papers, and reviewers who move the radio rather
than ticking a verdict — buys more than any further tuning of the extractor would.

## Repairing the split deterministically, at stage 1

The rubric above measures direction. This section is about removing one cause of getting it
wrong, rather than scoring it after the fact.

`Analysis` requires a separate record per normalized direction — direction is the first
discriminator in the splitting rule. An analysis holding both a positive and a negative
statistic is therefore not a judgement call; it is two analyses that were never separated.

**The pass being asked to split cannot see what it is being asked to split on.**
`prompt/render.py`'s stage-1 block already instructs the model to split an entry "when the
table distinguishes the rows it covers by a column the entry's name does not mention", and
names the trigger explicitly: "one entry would otherwise hold effects of opposite sign". But
the same block states the constraint that makes this impossible — "the parse had the contrast
name and not the rows". The extraction passes are shown captions, never cell values. The
signal is real and the pass is blind to it.

Stage 1 is not. `corpus/tables.py` holds the parsed rows, so `split_opposite_signs` partitions
them there, before `analyses.json` is written and therefore before anything downstream sees
the list. Nothing about Autonima's prompt changes: the split is a post-pass over what
`parse_single_table` returns, and the partition is arithmetic rather than a second opinion.

### What the rule is, and what it refuses to guess

`p-value` is the only parsed kind that cannot carry a sign. Every other kind can —
`t-statistic`, `z-statistic`, `correlation`, `beta`, and the `other` catch-all.

Two temptations were tried against the corpus and rejected, both of them the same mistake:

- **Excluding a kind because this corpus shows no negatives for it.** `z-statistic` has 42
  values and no negative one; so do 56 of the analyses carrying `t-statistic`. That is
  evidence about how these tables print statistics — most print |t| and put the direction in
  the caption — and none at all about whether the quantity is signed. A z is signed by
  definition. `other` is included for the converse reason: one study contributes 124 of them
  spanning 0.61–3.75, which is a t or z that lost its heading, and dropping the kind would
  discard real directions.
- **Requiring both groups to be large enough.** A single row in the minority direction looks
  like a mis-parse and is not: one surviving cluster is an ordinary result of thresholding,
  and 11% of parsed analyses have exactly one point already — `Tuina > Sham`, `FESZ>SZ`, four
  separate `CBT change` contrasts. Gating on group size would have suppressed the split on
  `JzsUUQbDr2bm`, leaving a negative t inside a contrast the paper named `>`.

What it does refuse: a **partial** partition. If any row carries no sign — only a p-value, or
statistics that disagree with each other — the analysis is reported and left whole. Filing
the signed rows and stranding the rest is worse than leaving the defect visible.

Names follow the convention the extraction prompt already states for a split entry,
`<given name> (<level>)`, with the direction as the level. Nothing here writes prose:
`definition` must be source-grounded and is left to the pass that can quote the paper.

### Measured effect

Applied with `--resplit`, which re-partitions stage-1 output already on disk and needs no
model call — re-parsing would resample every other decision the parse makes at the same time:

| study | analysis | split |
|---|---|---|
| `YwwKWoEFwY3G` | `Encoding` — *"peak response in each significant age correlation cluster"* | 2+ / 2− |
| `TgcHKMRfrVog` | `Baseline rsFC with rACC` | 1+ / 2− |
| `JzsUUQbDr2bm` | `FESZ > NC` | 8+ / 1− |

Three of 88 parsed analyses, so this is a narrow high-confidence repair and not a general
fix. The ceiling is structural: only the analyses whose tables print signed statistics can be
partitioned this way, and where both directions are printed as positive magnitudes with the
direction in a row-grouping header — `JzsUUQbDr2bm`'s Tables S2 and S3 — no arithmetic can
separate them and the direction has to be read from prose.

`YwwKWoEFwY3G` is the case worth keeping in mind: an age-correlation table with r = +0.42,
+0.40, −0.41, −0.40 in one analysis. Positive and negative correlations on one continuous
term are two analyses, and one of them was silently the other's negation.


## Error investigation: what the 17 flips actually were

Every sign flip across the deployed record and three replicates, traced to cause. **Three of
17 were extraction errors, and none of those three was the extractor's fault.**

### A. The comparison operator was being deleted — 14 of 17 flips

`normalize()` strips punctuation before comparing strings, and `>` and `<` are punctuation to
a regex. So:

    normalize("FESZ>NC")  ->  "fesz nc"
    normalize("FESZ<NC")  ->  "fesz nc"        fuzzy(...) == 1.0

The operator is the *entire* semantic difference between an analysis and its mirror — the
reason the schema makes them separate Analyses at all — and the comparison could not see it.
The §1 assignment then paired `analysis_01` with the candidate's `< ` contrast, and every
cell beneath it read as a flip. `DIRECTION_LEAKING` exists to let the primary aligner read
`name` and `definition` for exactly this discrimination; normalization was removing the only
character that carried it, so the mechanism was inert.

Operators are now folded to word tokens (`gt`, `lt`, `gte`, `lte`) *before* punctuation is
stripped, and spelled-out forms fold to the same tokens so `A > B` still meets
`A greater than B` at 1.0.

The effect on the measurement is the whole result:

| | before | after |
|---|---:|---:|
| rep 1 | 93.0% | 96.5% |
| rep 2 | **73.3%** | **96.7%** |
| rep 3 | 96.7% | 96.7% |
| replicate sd | **10.2%** | **0.1%** |

The "high variance" this pipeline was believed to have, on this metric, was the scorer pairing
contrasts with their own mirrors. rep 2 was never a degenerate run.

### B. Reference-level inversion in the gold — 2 flips

`kzMj26hGWacQ / analysis_tbl4_dan_treatment`, flipped in **all four** runs, which is what
made it look systematic. It is: all four are right.

The analysis is named `Baseline > week 6`, stage 1 parsed `t = +4.217`, and the paper says

> No areas within the DAN showed increased connectivity compared to baseline (Table 4).

Table 4 is therefore *decreased* connectivity, baseline exceeds week 6, and baseline takes
the plus side. The record's `baseline: positive, six-week follow-up: negative` is correct and
the reviewer's answer is inverted.

The ambiguity is real and worth naming: the same finding can be described as a contrast
(`baseline > week 6`) or as a change (`connectivity decreased over treatment`), and the two
readings put opposite signs on the same level. The schema settles it — `Analysis.definition`
states the ordering and the cells must match it — but nothing in the review task showed the
reviewer that ordering next to the radios.

### C. Row attribution handed one contrast another's rows — 1 flip

`YwwKWoEFwY3G / analysis-alpha-encoding-age`. Not a reviewer error: the reviewer read the
table correctly and the table they were shown was the wrong one.

Table 1 reports four encoding clusters across two frequency bands with opposite signs —
θ at `+.42` and `+.40`, α at `−.41` and `−.40`. The contrast grid highlighted the **θ**
rows under `analysis-alpha-encoding-age`, so a reviewer looking at two positive
correlations signed the alpha contrast `positive`. The task even carried the caution,
`weak name match 0.50/0.14, check the rows are this contrast's`.

The cause is a tie the linker could not break. A record analysis carries no per-row
attribution — all five age analyses say only `tables: [tbl1]` — so `link_analyses` matches
record names to stage-1 parse names. After the sign split those parses are `Encoding
(positive)` and `Encoding (negative)`, and a record analysis named by *frequency band*
scores identically against both. The tie then fell to whichever `local_id` sorted first,
and `analysis-alpha-encoding-age` sorts before `analysis-theta-encoding-age`.

The fix is that the sign the split was made on is also what identifies the parse. A record
analysis's own `Cell.direction` now breaks the tie, applied only to siblings carrying
`split_direction` so nothing else in the assignment moves. Attribution afterwards:

| analysis | rows now highlighted |
|---|---|
| `analysis-theta-encoding-age` | Left primary visual `+.42`, Left dlPFC `+.40` |
| `analysis-alpha-encoding-age` | Left middle frontal `−.41`, Left ant. cingulate `−.40` |

This matches the paper: theta *"became significantly stronger as a function of age in
the left primary visual cortex"* and *"older individuals showed significantly stronger
decreases in alpha activity"*. The record's `negative` for alpha was right all along; the
gold answer was taken against rows belonging to the theta contrast and needs re-review.

Worth keeping the near-miss on record: the prose here **is** double-negative — "stronger
decreases" pairs an increasing word with a decreasing one — and that was the first
explanation reached for. It was wrong. The reviewer never had to parse that sentence,
because the table in front of them showed `+.42` and `+.40`. A plausible cause is not a
cause, and the row highlighting is what distinguished them.

### What this says about where to spend effort

All three surviving flips are gold errors, and **all three are `tier: changed` rows** — two
where the exporter had emptied the pre-fill to `absent`, one where the grid highlighted
another contrast's rows. A reviewer
given no anchor on a double-negative sentence or an unstated contrast ordering gets it
backwards, and the two failures compound.

The ranked fixes therefore do not concern the extractor:

1. **Re-adjudicate the 21 flagged tasks** now that pre-fills are correct. These 3 cells are
   among them.
2. **Show the ordering in the task.** `Analysis.definition` states which side is the plus
   side and the contrast grid never displays it. A reviewer signing `baseline` cannot see
   that the analysis is named `Baseline > week 6`.
3. **Treat the weak-match caution as blocking, not advisory.** It fired on exactly the task
   that produced a wrong answer. A grid that cannot say which rows are this contrast's is
   asking the reviewer to sign a direction for rows it has not identified, and the honest
   move is to withhold the question rather than print a warning above it.

Corrected for the three gold errors, all four runs score **100%** on this metric. That is not
a claim the extractor is solved -- it is a 57-90 cell benchmark whose remaining signal is
mostly `accepted`-tier rows -- but the polarity errors this benchmark set out to find are, on
this corpus, in the gold and the instrument rather than in the model.
