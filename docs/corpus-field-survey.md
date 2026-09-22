# What 1,817 records say about every field the extractor fills

> Where this sits: [pipeline-architecture.md](pipeline-architecture.md) covers how a record is
> produced, [normalization-pipelines.md](normalization-pipelines.md) how a field is normalized
> once it exists. This is the survey that says *which* fields need it, measured rather than
> guessed. Companion to [field-extraction-audit.md](field-extraction-audit.md), which asks
> whether a value is recoverable from the page; this asks what shape the values arrived in.

## The corpus

`beast:/data/james/pondie-vs-fulltext/repos/autonima-results/experiments/record_arms/records`
— 1,817 records over five projects (cue_reactivity 550, emotion_regulation_2022 525,
dementia 448, vbm_of_substance_use 244, vbm_of_ptsd 50), all from
`gpt-5.6-luna` at `extractor_version: pondie-1`, dated 2026-09-09 and 2026-09-11.

That is 5,989 analyses, 4,237 groups, 12,098 effect cells and 1,014 distinct dotted paths —
between 60 and 300 times the 16-record benchmark sets every earlier coverage number in
`pondie/normalization/` was measured on.

## The schema is not as outdated as it looks

Of the 1,014 observed paths, exactly **one** is a field the schema has since renamed. Every
other path either resolves against the current schema or fails because a handful of papers
put a value somewhere it has never belonged.

| | paths | occurrences |
|---|---:|---:|
| resolve against the current schema | 258 | — |
| `local_id`, the extraction projection's name for `id` | 19 | — |
| **one genuine rename** | 1 | 1,418 |
| per-paper record damage (below) | 736 | 1,795 |

**The rename is `Task.response_mode` → `Task.response_modality`**, in 1,256 of 1,817 papers.
It needs no value normalization at all: of the 1,418 values, 1,415 are already exactly the
current `ResponseModality` permissible values (`button_press` 806, `none` 440,
`hand_movement` 66, `speech` 62, `covert_response` 26, `eye_movement` 9, `oral_nonspeech` 6).
The three that are not are `pointing` ×2 and `oral_nonspee` ×1, a truncation. **This is a key
rename, not a normalization job.**

Everything else that fails to resolve is one of a handful of papers putting a value in the
wrong place, and the counts are in papers, not percent:

- `analyses.effect.<Analysis slot>` — `coordinate_space`, `tasks`, `tables`,
  `inference_settings`, `acquisitions`, `source_table_analysis` nested one level too deep,
  8–17 papers each.
- `model_estimations.study.*` — an entire Study nested inside a model estimation, 7 papers.
- `analyses.required_entities.*` — a container that has never existed, 3 papers.
- `preprocessing` singular instead of `preprocessings`, 4 papers.

No unresolved path other than the rename reaches more than 17 of 1,817 papers. These belong
to `extraction/record/fix`, not to normalization.

Fifteen current slots are never filled at all, and the list is informative rather than
alarming: `acquisitions.brain_coverage`, `short_channel_count`,
`source_detector_distances_mm` (no fNIRS study in the corpus), `analyses.statistical_maps`,
`groups.population_characteristics`, `other_characteristics`, `sample_source`,
`sample_source_reference`, `sample_overlap_count`, `tables.coordinate_count`,
`tables.source_path`, `tasks.references`, `tasks.stimulus_modality`,
`tasks.response_modality` (the rename above) — and **`tables.coordinate_space`**.

That last one confirms at corpus scale what `coordinate_space.resolve`'s docstring claims
from 16 records: the table step in its precedence cascade "is empty in every table measured,
so the middle step never fires". Empty in all 2,266 tables here too.

## Two bugs the corpus exposed in code that already ships (both now fixed)

Both are the same mistake, and it is worth naming because it will recur: **a pattern written
with `[\s-]` or a trailing `\b` does not match the schema's own underscore-joined
permissible values.**

**`correction_scope` answers half of what it sees.** `whole[\s-]?brain` does not match
`whole_brain`, which is what the field actually holds 212 times. Measured over the corpus's
433 values, coverage is **50.8%**, and every miss is the literal enum value:

| value | n | `normalize` says |
|---|---:|---|
| `whole_brain` | 212 | UNKNOWN, unmatched |
| `searchlight` | 1 | UNKNOWN, unmatched |

Admitting `_` in every separator class, and adding `searchlight` to the OTHER rule on the
same grounds the module already gives for `cluster level` — it names a geometry, not a volume
— takes the field to **100%** of those 433 values: RESTRICTED 220, WHOLE_BRAIN 212, OTHER 1,
nothing unmatched.

**`coordinate_space` is at 98.8%, and the residual is three clean classes.** Over 4,802 real
values: MNI 3,760, TAL 973, OTHER 7, UNKNOWN 62. Only 22 surface forms and 57 occurrences go
unmatched, and they sort into:

| class | forms | what they should be |
|---|---|---|
| study-specific templates | `customized template` ×11, `study-specific template brain`, `symmetric DARTEL template space`, `SPM5 template`, `FDG-PET-specific template`, `SUIT template space`, `FMRIB58_FA standard space`, `fsaverage6` | **OTHER** — a third space, refuse to transform |
| a known space, spelled out or misspelled | `International Consortium for Brain Mapping template` ×3, `standard proportional stereotaxic space` ×3, `Talaraich` ×2, `NMI` ×2, `Colin27 Brain` ×2 | MNI, TAL, TAL, MNI, OTHER |
| contentless | `template image space` ×6, `reference atlas` ×4, `stereotactic` ×2 | UNKNOWN, correctly |

Two of these were the trailing-boundary lesson the module already documents for `\bmni`:
`\bfsaverage\b` missed `fsaverage6`, and the `\bicbm` rule added this week caught the
acronym but not `International Consortium for Brain Mapping`, exactly as MNI needed
`montreal\s+neurolog` alongside `\bmni`.

Widening all three rules takes the field from **98.81% to 99.67%**. Every one of the 41
occurrences that moved went from UNKNOWN to an answer — 17 surface forms, no previously
correct value reclassified. The 16 that remain are the contentless class, still UNKNOWN and
correctly so: `template image space` ×6, `reference atlas` ×4,
`standard proportional stereotaxic space` ×3, `stereotactic` ×2, `unbiased atlas space` ×1.
The last of those is the line the OTHER rule is drawn on: a paper that names *no* template
has not told us it used a third space.

The other shipped normalizers hold up: `modality` 100% of 1,345 values,
`prespecification` 98.3% (residual is three typos — `prereglstered`, `prere_registered`,
`preregestistered`), `multiple_comparison_method` 92.9% of 1,562, whose residual is one
coherent class — `random field theory`, `3dClustSim`, `cluster thresholding`,
`clusterwise correction` — that wants an RFT/cluster-extent rule.

## Fields with an analog that have no normalizer, ranked by what a lexicon buys

Each coverage number below is a prototype lexicon scored against every value in the corpus,
not an estimate. The prototypes are ~6 lines each and use the existing `ClosedField`/`Rule`
machinery.

| field | values | distinct | off-enum now | lexicon coverage | verdict |
|---|---:|---:|---:|---:|---|
| `tasks.design_type` | 1,349 | 166 | **73%** | 91% | **highest yield in the corpus** |
| `model_estimations.stage` | 3,043 | 99 | n/a (string) | 93% | closed target, never declared as one |
| `inference_settings.height_threshold_type` | 1,420 | 24 | n/a (string) | 99.9% | trivial, and it gates the value slot |
| `model_estimations.model_family` | 1,367 | 70 | 12% | 97% | |
| `analyses.effect.statistic.family` | 5,280 | 35 | 2% | 99% | |
| `analyses.spatial_scope` | 1,431 | 15 | 3% | 98% | |
| `groups.age_unit` | 3,309 | 8 | n/a (string) | 100% | eight forms, one line |
| `model_estimations.software` | 1,471 | 1,279 | n/a (string) | 82% | **multi-label, not single-answer** |

**`tasks.design_type` is the one to do first.** The enum is
`block | continuous | event_related | mixed`, and 989 of 1,349 values are outside it —
almost entirely as punctuation:

```
event-related 556    block design 68    blocked 44    block-design 18
blocked design 11    rapid event-related 9    slow event-related 6
continuous with no modelled events 64       resting-state 8    resting state 6
```

`event[\s_-]?related`, `\bblock`, `continuous|resting[\s_-]?state` and a **decisive** `mixed`
rule for `mixed block/event-related` reach 91%. The 9% left over is a different question
being answered in the slot — `two-alternative forced choice`, `forced-choice`, `factorial`,
`naturalistic` describe the *task*, not its timing, and belong in `UNKNOWN` with a residual
entry rather than forced onto the scale.

**`model_estimations.software` is the shape the package does not yet have a name for.** Its
82% is a ceiling imposed by the question, not the rules: the residual is entirely strings
naming *several* tools — `SPM99 running in MATLAB 6.1`, `SPM8 and AFNI 3dClustStim`,
`AFNI/SUMA; FreeSurfer surface reconstruction`. A pipeline genuinely uses two or three
packages, so first-match-wins is the wrong model. This wants **set extraction**: run every
rule, keep every hit, return a sorted tuple. `model_estimations.stage` has the same problem
in miniature (`subject and group` ×72). Neither fits the four shapes in
`pondie/normalization/__init__.py`; adding a fifth, `multi-label`, is the honest fix.

Note that going from `\bspm\b` to `\bspm` alone moved software from 48.6% to 82.0%, because
`\bspm\b` does not match `SPM8`. Third instance of the same bug class.

## Numeric fields: the unit problems are real but smaller than they look

**`echo_time_seconds` is the genuine one.** 112 of 1,531 values (**7.3%**) exceed 0.5 s, with
a maximum of 104.6. The distribution is cleanly bimodal — 1,417 values at or below 0.2 s
(median 0.03, a correct fMRI TE) and 112 above 0.5 — with two values in between. A TE over
0.5 s is not physically achievable, so a divide-by-1000 rule on that upper mode is
deterministic, not a guess. It should be a `fix/derive`-style repair with a report line, not
a silent rewrite.

**`repetition_time_seconds` is mostly fine, and the obvious reading of it is wrong.** 183 of
1,402 values sit between 5 and 60 ms, which looks like a unit error and is not: 169 of those
183 acquisitions are `sMRI`, and a 3D gradient-echo structural genuinely has a TR of 9–25 ms.
The real suspects are far fewer — 9 values under 5 ms, 4 at or above 100 (2,500 / 2,050 /
1,960 / 8,986, milliseconds left unconverted), and 16 between 10 and 100 s. About 2%, not 14%.

Shape violations in the same slot, which the schema declares as a non-multivalued float:
16 records hold a list, and 5 hold the string `'ambiguous'`.

Smaller numeric findings, each a one-line bound check rather than a pipeline:

| field | range seen | problem |
|---|---|---|
| `groups.sex_distribution.percentage` | 0 – 133.3 | one value above 100 |
| `inference_settings.alpha_level` | 0 – 0.502 | one at 0.502, one at exactly 0 |
| `acquisition_voxel_size_mm` | 0 – 30 | 5 values above 10 mm, 4 at or below 0 |
| `groups.age_mean` with `age_unit = months` | 86.2 – 91.4 | 4 values; years mislabelled, or a duration in the age slot |

## `clusterwise_threshold_value` cannot be normalized, and that is the finding

927 values spanning 0 to 15,940 in one slot: p-values (median 0.05) and voxel extents in the
same column. The partition that would separate them exists —
`clusterwise_threshold_type` — and is **`None` in all 927 cases**.

This is worth contrasting with `height_threshold_value`, which has the identical two-kinds
problem and is *already solved* by its type slot being filled: split by
`height_threshold_type`, 1,247 p-like values (`p`, `P`, `q`, `alpha`, `FDR`, `uncorrected p`)
are ≤ 1 and 151 statistic-like values (`Z`, `z`, `t`, `T`, `F`) are > 1. Three records
disagree with their own declared type — `z` = 0.01, `T` = 0.001, and a `percentile` of 30
that is neither. Three in ~1,400: the type slot does its job when it is filled.

So the fix for the clusterwise pair is not a normalizer. It is either (a) make the type slot
required whenever the value is present, or (b) derive the type from the value's own
magnitude, which is only defensible because the two modes do not overlap — `≤ 1` is a
p-value, `> 1` is an extent in voxels. (a) is honest; (b) is available for the 927 already
written.

Note `height_threshold_type` itself still needs the lexicon in the table above: 24 spellings
(`p` 1,137, `P` 96, `Z` 61, `z` 58, `t` 23, `q` 11, `T` 10, `alpha`, `FDR`, `FDR q`,
`p uncorrected`, `bootstrap ratio`, `Z score`, `PFWE`) for what is six answers.

## Fields that want a vocabulary or a clustering, not a lexicon

| field | values | distinct | shape |
|---|---:|---:|---|
| `assessments.assessment_type` | 4,332 | 589 | **closed-ish**: `questionnaire` 896, `clinical scale` 594, `diagnostic interview` 520, `cognitive test` 218 — the head is six categories with modifier noise (`self-report questionnaire`, `structured clinical interview`, `neuropsychological assessment`). A lexicon over the head plus a residual report is the right first pass. |
| `regions.name` | 3,238 | 1,317 | **link** — an atlas vocabulary exists; this is the `medical_condition` pipeline applied to a different target. |
| `assessments.name` | 4,359 | 2,607 | **link** to an instrument vocabulary; the head is standard scales. |
| `groups.name`, `tasks.conditions.name` | 4,237 / 3,013 | 2,301 / 1,655 | **cluster** — the `task` pipeline's problem exactly. |
| `groups.description`, `assessments.description`, `regions.description` | ~4,000 each | 0.92–0.98 distinct/value | free prose; normalizing the *description* is not a goal, it is input to the clustering of the name. |

One field that looks like a normalization target and is not: `groups.diagnostic_instrument`
has 596 distinct values, but its range is `Assessment` — those are `asm_*` references, and
the cardinality is id cardinality. Worth noting only because two of its ids are truncated at
32 characters (`asm_structured_clinical_intervie`, `asm_mini_international_neuropsyc`), which
is an id-generation bug, not a value problem.

## Hygiene: sentinel strings written into value slots

368 occurrences over 32 field/value pairs hold a string that the slot does not permit and
that means "no value" — the state `extraction_status` exists to carry:

```
135  inference_settings.multiple_comparison_method = 'none'
 63  analyses.name = 'None'
 84  model_estimations.hrf_model = 'not applicable' / 'not_applicable'
 15  analyses.effect.cells.direction = 'not_reported'
 18  groups.medication_status = 'Not reported' / 'not reported'
  8  analyses.spatial_scope = 'not_applicable'
  5  analyses.effect.statistic.family = 'not_reported'
```

This count deliberately excludes the cases where `not_applicable` **is** a permissible value
— `design.blinding` ×930 and `design.allocation` ×517 are correct and must not be swept up by
a fix. A guard belongs in `extraction/record/fix`: a value equal to a sentinel the slot does
not permit becomes `extraction_status: not_reported` with the value dropped.

## What I would build, in order (1 and 3 are done)

1. ~~**`correction_scope`: add `_` to the whole-brain class and a `searchlight` rule.**~~
   **Done** — 50.8% → 100% on 433 values.
2. **`tasks.design_type`**, a new `ClosedField` with four rules and a decisive `mixed`. 73%
   of its values are outside the enum and 91% are recoverable.
3. ~~**`coordinate_space`: three rules** — spelled-out ICBM, `\bfsaverage` without the
   trailing boundary, and an OTHER rule for the study-specific templates.~~ **Done** —
   98.81% → 99.67%, and the templates now land in OTHER, where a transform refuses them,
   rather than UNKNOWN, where a caller may default.
4. **`height_threshold_type`, `age_unit`, `statistic.family`, `spatial_scope`,
   `model_family`, `stage`** — five more `ClosedField`s, each 6 lines, all ≥ 93%.
5. **A fifth shape, multi-label**, for `software` and `stage`: every rule runs, every hit is
   kept. Then `software` at 82% is a real 82% and not a ceiling.
6. **`echo_time_seconds` unit repair** in `fix/derive`, on the ≥ 0.5 s mode only, reported
   per record.
7. **`clusterwise_threshold_type`**: require it when the value is present, and backfill the
   927 existing records from the value's magnitude.
8. **The sentinel guard**, permissible-value aware.

Items 1–4 are the same mechanism the package already has, applied to fields nobody has
declared closed yet. Nothing here needs an encoder.

## Method and limits

`beast:/tmp/profile_records.py` walks every record and records, per dotted path, the fill
count, `extraction_status` histogram, value-type histogram, the 250 most common short values
and numeric quantiles. Paths are resolved against the current schema with
`pondie.schema.reader`, following `is_a` and descending into abstract ranges
(`AnalysisDetails`, `Acquisition`) so a slot declared on `MRI` resolves under `acquisitions`.
Coverage figures come from running the real normalizers, and the prototypes, over the
captured value counts.

Three limits worth stating. The value capture keeps strings of 200 characters or fewer, so
the long free-text fields are profiled by their length and cardinality rather than their
content — `groups.medication_status` shows 785 of its 1,544 values for that reason. Five
projects in two domains (addiction/cue-reactivity and dementia/VBM) is not the whole
literature, and the design-type and modality mixes will shift elsewhere. And every value here
came from one model at one version; a surface-form distribution is partly a fact about the
extractor, which is an argument for the residual report, not against the lexicon.
