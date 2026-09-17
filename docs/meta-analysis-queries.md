# Can the records answer the queries a meta-analyst runs?

Each meta-analysis in neurometabench states its inclusion and exclusion criteria in prose.
`scripts/query_metaanalyses.py` decomposes them into deterministic predicates over the
extraction schema, runs them against the 1,817 committed records, and scores the result
against the benchmark's own included set. The criteria each predicate translates are quoted
in its docstring, so the translation can be checked rather than trusted.

```
python scripts/query_metaanalyses.py --records '<dir>/*/*.extraction.json' \
    --bench <neurometabench>/data [--repair] [--literal]
```

**The measurement turns on a third answer.** A predicate returns True, False, or **None for
"this record cannot say"**, and the difference between the last two is the whole point: a
paper excluded because it used an ROI is a correct exclusion, and a paper excluded because no
analysis states its scope is a hole in the record. A `WHERE` clause collapses them into one
number and the schema then looks like it works.

## The result

Recall against the benchmark's included set, over the papers that reached screening:

| | papers | gold present | strict recall | permissive recall |
|---|---|---|---|---|
| `vbm_of_ptsd` | 50 | 17 | **52.9%** | 82.4% |
| `dementia` | 448 | 58 | **53.4%** | 89.7% |
| `cue_reactivity` | 550 | 140 | **57.1%** | 72.9% |
| `vbm_of_substance_use` | 244 | 76 | **59.2%** | 71.1% |
| `emotion_regulation_2022` | 525 | 87 | **36.8%** | 59.8% |

*strict* excludes a paper the record cannot answer for; *permissive* admits it. **The gap
between the two columns is entirely records that cannot say**, and it is 20 to 36 points. On
`emotion_regulation_2022`, 40 of 87 gold papers are unanswerable on at least one criterion and
only 15 are contradicted.

## What breaks a query, in order

### 1. A scalar enum holding a one-item list, and it is the largest effect measured here

`--literal` compares a scalar slot to a string the way a query actually does, instead of
flattening a one-item list as the harness otherwise kindly does. The whole-brain criterion --
the one criterion every meta-analysis in this benchmark states -- then becomes unanswerable on
most of the corpus:

| | `whole brain` unanswerable | after `unwrap_singletons` |
|---|---|---|
| `cue_reactivity` | 388 of 550 (71%) | **15** |
| `emotion_regulation_2022` | 355 of 525 (68%) | **17** |
| `dementia` | 315 of 448 (70%) | **24** |
| `vbm_of_substance_use` | 147 of 244 (60%) | **7** |
| `vbm_of_ptsd` | 20 of 50 (40%) | **1** |

And strict recall under a literal comparison:

| | before | after |
|---|---|---|
| `vbm_of_ptsd` | 5.9% | **47.1%** |
| `dementia` | 10.3% | **51.7%** |
| `cue_reactivity` | 5.0% | **52.9%** |
| `vbm_of_substance_use` | 10.5% | **59.2%** |
| `emotion_regulation_2022` | 5.7% | **35.6%** |

An eightfold difference from one shape defect, on the criterion the whole literature is
selected by. `unwrap_singletons` closes it; the point of measuring it here is that a checker
counting 21,701 malformed fields does not convey that.

### 2. `Analysis.coordinate_space` is absent where the criterion is universal

"reporting foci as 3D coordinates in Talairach or MNI stereotaxic space" is stated by every
one of the five. Unanswerable on **187 of 448** dementia records (42%), 86 of 244 substance
use (35%), 81 of 550 cue (15%), 77 of 525 emotion regulation (15%), 13 of 50 PTSD (26%).

The space is *known* -- stage 1 parses it off the table, and `Table.coordinate_space` is a
`deterministic` slot for exactly that, filled by `derive_coordinate_spaces`. The analysis-level
slot is the one a query reads and it is the one that is empty.

### 3. Age, which four of the five state, is unanswerable on a third of records

"age 18-60", "studies in children/adolescent (<18 y)" excluded, "adult humans". Unanswerable
on **151 of 525** emotion regulation records (29%) and 21 of 50 PTSD (42%) -- and on **17 of
87** and **7 of 17** gold papers respectively. `Group.age_minimum` falling back to `age_mean`;
neither is present.

### 4. The between-group contrast, which is what these meta-analyses pool

"a between-subjects contrast comparing smokers to matched nonsmoking participants",
"bvFTD < HC", "users < non-users". Expressed as two cohorts on one term's levels, which is
the only encoding a query can traverse. Unanswerable on **141 of 448** dementia records (31%)
and 17 of 58 of its gold. This is `docs/record-defects.md` finding 2 seen from the query side:
`FactorLevel.groups` is carried by 47.5% of levels and the rest are bare strings.

`name_links` recovers 29 of these across the corpus and does not move the number -- the 715
links it writes are mostly `conditions`, and the cohort half of the join is the half that is
missing.

### 5. `InferenceSettings.correction_scope`, for the SVC exclusion

"we excluded studies using region of interest (ROI) or small volume correction (SVC)".
Unanswerable on 113 of 550 cue records (21%) and **25 of 140** of its gold. The slot exists and
is right -- the correction's own domain rather than the analysis's -- and is unfilled.

## What the schema has no slot for

Three criteria classes cannot be written at all.

**Language.** "English language" is an inclusion criterion in at least six of the sixteen
meta-analyses. There is no slot on `Study`, and PubMed returns it.

**Sample overlap.** "overlapping samples to previous studies" excluded (VBM of PTSD);
"Published papers that examined one or [more overlapping cohorts]" (cannabis). `ExternalDataset`
carries `name` and `url` and nothing a query could join two papers on -- no accession, no
cohort identifier -- and it is present on 132 of 1,817 records. Two papers reporting the same
scans are indistinguishable from two independent ones, and this is the criterion that most
directly biases a pooled result.

**Whether an analysis reported anything.** "null effects" excluded (VBM of PTSD); "BOLD/rCBF
increases" required (problem solving). There is no analysis-level count of reported foci.
`Table.coordinate_count` is the nearest slot, it is per table rather than per analysis, and it
is **absent on all 2,267 tables**.

## What the schema has and nothing fills

**`Study.study_type` is `None` on all 1,817 records, and nothing in `pondie/` writes it.**
It is `deterministic`, its description names the PubMed E-utilities publication types, and its
vocabulary is exactly what six criteria need: `Review`, `Systematic Review`, `Meta-Analysis`,
`Case Reports`, `Editorial`, `Letter`. "editorial letters, case-reports, systematic reviews,
meta-analyses, and methodological studies" (sleep deprivation) and "systematic reviews or
meta-analyses" (social) are unexpressible today and cost one API call per paper.

**`Table.coordinate_count` is absent on all 2,267 tables**, also `deterministic`. The stage-1
parse holds the points per analysis, and the `Tables` stage already reads that parse -- it
copies `table_number`, `caption` and `footer` and could copy the count in the same pass.

## Proposals, ordered by what they buy a query

1. **Run `unwrap_singletons` over the corpus.** Measured: strict recall 5–10% → 36–59%. The
   repair exists; the records predate it.
2. **Fill `Study.study_type` from PubMed** and add a `language` slot filled the same way. Both
   are deterministic, both are one lookup, and together they express the publication-type and
   language criteria that every meta-analysis in the benchmark states and none can currently
   be asked about.
3. **Fill `Table.coordinate_count` in the `Tables` stage** from the parse it already reads, and
   consider an analysis-level count, since "did this contrast report any foci" is a criterion
   and a per-table number cannot answer it.
4. **Propagate `coordinate_space` from the table to the analysis** where the analysis is silent
   and its cited tables agree. `derive_coordinate_spaces` already does the table half.
5. **`FactorLevel.groups`** is the join four of the five criteria need and 52% of levels lack.
   Finding 7 of `record-defects.md` is the larger half of it and is a prompt problem.
