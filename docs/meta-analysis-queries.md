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

## Done: the corpus backfilled

`scripts/backfill_records.py` applied `unwrap_singletons` and filled `Study.study_type` from
PubMed over all 1,817 records — 21,701 unwrapped fields and 1,817 filled slots, stamped
`repaired_by: pondie-repair-1+backfill-1`, because a backfilled record is no longer the one
the extractor produced.

**The records are older than the schema, so both fills were argued rather than assumed.**
`unwrap_singletons` reads the *current* schema to decide which wrappers are scalar, which is
only safe if no slot it touches changed cardinality since extraction: the records date from
2026-09-09 and 09-11, the storage commits after that are the cohort-trait fields, the modality
rename and the design enum, and `Region.region_type` and `Region.definition_method` — which
hold the most multi-item values — were scalar at extraction too. So the repair fixes a defect
that was already one. `study_type` comes from PubMed rather than the paper and does not depend
on the extraction-time schema at all.

What the backfill does not fix is reported rather than hidden: 4,237 groups still carry no
`population_characteristics` and 1,405 tasks still say `response_mode`, because those slots
landed after these records were written.

Strict recall under a literal scalar comparison, from the records themselves with no repair
applied at read time:

| | before the backfill | after |
|---|---|---|
| `vbm_of_ptsd` | 5.9% | **47.1%** |
| `dementia` | 10.3% | **51.7%** |
| `cue_reactivity` | 5.0% | **52.9%** |
| `vbm_of_substance_use` | 10.5% | **59.2%** |
| `emotion_regulation_2022` | 5.7% | **35.6%** |

PubMed answered for all 1,804 distinct pmids (1,817 files: 13 papers are screened by two
projects each). The publication-type exclusion now runs, and excludes 23 papers across the
five as non-original research — **21 of the 23 are correctly outside the gold set**.

The two that are not are in `vbm_of_substance_use`'s included set and are reviews:
`23142417` "The role of default network deactivation in cognition and disease" (`Review`) and
`27793597` "Impact of general cognition and executive function deficits on addiction treatment
outcomes: Systematic review and meta-analysis" (`Systematic Review`). That meta-analysis's own
criterion is "only empirical English language MRI studies", so the criterion and the gold set
disagree and the criterion looks right. Two of 79, inferred from the title and the PubMed
type rather than from reading the papers.

## Proposals, ordered by what they buy a query

1. ~~Run `unwrap_singletons` over the corpus.~~ Done, above.
2. ~~Fill `Study.study_type` from PubMed.~~ Done, above. Language was dropped as out of
   scope.
3. **Fill `Table.coordinate_count` in the `Tables` stage** from the parse it already reads, and
   consider an analysis-level count, since "did this contrast report any foci" is a criterion
   and a per-table number cannot answer it.
4. **Propagate `coordinate_space` from the table to the analysis** where the analysis is silent
   and its cited tables agree. `derive_coordinate_spaces` already does the table half.
5. **`FactorLevel.groups`** is the join four of the five criteria need and 52% of levels lack.
   Finding 7 of `record-defects.md` is the larger half of it and is a prompt problem.

# Three selectors, one denominator

`scripts/compare_screening_to_queries.py` scores a language model reading the full article,
the same model reading the extraction record, and the deterministic query, against the same
gold and over the same pool: **the papers all three can see** -- extracted *and* reached
full-text screening. That matters because `compare_arms.py` scores over the papers all arms
screened in common, which omits gold lost upstream, and its PTSD recall reads 0.941 where
end-to-end it is 16/22.

Mean over the five projects:

| selector | precision | recall | F1 |
|---|---|---|---|
| autonima, full text | 40.0% | **95.3%** | 0.555 |
| autonima, record + evidence | 44.4% | 84.1% | 0.567 |
| autonima, record, no evidence | 45.3% | 82.3% | 0.570 |
| query, strict | **48.8%** | 45.8% | 0.430 |
| query, permissive | 33.0% | 69.0% | 0.409 |
| query veto + full text | 50.4% | 67.4% | 0.532 |
| query veto + record + evidence | **52.1%** | 62.0% | 0.527 |

## F1 ranks these wrongly, and it is worth saying so

The veto compositions and the record arms beat full text on F1. **They should not be
preferred on that basis.** A screening stage feeds a shortlist to a human and to a
coordinate extraction; a missed study biases the pooled estimate and cannot be recovered
downstream, while a false positive is removed by the next reader at the cost of their time.
Those errors are not exchangeable, and F1 assumes they are.

On the objective that matches the task, **full text wins clearly: 95.3% recall against
84.1% and 82.3% for the record arms and 45.8% for the query.** Every alternative here buys
precision with recall. That is the same conclusion `AUDIT.md` reached from the arm contrast
and it survives the addition of a third selector.

## What the query is good for is measurement

The query's 23-point gap between strict (45.8%) and permissive (69.0%) is **entirely
records that cannot answer**, so it is a direct read on record completeness rather than on
the query's logic. Used that way it localises the gap: `vbm_of_ptsd` strict precision is
**81.8%** -- the highest single figure in the table -- on 17 gold papers, because that
project's records answer most criteria.

It is also free. The arms cost a model pass per paper; the query costs nothing at screening
time, and the veto removes 17 to 262 papers per project that a screener then never reads.
As a prefilter ahead of the model, on a corpus where the records are complete, that is a
real saving. On these records it is not safe: the veto drops 38 of cue reactivity's 140
gold papers and 47 of substance use's 65.

## Correcting two of my own translations made the query worse, and that is the finding

Two predicates were mistranslations rather than record defects, and fixing them moved the
failure rather than removing it.

**`all cohorts healthy` → `a healthy cohort`.** Emotion regulation's criterion explicitly
admits a paper with patients "if they reported results for a control group separately", and
requiring every cohort to be healthy vetoed 16 of its 87 gold papers. Faithful now.

**`no pharmacological arm` read off `allocation`, not off whether an Arm is declared.** The
arm version vetoed 7 of substance use's gold against 12 non-gold, barely better than chance,
because the records invent Arms for diagnostic cohorts -- 47 of 164 `parallel`-with-arms
records have every arm name identical to a cohort name.

The corrected version is worse, and for a reason already documented: **45 of substance use's
76 gold papers carry `allocation: non_randomized`** for observational studies where nothing
was administered. `non_randomized` cannot distinguish "assigned to a drug non-randomly" from
"split by diagnosis", so "presence of pharmacological manipulations" is unanswerable from
these records either way, and query recall there falls from 66.2% to **23.1%**.

That is finding 9 of this document, and `AssignmentStructure.observational_cohorts` with the
tightened `Allocation.non_randomized` description is its fix -- for future extractions. These
records predate it, and no deterministic backfill can recover a label the model was steered
into by a vocabulary with no right answer in it.
