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

Recall against the benchmark's included set, over the papers that reached screening. These
are the current numbers, after the predicate audit recorded in
[what the records changed](#what-interrogating-the-records-against-the-gold-changed); the
column beside each says what it read before that audit, and the two that moved are the two
where a predicate, not a record, was wrong.

| | papers | gold present | strict recall | permissive recall | strict, before the audit |
|---|---|---|---|---|---|
| `vbm_of_ptsd` | 50 | 17 | **52.9%** | 82.4% | 52.9% |
| `dementia` | 448 | 58 | **53.4%** | 89.7% | 53.4% |
| `cue_reactivity` | 550 | 140 | **81.4%** | 95.0% | 57.1% |
| `vbm_of_substance_use` | 244 | 76 | **22.4%** | 68.4% | 59.2% |
| `emotion_regulation_2022` | 525 | 87 | **43.7%** | 73.6% | 36.8% |

*strict* excludes a paper the record cannot answer for; *permissive* admits it. **The gap
between the two columns is records that cannot say**, and it is 13 to 46 points.
`vbm_of_substance_use` fell because its pharmacological-manipulation predicate stopped
guessing from `StudyDesign.allocation`: it now answers from `Arm.arm_kind` where the record
declares an arm and says "cannot say" where it does not, which is 49 of its 76 gold papers.

## What breaks a query, in order

### 1. A scalar enum holding a one-item list, and it is the largest effect measured here

`--literal` compares a scalar slot to a string the way a query actually does, instead of
flattening a one-item list as the harness otherwise kindly does. The whole-brain criterion --
the criterion four of the five state -- then becomes unanswerable on
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

### 5. ~~`InferenceSettings.correction_scope`, for the SVC exclusion~~ -- withdrawn

This read "we excluded studies using region of interest (ROI) or small volume correction
(SVC)" as cue reactivity's criterion, and it is not: that review **includes** small-volume
corrected experiments and says so. The field is unanswerable on 113 of its 550 records, but
the criterion never asked, so the predicate has been removed rather than counted. See
[what the records changed](#what-interrogating-the-records-against-the-gold-changed),
correction 3.

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
| full text → query veto | 50.4% | 67.4% | 0.532 |
| record + evidence → query veto | **52.1%** | 62.0% | 0.527 |

The last two rows are a **pipeline, not a fourth selector**:

```python
selected = arm_included - vetoed
vetoed   = {paper : some predicate answered False on its record}
```

Take every paper the arm included, then drop the ones whose record *contradicts* a stated
criterion. Three properties follow, and the first is the design:

- **The veto fires on `False` alone**, never on "cannot say". A record that is silent is left
  to the screener; only a record that positively says `spatial_scope: roi` or
  `allocation: non_randomized` overrides a model. The query's confident exclusions and its
  silences are different things and only the first is worth acting on.
- It is a **subset of the arm**. The query can only remove a paper, never add one the arm
  excluded, so recall can only fall and precision can only rise or hold.
- In this order the arm's model pass still happens on every paper. The saving -- a paper the
  screener never reads -- needs the veto to run *first*, which is a deployment and not what
  is measured here.

They were labelled `query veto + <arm>` until someone asked what it meant: `record +
evidence` has a plus in its own name, so the composition operator and the arm name were the
same symbol.

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

## Five papers the strict query wrongly included, and what they show

Asked for false positives I would defend as false, five hold up — and all five turned out to
fail a criterion the **records already answer**, through a predicate I had not written. The
query was too coarse; the records were not too thin.

| pmid | the query's error | what the record says |
|---|---|---|
| `32977211` | PTSD: "focused on gray matter structural differences", and this uses **diffusion tensor imaging and region-based morphometry**. Also excluded as a "non-voxel-based morphometry method". | `Measure: diffusion_metric/diffusion` — and no grey-matter measure at all |
| `28971228` | Substance use: "assessing GM volume differences". Measures **cortical thickness**, and its own abstract says it "focused on empathy-related brain areas", an ROI restriction. | `Measure: cortical_thickness` |
| `26507433` | Substance use: no non-using control group — it contrasts **early-onset against late-onset users** — and measures cortical thickness, GWR and gyrification, not volume. | `Measure: cortical_thickness`, `Effect.kind: interaction` |
| `26947584` | Substance use: not a users-versus-controls contrast. It **regresses grey matter on a continuous AUDIT score** across a 436-person range of severity. | `Effect.kind: cross_subject_regression`, `ModelTerm: AUDIT score:continuous` |
| `20424827` | Substance use: contrasts a **smoking-cessation treatment outcome** (abstinence at 4 weeks) within smokers, so it is both a treatment study and not a users-versus-controls design. | terms are `cigarettes per day:continuous`, `smoking cessation outcome:categorical` — no user/control group factor |

A sixth, `30456877`, tests a **gene × smoking interaction** rather than a main effect of
smokers against non-smokers: `ModelTerm: rs1137070 genotype:categorical` and
`smoking status:categorical`.

### The correction, and what it cost

Two predicates were missing and both were expressible:

- **`measures(pattern)`** on `Measure.type`. Without it, a criterion naming grey-matter
  *volume* admitted any structural study, because the modality predicate saw "structural
  MRI" and stopped.
- **`group_contrast`** requiring `Effect.kind == contrast` before asking for two cohorts on
  a term, because a criterion naming a group difference does not admit a regression on a
  continuous exposure or an interaction.

| | strict precision before | after |
|---|---|---|
| `vbm_of_ptsd` | 81.8% | **90.0%** |
| `vbm_of_substance_use` | 68.2% | **86.7%** |
| mean over five | 48.8% | **54.1%** |

Recall fell with it -- substance use from 23.1% to 20.0% -- so the query is now precise and
narrow rather than precise and broad, and it is still far short of any screener on the
objective that matters. What changed is the diagnosis: a meaningful part of what I had
attributed to incomplete records was an incomplete translation of the criteria, and
`Measure.type` and `Effect.kind` were sitting in every record the whole time.

One paper I will not claim: **`16371250`**, "Gray matter density reduction in the insula in
fire survivors with posttraumatic stress disorder", 12 fire victims with PTSD against 12
matched victims of the same fire without PTSD, by VBM. It meets every criterion the PTSD
meta-analysis states and is not in its included set, so it reads as a gold-set omission
rather than a query error -- which, with the two reviews the publication-type filter found
inside `vbm_of_substance_use`'s included set, makes three disagreements with the benchmark
that look like the benchmark's.

## Five papers the strict query wrongly excluded

The mirror of the last section, and the five split three ways rather than one.

| pmid | the record's claim | the paper | verdict |
|---|---|---|---|
| `17133391` | `spatial_scope: roi` on **all 12** analyses, `correction_scope: roi` | whole-brain fMRI of voluntary emotion regulation, reporting prefrontal and amygdala effects | **record defect** |
| `15127179` | `spatial_scope: roi` on both analyses | cue-induced striatal and medial-prefrontal activation, 10 abstinent alcoholics against controls, included by a meta-analysis that excludes ROI studies | **record defect** |
| `16687507` | `spatial_scope: roi` on its one analysis | "Individual differences in reward drive predict neural responses to images of food" -- a correlation across the brain whose *findings* are in orbitofrontal, striatal and amygdala regions | **record defect** |
| `11822992` | `gray_matter_density`, `spatial_scope: whole_brain`, `correction_scope: whole_brain` -- all correct | "voxel based morphometry ... gray and white matter **concentration**" in cocaine-dependent against cocaine-naive | **my predicate**: it asked for `gray_matter_volume` and VBM measures density |
| `14667419` | `stimuli: "Alcohol-related and neutral words"`, `spatial_scope: whole_brain` -- correct | fMRI response to alcohol-related **words** in alcohol-dependent young women | **my predicate**: written words on a screen are a visual cue and the pattern did not admit text |

And a sixth that is neither: **`21686071`**, "How grossed out are you? The neural bases of emotion regulation **from childhood to adolescence**", record `age_minimum: 7`, `age_mean: 13.03`. The emotion-regulation meta-analysis's first criterion is "studies of healthy **adults**". The query is right and the paper is in the included set -- a fourth benchmark disagreement, after `16371250` and the two reviews.

### The three record defects are one defect, and it is the most consequential one

`spatial_scope: roi` where the paper is whole-brain contradicts **38 gold papers**: 20 in cue
reactivity, 12 in emotion regulation, 4 in dementia, 2 in PTSD. "Whole brain, not ROI" is the
single criterion four of the five state, so this one field decides more
inclusions than any other, and all three examples above share a shape: **the paper names
regions because that is how a whole-brain result is reported, and the record reads the names
as a restriction.**

The largest driver overall is not this one. `no pharmacological arm` contradicts **39 of
substance use's 65 gold papers**, and that is the `allocation: non_randomized` defect of
finding 9 -- a diagnosis read as an assignment. `observational_cohorts` is its fix and these
records predate it.

### A diagnostic that failed, and one that partly works

I expected `spatial_scope: roi` with no `regions` reference to flag the mislabel -- an ROI
analysis must have a region. **It does the opposite.** Of 1,865 `roi` analyses, the ones in
gold papers name no region 3% of the time against 10% in non-gold. So the mislabelled records
are internally coherent: they name regions and still disagree with the meta-analysts, and no
consistency check inside the record will find them.

What does carry signal is the table metadata. Among gold papers whose record says **no**
whole-brain analysis anywhere, **43% declare no Table at all**, against 24% of the gold papers
whose record does say whole-brain. Missing table metadata nearly doubles the rate of the
mislabel, which is the 17133391 diagnosis generalising: the caption stating "whole-brain
analysis" never reached the extractor. That makes the `Tables` stage's parse fallback a
testable prediction rather than a tidy-up -- 654 of the 1,143 dangling `Analysis.tables`
references came from exactly these papers. 47% of the mislabels have captioned tables, so it
is not the whole cause.

### The two predicate errors, fixed

`measures` now asks for `gray_matter` rather than `gray_matter_volume`, because VBM measures
density or concentration and the criterion says volume -- the narrow pattern dropped 19 gold
papers. And `visual_stimuli` admits text, and disqualifies a task by finding gustatory,
olfactory or tactile cues rather than by failing to find the word "visual" -- which is what
the criterion actually says. That dropped 10.

| | before | after |
|---|---|---|
| `cue_reactivity` strict recall | 57.1% | **60.7%** |
| `vbm_of_substance_use` strict recall | 20.0% | 21.5% |
| mean strict | 45.2% / 54.1% precision | **46.2% / 54.3%** |

A point of recall for no precision, which is the right direction and a small effect: after
four rounds of correcting my own translations, what is left is dominated by two record
defects rather than by the difficulty of writing the query.

## The comparison again, after the audit

Six predicate corrections later -- four found by reading the query's own errors, two more by
reading the criteria against the records, all of them in
[what the records changed](#what-interrogating-the-records-against-the-gold-changed) -- the
comparison over the four projects that have all three arms:

| selector | precision | recall | F1 |
|---|---|---|---|
| autonima, full text | 43.3% | **95.2%** | 0.588 |
| autonima, record + evidence | 48.6% | 84.2% | 0.606 |
| autonima, record, no evidence | 49.4% | 82.0% | 0.606 |
| query, strict | **60.2%** | 53.3% | 0.475 |
| query, permissive | 41.6% | 85.8% | 0.529 |
| full text → query veto | 53.5% | 84.5% | 0.634 |
| record + evidence → query veto | 56.5% | 78.0% | **0.637** |

**Where the records can answer, the deterministic criteria match a model reading the same
records and now slightly beat them.** `query, permissive` reaches **85.8% recall against the
record arms' 84.2% and 82.0%** -- admitting the papers the record is silent about, the
published criteria applied mechanically select as much of the gold as a language model
reading the record does. It does so at 41.6% precision against their ~49%, so it is not a
replacement; it is evidence that the criteria are expressible and that what the model adds
over them is judgement about silence.

**`vbm_of_ptsd` is what this looks like when the records are good.** Query strict reaches
88.9% precision there, and both veto compositions reach F1 0.70 against full text's 0.606.

### The conclusion does not change

Full text still wins on recall -- 95.2% -- and recall is the objective screening has. A
missed study biases the pooled estimate and cannot be recovered downstream; a false positive
costs the next reader some time. Every alternative in the table buys precision with recall,
the veto included, and the F1 column should not be read as a ranking.

What the corrections established is where the remaining gap lives. It is not in the
difficulty of writing the criteria down -- every correction so far has been a predicate
reading the wrong field or the wrong side of a negation, and each was worth 10 to 20 points
of recall. What is left is two record defects: `spatial_scope: roi` on whole-brain papers,
and a `pharmacological` arm that four fifths of the records do not declare either way.

# Analysis selection, which is the stage that actually decides the map

Screening picks papers. A coordinate meta-analysis pools **contrasts**, and the benchmark
names them: "non-PTSD > PTSD", "bvFTD < HC (all & by modality)", "users < non-users". The
ground truth is `nimads/<project>/merged/`, whose annotation carries one boolean per key per
analysis, so the analyses a published meta-analysis pooled are recorded per key. In autonima
this stage is `annotation_results.json` -- an `include` per analysis per key -- and it has not
been measured against that gold before now. `scripts/query_analysis_selection.py` does it,
alongside a deterministic query over `Effect.cells`.

Each key is a signed contrast -- between two cohorts, or between two conditions -- optionally
restricted by modality, and all three are things the schema encodes: a `Cell.direction` on a
`FactorLevel` that reaches a `Group` or a `Condition`, and `Analysis.measure` →
`Measure.family`. So the question is not whether the schema can state the contrast. It can.
The specs are in `scripts/query_contrasts.py`, one per key, read by both scripts that ask.

**Both selectors go through one scorer.** An earlier version of this table read autonima's
columns out of a csv built elsewhere and the query's from the records, which is how two
numbers meant to be compared end up measuring different things. `query_analysis_selection.py`
now reads each arm's own `annotation_results.json` and `coordinate_parsing_results.json`,
runs the query over the same papers, and scores both the same way -- including the
coordinates, which the first version could not do at all:

| project / key | selector | saw | ≥1 | =count | coord P | coord R | coord F1 |
|---|---|---|---|---|---|---|---|
| `vbm_of_ptsd` / non-PTSD > PTSD (17) | autonima, full text | 8 | 41% | 35% | **93.2%** | 48.9% | 0.642 |
| | autonima, record + evidence | 7 | 41% | 35% | 91.9% | 48.9% | 0.638 |
| | query, strict | 17 | **94%** | **71%** | 88.6% | **50.4%** | **0.642** |
| | query, permissive | 17 | 100% | 47% | 53.4% | 67.6% | 0.597 |
| `dementia` / decrease (14) | autonima, record + evidence | 6 | 43% | 29% | **91.9%** | 26.8% | 0.415 |
| | query, strict | 14 | **86%** | **43%** | 56.2% | **35.3%** | **0.433** |
| `dementia` / functional (11) | autonima, record + evidence | 6 | 45% | **36%** | **95.6%** | 18.9% | 0.315 |
| | query, strict | 11 | **82%** | 27% | 59.7% | **33.8%** | **0.431** |
| `vbm_of_substance_use` / all drug classes (74) | autonima, record + evidence | 53 | 62% | **43%** | **83.9%** | **65.4%** | **0.735** |
| | query, strict | 74 | **73%** | 32% | 71.5% | 62.2% | 0.665 |
| `vbm_of_substance_use` / alcohol (19) | autonima, record + evidence | 16 | 74% | **53%** | **84.0%** | 72.4% | 0.778 |
| | query, strict | 19 | **84%** | 32% | 74.4% | **86.8%** | **0.801** |
| `cue_reactivity` / drug > neutral (106) | autonima, full text | 82 | **76%** | 25% | 54.3% | **65.1%** | **0.592** |
| | autonima, record + evidence | 82 | 72% | 21% | 53.8% | 62.9% | 0.580 |
| | query, strict | 105 | 60% | **32%** | **70.9%** | 41.7% | 0.525 |
| | query, permissive | 105 | 86% | 23% | 49.6% | 52.7% | 0.511 |

`saw` is how many of the gold papers the selector had anything to choose from, and it is the
column that keeps the rest honest: **an annotation cannot include an analysis its own arm
never parsed**. Autonima saw 6 to 16 of the between-group projects' gold papers and the query
saw all of them, because the query reads a record that exists for every paper. The two
selectors are therefore not competing on equal footing at this stage -- the comparison that
does put them on one is the arms comparison below, where both run end to end.

The full table, all nine keys and five selectors, is what the script prints.

## CORRECTION: this is not analysis selection, and the claim above was wrong

I wrote that analysis selection is the bottleneck. **It is not.** Auditing the metric found
two things, and the second overturns the conclusion.

**First: where the annotation does select, the coordinates are exact.** The first version of
the table above compared foci *counts*, which two different analyses can share -- which is
why the table now carries coordinate precision and recall instead. Comparing the coordinate
*sets* for `vbm_of_ptsd`, over the 7 gold studies both the gold and the annotation contain:

```
count agrees on 7/7, the SET agrees on 7/7
foci: gold 73, auto 73, shared 73  ->  recall 100%, precision 100%
```

So extraction, parsing and selection are exactly right on every study that gets through. The
count agreement was not luck, and there is no coordinate-level disagreement to explain.

**Second: the studies that do not get through are almost never lost at annotation.** Placing
each gold study in the channel that lost it:

| project / key | contributed | annotation selected none | **absent from the coordinate parse** | rejected at screening | never reached screening | gold |
|---|---|---|---|---|---|---|
| `vbm_of_ptsd` / non-PTSD>PTSD | 7 | 1 | **8** | 1 | 5 | 22 |
| `cue_reactivity` / reward | 89 | 2 | **23** | 4 | 42 | 160 |
| `cue_reactivity` / drug | 81 | 1 | **21** | 3 | 7 | 113 |
| `cue_reactivity` / natural | 10 | 0 | **4** | 1 | 35 | 50 |
| `dementia` / all | 7 | 1 | **10** | 0 | 3 | 21 |
| `dementia` / decrease | 6 | 1 | **8** | 0 | 2 | 17 |
| `vbm_of_substance_use` / all | 47 | 8 | **8** | 2 | 12 | 77 |

**`annotation selected none` is the smallest channel in every row** -- 0 to 8 studies. The
large ones are search and **the coordinate parse**: for PTSD, 8 of 22 gold studies were
screened in and are simply *not in `coordinate_parsing_results.json` at all*, which holds 11
studies against a 29-study studyset. No table was parsed, so no analysis existed to annotate.

So the loss ranking is **search > coordinate parse > screening ≈ annotation**, and the honest
statement is the one `AUDIT.md` already made, extended by one stage: full-text screening is a
small channel, and the stage after it -- getting a coordinate table parsed at all -- is a
large one that no comparison in this experiment had counted.

That also connects the thread rather than opening a new one. The `Tables` stage work found 654
of 1,143 dangling `Analysis.tables` references in papers with **no table manifest for their
flavour**, and this is the same root cause reaching the map: no manifest, no parse, no
analyses, no contribution. The parse fallback added there is aimed at the second-largest
channel, which is worth knowing before anyone tunes the annotation prompt.

## What the query comparison does and does not show

The deterministic query finds a matching contrast more often than the annotation on dementia
(86%, 82%, 67% against 57%, 64%, 56%) and matches the gold count as often on PTSD. But the
annotation's apparent misses are mostly papers with no parsed analyses to annotate, and the
query is reading a *record*, which exists for those papers. So the two are not selecting from
the same candidate set, and the comparison overstates the query. It remains true that the
criteria are expressible; it is no longer evidence that the annotation pass underperforms them.

## Two measurement notes, because both bit

**Merged gold studies cannot be attributed per paper.** Dementia's gold merges up to **27
pmids into one study**; assigning that study's foci to each of them made every per-paper
comparison meaningless, and the foci column read 2-6% before I noticed. Those analyses are now
dropped and counted -- 8 in dementia, 3 in substance use -- which is why dementia's samples
fall to 9-14 papers. PTSD is 22 single-paper studies and cue reactivity 191, so neither is
affected.

**The query had no foci column**, and it should have. `Analysis.source_table_analysis` is the
exact join -- "the only exact route from an analysis to its coordinates", per its own repair --
so the record can be taken to its coordinates without a string match. It needs the stage-1
parse beside the records, and with that input the column is measured below in
[Both gates at once](#both-gates-at-once-what-the-query-alone-produces).

# What would make these queries simpler, and which normalizations to avoid

Written after translating sixteen published criteria into predicates, getting four of them
wrong, and correcting each against the gold. The evidence for every item below is a measured
cost in this document, not a preference.

## What made the queries hard to write

**Every group-contrast predicate reconstructs the same thing by hand.** `group_contrast` walks
`analyses[] -> effect.cells[] -> cell.term -> model_estimations[].terms[].levels[] ->
FactorLevel.groups -> groups[].medical_condition`, and joins `Cell.direction` on the way, to
answer "did this analysis compare a patient cohort against a control cohort, and which way".
That is 40 lines, it is the same 40 lines in `query_metaanalyses.py` and
`query_analysis_selection.py`, and **it is what every one of the sixteen criteria asks**.

A derived, deterministic `Analysis` summary -- which cohorts, which conditions, which
direction -- would collapse it to one field test. It needs no model: all four inputs are in
the record, the derivation is the walk above, and `Effect.kind` already does the same job for
a coarser question. This is the single largest simplification available.

**Two fields a query needs and cannot reach.** "Did this analysis report any foci" decides the
null-effects exclusion and cannot be answered from the record: `Table.coordinate_count` is
per-table and absent on all 2,267 tables. And `Analysis.coordinate_space` is unanswerable on
15-42% while the space is parsed off the table -- the table slot is filled and the analysis
slot a query reads is not.

## Normalizations that would help, in order of what they cost me

| field | what I had to write | what it cost |
|---|---|---|
| `Group.medical_condition` | `alcohol\|nicotine\|tobacco\|smok\|cocaine\|cannabis\|opioid\|heroin\|methamphetamine\|substance\|depend\|abuse\|addict` | a MONDO subsumption query replaces the whole alternation; it still vetoed 6 gold papers whose condition wording I had not anticipated |
| `Measure.type` | `gray_matter_volume` first, then `gray_matter` | the narrow form dropped **19 gold papers**, because VBM measures density or concentration and the criterion says volume |
| `Analysis.coordinate_space` | `mni\|talairach\|tal\b\|icbm\|mni152\|asym` | a two-value enum would make this `== MNI`; the regex is the reason I cannot tell a missing space from an unrecognised one |
| `Task.stimulus_modality` | a prose fallback over `stimuli` and `description` | the fallback dropped **10 gold papers**; the field now exists and these records predate it |
| `StudyDesign.allocation` | `not_applicable\|single_arm` | not a normalization problem -- a vocabulary gap, and it cost **45 of 76** gold papers in one project |

`Group.is_healthy` is the counter-example that shows the shape working: because it is derived
from `medical_condition` in code, the healthy-cohort predicate is three lines and needed no
negation lexicon of its own.

## Normalizations that would be harmful, and the field each would damage

These are not hypotheticals. Each is a normalization that looks obviously good and is
contradicted by a measurement here.

**`Analysis.spatial_scope` -- never infer it.** Inferring `roi` from region mentions is the
single largest record defect found: **38 gold papers** contradicted, across four projects, on
the one criterion every meta-analysis in the benchmark states. And the obvious consistency
rule fails: `roi` with no `regions` reference is *rarer* in gold papers (3%) than in non-gold
(10%), so the mislabelled records are internally coherent and a normalization pass has nothing
to key on. Any rule that reads scope from prose, or from `Analysis.regions` being populated,
makes this worse.

**`Cell.level` and `FactorLevel.level` -- do not canonicalise.** They are a join key, and
`spans.fold_label` already states the rule: "`Healthy controls` and `healthy controls` are the
same level; `AD` and `AD group` are not, and calling them equal here would hide the join
failure rather than report it." 1,713 levels reach no entity and 715 are recoverable by an
*exact* fold; loosening the fold converts a visible join failure into a silent wrong join, and
`normalize_open_fields.py` measured what containment does -- it merged the corpus's most
frequent task term into a rarer variant with 38 candidate hosts.

**`Cell.direction` -- do not derive it more aggressively.** 163 cells already state opposite
signs in `level` and `direction`. Direction decides whether a coordinate enters the increase
or the decrease map, so a normalization that resolves those contradictions by rule would pick
a side on the one field where a wrong answer changes the result.

**`Group.medical_condition` -- do not let an ontology absorb the negations.** 235 mentions
(5%) assert the *absence* of a condition, and matched against a disease ontology every one
retrieves something at plausible similarity. Normalizing before triaging turns "no psychiatric
history" into a psychiatric diagnosis.

**`Group.population_characteristics` -- do not subset-merge.** Measured: 86 of 148 merges had
more than one candidate host, and containment inverts the relation that matters -- `emotion
regulation`, the most frequent term in the corpus, is absorbed into a rarer, more specific
variant.

**`StudyDesign.assignment_structure` -- do not normalize toward the existing vocabulary.**
That is what produced the defect: the vocabulary had no value for an observational
multi-cohort design, so 1,120 records said `parallel` and 956 of them declared no arm. A
normalization pass mapping free text onto the old four values would have entrenched it.
The fix was to add a value, not to map harder.

The pattern across all six: **normalize a field whose target vocabulary exists outside the
corpus, and leave alone any field that is a join key, a sign, or a judgement the corpus is the
only evidence for.**

# Both gates at once: what the query alone produces

The two gates above were scored apart. `scripts/query_workflow.py` runs them in series --
the published inclusion criteria over the record, then the published contrast over the
analyses of the papers that passed -- joins each selected analysis to its coordinates
through `Analysis.source_table_analysis`, and scores the pooled foci against the map the
meta-analysis published. That is the whole workflow, and the foci column is the one the
measurement notes above recorded as missing.

```
python scripts/query_workflow.py --records '<dir>/*/*.extraction.json' \
    --bench <neurometabench>/data --stage1 <corpus>
```

**Both gates run strict and permissive, so each map is scored four ways.** Strict drops
what the record cannot answer and permissive keeps it, and running the pair at each gate
says where a silence costs something: a paper the screener cannot judge is a different
loss from a contrast the record cannot sign.

Three inputs and one pin. The records are the 1,817 in `record_arms`; the parse is
`<pmid>/stage1/analyses.json` from the corpus; the benchmark is **pinned at 00398b9**,
the commit the record-arms run scored against; its current head no longer carries emotion
regulation at all. Foci are compared in MNI at a 2mm tolerance -- each side moves its own
Talairach coordinates and the two transforms are not the same one, so the benchmark's
(-54.7, -59.4, -15.8) and `tal2mni`'s (-55.5, -59.7, -15.2) are one focus.

## The gates in series, per published map

`anlys` is analyses selected and `nofoci` how many of them reached no coordinates. Paper and
foci precision and recall are against the annotation's own key, over the gold papers that
have a record.

**vbm_of_ptsd**, `non-PTSD > PTSD`: 17 gold papers with a record, 139 gold foci, ceiling 67.6%.

| screen / select | anlys | nofoci | paper P | paper R | foci P | foci R | foci F1 |
|---|---|---|---|---|---|---|---|
| strict / strict | 11 | 2 | **90.0%** | 52.9% | **89.7%** | 37.4% | 0.528 |
| strict / permissive | 24 | 3 | **90.0%** | 52.9% | 61.0% | 54.0% | 0.573 |
| permissive / strict | 38 | 9 | 70.0% | **82.4%** | 78.3% | 46.8% | **0.586** |
| permissive / permissive | 83 | 19 | 58.3% | **82.4%** | 40.6% | **64.0%** | 0.497 |

**dementia**, `bvFTD vs HC`, four keys: 16, 14, 9 and 11 gold papers with a record; ceilings
37.6%, 37.6%, 48.6%, 40.8%.

| key | screen / select | anlys | nofoci | paper P | paper R | foci P | foci R | foci F1 |
|---|---|---|---|---|---|---|---|---|
| all | strict / strict | 163 | 85 | 10.7% | 56.2% | 17.3% | 30.5% | **0.221** |
| all | permissive / strict | 247 | 154 | 9.4% | 75.0% | 15.6% | 30.9% | 0.207 |
| all | permissive / permissive | 867 | 529 | 5.0% | **81.2%** | 5.4% | **31.2%** | 0.093 |
| decrease | strict / strict | 119 | 60 | 13.6% | 64.3% | **26.8%** | 28.8% | **0.278** |
| decrease | permissive / strict | 169 | 99 | 11.2% | 78.6% | 22.9% | 28.8% | 0.255 |
| decrease | permissive / permissive | 766 | 452 | 4.8% | **85.7%** | 5.7% | 29.2% | 0.096 |
| structural | strict / strict | 73 | 39 | 9.1% | 44.4% | 5.0% | 11.5% | **0.070** |
| structural | permissive / permissive | 526 | 323 | 2.7% | 55.6% | 1.3% | 11.5% | 0.024 |
| functional | strict / strict | 79 | 39 | 19.0% | 72.7% | **42.7%** | 33.3% | **0.374** |
| functional | permissive / strict | 106 | 60 | 16.7% | 81.8% | 41.6% | 33.8% | 0.373 |
| functional | permissive / permissive | 307 | 196 | 10.0% | **90.9%** | 19.2% | **34.2%** | 0.246 |

**cue_reactivity**, three keys: 118, 106 and 15 gold papers with a record; ceiling 66% on all
three.

| key | screen / select | anlys | nofoci | paper P | paper R | foci P | foci R | foci F1 |
|---|---|---|---|---|---|---|---|---|
| reward | strict / strict | 228 | 39 | **43.2%** | 48.3% | **37.0%** | 32.3% | 0.345 |
| reward | permissive / strict | 255 | 47 | 42.5% | 52.5% | 36.3% | 36.0% | **0.361** |
| reward | permissive / permissive | 932 | 232 | 29.4% | **81.4%** | 19.8% | **49.8%** | 0.283 |
| drug | strict / strict | 169 | 29 | **54.6%** | 50.0% | 44.9% | 33.2% | 0.382 |
| drug | permissive / strict | 191 | 36 | 53.2% | 54.7% | 44.4% | 36.4% | **0.400** |
| drug | permissive / permissive | 934 | 235 | 26.1% | **80.2%** | 16.2% | **47.3%** | 0.242 |
| natural | strict / strict | 65 | 13 | 12.2% | 33.3% | 15.3% | 26.9% | 0.195 |
| natural | permissive / strict | 77 | 15 | **13.0%** | 40.0% | **16.7%** | 37.3% | **0.231** |
| natural | permissive / permissive | 946 | 230 | 3.7% | **80.0%** | 3.6% | **64.6%** | 0.068 |

**vbm_of_substance_use**, `all drug classes`: 74 gold papers with a record, 511 gold foci,
ceiling 81.2%.

| screen / select | anlys | nofoci | paper P | paper R | foci P | foci R | foci F1 |
|---|---|---|---|---|---|---|---|
| strict / strict | 35 | 0 | **88.2%** | 20.3% | 46.9% | 20.7% | 0.288 |
| strict / permissive | 68 | 0 | **89.5%** | 23.0% | 38.4% | 25.0% | 0.303 |
| permissive / strict | 129 | 35 | 65.3% | 63.5% | **53.4%** | 60.3% | **0.566** |
| permissive / permissive | 285 | 76 | 61.0% | **67.6%** | 36.0% | **71.4%** | 0.478 |

**emotion_regulation_2022** stops at screening: 43.7% strict and 73.6% permissive recall over
87 gold papers, and the benchmark ships no analysis-level annotation for it, so there is no
map to score against.

## What the four cells say

**Permissive costs little at the first gate and a great deal at the second.** Strict
selection has the better foci F1 in sixteen of the eighteen comparisons -- nine maps by two
screening modes -- and the losses from relaxing it are not small: foci precision falls from
89.7% to 61.0% on PTSD, 44.9% to 16.4% on cue drug, 42.7% to 21.8% on dementia functional.
The two silences are not equivalent. A record that cannot answer "was this whole-brain" is
usually still the right paper; an analysis whose cells cannot be signed is usually not the
contrast the map pooled, and admitting it pools the wrong coordinates.

**Permissive screening is worth it where screening is the binding constraint.** It buys 3 to
43 points of paper recall, and it improves foci F1 on five of the nine maps: PTSD 0.528 to
0.586, all three cue keys, and substance use 0.288 to **0.566**, which is the largest single
effect in this document and comes from one predicate that can only say "cannot say" (see
below). It costs 0.001 to 0.023 on dementia, where selection was already losing more than
screening was.

**So the cell to run is permissive screening with strict selection.** It has the best foci F1
on six of the nine maps and is within 0.002 on a seventh. It is the same shape as the veto
argument above, reached from the other side: let a silent record through, and act only on
what the record positively says.

**The best map is still not the published one.** Foci recall tops out at 71.4% (substance
use, permissive/permissive, at 36.0% precision) and foci F1 at 0.566 (substance use,
permissive/strict). Four maps are under 0.25 either way.

## Where the foci go, and it is not the query

Of the 990 analyses the strict/strict pipelines select across the fourteen maps, **684 reach
coordinates and 306 do not**:

| | analyses |
|---|---|
| joined to coordinates | 684 |
| **`source_table_analysis` empty** | **234** |
| its address names no row group in the parse | 40 |
| the row group it names carries no coordinates | 32 |

The empty slot is dementia's: 75 of the 163 analyses selected for `all`, 52 of 119 for
`decrease`, and 195 of the 234 in that project alone. Substance use loses none of its 83. `resolve_source_table_analysis` fills the slot only where exactly one parsed
entry under the cited tables carries the analysis's name, and on that project it usually
cannot -- which is the same dangling-`Analysis.tables` finding the `Tables` stage work
measured, arriving at the map.

The ceiling makes the same point without the query in the way. Taking **every** analysis of
**every** gold paper and joining it to the parse:

| map | foci recall at the ceiling |
|---|---|
| `vbm_of_substance_use` / all drug classes | **81.2%** |
| `vbm_of_ptsd` / non-PTSD > PTSD | 67.6% |
| `cue_reactivity` / drug | 66.3% |
| `cue_reactivity` / reward | 66.3% |
| `cue_reactivity` / natural | 66.4% |
| `dementia` / structural | 48.6% |
| `dementia` / functional | 40.8% |
| `dementia` / all, decrease | **37.6%** |

No selection rule can beat those numbers, and four of the nine are under half the published
map. The gap between a cell in the tables above and its ceiling is what the two queries
cost; the gap between the ceiling and 100% is extraction and parsing, and on dementia it is
the larger of the two.

## Two things this run had to fix to be measurable

**The benchmark stores MNI where the paper published Talairach**, and the record stores what
the paper said. Cue reactivity's studyset is 3,270 MNI points and nothing else, and on 35 of the
123 gold records behind its drug key the record resolves to `TAL` instead; substance use
disagrees on 15 of 74, dementia on 4 of 16, PTSD on 2 of 17. Moving both sides into MNI with `query.engine._points` -- the same transform `pondie
query` uses -- is what makes the comparison mean anything: it took cue reactivity's ceiling
from 47.9% to 66.3%. The residual disagreements are counted per map in the script's output
rather than smoothed away, because a space the record and the benchmark disagree about
displaces foci by 5-10mm and no tolerance short of a smoothing kernel recovers them.

**Two of dementia's keys are not direction-restricted.** `structural` and `functional` are
`all` split by modality, `all` contains five analyses `decrease` does not, and `decrease` is
a subset of `all`. So the contrast spec for those three admits `bvFTD` on either side and
only `decrease` requires it lower. The earlier analysis-selection table treated all four as
lower-only; `scripts/query_contrasts.py` now holds one spec per key and both scripts read
it, so the two cannot disagree again.

# What interrogating the records against the gold changed

Every number above the arms comparison was rerun after this. The method was the same each
time: take the gold papers a query misses, put each miss in the channel that lost it, and
read the records in the largest channel. Six of the predicates were wrong, and each was
wrong in a way the criterion itself settles.

**1. A control group is named after the condition it does not have.** "nonsmoking control
subjects", "comparison group, non-use of marijuana", "cannabis non-consuming group",
"healthy controls with no history of alcohol misuse". A cohort pattern matched on words
alone puts both sides of the contrast in the cohort, which leaves the contrast with no
other side. It cost 13 of the 15 gold papers under substance use's nicotine key, 4 more
under cannabis, and 7 of PTSD's 17. `is_healthy` derives half of the distinction from
`medical_condition`; `names_cohort` in `scripts/query_contrasts.py` reads the negated
namings a parser does not see, because they are morphology rather than syntax. **115
records in the five projects have a control group that matches their own cohort pattern.**

**2. A neutral condition is described by what it is not.** Cue reactivity's records carry
`Condition.description` like "pictures of people not smoking cigarettes" and "nonalcoholic
beverage pictures matched to the alcohol pictures". A cue pattern run over that prose
matches the control side too. Matching `Condition.name` instead -- the field whose job is
to name the thing -- took the drug key from 46 to 67 of 106 gold papers.

**3. Cue reactivity does not require whole-brain analyses, and I had made it.** Its
criterion reads "experiments reporting coordinates from whole-brain **or small-volume
corrected** analyses ... were included (studies involving regions of interest [ROIs]
derived from a brain parcellation scheme were excluded **given the absence of
coordinates**)". The predicate had `spatial_scope == whole_brain` and a second test that
rejected SVC outright, which inverts the sentence. Strict recall on that project: 60.7% to
**81.4%**. The other three projects do state whole-brain only -- "(5) performed a
whole-brain analysis", "as ROI analyses violate the ALE null-hypothesis", "papers reporting
a priori regions of interest (ROIs)" excluded -- so the predicate stays for them.

**4. "Presence of pharmacological manipulations" has a field, and it is not `allocation`.**
`ArmKind` has the value `pharmacological`, glossed "an administered drug or other agent".
Reading the criterion off `StudyDesign.allocation in {not_applicable, single_arm}` vetoed
**46 of substance use's 76 gold papers and 105 non-gold**, because an observational cohort
study is recorded `non_randomized`. Five arms in that whole 244-record project are
`pharmacological`, which is what a VBM literature should look like.

**5. Half a contrast is not the other contrast.** A simple effect that carries the cue side
and no comparison -- `heroin-related, positive`, in a paper whose table is a cue-minus-
neutral map -- does not say what it was contrasted against. That is now "cannot say" rather
than False, so permissive admits it and strict does not.

**6. A level spelled `H` against `O`.** 111 conditions in the corpus have a name with no
three-letter word in it, and the task's own "foods of high hedonic value" and "neutral
nonfood objects" sit beside them unlinked. Reading the label as the other side of the
contrast reports a contradiction where the record is silent, so those are unresolved.

One more change is not a correction but a definition: the cue keys the benchmark maps are
the `_wbonly` ones, so the contrast spec for them requires the analysis's own
`spatial_scope` to be whole-brain. That is the key's meaning, not the review's criterion.

## What the records got wrong

Counted over all 1,817 records. Everything here is the record's side of a query that is now
faithful to the criterion.

| | count | of |
|---|---|---|
| cells naming a level no `ModelTerm` declares | **2,164** | 12,098 cells |
| levels reaching no entity at all -- no group, no condition | **2,000** | 6,854 levels |
| analyses with no `coordinate_space` | 1,187 | 5,989 analyses |
| cells carrying no sign: `undirected` | 900 | 12,098 cells |
| cells carrying no sign: `direction` absent | 212 | 12,098 cells |
| selected analyses whose `source_table_analysis` is empty | 234 | 990 selected |
| conditions whose `name` has no three-letter word in it | 111 | 3,013 conditions |
| cells whose `direction` is double-wrapped -- `{"value": "positive"}` with no `extraction_status` | 30 | 12,098 cells |

`held` is not in that table: 476 cells carry it and it is the vocabulary working -- a
factor held constant while another is contrasted, which is how an interaction is written.

The double-wrapped direction is a shape defect rather than a judgement: the wrapper is missing its status,
so every consumer that unwraps an `ExtractedValue` reads a dict where a string belongs and
the cell silently loses its sign. It is 30 cells in two records and it is worth fixing
because it is free to detect.

The join failures are the expensive ones, and they are the same failure twice. A `Cell`
names `<term>|<level>`; 18% of cells name a level that does not exist, and 29% of the
levels that do exist reach neither a `Group` nor a `Condition`. Either one leaves an
analysis whose contrast cannot be read at all -- not contradicted, unreadable -- which is
why the permissive column moves so much.

**The sign of a contrast is relative to a polarity the record does not state.** 103
analyses are named for a loss or an atrophy while their measure is a volume, a density or a
thickness. `26673947` records "Regions of GM atrophy in bvFTD versus control subjects" as
bvFTD **positive**; `19884571` records "bvFTD group greater frontal grey matter loss than
SMD group" as bvFTD **positive** on a `gray_matter_volume` measure. A query asking for the
contrast the meta-analysis pools -- patients lower on grey matter -- reads both as the
opposite contrast. Three records go further and give the measure a deficit-polarity type,
`gray_matter_atrophy`, which makes the sign correct and the vocabulary inconsistent.

## What the schema could hold better

Three of these are new fields and three are changes to fields that exist. Each is here
because a query needed it and had to be written around its absence, and the cost is the
measurement beside it.

**~~`Group.role`~~, and why a field cannot hold it.** Implemented, measured and removed --
the case below is real and the field is not where the answer goes. See
[was it worth it](#was-grouprole-worth-it-not-to-the-queries). Every cohort
criterion needs it, and the only route today is a lexicon: `is_healthy` over
`medical_condition`, plus a negation reader over the group's own name. 115 records have a
control group that matches their cohort pattern, and before the negation reader those
records lost the contrast entirely. A two-value enum -- `case` / `comparison` -- would
replace both readers with a field test. `is_healthy` is not a substitute: a study of
depressed against schizophrenic cohorts has no healthy group and still has a comparison.

**~~`Measure.polarity`~~, and the test that killed it.** "Higher value means more tissue"
against "higher value means more deficit" decides which map a coordinate enters, and 103
analyses are named for a loss while measuring a volume. Synthesising the field says the
field is the wrong fix: only 3 records type a measure as a deficit, flipping those changed
no key, and inferring polarity from the analysis name instead lost gold on four keys and
gained it on one. What is left is a vocabulary defect -- drop `gray_matter_atrophy` from
the type vocabulary, so the sign is always relative to the substance being measured.

**`Analysis.spatial_scope` needs to split `roi`.** Cue reactivity's criterion admits an ROI
analysis that reports coordinates (a small-volume correction) and excludes one that does
not (a parcellation ROI). The record has one value for both, so the criterion is
unanswerable as written; the predicate approximates it by asking whether the analysis
reaches a parsed row group. `roi_small_volume` against `roi_parcellation`, or an
`Analysis.reports_coordinates` boolean, answers it directly. 1,884 analyses say `roi`.

**`FactorLevel` should be required to reach an entity, or to be readable without one.** The
2,000 levels that link neither a group nor a condition are the single largest reason a
contrast cannot be read, and 111 conditions carry a name a human cannot read either. The
repair that would help is not a normalization -- `spans.fold_label` is right that
canonicalising a join key hides the failure -- it is a builder check that a level either
links an entity or spells its own name out.

**An analysis-level coordinate count.** "Did this contrast report any foci" is a stated
criterion in two of these reviews and cannot be answered: `Table.coordinate_count` is
per-table and empty on all 2,267 tables. The count can be derived -- 3,601 of 5,989
analyses join a row group -- and the criterion still should not be run, because the 2,388
that do not join are unjoined rather than silent: applied to PTSD it drops 5.8 points of
recall to remove one false positive. Fix the join first; the field is worth adding when
something reads it.

**The derived contrast summary, still the largest simplification.** Which cohorts, which
conditions, which direction, per analysis. Three of the six corrections above are in code
that walks `analyses → effect.cells → cell.term → model_estimations[].terms[].levels →
groups/conditions`, and the walk is why they were possible to get wrong. It needs no model:
every input is in the record.

# The query as a fifth and sixth arm, in the record-arms figures

The record-arms experiment compares full-text autonima against autonima reading the
extraction record, and it compares them as *maps*: each arm's studyset goes through MKDA
density and the result is correlated with the meta-analysis's published map. The query
belongs in that comparison rather than beside it, so it is written as two more arms.

`scripts/query_studysets.py` writes, per project and per mode, the two files a run needs --
`nimads_studyset.json` and `nimads_annotation.json`, plus a
`fulltext_screening_results.json` in the shape the experiment's scorer reads -- into
`projects/<project>/query-{strict,permissive}/outputs/`. From there nothing is
query-specific: `code/run_mkda.py` builds the maps, `recordarms score` writes the CSVs and
`make_figure4_record_arms.py` and `make_record_arm_figures.py` draw them.

**An arm is end to end**, so a query arm is both gates: the published inclusion criteria
choose the papers, the published contrast chooses their analyses. Two conventions are
copied from autonima rather than improved on, because the figure compares selectors and
every other difference is noise in it: coordinates go in as the paper published them with a
`space` label beside them, and the note keys are autonima's annotation names from
`nmb_mappings.json`.

Two things had to be decided to make the comparison fair:

**A map from two experiments is not a meta-analysis.** Substance use's cannabis query-strict
studyset is 2 analyses; its map has almost no suprathreshold voxel, reads dice 0.000 against
the manual map and r2_nonzero **0.756**, because the masked correlation is computed over the
handful of voxels either map left nonzero. `results.MIN_ANALYSES = 5` drops it and the
4-analysis opioids map, and nothing else in the comparison: those two are the only maps in
84 below five. Leaving them in put the query's mean advantage over the baseline at +0.132;
taking them out puts it at +0.041, and the second is the honest number.

**A query arm must not remove a column from the arms' own comparison.** `map_columns` used
to require every arm before it would use a benchmark column, so a column where the query
maps nothing would have dropped that column for full text too. It now requires the three
model arms and a baseline, and includes a query arm where it exists.

Mean ΔR² against each column's own re-estimated baseline, over the columns each arm maps:

| arm | Δ R² vs best baseline | 95% CI | columns |
|---|---|---|---|
| autonima, full text | +0.114 | [+0.034, +0.225] | 17 |
| autonima, record + evidence | **+0.116** | [+0.056, +0.231] | 17 |
| autonima, record, no evidence | +0.112 | [+0.052, +0.222] | 17 |
| query, strict | +0.041 | [−0.040, +0.124] | 12 |
| query, permissive | +0.049 | [+0.035, +0.102] | 14 |

Per project, mean R² against the manual map over the columns each arm maps -- which is what
figure 7 panel b plots:

| project | full text | record + ev. | record, no ev. | query, strict | query, permissive |
|---|---|---|---|---|---|
| `vbm_of_ptsd` | 0.185 | 0.179 | 0.148 | 0.185 | **0.219** |
| `dementia` | 0.076 | 0.129 | 0.121 | **0.204** | 0.122 |
| `cue_reactivity` | 0.332 | 0.332 | **0.339** | 0.279 | 0.290 |
| `vbm_of_substance_use` | **0.158** | 0.124 | 0.124 | 0.014 | 0.091 |

The query's map is better than every model arm's on dementia, level with full text on PTSD,
somewhat behind on cue reactivity, and far behind on substance use -- where its screening
gate admits 19 papers of 244 under strict and the maps it does build are small. There is no
arm that wins everywhere, and the spread within a project is mostly larger than the spread
between the three model arms, which is the same thing `AUDIT.md` found from the other
direction.

The funnel figure keeps three arms. The query has no search, no abstract screen and no
retrieval -- it starts from the extracted corpus -- and drawing it as a flat line through
three stages it never ran would say something false about where it loses papers.

# Testing the proposed fields, and filling three of them

The section above proposed six changes. Each was then synthesised over the 1,817 records
and the query that failed without it was re-run. Three of the six do not survive that, and
saying so is the point of running it.

| proposal | what the test did | outcome |
|---|---|---|
| `Group.role` | derive case/comparison from the cohorts' own words, re-run every key | **built, measured, removed**: it reproduces the readers' 164 gold papers at 0.547 mean F1 against their 0.549, and cannot express a role that depends on which map is being built |
| `Study.language` | fetch `lang` from PubMed for all 1,804 ids, apply the criterion | **kept**: 1,703 are `eng`, one is not, and the criterion six reviews state becomes expressible |
| `Analysis.coordinate_count` | count through `source_table_analysis` into the parse | **built and dropped**: fillable on 3,601 of 5,989 analyses, and nothing reads it -- the consumers join the parse directly -- while its one criterion, PTSD's null-effects exclusion, costs 5.8 points of recall because the 40% that cannot be joined look silent |
| `Measure.polarity` | flip the expected sign where the measure is a deficit | **dropped**: only 3 records type a measure that way and flipping changed nothing; inferring polarity from the analysis name instead lost gold on four keys and gained it on one |
| `Analysis.coordinate_space` propagation | run `derive_coordinate_spaces` over the corpus | **dropped**: fills 0. The 1,187 analyses with no space are the analyses whose paper has no parsed coordinates either, and `coordinate_space.resolve` reaches only 11 of them |
| `spatial_scope` split | replace the "reaches a parsed row group" approximation with the field | **deferred**: identical by construction, so it buys clarity and no recall |

The deterministic repairs already in the package were tested the same way and are the
clearest negative result here. `unwrap_singletons`, `link_entities_by_name` and
`drop_redundant_cell_levels` write 876 links and 728 drops and take the levels that reach
no entity from 2,000 to 1,399 -- and **change no screening number on any of the five
projects and no selection outcome on any of the fourteen keys**. A prototype repair that
declares the `FactorLevel` a `Cell` already names, where its term declares none, adds 1,092
levels and 1,036 further links and also changes nothing: the levels it creates are bare,
and the entity link is the half that is missing.

## What is filled now, and by what

| field | filled by | coverage |
|---|---|---|
| `Study.study_type` | `pondie.extraction.pubmed`, E-utilities `esummary` | 1,804 of 1,817 |
| `Study.language` | the same call, same response | 1,804 of 1,817 |

Two more were built and taken out again, each for the same reason -- a field nothing reads
and no measurement supports is a slot to maintain. `Group.role` reproduced two readers and
could not express a role that depends on the question; `Analysis.coordinate_count` filled
correctly on 60% of analyses and had no consumer, because the workflow joins the parse
itself. The derivation for the second is four lines and this paragraph is where to find
that it was tried: count the points of the row group `source_table_analysis` names, and
leave it absent where the join does not reach one.

`scripts/backfill_records.py` applies both to a corpus extracted before they existed. The E-utilities call is now a POST: 200 ids is a 2kB URL and NCBI
answers a GET of that length with a 500 often enough to lose a whole batch.

## Old queries against new, on old records and new

Four runs, one table. *strict* recall and precision against each benchmark's included set.

| project | old queries, as extracted | new queries, as extracted | new queries, backfilled |
|---|---|---|---|
| `vbm_of_ptsd` | 52.9% / 90.0% | 0.0% / 0.0% | 52.9% / **90.0%** |
| `dementia` | 53.4% / 31.6% | 53.4% / 31.6% | 53.4% / 31.6% |
| `cue_reactivity` | 60.7% / 34.1% | 0.0% / 0.0% | **81.4%** / 30.1% |
| `vbm_of_substance_use` | 18.4% / 87.5% | 0.0% / 0.0% | **22.4%** / **89.5%** |
| `emotion_regulation_2022` | 43.7% / 28.6% | 43.7% / 28.6% | 43.7% / 28.6% |

**The zeroes are the point.** The new query asks three of the five for a language, because
three of them state one, and a record that does not carry `Study.language` cannot answer --
so strict, which drops what a record cannot answer, drops everything. The same query over
the same records, backfilled from PubMed, is back to where it was and one paper better: the
permissive cue selection goes from 462 papers to 461, and the paper it loses is the Chinese
one. That is the whole measured effect of the language criterion, and it is the honest
shape of a criterion that is stated and nearly always satisfied.

Old queries read the backfilled records identically to the as-extracted ones, on every
project, because they ask for none of the three fields.

At the second gate, coordinate precision and recall against each published map, old
contrast specs and new, both through the same scorer:

| project / key | old F1 | new F1 | what changed |
|---|---|---|---|
| `vbm_of_ptsd` / non-PTSD > PTSD | 0.306 | **0.642** | the cohort's negated naming |
| `dementia` / all, decrease, structural, functional | 0.436, 0.433, 0.310, 0.431 | unchanged | no dementia control is named `non-FTD` |
| `vbm_of_substance_use` / all drug classes | 0.476 | **0.661** | the cohort's negated naming |
| `cue_reactivity` / reward | 0.372 | **0.503** | the condition matched by name, not description |
| `cue_reactivity` / drug | 0.350 | **0.525** | the same |
| `cue_reactivity` / natural | 0.500 | 0.418 | the `_wbonly` key's own whole-brain gate, which the old spec did not apply |
| `vbm_of_substance_use` / 5 per-drug keys | none | 0.456 to 0.801 | the old specs did not cover them |

The figures were rebuilt from the backfilled records with this code -- studysets, maps,
scoring, all three figures -- and every arm number is the same to three decimals. Nothing
the new fields change reaches a map: the language criterion removes one paper from a
462-paper selection. The rebuild is worth having anyway, because the figures now come from
the corpus the pipeline produces rather than from the one it produced before these fields
existed.

## Was `Group.role` worth it? Not to the queries

Over the 11 cohort keys, 212 gold papers, four ways of deciding which cohort is the case:

| | gold papers found | mean coordinate F1 |
|---|---|---|
| the cohort pattern alone | 120 | 0.445 |
| **+ `is_healthy`** | **160** | **0.546** |
| + the negated-naming rule | 164 | 0.549 |
| `Group.role`, read off the record | 164 | 0.547 |

**`is_healthy` is nearly the whole fix, and it already existed.** The queries were not
reading a derived field the package had had all along; using it is worth 40 gold papers
and 0.101 of F1. The negated-naming rule on top is worth four more papers -- one each on
PTSD, the pooled substance-use key, nicotine and cannabis.

**The field reproduces that and cannot do better, because a role is relative to the
question.** 22445480 has `Control Smoker` beside `MA-dependent Smoker`: the same cohort is
the comparison for the methamphetamine map and the case for the nicotine one. A per-group
enum picks one, picks comparison, and loses the nicotine contrast; 19645730 loses its
contrast to `Non-alcoholic control status`, where the denial is real and the word form
(`alcoholic` against `alcohol`) hides it. So the queries keep the two readers, which have
the pattern in hand, and the field stays for consumers that want the common two-cohort
answer without a lexicon. On this corpus it is a convenience, not a result.

# Which meta-analysis the query does worst on, and why

`vbm_of_substance_use`, and not for the reason the other numbers suggest.

| | screening recall (strict) | mean selection F1 | map R² (query, strict) | best model arm |
|---|---|---|---|---|
| `vbm_of_ptsd` | 52.9% | **0.642** | 0.185 | 0.185 |
| `dementia` | 53.4% | 0.402 | **0.204** | 0.129 |
| `cue_reactivity` | **81.4%** | 0.482 | 0.279 | 0.339 |
| `vbm_of_substance_use` | **22.4%** | 0.627 | **0.014** | 0.158 |
| `emotion_regulation_2022` | 43.7% | -- | -- | -- |

It has the second-best contrast selection of the five and the worst map by an order of
magnitude. Those two facts are consistent, and the thing joining them is one predicate.

**The screening gate starves it.** "Presence of pharmacological manipulations" is answered
from `Arm.arm_kind`, and 153 of the project's 244 records declare no arm at all, so the
criterion reads "cannot say" on 45 of its 76 gold papers. Strict drops what a record
cannot answer, so it admits 19 papers; the studysets that follow are 2 to 20 analyses per
drug class and two of the six columns are too small to map at all.

**Given the right papers, the same selection makes the best map on the project.** Running
the query's contrast selection over the benchmark's own included set -- its screening gate
replaced by the gold list, everything else identical -- gives 52 contributing papers and 97
analyses against strict's 18 and 39:

| key | analyses | query over gold papers | query, strict | query, permissive | full text | record + evidence |
|---|---|---|---|---|---|---|
| alcohol | 35 | **0.505** | 0.024 | 0.170 | 0.493 | 0.390 |
| all drug classes | 84 | 0.176 | 0.006 | 0.174 | **0.277** | 0.226 |
| cannabis | 9 | **0.285** | -- | 0.105 | 0.011 | 0.004 |
| nicotine | 30 | 0.000 | 0.003 | 0.002 | **0.069** | 0.041 |
| opioids | 13 | 0.012 | -- | 0.001 | 0.004 | 0.006 |
| stimulants | 23 | 0.082 | 0.024 | 0.096 | **0.090** | 0.077 |
| **mean** | | **0.177** | 0.014 | 0.091 | 0.158 | 0.124 |

So the deterministic contrast is not what fails here. It reads this project's records better
than it reads any other project's except PTSD's, and its map beats full text's when it is
handed the papers. What fails is a single criterion answered from a field four fifths of the
records leave empty -- and the honest reading of that field is what makes it fail, because
the previous reading answered from `allocation` and was wrong on 46 gold papers instead of
silent on 45.

Two smaller notes from the same table. `nicotine` reads 0.000 even over the gold papers, and
every arm is under 0.07 there, so that column is not a query problem. And the query's
`cannabis` map is the best of any arm by a factor of twenty-five, on 9 analyses -- which is
under the five-analysis floor for the arms and above it here only because the gold papers
supply more of them.

# Re-extracting five of the papers that have no arm

The substance-use failure is one criterion -- "presence of pharmacological manipulations",
answered from `Arm`, and 153 of that project's 244 records declare no arm. Two questions
follow: can the pipeline extract an arm at all, and would it now.

## The demands pass could not ask for one, and that was a mistake

`demands` is the only pass that declares an entity; `satisfy` emits exactly what was
declared, and `fill` walks existing entities and skips nested ranges, so it cannot create
`design.arms`. Nothing in the code forbids declaring an `Arm` -- the only kind `satisfy`
refuses is `Table` -- so the restriction was entirely in the prompt, in two places: the
`required_entities` example showed four kinds and never an arm, and `satisfy` said "emit
one entity per declared entry, under the right **top-level list**" while rule 2 puts
`arms` inside `study.design`. A declared arm had nowhere the instructions agreed on.

Both are fixed: the example now carries an `Arm` and a `Timepoint`, the declarable set is
stated (everything except `Table`, which already exists), and `satisfy` names
`study.design.arms` as their destination.

## Running the five papers said the arms were never the problem

Five gold substance-use papers with no arm, re-extracted end to end with today's pipeline
-- once on the shipped prompt, once on the patched one, five workers each:

| | committed corpus | re-extracted, shipped prompt | re-extracted, patched prompt |
|---|---|---|---|
| declared an `Arm` | 0 of 5 | 0 of 5 | 0 of 5 |
| `assignment_structure` | `parallel` on 4 | `observational_cohorts` on 3 of 4 | **`observational_cohorts` on 5 of 5** |
| `allocation` | `non_randomized` on 3 | mixed | `observational_cohorts` on 4 of 5 |

**No arm is the right answer for these papers.** They are observational VBM studies;
nobody was allocated to anything, so the patched instruction -- declare an arm whenever
the paper administered something -- correctly does not fire. The prompt gap was real and
worth closing for drug studies, and it is not what makes this project fail.

**What makes it fail is a stale corpus.** `assignment_structure: observational_cohorts` is
the value the schema added for exactly this design, and today's pipeline writes it on five
of five. The committed records say `parallel` because they were extracted before the value
existed. The criterion is answerable from these records and was not from those: over the
five papers, "no pharmacological manipulation" answers on **4 of 5 re-extracted against 1
of 5 committed**.

**And a new defect, which the run found rather than the corpus.** Four of the five write
`observational_cohorts` into `allocation`, which is `assignment_structure`'s vocabulary,
not allocation's. The value is accepted because the vocabulary is open, and it answers a
different question: `allocation` is the only record of whether anything was administered
at all, and its own value for that is `not_applicable`.

**Saying that in the slot fixed it, on five of five.** The slot's description now states
that a study with no arms takes `not_applicable` here and says `observational_cohorts` in
its neighbour, and that this slot is the only record of whether anything was administered.
A third run of the same five papers:

| paper | committed | re-extracted | + prompt fix | + the slot description |
|---|---|---|---|---|
| 11822992 | `not_applicable` / `parallel` | `observational_cohorts` / `observational_cohorts` | none / none | **`not_applicable` / `observational_cohorts`** |
| 14706428 | `non_randomized` / `parallel` | `not_applicable` / `observational_cohorts` | `observational_cohorts` / `observational_cohorts` | **`not_applicable` / `observational_cohorts`** |
| 15607838 | `non_randomized` / `parallel` | `not_applicable` / `observational_cohorts` | `observational_cohorts` / `observational_cohorts` | **`not_applicable` / `observational_cohorts`** |
| 15982446 | none / none | none / none | `observational_cohorts` / `observational_cohorts` | **`not_applicable` / `observational_cohorts`** |
| 16369836 | `non_randomized` / `parallel` | `observational_cohorts` / `observational_cohorts` | `observational_cohorts` / `observational_cohorts` | **`not_applicable` / `observational_cohorts`** |

Five of five exactly right, against one of five in the committed corpus. The criterion
answers on all five, and one of the five now passes strict screening outright; the four
that do not are blocked by `Analysis.coordinate_space`, by `Study.language` (this harness
did not run the PubMed backfill over these records), and on one paper by a cohort the
record names but does not diagnose.

## So: does this need a new field?

No. The measurement says the field exists, is filled on 92% of the corpus, and was filled
wrongly: on the 76 substance-use gold papers -- every one of which administered nothing,
because the review excluded pharmacological manipulations -- 46 records say `randomized`
or `non_randomized`, which the schema reads as "something was still administered".
Coverage was never the problem and a new question would have inherited the same
confusion. What the slot needed was to say which question it answers.

`observational_cohorts` alone would not have been enough either. It covers the commonest
design and not the criterion: `single_group` says nothing about administration, and
`crossover` and `within_subject` take priority over it by the field's own description, so
a cohort study with two sessions loses the signal -- 44 records corpus-wide read "the
structure implies arms and none are declared".

What is worth adding is not a question but a derivation: `StudyDesign.administered`,
`deterministic`, true where any `Arm.arm_kind` is not `no_intervention` or an `Arm.agent`
is named or `allocation` is an allocation, false where `allocation: not_applicable` or
`assignment_structure: observational_cohorts`. One field to test instead of four to join,
and it cannot contradict its sources. It would not have rescued the committed corpus --
a derivation cannot repair its inputs -- which is why the fix that matters here is the
slot description and a re-extraction.

Two query changes follow from the run, both faithful to the criterion:

* `no_pharmacological` reads `assignment_structure: observational_cohorts` as well as
  `allocation: not_applicable` -- the schema glosses the first as "not assigned to
  anything ... and nothing was administered", which is the criterion in its own words.
* `group_condition` reads the cohort's **name** as well as its `medical_condition`, because
  re-extracting 15607838 gives its case cohort the name "marijuana group" and an empty
  condition. The name may only say yes: letting it say no contradicted two gold papers
  whose cohorts are "successful quitters" and "all participants".

The five re-extracted records still do not pass strict screening, and what blocks them is
the rest of this document: `Analysis.coordinate_space` empty on two, `spatial_scope` on
one, no analyses at all on one, and `Study.language` missing because this harness did not
run the PubMed backfill over them.
