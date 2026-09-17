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
single criterion every meta-analysis in this benchmark states, so this one field decides more
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

## The comparison again, after four predicate corrections

Two rounds of examining the query's own errors corrected four predicates -- `measures`,
`group_contrast`, a wider grey-matter pattern, and a visual-cue test that admits text. The
comparison with both autonima arms, rerun:

| selector | precision | recall | F1 |
|---|---|---|---|
| autonima, full text | 40.1% | **95.3%** | 0.555 |
| autonima, record + evidence | 44.4% | 84.1% | 0.566 |
| autonima, record, no evidence | 45.3% | 82.3% | 0.569 |
| query, strict | **54.3%** | 46.2% | 0.437 |
| query, permissive | 40.5% | 70.0% | 0.441 |
| full text → query veto | 52.1% | 68.2% | 0.537 |
| record + evidence → query veto | 53.9% | 62.7% | 0.532 |

Against the first run, the query's strict precision rose from 48.0% to 54.3% and its recall
fell from 53.3% to 46.2%. **The whole of that recall loss is one project.**
`vbm_of_substance_use` went from 58.1% / 66.2% to 87.5% / 21.5%, because reading "no
pharmacological manipulations" off `allocation` rather than off declared Arms exposed the
defect of finding 9: 45 of its 76 gold papers say `allocation: non_randomized` for a study
that administered nothing. The predicate is now faithful to the criterion and the criterion
is unanswerable from these records.

Excluding that project, so the mean is not carrying one unanswerable criterion:

| selector | precision | recall | F1 |
|---|---|---|---|
| autonima, full text | 39.0% | **94.9%** | 0.541 |
| autonima, record + evidence | 41.6% | 81.7% | 0.534 |
| autonima, record, no evidence | 42.3% | 81.3% | 0.540 |
| query, strict | 46.0% | 52.4% | 0.460 |
| query, permissive | 33.6% | **81.0%** | 0.457 |
| full text → query veto | 46.7% | 78.8% | **0.575** |
| record + evidence → query veto | **49.2%** | 72.2% | 0.573 |

Three things this says that the five-project mean does not.

**The veto pipelines lead on F1 and the arms are indistinguishable from each other.** 0.575
and 0.573 against 0.534 to 0.541 for all three arms, whose spread is 0.007 -- well inside the
0.034 run-to-run noise `AUDIT.md` measured. The arm contrast this experiment was built to
settle does not resolve at this precision, and the veto does.

**Where the records can answer, the deterministic criteria match a model reading the same
records.** `query, permissive` reaches **81.0% recall against the record arms' 81.7% and
81.3%** -- admitting the papers the record is silent about, the published criteria applied
mechanically select as much of the gold as a language model reading the record does. It does
so at 33.6% precision against their ~42%, so it is not a replacement; it is evidence that the
criteria are expressible and that what the model adds over them is judgement about silence.

**`vbm_of_ptsd` is what this looks like when the records are good.** Query strict reaches
**90.0% precision**, and `full text → query veto` reaches **F1 0.778**, the highest figure
anywhere in this document and above full text's own 0.727. That project is 49 papers with
records that answer most criteria, and it is the only place the composition is clearly worth
deploying.

### The conclusion does not change

Full text still wins on recall -- 95.3%, or 94.9% without substance use -- and recall is the
objective screening has. A missed study biases the pooled estimate and cannot be recovered
downstream; a false positive costs the next reader some time. Every alternative in both
tables buys precision with recall, the veto included, and the F1 column should not be read as
a ranking.

What four rounds of correction established is where the remaining gap lives. It is not in the
difficulty of writing the criteria down: two record defects account for most of it --
`spatial_scope: roi` on whole-brain papers, contradicting 38 gold, and
`allocation: non_randomized` on observational studies, contradicting 39 in one project.

# Analysis selection, which is the stage that actually decides the map

Screening picks papers. A coordinate meta-analysis pools **contrasts**, and the benchmark
names them: "non-PTSD > PTSD", "bvFTD < HC (all & by modality)", "users < non-users". The
ground truth is `nimads/<project>/merged/`, whose annotation carries one boolean per key per
analysis, so the analyses a published meta-analysis pooled are recorded per key. In autonima
this stage is `annotation_results.json` -- an `include` per analysis per key -- and it has not
been measured against that gold before now. `scripts/query_analysis_selection.py` does it,
alongside a deterministic query over `Effect.cells`.

Every key here is a directional between-group contrast, optionally restricted by modality, and
both are things the schema encodes: a `Cell.direction` on a `FactorLevel` that reaches a
`Group`, and `Analysis.measure` → `Measure.family`. So the question is not whether the schema
can state the contrast. It can.

| project / key | papers | autonima ≥1 | autonima = gold count | **autonima = gold foci** | query ≥1 | query = gold count |
|---|---|---|---|---|---|---|
| `vbm_of_ptsd` / non-PTSD > PTSD | 17 | 41% | 35% | **35%** | 53% | 35% |
| `vbm_of_substance_use` / all drug classes | 74 | 64% | 41% | **32%** | 46% | 24% |
| `dementia` / decrease | 14 | 57% | 29% | **21%** | 86% | 43% |
| `dementia` / functional | 11 | 64% | 27% | **9%** | 82% | 27% |
| `dementia` / structural | 9 | 56% | 33% | **11%** | 67% | 22% |

## This is the bottleneck, and it is not screening

**On the gold papers it selected, autonima reproduces the foci the published meta-analysis
pooled for 9% to 35% of them.** Screening recall on the same corpus is 95.3%. So the stage that
finds the right papers works, and the stage that decides which coordinates enter the map --
which is the stage the map is made of -- agrees with the published analysis on a third of
papers at best.

That reframes the map results this experiment has been reporting. `figure7`'s R² and the
Dice figures compare a map built from the wrong contrasts against one built from the right
ones, on papers that were mostly correctly included. The arm contrast -- full text against
records -- was measured at screening, where all three arms exceed 82% recall and differ by
less than the run-to-run noise. The place where a map is won or lost was never in that
comparison.

## The query is competitive here, which says the same thing again

The deterministic query finds a matching contrast more often than the annotation does on
dementia (86%, 82%, 67% against 57%, 64%, 56%) and matches the gold count as often on PTSD
(35% each), while losing on substance use (24% against 41%). Given that it is five lines of
predicate over `Effect.cells` and `Cell.direction`, with no model and no prompt, that is not a
claim that it should replace the annotation pass -- the samples are 9 to 74 papers and
dementia's are small. It is the third time in this document that criteria applied mechanically
match a model reading the same record, and the reading is the same each time: the records
carry the facts and the pass that reads them is not extracting the advantage.

## Two measurement notes, because both bit

**Merged gold studies cannot be attributed per paper.** Dementia's gold merges up to **27
pmids into one study**; assigning that study's foci to each of them made every per-paper
comparison meaningless, and the foci column read 2-6% before I noticed. Those analyses are now
dropped and counted -- 8 in dementia, 3 in substance use -- which is why dementia's samples
fall to 9-14 papers. PTSD is 22 single-paper studies and cue reactivity 191, so neither is
affected.

**The query has no foci column yet**, and it should. `Analysis.source_table_analysis` is the
exact join -- "the only exact route from an analysis to its coordinates", per its own repair --
so the record can be taken to a focus count without a string match. Computing it needs the
stage-1 parse beside the records, which is the one input this harness does not have locally.
