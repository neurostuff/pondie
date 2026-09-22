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

## CORRECTION: this is not analysis selection, and the claim above was wrong

I wrote that analysis selection is the bottleneck. **It is not.** Auditing the metric found
two things, and the second overturns the conclusion.

**First: where the annotation does select, the coordinates are exact.** The table above
compares foci *counts*, which two different analyses can share. Comparing the coordinate
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

`anlys` is analyses selected and `nofoci` how many of them reached no coordinates. Paper
and foci precision and recall are against the annotation's own key, over the gold papers
that have a record.

**vbm_of_ptsd**, `non-PTSD > PTSD`: 17 gold papers with a record, 139 gold foci.

| screen / select | anlys | nofoci | paper P | paper R | foci P | foci R | foci F1 |
|---|---|---|---|---|---|---|---|
| strict / strict | 7 | 2 | **100.0%** | 29.4% | 80.0% | 11.5% | 0.201 |
| strict / permissive | 18 | 3 | **100.0%** | 41.2% | 33.3% | 15.8% | 0.215 |
| permissive / strict | 12 | 5 | 77.8% | 41.2% | **80.8%** | 15.1% | **0.255** |
| permissive / permissive | 47 | 12 | 57.9% | **64.7%** | 21.1% | **19.4%** | 0.202 |

**dementia**, `bvFTD vs HC`, four keys: 16, 14, 9 and 11 gold papers with a record.

| key | screen / select | anlys | nofoci | paper P | paper R | foci P | foci R | foci F1 |
|---|---|---|---|---|---|---|---|---|
| all | strict / strict | 161 | 84 | 11.0% | 56.2% | 17.3% | 30.5% | 0.221 |
| all | strict / permissive | 297 | 142 | 10.1% | 56.2% | 10.4% | 30.5% | 0.156 |
| all | permissive / strict | 236 | 144 | 9.9% | 75.0% | 15.6% | 30.9% | 0.208 |
| all | permissive / permissive | 699 | 404 | 5.5% | **81.2%** | 6.2% | **30.9%** | 0.103 |
| decrease | strict / strict | 117 | 59 | 14.1% | 64.3% | **26.9%** | 28.8% | **0.278** |
| decrease | strict / permissive | 253 | 117 | 11.4% | 64.3% | 12.9% | 28.8% | 0.178 |
| decrease | permissive / strict | 165 | 96 | 11.6% | 78.6% | 23.0% | 28.8% | 0.256 |
| decrease | permissive / permissive | 628 | 356 | 5.5% | **85.7%** | 6.7% | 28.8% | 0.109 |
| structural | strict / strict | 73 | 39 | 9.1% | 44.4% | 5.0% | 11.5% | 0.070 |
| structural | permissive / permissive | 434 | 255 | 3.1% | 55.6% | 1.5% | 11.5% | 0.027 |
| functional | strict / strict | 77 | 38 | 20.0% | 72.7% | **42.9%** | 33.3% | **0.375** |
| functional | strict / permissive | 136 | 67 | 16.3% | 72.7% | 27.4% | 33.3% | 0.301 |
| functional | permissive / strict | 103 | 58 | 17.6% | 81.8% | 41.8% | 33.8% | 0.374 |
| functional | permissive / permissive | 241 | 145 | 11.0% | **90.9%** | 23.3% | **33.8%** | 0.276 |

**cue_reactivity**, three keys: 140, 123 and 20 gold papers with a record.

| key | screen / select | anlys | nofoci | paper P | paper R | foci P | foci R | foci F1 |
|---|---|---|---|---|---|---|---|---|
| reward | strict / strict | 154 | 30 | **45.2%** | 27.1% | 28.9% | 14.0% | 0.189 |
| reward | strict / permissive | 501 | 119 | 33.2% | 45.0% | 17.7% | 18.9% | 0.183 |
| reward | permissive / strict | 191 | 38 | 43.4% | 32.9% | **30.9%** | 19.8% | **0.241** |
| reward | permissive / permissive | 646 | 179 | 32.3% | **57.9%** | 19.4% | **25.8%** | 0.221 |
| drug | strict / strict | 112 | 23 | **54.7%** | 28.5% | 30.8% | 12.7% | 0.180 |
| drug | strict / permissive | 461 | 112 | 31.7% | 46.3% | 16.5% | 18.5% | 0.175 |
| drug | permissive / strict | 143 | 30 | 51.8% | 35.0% | **33.8%** | 18.7% | **0.240** |
| drug | permissive / permissive | 600 | 171 | 30.0% | **58.5%** | 17.8% | **24.7%** | 0.207 |
| natural | strict / strict | 55 | 13 | 13.3% | 20.0% | 20.9% | 20.4% | 0.207 |
| natural | permissive / strict | 68 | 15 | 13.9% | 25.0% | **21.5%** | 28.2% | **0.244** |
| natural | permissive / permissive | 531 | 158 | 4.7% | **50.0%** | 6.1% | **33.7%** | 0.103 |

**vbm_of_substance_use**, `all drug classes`: 74 gold papers with a record, 511 gold foci.

| screen / select | anlys | nofoci | paper P | paper R | foci P | foci R | foci F1 |
|---|---|---|---|---|---|---|---|
| strict / strict | 11 | 0 | **100.0%** | 9.5% | **59.4%** | 7.4% | **0.132** |
| strict / permissive | 34 | 0 | 91.7% | 14.9% | 33.3% | 7.8% | 0.127 |
| permissive / strict | 21 | 6 | 64.3% | 12.2% | 41.3% | 7.4% | 0.126 |
| permissive / permissive | 61 | 13 | 71.4% | **20.3%** | 17.9% | 7.8% | 0.109 |

**emotion_regulation_2022** stops at screening: 43.7% strict and 73.6% permissive recall
over 87 gold papers, and the benchmark ships no analysis-level annotation for it, so there
is no map to score against.

## What the four cells say

**Permissive costs little at the first gate and a great deal at the second.** Strict
selection has the better foci F1 in seventeen of the eighteen comparisons -- nine maps by
two screening modes, the exception being PTSD under strict screening -- and the losses are
not small: 30.8% to 16.5% foci precision on cue drug, 80.0% to 33.3% on PTSD, 59.4% to
33.3% on substance use. The two silences are not equivalent. A record that cannot answer
"was this whole-brain" is usually still the right paper; an analysis whose cells cannot be
signed is usually not the contrast the map pooled, and admitting it pools the wrong
coordinates.

**The screening gate is a wash and the choice is per map.** Permissive screening buys 3 to
19 points of paper recall everywhere. It improves foci F1 on the four maps where screening
is the binding constraint -- PTSD 0.201 to 0.255, and all three cue keys, the largest
0.207 to 0.244 -- and costs at most 0.022 of it on dementia and substance use, where
selection was already losing more than screening was. Foci precision survives it except on substance
use, where 16 screened papers become 26 and it falls from 59.4% to 41.3%.

**Nothing here approaches a usable map.** The best foci recall on any map is 33.8%
(dementia functional, permissive/permissive) and the best F1 is 0.375 (dementia functional,
strict/strict). A pooled ALE on a third of the foci with three false foci in four is not
the published result.

## Where the foci go, and it is not the query

Of the 767 analyses the strict/strict pipelines select across the nine maps, **479 reach
coordinates and 288 do not**:

| | analyses |
|---|---|
| joined to coordinates | 479 |
| **`source_table_analysis` empty** | **230** |
| its address names no row group in the parse | 40 |
| the row group it names carries no coordinates | 18 |

The empty slot is dementia's: 74 of the 161 analyses selected for `all`, 51 of 117 for
`decrease`. `resolve_source_table_analysis` fills the slot only where exactly one parsed
entry under the cited tables carries the analysis's name, and on that project it usually
cannot -- which is the same dangling-`Analysis.tables` finding the `Tables` stage work
measured, arriving at the map.

The ceiling makes the same point without the query in the way. Taking **every** analysis of
**every** gold paper and joining it to the parse:

| map | foci recall at the ceiling |
|---|---|
| `vbm_of_substance_use` / all drug classes | **81.2%** |
| `vbm_of_ptsd` / non-PTSD > PTSD | 67.6% |
| `cue_reactivity` / drug | 66.8% |
| `cue_reactivity` / reward | 66.3% |
| `cue_reactivity` / natural | 63.3% |
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
