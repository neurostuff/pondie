# Judging the top-1 evidence sentence by hand

Seventy field instances, drawn from nine papers, run through the cross-encoder retriever
(`cross-encoder/ms-marco-MiniLM-L12-v2`) over sentence units and sentence-ified table rows.
For each one this file records the value that was extracted, the single sentence the
retriever ranked first, the sentence the incumbent LLM evidence pass cited, and my
own reading of whether the picked sentence supports the value.

The point of doing it by hand is that the automated number — "top-1 unit equals the
LLM pass's unit" — was 12/70 (17%), and that number is wrong about what it claims to
measure.

> **What the comparison target is.** Everything scored here is measured against the
> evidence spans in the extraction records, and those were produced by the incumbent
> LLM evidence pass (`review/add_evidence.py`), not by a human. So every agreement
> number below says *how often the retriever picks the same sentence the LLM picked* --
> not how often either is right. The hand judgements in the case list are the only part
> of this file that assesses truth, because there I read the paper and decided myself;
> that is why they disagree with the agreement metric by 2.4x, and why four of the
> retriever's picks are better than the LLM's. Human-written evidence quotes do exist --
> 143 of them across 14 `corrections/*.json` files -- and scoring both against those is
> the measurement this file is missing.

## What the hand count says

| verdict | n | share of in-scope (66) |
|---|---|---|
| matches the LLM pass's sentence | 12 | 18% |
| supports the value from a *different* sentence | 17 | 26% |
| partially supports it | 11 | 17% |
| does not support it | 26 | 39% |
| out of scope (contrast levels/directions) | 4 | — |

**44% of top-1 picks are citations I would accept as they stand**, against the 18% the
exact-unit metric reports. The metric undercounts by roughly 2.4x, because a paper usually
states a fact in more than one place and the LLM pass picked one of them.

Four picks are *better* than the incumbent LLM's own span (#27, #37, #41, #60). In #60 the
retriever cites the demographics row reading "FESZ patients (n=43)" while the LLM cited a coordinate table row that never mentions 43. So exact-unit agreement is not
merely a strict metric, it is measuring against a noisy target, and it has no 100%.

## The failure taxonomy

The 26 genuine failures are not one problem.

**A — the value is never worded in the paper (12 cases).** `undirected` (#16), `whole_brain`
(#7, #53, #68), `non_randomized` (#48), `human` (#28), `participant_blind` (#34),
`active_comparator` (#65), `exploratory` (#5), `network` (#13), `within_subject`. These are
controlled terms the extractor *inferred* — from a correlation being computed, from a
voxelwise analysis with no ROI mask, from the absence of a preregistration. There is no
sentence to retrieve. No reranker, NLI verifier or embedding model fixes this; the honest
outcomes are either `not_found` or an evidence span for the *premise* of the inference
(cite "connectivity was calculated as a sliding-window correlation" for `undirected`) with
the inference recorded as such.

**B — homonym traps (4 cases).** #49 matched `parallel` in "which is parallel with our
findings" — the discourse sense, not the design sense. #48 matched `randomly` in "trials
were presented randomly in E-prime" — trial order, not allocation. #40 matched
`pharmacological` in a limitations sentence about shared mechanisms. #20 matched the bare
numeral `2` in "k = 2 to 20". Every one is a lexical collision the cross-encoder did not
disambiguate, and in #48 the result is a citation that argues *against* the recorded value.

**C — abbreviation and unit gaps (2 cases, systematically).** #20: the value is
`repetition_time_seconds = 2`, the paper says `TR = 2 s`, and the query never contains "TR".
#21 is worse — the value is `0.015` and the paper says `TE = 15 ms`. Nothing bridges those
without normalising units and expanding the schema field name to its field aliases *before*
retrieval.

**D — the field's name decides the outcome (#8 vs #29).** Both need the same table row.
`analyses.groups.n = 31` retrieved it; `groups.acquired_count = 44` did not — "acquired
count" has no lexical anchor in "For N: TD is 44; ASD is 31". The query template is not a
neutral wrapper.

**E — misses that a substring search would not have made (2 cases).** #23 was asked for the
assessment "Wechsler Abbreviated Scale of Intelligence, WASI-IV" and returned the fragment
"2013DFA11140, to BH)." — a grant number — while a sentence containing the name
character-for-character sat in the text. #26 wanted a URL and picked a sentence with no URL
in it. This is the clearest argument in the file for a hybrid: try literal and regex match
first, fall back to the reranker only when that finds nothing or finds too many.

**F — top-1 is the wrong shape (#64, #67).** In #67 the picked sentence has the interaction
and the regions; the direction ("lower in the patients") is in the next sentence. In #64 the
needle specification and the acupoint list are in different sentences. `evidence.sets[]`
already holds several spans — forcing the retriever to commit to one throws away a fact that
is simply spread across two.

**G — Introduction and Discussion outrank Methods (roughly 8 of the failures).** #1, #2,
#3, #4, #12, #13, #36, #56 all picked a background or speculation sentence with high term
overlap over the Methods sentence that states the fact. The retriever has no notion of "this
study did X" versus "X is a thing that can be done". Section-restricting the candidate pool
by field — acquisition parameters from Methods, interpretations from Results/Discussion —
is the cheapest available fix and is orthogonal to everything above.

## What this changes

- Exact-unit agreement should stop being the headline. Score against *any* sentence that
  supports the value, or accept that the reported number is ~2.4x pessimistic.
- Retrieval-only failures (A) are ~46% of the failures and are not retrieval problems.
  Separating them out is the difference between "34% of evidence is wrong" and "the
  retrievable half is 70% right".
- B, C, D and E are all fixable before the model runs: normalise units, expand the query
  from schema aliases rather than the field name, and try literal match first.
- G is fixable with a section filter.

Raw dumps: `data/eval/top1-judgements.json`.

## The fixes, and what they were worth

Four of the failure classes above are fixable before any model runs. They are
implemented in `review/evidence_retrieval.py` and tested in
`study-schema/test_evidence_retrieval.py`; each test is one of the cases below.

Measured over **4,074 field instances across all 21 papers**, scoring the retriever's
pick against the incumbent LLM evidence pass's unit. First, each fix alone against the baseline
declarative query:

| arm | top-1 | recall@12 |
|---|---|---|
| base | 30.2% | 60.8% |
| + alias and unit expansion | 31.2% | 62.2% |
| + literal-match bonus | 31.7% | 61.7% |
| + section prior | 32.4% | 63.6% |
| all four | 34.2% | 65.2% |

They stack, and the section prior is the largest single contributor.

### The query itself was the bigger lever

A separate sweep over query formulations found that the declarative claim I had been
sending was the problem. `ms-marco-MiniLM` is trained on short web queries and every
term past the field name and its value dilutes the match:

| query formulation | r@1 | r@5 | r@12 | r@25 | r@50 |
|---|---|---|---|---|---|
| claim (declarative) | 36% | 56% | 65% | 74% | 84% |
| keyword (short) | 35% | 56% | 65% | 75% | 83% |
| **value only** (`{leaf} {value}`) | **43%** | **62%** | **70%** | 75% | 81% |
| + ancestor entity names | 31% | 49% | 58% | 66% | 75% |

Dropping the entity name from the query is worth more than any of the four fixes.

### Does that break entity disambiguation?

It should. Two sibling entities that share a value produce a byte-identical query and
must therefore get an identical top-1, so at most one can be right. Stratifying by
whether the value can discriminate at all:

| stratum | n | claim | value only | entity + value |
|---|---|---|---|---|
| singleton (one entity of that class) | 687 | 49.6% | **51.8%** | 49.8% |
| sibling, distinct value | 1592 | 37.5% | **40.9%** | 38.2% |
| sibling, **same** value | 1795 | 16.3% | **21.7%** | 14.9% |
| all | 4074 | 30.2% | **34.3%** | 29.9% |

`value only` wins in every stratum including the one where it cannot discriminate,
because siblings there usually share a gold sentence -- "For N: TD is 44; ASD is 31" is
the evidence for both groups' `n`.

The result that matters is the third column. In the one stratum where the entity name
is the *only* possible discriminator, putting it in the query scores 14.9% -- below
both alternatives. The cross-encoder cannot use it as a term. So the entity is scored
as a separate signal instead, the way the literal bonus already works.

### Final

| arm | top-1 | recall@12 |
|---|---|---|
| base | 30.2% | 60.8% |
| all fixes, entity in query | 34.6% | 65.6% |
| lean query, entity dropped | 36.8% | 65.3% |
| **lean + entity scored separately** | **37.2%** | **66.8%** |
| lean, no aliases | 37.0% | 64.6% |

**+7.0 top-1 and +6.0 recall@12 over the baseline.**

The alias table is the one component that does not pay for itself: 36.8% with it
against 37.0% without at top-1. It survives only because it is worth +0.7 at
recall@12, which is where a two-stage pipeline actually reads. Unit normalisation is
not part of that comparison -- both lean arms use it -- and is not in question.

None of this touches failure class A. Roughly 46% of the remaining failures are values
the paper never words, and they need a decision about what evidence for an inferred
term should even mean, not a better retriever.

Raw shard results: `data/eval/ablate.json`, `data/eval/ablate2.*.json`,
`data/eval/strat.*.json`.

## The cases

### 1. `description` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** _(study-level)_
- **value** `Resting-state fMRI study using group independent component analysis and time-varying functional connectivity to investig`
- **query sent to the reranker** For the study, the description is Resting-state fMRI study using group independent component analysis and time-varying functional connectivity to investigate differences in brain connectivity states between children with autism spectrum disorder and typically developing children..
- **top-1 picked** > Time-varying connectivity analysis in resting-state fMRI can identify the influence of excitation and inhibition balance on whole brain connectivity state, and abnormal connectivity at resting-state in the ASD.
- **LLM evidence pass cited** > To investigate the influence of ASD on brain connectivity states, we performed group independent component analysis (GICA) and dynamic network analysis on fMRI data of ASD and TD children.
- **judgement** (NO) Picked is a general claim about the method, not a statement of what this study did. Luna's 'we performed GICA and dynamic network analysis on fMRI data of ASD and TD children' is correct.

### 2. `design.description` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** _(study-level)_
- **value** `Observational, cross-sectional comparison of autism spectrum disorder and typically developing children using an open-ac`
- **query sent to the reranker** For the study, the description is Observational, cross-sectional comparison of autism spectrum disorder and typically developing children using an open-access ABIDE I dataset..
- **top-1 picked** > Introduction Autism Spectrum Disorder (ASD) is a lifelong developmental disorder.
- **LLM evidence pass cited** > Participants and functional MRI data acquisition Data of participants were obtained from open accessed dataset collected by NYU Langone Medical Center, a collection site of Autism Brain Image Data Exchange I( ABIDE I ) (Di Martino et al., 2014 ).
- **judgement** (NO) 'ASD is a lifelong developmental disorder' is background. Nothing in the picked sentence bears on design.

### 3. `analyses.name` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `Mean dwell time of connectivity states, ASD against TD`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the name is Mean dwell time of connectivity states, ASD against TD.
- **top-1 picked** > Under the same cognitive function, brain states of ASD might diverge from TD due to the disturbed convergence of functional connectivity in the ASD.
- **LLM evidence pass cited** > Statistical analysis Two-sample t -test was used to compare the MDTs of ASD and TD children.
- **judgement** (NO) A Discussion speculation. The analysis name is defined by 'Two-sample t-test was used to compare the MDTs of ASD and TD children.'

### 4. `analyses.definition` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `Two-sample t-test comparing the mean dwell time of each k-means connectivity state between ASD and TD children, over k =`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the definition is Two-sample t-test comparing the mean dwell time of each k-means connectivity state between ASD and TD children, over k = 2 to 20..
- **top-1 picked** > Under the same cognitive function, brain states of ASD might diverge from TD due to the disturbed convergence of functional connectivity in the ASD.
- **LLM evidence pass cited** > Statistical analysis Two-sample t -test was used to compare the MDTs of ASD and TD children.
- **judgement** (NO) Same sentence as #3; same failure.

### 5. `analyses.prespecification` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `exploratory`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the prespecification is exploratory.
- **top-1 picked** > Under the same cognitive function, brain states of ASD might diverge from TD due to the disturbed convergence of functional connectivity in the ASD.
- **LLM evidence pass cited** > Hence, we performed time-varying connectivity analysis on resting-state fMRI data to investigate brain states mutation in ASD children.
- **judgement** (NO) No sentence states prespecification. 'exploratory' is inferred from the absence of a preregistration -- there is nothing to retrieve.

### 6. `analyses.interpretations` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `Mean dwell time differed between groups at k = 3, 5, 8, 13, 14 and 18, with ASD children holding the weak-connectivity s`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the interpretations is Mean dwell time differed between groups at k = 3, 5, 8, 13, 14 and 18, with ASD children holding the weak-connectivity states for longer than TD children..
- **top-1 picked** > Mean dwell time (MDT) was calculated at each k -value.
- **LLM evidence pass cited** > Results Significant differences in the MDTs were found when the k -values of k-means clustering were 3, 5, 8, 13, 14, and 18.
- **judgement** (NO) Picked is the method ('MDT was calculated at each k-value'), the value is the result. High term overlap, wrong assertion.

### 7. `analyses.spatial_scope` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `whole_brain`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the spatial scope is whole_brain.
- **top-1 picked** > Under the same cognitive function, brain states of ASD might diverge from TD due to the disturbed convergence of functional connectivity in the ASD.
- **LLM evidence pass cited** > ASD showed an imbalance of connectivity state and aberrant ratio of connectivity with different strengths in the whole brain network, and decreased connectivity associated precuneus/posterior cingulate gyrus with medial prefrontal gyrus in default mode network.
- **judgement** (NO) Discussion speculation. Luna's sentence at least contains 'whole brain network'.

### 8. `analyses.groups.n` — matches LLM pass

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `31`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the n is 31.
- **top-1 picked** > For N: TD is 44; ASD is 31; P-value is -.
- **LLM evidence pass cited** > For N: TD is 44; ASD is 31; P-value is -.
- **judgement** (OK) Sentence-ified table row. Table conversion is doing its job.

### 9. `analyses.effect.cells.level` — out of scope

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `ASD`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the level is ASD.
- **top-1 picked** > To determine the connectivity states, covariance matrices of ASD and TD were clustered by k-means clustering algorithm based on Manhattan distance.
- **LLM evidence pass cited** > Results Significant differences in the MDTs were found when the k -values of k-means clustering were 3, 5, 8, 13, 14, and 18.
- **judgement** (SKIP) Contrast level -- out of scope; a level is a schema construct, not a claim a sentence can support.

### 10. `analyses.effect.cells.direction` — out of scope

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `positive`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the direction is positive.
- **top-1 picked** > Our study might reflect some characteristics of time-varying functional state in the ASD and the differences of connectivity states in dynamic network analysis between ASD and TD groups might suggest the imbalance between excitation and inhibition.
- **LLM evidence pass cited** > Results Significant differences in the MDTs were found when the k -values of k-means clustering were 3, 5, 8, 13, 14, and 18.
- **judgement** (SKIP) Contrast direction -- out of scope for the same reason.

### 11. `analyses.effect.statistic.family` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `t`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the family is t.
- **top-1 picked** > To determine the connectivity states, covariance matrices of ASD and TD were clustered by k-means clustering algorithm based on Manhattan distance.
- **LLM evidence pass cited** > Statistical analysis Two-sample t -test was used to compare the MDTs of ASD and TD children.
- **judgement** (NO) Picked describes k-means clustering. 'Two-sample t-test was used' is two sentences away and a literal string match on 't-test' would have found it.

### 12. `analyses.details.connectivity_method` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `time-varying sliding-window correlation between ICA components`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the connectivity method is time-varying sliding-window correlation between ICA components.
- **top-1 picked** > Our study might reflect some characteristics of time-varying functional state in the ASD and the differences of connectivity states in dynamic network analysis between ASD and TD groups might suggest the imbalance between excitation and inhibition.
- **LLM evidence pass cited** > Time-varying functional connectivity was calculated based on segmented time courses in 148 windows created by a tapered window [a rectangle (width = 22 TRs) with a Gaussian (=3 TRs)] sliding in steps of 1 TR.
- **judgement** (NO) Picked is a Discussion summary. The gold is the actual sliding-window methods sentence, which is unambiguously right.

### 13. `analyses.details.inference_target` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `network`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the inference target is network.
- **top-1 picked** > Our study might reflect some characteristics of time-varying functional state in the ASD and the differences of connectivity states in dynamic network analysis between ASD and TD groups might suggest the imbalance between excitation and inhibition.
- **LLM evidence pass cited** > Statistical analysis Two-sample t -test was used to compare the MDTs of ASD and TD children.
- **judgement** (NO) Same Discussion sentence again; 'network' is inferred, not stated.

### 14. `analyses.model_representation_notes` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Mean dwell time of connectivity states, ASD against TD
- **value** `Mean dwell time is a parameter derived from the state sequence rather than a measured signal; the schema records the qua`
- **query sent to the reranker** For the Mean dwell time of connectivity states, ASD against TD, the model representation notes is Mean dwell time is a parameter derived from the state sequence rather than a measured signal; the schema records the quantity in Measure.source_label and specific_metric and has no slot saying it is derived, which is one of the known limits in extraction-readme.md §5..
- **top-1 picked** > In our study, ASD showed more divergent connectivity strength of brain state than TD (Table 2 , Figure 2 ).
- **LLM evidence pass cited** > MDT was the average number of windows that were continuous on the time distribution and classified as the same state, representing the duration of each state.
- **judgement** (NO) Picked is a result. The value is about what MDT *is*, which luna's definition sentence gives.

### 15. `analyses.details.parameter_change` — partial

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Recurrently increased connectivity in ASD
- **value** `increased`
- **query sent to the reranker** For the Recurrently increased connectivity in ASD, the parameter change is increased.
- **top-1 picked** > Also, ratios of connectivity with different strengths changed more drastically in the ASD (Figures 3 , 4 ).
- **LLM evidence pass cited** > For 1: Related components is 30, 48; Frequency of of occurrence is 10; Increase or decrease is Increase.
- **judgement** (PART) 'ratios of connectivity changed more drastically in the ASD' is topically right but does not say increased for this entity. Luna's gold is a table row that says 'Increase or decrease is Increase' -- again a table.

### 16. `analyses.details.edges.directionality` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Recurrently increased connectivity in ASD
- **value** `undirected`
- **query sent to the reranker** For the Recurrently increased connectivity in ASD, the directionality is undirected.
- **top-1 picked** > In our results, ASD showed decreased connectivity between posterior and frontal regions in DMN (PCUN/PCG and SFGmed.L, Figure 5F ).
- **LLM evidence pass cited** > Time-varying functional connectivity was calculated based on segmented time courses in 148 windows created by a tapered window [a rectangle (width = 22 TRs) with a Gaussian (=3 TRs)] sliding in steps of 1 TR.
- **judgement** (NO) 'undirected' is inferred from the fact that connectivity was computed as a correlation. The word never appears; retrieval cannot reach it.

### 17. `acquisitions.modality` — partial

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** fMRI
- **value** `fMRI`
- **query sent to the reranker** For the fMRI, the modality is fMRI.
- **top-1 picked** > However, time-varying functional connectivity analysis of resting-state functional Magnetic Resonance Imaging (fMRI) have been rarely performed on the Autism Spectrum Disorder (ASD).
- **LLM evidence pass cited** > BOLD fMRI data of each participant were acquired with a whole-brain echo planar imaging (EPI) sequence and interleaved slice acquisition (TR = 2 s, TE = 15 ms, flip angle = 90°, slice thickness = 4 mm, FoV = 240 mm, 180 volumes) on a 3T Allegra scanner.
- **judgement** (PART) The picked sentence does establish fMRI, but as an Introduction aside. The acquisition paragraph is the correct provenance.

### 18. `acquisitions.magnetic_field_strength_tesla` — matches LLM pass

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** fMRI
- **value** `3`
- **query sent to the reranker** For the fMRI, the magnetic field strength tesla is 3.
- **top-1 picked** > BOLD fMRI data of each participant were acquired with a whole-brain echo planar imaging (EPI) sequence and interleaved slice acquisition (TR = 2 s, TE = 15 ms, flip angle = 90°, slice thickness = 4 mm, FoV = 240 mm, 180 volumes) on a 3T Allegra scanner.
- **LLM evidence pass cited** > BOLD fMRI data of each participant were acquired with a whole-brain echo planar imaging (EPI) sequence and interleaved slice acquisition (TR = 2 s, TE = 15 ms, flip angle = 90°, slice thickness = 4 mm, FoV = 240 mm, 180 volumes) on a 3T Allegra scanner.
- **judgement** (OK) Exact acquisition sentence.

### 19. `acquisitions.pulse_sequence_type` — matches LLM pass

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** fMRI
- **value** `EPI`
- **query sent to the reranker** For the fMRI, the pulse sequence type is EPI.
- **top-1 picked** > BOLD fMRI data of each participant were acquired with a whole-brain echo planar imaging (EPI) sequence and interleaved slice acquisition (TR = 2 s, TE = 15 ms, flip angle = 90°, slice thickness = 4 mm, FoV = 240 mm, 180 volumes) on a 3T Allegra scanner.
- **LLM evidence pass cited** > BOLD fMRI data of each participant were acquired with a whole-brain echo planar imaging (EPI) sequence and interleaved slice acquisition (TR = 2 s, TE = 15 ms, flip angle = 90°, slice thickness = 4 mm, FoV = 240 mm, 180 volumes) on a 3T Allegra scanner.
- **judgement** (OK) Same sentence; 'echo planar imaging (EPI)' is verbatim.

### 20. `acquisitions.repetition_time_seconds` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** fMRI
- **value** `2`
- **query sent to the reranker** For the fMRI, the repetition time seconds is 2.
- **top-1 picked** > So clustering was performed at k = 2 to 20, and repeated 150 times per k -value.
- **LLM evidence pass cited** > BOLD fMRI data of each participant were acquired with a whole-brain echo planar imaging (EPI) sequence and interleaved slice acquisition (TR = 2 s, TE = 15 ms, flip angle = 90°, slice thickness = 4 mm, FoV = 240 mm, 180 volumes) on a 3T Allegra scanner.
- **judgement** (NO) Matched the bare numeral 2 in 'k = 2 to 20'. The gold says 'TR = 2 s' and never says 'repetition time' -- an abbreviation gap the query template does not bridge.

### 21. `acquisitions.echo_time_seconds` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** fMRI
- **value** `0.015`
- **query sent to the reranker** For the fMRI, the echo time seconds is 0.015.
- **top-1 picked** > However, time-varying functional connectivity analysis of resting-state functional Magnetic Resonance Imaging (fMRI) have been rarely performed on the Autism Spectrum Disorder (ASD).
- **LLM evidence pass cited** > BOLD fMRI data of each participant were acquired with a whole-brain echo planar imaging (EPI) sequence and interleaved slice acquisition (TR = 2 s, TE = 15 ms, flip angle = 90°, slice thickness = 4 mm, FoV = 240 mm, 180 volumes) on a 3T Allegra scanner.
- **judgement** (NO) Worse than #20: the value is 0.015 (seconds) and the paper says 'TE = 15 ms'. No lexical or semantic matcher reaches that without unit normalisation.

### 22. `acquisitions.number_of_volumes` — matches LLM pass

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** fMRI
- **value** `180`
- **query sent to the reranker** For the fMRI, the number of volumes is 180.
- **top-1 picked** > BOLD fMRI data of each participant were acquired with a whole-brain echo planar imaging (EPI) sequence and interleaved slice acquisition (TR = 2 s, TE = 15 ms, flip angle = 90°, slice thickness = 4 mm, FoV = 240 mm, 180 volumes) on a 3T Allegra scanner.
- **LLM evidence pass cited** > BOLD fMRI data of each participant were acquired with a whole-brain echo planar imaging (EPI) sequence and interleaved slice acquisition (TR = 2 s, TE = 15 ms, flip angle = 90°, slice thickness = 4 mm, FoV = 240 mm, 180 volumes) on a 3T Allegra scanner.
- **judgement** (OK) '180 volumes' verbatim.

### 23. `assessments.name` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Wechsler Abbreviated Scale of Intelligence, WASI-IV
- **value** `Wechsler Abbreviated Scale of Intelligence, WASI-IV`
- **query sent to the reranker** For the Wechsler Abbreviated Scale of Intelligence, WASI-IV, the name is Wechsler Abbreviated Scale of Intelligence, WASI-IV.
- **top-1 picked** > 2013DFA11140, to BH).
- **LLM evidence pass cited** > The criteria of included subjects are: male scores of full intelligence quotient (FIQ, estimated by the fourth subtests of the Wechsler Abbreviated Scale of Intelligence, WASI-IV) above 85 right-handedness aged 7–18 (not including 18 years old) TD children were matched with ASD children for age, gen
- **judgement** (NO) Picked a grant-number fragment ('2013DFA11140, to BH).'). The gold sentence contains the assessment name character-for-character. A plain substring search solves this and the reranker did not.

### 24. `assessments.description` — matches LLM pass

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Wechsler Abbreviated Scale of Intelligence, WASI-IV
- **value** `Full intelligence quotient estimated from four subtests.`
- **query sent to the reranker** For the Wechsler Abbreviated Scale of Intelligence, WASI-IV, the description is Full intelligence quotient estimated from four subtests..
- **top-1 picked** > The criteria of included subjects are: male scores of full intelligence quotient (FIQ, estimated by the fourth subtests of the Wechsler Abbreviated Scale of Intelligence, WASI-IV) above 85 right-handedness aged 7–18 (not including 18 years old) TD children were matched with ASD children for age, gen
- **LLM evidence pass cited** > The criteria of included subjects are: male scores of full intelligence quotient (FIQ, estimated by the fourth subtests of the Wechsler Abbreviated Scale of Intelligence, WASI-IV) above 85 right-handedness aged 7–18 (not including 18 years old) TD children were matched with ASD children for age, gen
- **judgement** (OK) Correct -- and note it is the same sentence #23 needed. The description query found it; the name query, which was nearly a copy of the value string, did not.

### 25. `external_datasets.name` — matches LLM pass

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Autism Brain Image Data Exchange I (ABIDE I)
- **value** `Autism Brain Image Data Exchange I (ABIDE I)`
- **query sent to the reranker** For the Autism Brain Image Data Exchange I (ABIDE I), the name is Autism Brain Image Data Exchange I (ABIDE I).
- **top-1 picked** > Participants and functional MRI data acquisition Data of participants were obtained from open accessed dataset collected by NYU Langone Medical Center, a collection site of Autism Brain Image Data Exchange I( ABIDE I ) (Di Martino et al., 2014 ).
- **LLM evidence pass cited** > Participants and functional MRI data acquisition Data of participants were obtained from open accessed dataset collected by NYU Langone Medical Center, a collection site of Autism Brain Image Data Exchange I( ABIDE I ) (Di Martino et al., 2014 ).
- **judgement** (OK) Exact.

### 26. `external_datasets.url` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** Autism Brain Image Data Exchange I (ABIDE I)
- **value** `http://fcon_1000.projects.nitrc.org/indi/abide/`
- **query sent to the reranker** For the Autism Brain Image Data Exchange I (ABIDE I), the url is http://fcon_1000.projects.nitrc.org/indi/abide/.
- **top-1 picked** > Participants and functional MRI data acquisition Data of participants were obtained from open accessed dataset collected by NYU Langone Medical Center, a collection site of Autism Brain Image Data Exchange I( ABIDE I ) (Di Martino et al., 2014 ).
- **LLM evidence pass cited** > More detailed information is available at http://fcon_1000.projects.nitrc.org/indi/abide/ .
- **judgement** (NO) Picked introduces ABIDE but contains no URL. A regex for http(s):// solves this outright.

### 27. `groups.name` — supports (different sentence)

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** typical development children
- **value** `typical development children`
- **query sent to the reranker** For the typical development children, the name is typical development children.
- **top-1 picked** > As compared to typical development children, weak relevance condition (the strength of a large number of connectivities in the state was less than means minus standard deviation of all connection strength) was maintained for a longer time between brain areas of ASD children, and ratios of weak conne
- **LLM evidence pass cited** > To investigate the influence of ASD on brain connectivity states, we performed group independent component analysis (GICA) and dynamic network analysis on fMRI data of ASD and TD children.
- **judgement** (SUP) Picked literally reads 'As compared to typical development children'; luna's gold says only 'TD children'. The top-1 is the better citation here.

### 28. `groups.species` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** typical development children
- **value** `human`
- **query sent to the reranker** For the typical development children, the species is human.
- **top-1 picked** > As compared to typical development children, weak relevance condition (the strength of a large number of connectivities in the state was less than means minus standard deviation of all connection strength) was maintained for a longer time between brain areas of ASD children, and ratios of weak conne
- **LLM evidence pass cited** > In previous studies, dynamic network analysis showed that connectivity state could be shifted in humans with long-term training and experience, such as taxi drivers (Shen et al., 2016 ).
- **judgement** (NO) 'human' is inferred. Luna's gold is worse still -- it is about taxi drivers in someone else's study.

### 29. `groups.acquired_count` — does not support

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** typical development children
- **value** `44`
- **query sent to the reranker** For the typical development children, the acquired count is 44.
- **top-1 picked** > As compared to typical development children, weak relevance condition (the strength of a large number of connectivities in the state was less than means minus standard deviation of all connection strength) was maintained for a longer time between brain areas of ASD children, and ratios of weak conne
- **LLM evidence pass cited** > For N: TD is 44; ASD is 31; P-value is -.
- **judgement** (NO) Same Discussion sentence as #27, no count. The gold is the table row that #8 retrieved correctly. Identical evidence, different field name: 'n' hit, 'acquired count' missed.

### 30. `groups.age_mean` — matches LLM pass

- **paper** 6oTrCJA43Jcd · doi:10.3389/fnhum.2016.00463
- **entity** typical development children
- **value** `12.46`
- **query sent to the reranker** For the typical development children, the age mean is 12.46.
- **top-1 picked** > For Age (Mean ± SD): TD is 12.46 ± 3.1; ASD is 11.51 ± 2.64; P-value is 0.1693.
- **LLM evidence pass cited** > For Age (Mean ± SD): TD is 12.46 ± 3.1; ASD is 11.51 ± 2.64; P-value is 0.1693.
- **judgement** (OK) Table row with the mean and SD.

### 31. `description` — partial

- **paper** 7HPLh5nJzmP5 · doi:10.1016/j.nicl.2022.103028
- **entity** _(study-level)_
- **value** `Randomized sham-controlled clinical intervention study examining delayed resting-state fMRI functional-connectivity chan`
- **query sent to the reranker** For the study, the description is Randomized sham-controlled clinical intervention study examining delayed resting-state fMRI functional-connectivity changes after intermittent theta-burst stimulation in patients with depression, and linking those changes to concurrent fNIRS oxy-Hb responses during treatment..
- **top-1 picked** > ## Discussion In this randomized sham-controlled study, we recorded blood oxygenation using fNIRS from a region within left and right dlPFC during dmPFC-iTBS treatment.
- **LLM evidence pass cited** > ## Objective To understand the network effects behind such a prefrontal fNIRS response during rTMS, and to test whether the fNIRS signal may be predictive of treatment response, we linked data from fNIRS and fMRI within a clinical intervention study.
- **judgement** (PART) Picked does describe this study but foregrounds fNIRS; the value is about the fMRI connectivity arm.

### 32. `design.description` — supports (different sentence)

- **paper** 7HPLh5nJzmP5 · doi:10.1016/j.nicl.2022.103028
- **entity** _(study-level)_
- **value** `Randomized, parallel-group, sham-controlled intervention study in patients with depression. Participants received active`
- **query sent to the reranker** For the study, the description is Randomized, parallel-group, sham-controlled intervention study in patients with depression. Participants received active or sham intermittent theta-burst stimulation twice daily for ten treatment days. fNIRS was recorded during treatment, while clinical assessments and resting-state fMRI were obtained at baseline and follow-up..
- **top-1 picked** > ## Discussion In this randomized sham-controlled study, we recorded blood oxygenation using fNIRS from a region within left and right dlPFC during dmPFC-iTBS treatment.
- **LLM evidence pass cited** > ### Procedures The patients were randomized to receive active or sham treatment in a blind treatment phase, with the iTBS protocol described below ( 2.3 Intermittent theta-burst stimulation ).
- **judgement** (SUP) 'In this randomized sham-controlled study' supports randomised + sham-controlled directly.

### 33. `design.allocation` — supports (different sentence)

- **paper** 7HPLh5nJzmP5 · doi:10.1016/j.nicl.2022.103028
- **entity** _(study-level)_
- **value** `randomized`
- **query sent to the reranker** For the study, the allocation is randomized.
- **top-1 picked** > Out of 101 eligible patients from these two studies, 69 were finally randomized (not meeting the inclusion criteria: n = 5, declined to participate: n = 26, other reasons: n = 1).
- **LLM evidence pass cited** > ### Procedures The patients were randomized to receive active or sham treatment in a blind treatment phase, with the iTBS protocol described below ( 2.3 Intermittent theta-burst stimulation ).
- **judgement** (SUP) '69 were finally randomized' is as good a citation for allocation=randomized as luna's.

### 34. `design.blinding` — does not support

- **paper** 7HPLh5nJzmP5 · doi:10.1016/j.nicl.2022.103028
- **entity** _(study-level)_
- **value** `participant_blind`
- **query sent to the reranker** For the study, the blinding is participant_blind.
- **top-1 picked** > ## Materials and methods ### Participants This project shares data with a recent randomized clinical trial comprising patients with depression or schizophrenia ( Bodén et al., 2021 ) and its add-on brain-imaging study ( Persson et al., 2020 ), with all patients being recruited from the psychiatric o
- **LLM evidence pass cited** > ### Procedures The patients were randomized to receive active or sham treatment in a blind treatment phase, with the iTBS protocol described below ( 2.3 Intermittent theta-burst stimulation ).
- **judgement** (NO) Picked is a data-provenance sentence. 'blind treatment phase' is in the gold and is the only anchor for participant_blind.

### 35. `design.arms.name` — supports (different sentence)

- **paper** 7HPLh5nJzmP5 · doi:10.1016/j.nicl.2022.103028
- **entity** active iTBS
- **value** `active iTBS`
- **query sent to the reranker** For the active iTBS, the name is active iTBS.
- **top-1 picked** > This resulted in 17 patients in the active iTBS group, and 17 patients in the sham iTBS group included in the analysis.
- **LLM evidence pass cited** > ### Procedures The patients were randomized to receive active or sham treatment in a blind treatment phase, with the iTBS protocol described below ( 2.3 Intermittent theta-burst stimulation ).
- **judgement** (SUP) Picked contains the arm name 'active iTBS' verbatim; luna's says only 'active or sham treatment'.

### 36. `description` — does not support

- **paper** 84rGLhCbUJTh · doi:10.1371/journal.pone.0035889
- **entity** _(study-level)_
- **value** `A diffusion tensor imaging study of white matter microstructure abnormalities in unmedicated obsessive-compulsive disord`
- **query sent to the reranker** For the study, the description is A diffusion tensor imaging study of white matter microstructure abnormalities in unmedicated obsessive-compulsive disorder and changes after 12-week SSRI treatment..
- **top-1 picked** > Diffusion tensor imaging (DTI) is a non-invasive method that maps the diffusivity of water molecules in tissue [5] .
- **LLM evidence pass cited** > ## Methodology and Principal Findings Parameter maps of DTI, including fractional anisotropy (FA), axial diffusivity (AD), radial diffusivity (RD) and mean diffusivity (MD), were acquired from 27 unmedicated OCD patients (including 13 drug-naïve individuals) and 23 healthy controls.
- **judgement** (NO) 'DTI is a non-invasive method that maps diffusivity' is a textbook definition, not a description of this study.

### 37. `design.assignment_structure` — supports (different sentence)

- **paper** 84rGLhCbUJTh · doi:10.1371/journal.pone.0035889
- **entity** _(study-level)_
- **value** `within_subject`
- **query sent to the reranker** For the study, the assignment structure is within_subject.
- **top-1 picked** > Pre and post-treatment and brain regions with abnormal DTI-derived parameters were set as within-subject factors.
- **LLM evidence pass cited** > ### Pharmacotherapy After receiving clinical feature measurements and DTI scans at baseline, 15 of the 27 OCD patients agreed to accept SSRI treatment only (6 of 15 OCD patients were drug-naive).
- **judgement** (SUP) Picked literally says '...were set as within-subject factors'. This is a better citation than the gold.

### 38. `design.arms.name` — supports (different sentence)

- **paper** 84rGLhCbUJTh · doi:10.1371/journal.pone.0035889
- **entity** SSRI therapy
- **value** `SSRI therapy`
- **query sent to the reranker** For the SSRI therapy, the name is SSRI therapy.
- **top-1 picked** > These parameters were compared for patients before and after 12-week Selective Serotonin Reuptake Inhibitor (SSRI) pharmacotherapy.
- **LLM evidence pass cited** > ### Pharmacotherapy After receiving clinical feature measurements and DTI scans at baseline, 15 of the 27 OCD patients agreed to accept SSRI treatment only (6 of 15 OCD patients were drug-naive).
- **judgement** (SUP) 'compared for patients before and after 12-week SSRI pharmacotherapy' names the arm.

### 39. `design.arms.description` — partial

- **paper** 84rGLhCbUJTh · doi:10.1371/journal.pone.0035889
- **entity** SSRI therapy
- **value** `Selective Serotonin Reuptake Inhibitor treatment for 12 weeks; fluvoxamine, fluoxetine, sertraline, or paroxetine were s`
- **query sent to the reranker** For the SSRI therapy, the description is Selective Serotonin Reuptake Inhibitor treatment for 12 weeks; fluvoxamine, fluoxetine, sertraline, or paroxetine were selected individually by psychiatrists..
- **top-1 picked** > These parameters were compared for patients before and after 12-week Selective Serotonin Reuptake Inhibitor (SSRI) pharmacotherapy.
- **LLM evidence pass cited** > ### Pharmacotherapy After receiving clinical feature measurements and DTI scans at baseline, 15 of the 27 OCD patients agreed to accept SSRI treatment only (6 of 15 OCD patients were drug-naive).
- **judgement** (PART) Covers SSRI and 12 weeks; misses the four drug names, which are in a separate sentence.

### 40. `design.arms.arm_kind` — partial

- **paper** 84rGLhCbUJTh · doi:10.1371/journal.pone.0035889
- **entity** SSRI therapy
- **value** `pharmacological`
- **query sent to the reranker** For the SSRI therapy, the arm kind is pharmacological.
- **top-1 picked** > Slight differences among the four SSRI drugs used in the present study should be considered, although they are all variations of SSRIs and share common pharmacological mechanisms.
- **LLM evidence pass cited** > SSRI medication included fluvoxamine, 6 patients;
- **judgement** (PART) Matched 'pharmacological' in a limitations sentence about mechanism similarity. Coincidental sense; it does establish the drugs are SSRIs.

### 41. `description` — supports (different sentence)

- **paper** AbuuV2nqkHKP · doi:10.1016/j.jadr.2023.100615
- **entity** _(study-level)_
- **value** `A placebo-controlled, double-blind trial examining regional cerebral blood flow during personalized traumatic and neutra`
- **query sent to the reranker** For the study, the description is A placebo-controlled, double-blind trial examining regional cerebral blood flow during personalized traumatic and neutral scripts in women with childhood sexual abuse with and without PTSD, and changes after paroxetine treatment..
- **top-1 picked** > Neural correlates of PTSD in women with childhood sexual abuse with and without PTSD and response to paroxetine treatment: A placebo-controlled, double-blind trial
- **LLM evidence pass cited** > The purpose of this study was to assess neural correlates of personalized traumatic memories in women with childhood sexual abuse with and without PTSD, and to assess response to treatment.
- **judgement** (SUP) Picked the paper title, which literally reads 'A placebo-controlled, double-blind trial'. Arguably the strongest citation available.

### 42. `hypothesis` — matches LLM pass

- **paper** AbuuV2nqkHKP · doi:10.1016/j.jadr.2023.100615
- **entity** _(study-level)_
- **value** `PTSD would be associated with a decrease in medial prefrontal cortex and other brain areas involved in the DMN at baseli`
- **query sent to the reranker** For the study, the hypothesis is PTSD would be associated with a decrease in medial prefrontal cortex and other brain areas involved in the DMN at baseline in comparison to non-PTSD..
- **top-1 picked** > We hypothesized based on prior studies involving behavioral, neuromodulation, and medication treatment of PTSD, including our prior study with paroxetine, that PTSD would be associated with a decrease in medial prefrontal cortex and other brain areas involved in the DMN at baseline in comparison to 
- **LLM evidence pass cited** > We hypothesized based on prior studies involving behavioral, neuromodulation, and medication treatment of PTSD, including our prior study with paroxetine, that PTSD would be associated with a decrease in medial prefrontal cortex and other brain areas involved in the DMN at baseline in comparison to 
- **judgement** (OK) Verbatim hypothesis sentence.

### 43. `design.description` — does not support

- **paper** AbuuV2nqkHKP · doi:10.1016/j.jadr.2023.100615
- **entity** _(study-level)_
- **value** `Women with childhood sexual abuse were divided into PTSD and non-PTSD groups. Women with PTSD were randomized to three m`
- **query sent to the reranker** For the study, the description is Women with childhood sexual abuse were divided into PTSD and non-PTSD groups. Women with PTSD were randomized to three months of double-blind paroxetine or placebo and underwent repeat imaging..
- **top-1 picked** > ## Discussion Women with childhood abuse-related PTSD in this study exposed to personalized traumatic scripts showed decreased function in parahippocampal gyrus, precuneus, cuneus, fusiform gyrus, insula, thalamus, posterior cingulate, dorsolateral prefrontal, parietal and temporal cortex compared t
- **LLM evidence pass cited** > Abused women were divided in those with and without PTSD.
- **judgement** (NO) Picked is a Discussion result. Luna's 'Abused women were divided in those with and without PTSD' is the design.

### 44. `design.allocation` — supports (different sentence)

- **paper** AbuuV2nqkHKP · doi:10.1016/j.jadr.2023.100615
- **entity** _(study-level)_
- **value** `randomized`
- **query sent to the reranker** For the study, the allocation is randomized.
- **top-1 picked** > All women underwent brain imaging at baseline, and the women with PTSD were randomized to a double-blind study of paroxetine versus placebo followed by repeat brain imaging at three months while still on study medication.
- **LLM evidence pass cited** > Women with PTSD were randomized to receive three months of double-blind paroxetine 10–40 mg variable dose given once per day or placebo.
- **judgement** (SUP) 'the women with PTSD were randomized to a double-blind study of paroxetine versus placebo' -- equal to the gold.

### 45. `design.blinding` — supports (different sentence)

- **paper** AbuuV2nqkHKP · doi:10.1016/j.jadr.2023.100615
- **entity** _(study-level)_
- **value** `double_blind`
- **query sent to the reranker** For the study, the blinding is double_blind.
- **top-1 picked** > All women underwent brain imaging at baseline, and the women with PTSD were randomized to a double-blind study of paroxetine versus placebo followed by repeat brain imaging at three months while still on study medication.
- **LLM evidence pass cited** > Women with PTSD were randomized to receive three months of double-blind paroxetine 10–40 mg variable dose given once per day or placebo.
- **judgement** (SUP) Same sentence, and it contains 'double-blind' verbatim.

### 46. `hypothesis` — matches LLM pass

- **paper** CLzPwUAJNDxC · doi:10.3389/fpsyt.2017.00287
- **entity** _(study-level)_
- **value** `We expected to observe steeper discounting of delayed rewards in IGD compared to HC and a modulation of delayed reward r`
- **query sent to the reranker** For the study, the hypothesis is We expected to observe steeper discounting of delayed rewards in IGD compared to HC and a modulation of delayed reward representations by the severity of IGD..
- **top-1 picked** > At the behavioral level, we expected to observe steeper discounting of delayed rewards in IGD compared to HC and a modulation of delayed reward representations by the severity of IGD.
- **LLM evidence pass cited** > At the behavioral level, we expected to observe steeper discounting of delayed rewards in IGD compared to HC and a modulation of delayed reward representations by the severity of IGD.
- **judgement** (OK) Verbatim.

### 47. `design.description` — supports (different sentence)

- **paper** CLzPwUAJNDxC · doi:10.3389/fpsyt.2017.00287
- **entity** _(study-level)_
- **value** `Observational case-control study comparing 18 Internet gaming disorder participants with 21 matched healthy controls; al`
- **query sent to the reranker** For the study, the description is Observational case-control study comparing 18 Internet gaming disorder participants with 21 matched healthy controls; all participants were male students and completed behavioral and fMRI measures..
- **top-1 picked** > This study aimed at elucidating the precise mechanisms underlying IGD by comparing intertemporal decision-making process between 18 IGD participants and 21 matched healthy controls (HCs).
- **LLM evidence pass cited** > Participants were right-handed male students (18 IGD and 21 HC) recruited through advertisements in Shanghai, PR China.
- **judgement** (SUP) 'comparing intertemporal decision-making between 18 IGD participants and 21 matched healthy controls' covers the value better than luna's recruitment sentence.

### 48. `design.allocation` — does not support

- **paper** CLzPwUAJNDxC · doi:10.3389/fpsyt.2017.00287
- **entity** _(study-level)_
- **value** `non_randomized`
- **query sent to the reranker** For the study, the allocation is non_randomized.
- **top-1 picked** > The trials in this study were presented randomly in E-prime (version 2.0, Psychology Software Tool, Figure 1 ).
- **LLM evidence pass cited** > Participants were right-handed male students (18 IGD and 21 HC) recruited through advertisements in Shanghai, PR China.
- **judgement** (NO) Matched 'randomly' in 'trials were presented randomly in E-prime' -- trial order, not group allocation. Actively misleading. non_randomized is in any case evidenced by absence.

### 49. `design.assignment_structure` — does not support

- **paper** CLzPwUAJNDxC · doi:10.3389/fpsyt.2017.00287
- **entity** _(study-level)_
- **value** `parallel`
- **query sent to the reranker** For the study, the assignment structure is parallel.
- **top-1 picked** > What’s more, impaired executive control and reward circuit have been detected in IGD ( 42 ), which is parallel with our findings.
- **LLM evidence pass cited** > This study aimed at elucidating the precise mechanisms underlying IGD by comparing intertemporal decision-making process between 18 IGD participants and 21 matched healthy controls (HCs).
- **judgement** (NO) Matched 'parallel' in 'which is parallel with our findings'. A pure homonym trap: the discourse sense, not the design sense.

### 50. `analyses.name` — partial

- **paper** CLzPwUAJNDxC · doi:10.3389/fpsyt.2017.00287
- **entity** delay > immediate
- **value** `delay > immediate`
- **query sent to the reranker** For the delay > immediate, the name is delay > immediate.
- **top-1 picked** > The RT stands for the difference between the response to delayed options and the response to immediate options (delay – immediate).
- **LLM evidence pass cited** > Brain activations change between IGD and HC (delay − immediate).
- **judgement** (PART) Right notation (delay - immediate) but the picked sentence defines a reaction-time difference; the analysis is a brain-activation contrast.

### 51. `analyses.name` — supports (different sentence)

- **paper** DTpwdoGbjqsq · doi:10.3233/JAD-200840
- **entity** AD < HC reduced GM volume
- **value** `AD < HC reduced GM volume`
- **query sent to the reranker** For the AD < HC reduced GM volume, the name is AD < HC reduced GM volume.
- **top-1 picked** > Regions of significantly reduced GM volume in AD patients compared to HC BA, Broadman areas;
- **LLM evidence pass cited** > Voxel-based between-group comparison of GM maps superimposed on the T1-weighted template in the axial plane for the AD < HC contrast.
- **judgement** (SUP) The table caption 'Regions of significantly reduced GM volume in AD patients compared to HC' names the analysis fine.

### 52. `analyses.definition` — matches LLM pass

- **paper** DTpwdoGbjqsq · doi:10.3233/JAD-200840
- **entity** AD < HC reduced GM volume
- **value** `Regions of significantly reduced GM volume in AD patients compared to HC`
- **query sent to the reranker** For the AD < HC reduced GM volume, the definition is Regions of significantly reduced GM volume in AD patients compared to HC.
- **top-1 picked** > Regions of significantly reduced GM volume in AD patients compared to HC BA, Broadman areas;
- **LLM evidence pass cited** > Regions of significantly reduced GM volume in AD patients compared to HC BA, Broadman areas;
- **judgement** (OK) Same caption; exact.

### 53. `analyses.spatial_scope` — partial

- **paper** DTpwdoGbjqsq · doi:10.3233/JAD-200840
- **entity** AD < HC reduced GM volume
- **value** `whole_brain`
- **query sent to the reranker** For the AD < HC reduced GM volume, the spatial scope is whole_brain.
- **top-1 picked** > Regions of significantly reduced GM volume in AD patients compared to HC BA, Broadman areas;
- **LLM evidence pass cited** > Results show, as expected, a significant diffuse pattern of volume loss in AD brains ( Fig.
- **judgement** (PART) Neither sentence states whole-brain; it is inferred from a voxelwise SPM analysis with no ROI mask.

### 54. `analyses.groups.n` — supports (different sentence)

- **paper** DTpwdoGbjqsq · doi:10.3233/JAD-200840
- **entity** AD < HC reduced GM volume
- **value** `20`
- **query sent to the reranker** For the AD < HC reduced GM volume, the n is 20.
- **top-1 picked** > For MMSE: AD group (n = 20) (mean±SD) is 23.1 (2.97); HC group (n = 17) (mean±SD) is –.
- **LLM evidence pass cited** > ### Statistical analysis A total of 20 mild stage AD patients and 17 matched HC were considered for both VBM-GM and C-PiB PET SUVR statistical parametric mapping analysis.
- **judgement** (SUP) The picked table row states 'AD group (n = 20)'. Supports the count as well as the gold does.

### 55. `analyses.effect.cells.level` — out of scope

- **paper** DTpwdoGbjqsq · doi:10.3233/JAD-200840
- **entity** AD < HC reduced GM volume
- **value** `AD group`
- **query sent to the reranker** For the AD < HC reduced GM volume, the level is AD group.
- **top-1 picked** > In a first approach, in order to test whether GM volume, C-PK11195 BP and C-PiB SUVR were different between AD and HC groups we performed at voxel-level group comparisons by computing two-sample t -tests in SPM12.
- **LLM evidence pass cited** > Results show, as expected, a significant diffuse pattern of volume loss in AD brains ( Fig.
- **judgement** (SKIP) Contrast level -- out of scope.

### 56. `design.timepoints.name` — does not support

- **paper** JzsUUQbDr2bm · doi:10.2147/NDT.S174356
- **entity** MRI scan
- **value** `MRI scan`
- **query sent to the reranker** For the MRI scan, the name is MRI scan.
- **top-1 picked** > 1 , 2 With the advances in noninvasive brain imaging technologies, such as multimodal magnetic resonance imaging (MRI), structural and functional brain abnormalities have been increasingly reported in both first-episode and chronic stages of SZ.
- **LLM evidence pass cited** > ### MRI data acquisition MRIs were acquired using a Philips 3T MR system (Philips, Achieva, the Netherlands) located at Guangzhou Brain Hospital.
- **judgement** (NO) Picked is an Introduction sentence about MRI advances. The timepoint name 'MRI scan' is a schema construct rather than a paper phrase.

### 57. `analyses.name` — does not support

- **paper** JzsUUQbDr2bm · doi:10.2147/NDT.S174356
- **entity** FESZ>NC
- **value** `FESZ>NC`
- **query sent to the reranker** For the FESZ>NC, the name is FESZ>NC.
- **top-1 picked** > Abbreviations: FESZ, first-episode drug-naive schizophrenia;
- **LLM evidence pass cited** > Table S1 — Significant differences in GMV among three groups
- **judgement** (NO) Picked the abbreviation list entry. Contains 'FESZ' but asserts nothing about a contrast.

### 58. `analyses.definition` — matches LLM pass

- **paper** JzsUUQbDr2bm · doi:10.2147/NDT.S174356
- **entity** FESZ>NC
- **value** `Significant differences in GMV: FESZ > NC`
- **query sent to the reranker** For the FESZ>NC, the definition is Significant differences in GMV: FESZ > NC.
- **top-1 picked** > Table S1 — Significant differences in GMV among three groups
- **LLM evidence pass cited** > Table S1 — Significant differences in GMV among three groups
- **judgement** (OK) Table title, exact.

### 59. `analyses.spatial_scope` — supports (different sentence)

- **paper** JzsUUQbDr2bm · doi:10.2147/NDT.S174356
- **entity** FESZ>NC
- **value** `roi`
- **query sent to the reranker** For the FESZ>NC, the spatial scope is roi.
- **top-1 picked** > ### ROI analysis of DC The ROI analysis indicated that the FESZ group showed significant DC reductions in the right FFG, the right SFGmed, the right PHG, the right calcarine (CAL) cortex, the right IFG, and the bilateral PCUN when compared with the NC group ( Figure 2I-D ;
- **LLM evidence pass cited** > ### ROI analysis of GMV The ROI analysis indicated that the FESZ group showed significant GMV reductions in the right fusiform gyrus (FFG), the left middle occipital gyrus (MOG), the left posterior cingulate gyrus (PCG), and the left parahippocampal gyrus (PHG) as well as significant GMV increases i
- **judgement** (SUP) Picked is the ROI-analysis heading for a different measure (DC rather than GMV), but the value being supported is just 'roi', and it supports that.

### 60. `analyses.groups.n` — supports (different sentence)

- **paper** JzsUUQbDr2bm · doi:10.2147/NDT.S174356
- **entity** FESZ>NC
- **value** `43`
- **query sent to the reranker** For the FESZ>NC, the n is 43.
- **top-1 picked** > For Age (years): FESZ patients (n=43) is 26.42±8.02; NC (n=56) is 25.07±5.85; CSZ patients (n=39) is 29.97±6.97; F value (χ2) is 5.94; P-value is 0.004???.
- **LLM evidence pass cited** > For 21: Anatomical region is Superior temporal gyrus; Hemisphere is R; Cluster size (voxels) is 151; MNI coordinates (mm) X is 46; MNI coordinates (mm) Y is −10; MNI coordinates (mm) Z is 0; t value is 2.907.
- **judgement** (SUP) Picked the demographics row stating 'FESZ patients (n=43)'. Luna's gold is a coordinate table row that never mentions 43 -- the top-1 is straightforwardly better than the gold.

### 61. `hypothesis` — matches LLM pass

- **paper** KryfAKT9dcby · doi:10.1038/s41598-017-18870-1
- **entity** _(study-level)_
- **value** `phantom acupuncture when credible could induce expectation-related placebo analgesic effects mediated by prefrontal cort`
- **query sent to the reranker** For the study, the hypothesis is phantom acupuncture when credible could induce expectation-related placebo analgesic effects mediated by prefrontal cortex activation.
- **top-1 picked** > Furthermore, we hypothesized that phantom acupuncture (when credible) could induce expectation-related placebo analgesic effects mediated by prefrontal cortex activation.
- **LLM evidence pass cited** > Furthermore, we hypothesized that phantom acupuncture (when credible) could induce expectation-related placebo analgesic effects mediated by prefrontal cortex activation.
- **judgement** (OK) Verbatim.

### 62. `design.assignment_structure` — supports (different sentence)

- **paper** KryfAKT9dcby · doi:10.1038/s41598-017-18870-1
- **entity** _(study-level)_
- **value** `parallel`
- **query sent to the reranker** For the study, the assignment structure is parallel.
- **top-1 picked** > Fifty-six non-specific low back pain patients received either real (REAL) or phantom (PHNT) acupuncture stimulation in a parallel group study.
- **LLM evidence pass cited** > ### Experimental Design Patients were randomized, i.e., parallel-group study, into real acupuncture (REAL, n = 33) and sham control phantom acupuncture (PHNT, n = 23) groups.
- **judgement** (SUP) 'received either real or phantom acupuncture stimulation in a parallel group study' -- textbook citation, better than the gold.

### 63. `design.arms.name` — partial

- **paper** KryfAKT9dcby · doi:10.1038/s41598-017-18870-1
- **entity** REAL
- **value** `REAL`
- **query sent to the reranker** For the REAL, the name is REAL.
- **top-1 picked** > For Fullness: REAL is 1.86 ± 1.56, p < 0.05; PHNT is 0.25 ± 0.72, p = 0.14; REAL Vs. PHNT is p < 0.01.
- **LLM evidence pass cited** > ### Experimental Design Patients were randomized, i.e., parallel-group study, into real acupuncture (REAL, n = 33) and sham control phantom acupuncture (PHNT, n = 23) groups.
- **judgement** (PART) Contains 'REAL' but only as a table column label; it does not introduce the arm.

### 64. `design.arms.description` — partial

- **paper** KryfAKT9dcby · doi:10.1038/s41598-017-18870-1
- **entity** REAL
- **value** `Real acupuncture stimulation with sterilized non-magnetic needles inserted and manually rotated at four acupoints.`
- **query sent to the reranker** For the REAL, the description is Real acupuncture stimulation with sterilized non-magnetic needles inserted and manually rotated at four acupoints..
- **top-1 picked** > For the REAL group, acupuncture stimulation with sterilized non-magnetic needles (0.3 mm × 30 mm, stainless steel, DongBang Co., Korea) was applied by a licensed acupuncturist who had more than 10 years of clinical experience and was trained to work in an fMRI experimental environment/protocol.
- **LLM evidence pass cited** > Needles were inserted ~2–3 cm deep into four acupoints: bilateral SP13, left SP11, and left ST36 (Fig.
- **judgement** (PART) Covers the needles and the acupuncturist; the 'four acupoints' clause is in luna's sentence. The fact is split across two sentences and top-1 can only hold one.

### 65. `design.arms.arm_kind` — does not support

- **paper** KryfAKT9dcby · doi:10.1038/s41598-017-18870-1
- **entity** REAL
- **value** `active_comparator`
- **query sent to the reranker** For the REAL, the arm kind is active_comparator.
- **top-1 picked** > REAL elicited greater activation in the bilateral pIns, ACC, SI, pOper/SII, SMA, and deactivation in the IPL and dlPFC (BA 8) compared to PHNT, whereas PHNT elicited greater activation in the bilateral vlPFC (BA 44, 45) and dlPFC (BA 46) compared to REAL.
- **LLM evidence pass cited** > The REAL group experienced somatosensory needling stimulation with needling credibility enhanced by visual stimulation, and the PHNT group experienced only the needling credibility from the visual stimulation without any somatosensory needling stimulation.
- **judgement** (NO) Picked is a results sentence. active_comparator is inferred from the REAL/PHNT design contrast, which is what luna cites.

### 66. `analyses.definition` — supports (different sentence)

- **paper** QQCjAAT6SwwQ · doi:10.1016/j.nicl.2022.103111
- **entity** Group-by-stage interaction — Wake
- **value** `Group-by-stage interaction; post-hoc comparison at Wake stage.`
- **query sent to the reranker** For the Group-by-stage interaction — Wake, the definition is Group-by-stage interaction; post-hoc comparison at Wake stage..
- **top-1 picked** > For Group-by-stage interaction: Brain region??? is Group-by-stage interaction; Post-hoc p values is Wake; Post-hoc p values.1 is N1; Post-hoc p values.2 is N2; Post-hoc p values.3 is N3.
- **LLM evidence pass cited** > Post hoc analysis showed that the functional connectivity in those areas was significantly changed across all four stages in the insomnia group compared to the control group.
- **judgement** (SUP) The sentence-ified table row maps onto the value term for term ('Group-by-stage interaction ... Post-hoc p values is Wake').

### 67. `analyses.interpretations` — partial

- **paper** QQCjAAT6SwwQ · doi:10.1016/j.nicl.2022.103111
- **entity** Group-by-stage interaction — Wake
- **value** `The connectivity of the aINS with the bilateral superior frontal gyrus, bilateral middle occipital gyrus, right inferior`
- **query sent to the reranker** For the Group-by-stage interaction — Wake, the interpretations is The connectivity of the aINS with the bilateral superior frontal gyrus, bilateral middle occipital gyrus, right inferior temporal gyrus and right postcentral gyrus was significantly lower in the patients than in the controls during wakefulness..
- **top-1 picked** > ## Results A significant group-by-stage interaction effect on aINS-based connectivity was observed in the bilateral frontal gyrus, right inferior temporal gyrus, bilateral middle occipital gyrus and right postcentral gyrus ( p < 0.05, corrected).
- **LLM evidence pass cited** > The connectivity of the aINS with the bilateral superior frontal gyrus, bilateral middle occipital gyrus, right inferior temporal gyrus and right postcentral gyrus was significantly lower in the patients than in the controls during wakefulness.
- **judgement** (PART) Picked has the interaction and the regions but not the direction; the value says connectivity was *lower* in patients, which is in luna's sentence. Direction and regions live in different sentences.

### 68. `analyses.spatial_scope` — partial

- **paper** QQCjAAT6SwwQ · doi:10.1016/j.nicl.2022.103111
- **entity** Group-by-stage interaction — Wake
- **value** `whole_brain`
- **query sent to the reranker** For the Group-by-stage interaction — Wake, the spatial scope is whole_brain.
- **top-1 picked** > For Group-by-stage interaction: Brain region??? is Group-by-stage interaction; Post-hoc p values is Wake; Post-hoc p values.1 is N1; Post-hoc p values.2 is N2; Post-hoc p values.3 is N3.
- **LLM evidence pass cited** > The ACF was developed and implemented into the 3dClustSim tool to determine the cluster-size threshold to use for a given voxelwise threshold.
- **judgement** (PART) Neither states whole-brain. The picked table row is at least the analysis in question; luna's cites the cluster-thresholding tool.

### 69. `analyses.groups.n` — supports (different sentence)

- **paper** QQCjAAT6SwwQ · doi:10.1016/j.nicl.2022.103111
- **entity** Group-by-stage interaction — Wake
- **value** `33`
- **query sent to the reranker** For the Group-by-stage interaction — Wake, the n is 33.
- **top-1 picked** > Group-by-stage interaction effect on the connectivity of the right aINS.(N=33 for the patients, N=31 for the controls.
- **LLM evidence pass cited** > #### fMRI data processing There were 33 patients with insomnia disorder and 31 age- and sex-matched healthy controls who completed all the processes and had available fMRI data.
- **judgement** (SUP) '(N=33 for the patients, N=31 for the controls' states the count directly.

### 70. `analyses.effect.cells.level` — out of scope

- **paper** QQCjAAT6SwwQ · doi:10.1016/j.nicl.2022.103111
- **entity** Group-by-stage interaction — Wake
- **value** `patients with insomnia disorder`
- **query sent to the reranker** For the Group-by-stage interaction — Wake, the level is patients with insomnia disorder.
- **top-1 picked** > MI-by-group interaction effect on the connectivity of the left aINS during wakefulness.(N=17 for the patients, N=10 for the controls).
- **LLM evidence pass cited** > Sleep discrepancy is associated with alterations in the salience network in patients with insomnia disorder: An EEG-fMRI study
- **judgement** (SKIP) Contrast level -- out of scope.
