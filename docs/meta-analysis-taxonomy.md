# What kind of meta-analysis is this, and does automation help?

Autonima screens and parses; pondie extracts and selects. Both pay off unevenly across the
literature, and the uneven part is not the papers — it is the *question* the meta-analysis
asks, because the question decides what has to be true of every row before it can be pooled.

This file sorts published neuroimaging meta-analyses into buckets by that criterion, so a
neurometabench sample can be drawn across the buckets rather than accidentally within one.

## The axis that actually predicts payoff

The intuition that "multiple task types" makes a meta-analysis hard is close, but task count
is a proxy rather than the cause, and it points the wrong way on its own. Pooling across
tasks is what a meta-analysis is *for* — Radua and Mataix-Cols' position, quoted approvingly
in the field's guideline paper, is that "a meta-analysis aims to pool across different
approaches and tasks in order to investigate effects consistent across strategies"
([Müller et al., 2018](https://doi.org/10.1016/j.neubiorev.2017.11.012)). A 47-study ALE of
music-evoked emotion spans joy, sadness, fear, tension, frissons, surprise and beauty and is
still an easy target for automation
([Koelsch, 2020](https://doi.org/10.1016/j.neuroimage.2020.117350)), because every one of
those tasks contributes the same kind of row: one group, one activation map, one direction.

What makes a meta-analysis hard to automate is whether **eligibility and sign are properties
of the contrast rather than of the paper**. Five axes, in rough order of how much damage each
does:

| # | Axis | The question it forces | Where it lands in pondie |
|---|---|---|---|
| 1 | **Unit of pooling** | Is the row a paper, a contrast, or a contrast-within-subgroup? | `Analysis` identity; `demands`/`satisfy` |
| 2 | **Sign** | Does "more" vs "less" change the answer, or is convergence unsigned? | `Cell.direction`, `Selection.direction` |
| 3 | **Group identity** | Must patients, controls, arms and timepoints be told apart? | `groups`, `Selection.contrast`, `arm_contrast` |
| 4 | **Construct-to-subtraction mapping** | Does one construct map to one subtraction, or many? | `task_family`, cell terms and levels |
| 5 | **Moderators** | Are numeric covariates needed per row (age, dose, severity)? | `groups.*`, `acquisitions.*` |

Axis 1 is the cliff, and it is the one the automated-synthesis literature names explicitly.
The original Neurosynth "aggregated all coordinates reported in a single set per study"
because heuristics cannot "differentiate coordinates belonging to distinct statistical
contrasts (e.g., 'Task A > Baseline' vs. 'Task B > Baseline'), specific subgroups (e.g.,
'Patients > Controls'), or activations and deactivations reported within the same
publication" — which is "effective for mapping broad cognitive domains but is often
insufficient for more targeted meta-analyses"
([Neurosynth Compose, 2026](https://doi.org/10.1162/IMAG.a.1114), via PubMed).

Axis 4 is the sharp version of the "multiple task types" intuition. The guideline's own
example: a Go/No-Go paper reports Go > Rest, No-Go > Rest and No-Go > Go, and only the last
two test supervisory control; picking No-Go > Rest instead pools "not only regions associated
with the process of interest but also other more general functions"
([Müller et al., 2018](https://doi.org/10.1016/j.neubiorev.2017.11.012)). The cost of task
diversity is not diversity itself — it is that each task family brings its own baseline, so
the eligible subtraction has to be re-decided per paper. A term-frequency method cannot make
that decision at all, and an LLM has to make it once per analysis rather than once per query.

Two axes are *not* on this list because they turn out to be cheap: the standard-space
determination and the whole-brain/ROI gate (rule 4). Both are Methods-section facts, both are
mechanical rules — "coordinates of experiments where authors used SPM (version SPM99 and
later) or FSL ... should be treated as being in MNI" — and pondie already keeps them as
`Selection.space` and `Selection.spatial_scope`. They are laborious for a human and nearly
free for a pipeline. They are the clearest place automation wins outright, including the
"hidden ROI" case of partial brain coverage that "hidden ROI analyses are often included"
warns about.

## The buckets

Ordered by how much of the human's work automation removes.

### A. Term-mappable domain convergence — automation is the whole method

"Where does the brain converge for X", one population, one direction, X named by a word that
studies use consistently. This is Neurosynth's native question, and the automated maps
matched hand-built ones for working memory, emotion and pain
([Yarkoni et al., 2011](https://doi.org/10.1038/nmeth.1635)). Curated exemplar: music-evoked
emotion ([Koelsch, 2020](https://doi.org/10.1016/j.neuroimage.2020.117350), via PubMed).

Needs: paper-level screening, coordinates, space, coverage. Nothing contrast-level.
**Autonima alone is close to sufficient; pondie adds the coverage and space gates.**

Failure mode is conceptual, not technical: term frequency cannot separate physical from
social pain, or disgust from negative affect — a specificity problem, not an extraction one.

### B. Multi-task convergence within one construct — automation moves the bottleneck

Same question, but the construct is realized by different subtractions in different tasks
(cognitive action control across Go/No-Go, Stop-Signal, Stroop, Simon). Here the inclusion
decision descends to the contrast, and the sample-dependence rule bites: multiple contrasts
from one subject group "are not independent", so they must be pooled into one experiment or
one must be chosen (rule 5).

Needs: contrast-level rows, per-contrast eligibility, sample identity within a paper.
**This is exactly the seam pondie's `demands`/`satisfy` split serves, and the bucket where
the extraction record earns its cost.** Autonima alone gets the wrong answer here — not
noisily, but systematically, by pooling one group's three contrasts as three studies.

Watch for the power interaction: pooling heterogeneous paradigms to reach the recommended
17–20 experiments risks results "driven by only a few experiments"
([Eickhoff et al., 2016](https://doi.org/10.1016/j.neuroimage.2016.04.072), as cited in
the guideline; that simulation study is the source of the ≥20-experiment recommendation).
Automation makes the numerator cheap, which makes this failure *easier* to fall into.

### C. Signed group difference — sign is the deliverable

Patients vs controls, and the direction is the finding. Müller et al.'s depression
re-analysis is the reference case: 57 studies, 99 experiments, whole-brain group comparisons
only, direction-separated analyses run "for meta-analyses with a minimum of 17 experiments
available", and no significant convergence anywhere
([Müller et al., 2017](https://doi.org/10.1001/jamapsychiatry.2016.2783), via PubMed). The
guideline adds a fork automation must respect: a meta-analysis *of group-comparison
experiments* gives "convergence of differences", while contrasting two single-group
meta-analyses gives "differences in convergence" — different objects, same inputs.

Needs: axes 1–3 together. Group typing, arm/timepoint sides, per-cell direction.
**`Selection.contrast="between_group"` plus `direction` is this bucket; per-cell direction is
also pondie's weakest measured field, which is not a coincidence — it is the field the paper
most often states only in prose.**

### D. Composed contrasts — interactions, treatment, longitudinal

Interaction terms, genotype × diagnosis, three-group designs, pre/post treatment. Exemplars:
imaging genetics where only the "genotype × diagnosis interaction" analysis yielded clusters
([Janouschek et al., 2018](https://doi.org/10.1007/s00429-018-1670-9), via PubMed), and
treatment-resistant vs treatment-sensitive depression vs controls — three groups, two
modalities, both directions, eight includable studies out of 1929 screened
([Miola et al., 2023](https://doi.org/10.1111/pcn.13530), via PubMed).

Needs: everything in C, plus the sign of a difference of differences, plus the half the paper
never printed. This is what pondie's `split` stage exists for.
**Highest value per paper, lowest reliability, and the bucket where a benchmark is most
informative.** See [interaction-simple-effects.md](interaction-simple-effects.md).

### E. Transdiagnostic / ill-posed constructs — automation cannot rescue the question

The construct has no canonical operationalization, so the meta-analysis fragments into a
dozen sub-analyses and finds nothing. Childhood irritability: 28 studies, ten separate ALEs,
"no evidence for neural activation convergence ... across neurocognitive functions related to
emotional reactivity, cognitive control, and reward processing, or within each domain", with
sensitivity analyses partialling out task, measure, stimulus and age all null
([Lee et al., 2022](https://doi.org/10.1016/j.jaac.2022.05.014), via PubMed).

This is the bucket the user-facing framing usually means by "multiple task types" — but note
what is actually broken. The extraction is no harder than bucket C. What fails is that the
inclusion criterion is a construct the literature does not measure the same way twice.
**Automation makes the null arrive faster and cheaper, which is a real service and not a
scientific one.** For neurometabench, these are the cases where an automated pipeline should
be scored on *reproducing the funnel*, not on reproducing a map.

### F. Region-seeded and database-native — automation is not optional but constitutive

MACM, functional decoding, large-scale clustering. The input is a whole database plus a seed
or a region, and no per-paper inclusion judgement is made at all
([NiMARE resources](https://nbclab.github.io/nimare-paper/04_resources.html)). Heterogeneity
is the signal rather than the noise here: the author-topic treatment of a left-IFJ seed
recovered "multiple task-dependent co-activation patterns" that a single ALE would have
averaged away ([Ngo et al., 2019](https://doi.org/10.1101/149567)).

**Extraction quality shows up as database coverage, not as per-record accuracy.** Bad rows do
not disqualify a study, they blur a map.

### G. Structural and image-based — a different data contract

VBM/SDM meta-analyses drop the task axis entirely (the guideline notes paradigm selection "is
not relevant for structural imaging studies") but keep sign, and SDM-family methods want
effect sizes, statistics and thresholds rather than peaks alone. IBMA wants unthresholded
maps, so its bottleneck is data availability, not text extraction
([Samartsidis et al., 2017](https://doi.org/10.1214/17-STS624)).

**Numeric-field extraction, not contrast reasoning.** `Selection.measure_type` plus statistic
fields; the audit's "model-only" fields matter more here than anywhere else.

### H. Moderated / meta-regression — the covariate is the point

Dose-response, age effects, symptom severity, medication status. Needs per-row numeric
covariates that are frequently in prose, frequently in a table, and frequently only in a
supplement. This is where the field-extraction audit's finding is decisive: 36 fields with
surface rate under 10% are "not on the page at all", and the large middle is a scoping
problem where the value is present and ambiguous
([field-extraction-audit.md](field-extraction-audit.md)).

## What this implies for neurometabench

1. **Sample by bucket, not by topic.** Sixteen papers drawn from bucket A measure a different
   pipeline than sixteen from D. The current corpus should be labelled with its bucket
   distribution before any headline number is quoted.
2. **Score bucket A on the funnel, buckets C/D on the cells.** In A, a wrong `direction` costs
   nothing because nothing filters on it; in C it inverts the result. A single macro-F1 over
   fields hides that, which is why the per-field table exists.
3. **Bucket E deserves a negative control.** A pipeline that produces a confident convergent
   map where the curated meta-analysis found none is failing in the expensive direction.
4. **The claim to test is axis 1, not task count.** Operationalized: for each benchmark paper,
   how many *eligible* analyses does the meta-analysis's own criterion admit, and does the
   paper report them for one sample or several? Papers where that number is 1 are bucket A/B;
   where it is >1 per sample, rule 5 applies and autonima's paper-level unit is wrong by
   construction.

## Sources

Retrieved via PubMed and web search, September 2026.

- Müller VI, Cieslik EC, Laird AR, Fox PT, Radua J, Mataix-Cols D, Tench CR, Yarkoni T,
  Nichols TE, Turkeltaub PE, Wager TD, Eickhoff SB. Ten simple rules for neuroimaging
  meta-analysis. *Neurosci Biobehav Rev* 2018;84:151–161.
  [10.1016/j.neubiorev.2017.11.012](https://doi.org/10.1016/j.neubiorev.2017.11.012)
- Neurosynth Compose: A web-based platform for flexible and reproducible neuroimaging
  meta-analysis. *Imaging Neuroscience* 2026.
  [10.1162/IMAG.a.1114](https://doi.org/10.1162/IMAG.a.1114)
- Yarkoni T, Poldrack RA, Nichols TE, Van Essen DC, Wager TD. Large-scale automated synthesis
  of human functional neuroimaging data. *Nat Methods* 2011;8:665–670.
  [10.1038/nmeth.1635](https://doi.org/10.1038/nmeth.1635)
- Müller VI, Cieslik EC, Serbanescu I, Laird AR, Fox PT, Eickhoff SB. Altered brain activity
  in unipolar depression revisited. *JAMA Psychiatry* 2017;74:47–55.
  [10.1001/jamapsychiatry.2016.2783](https://doi.org/10.1001/jamapsychiatry.2016.2783)
- Lee KS, et al. Systematic review and meta-analysis: task-based fMRI studies in youths with
  irritability. *J Am Acad Child Adolesc Psychiatry* 2023;62:208–229.
  [10.1016/j.jaac.2022.05.014](https://doi.org/10.1016/j.jaac.2022.05.014)
- Miola A, Meda N, Perini G, Sambataro F. Structural and functional features of
  treatment-resistant depression. *Psychiatry Clin Neurosci* 2023;77:252–263.
  [10.1111/pcn.13530](https://doi.org/10.1111/pcn.13530)
- Janouschek H, Eickhoff CR, Mühleisen TW, Eickhoff SB, Nickl-Jockschat T. Using
  coordinate-based meta-analyses to explore structural imaging genetics. *Brain Struct Funct*
  2018;223:3045–3061. [10.1007/s00429-018-1670-9](https://doi.org/10.1007/s00429-018-1670-9)
- Koelsch S. A coordinate-based meta-analysis of music-evoked emotions. *NeuroImage*
  2020;223:117350.
  [10.1016/j.neuroimage.2020.117350](https://doi.org/10.1016/j.neuroimage.2020.117350)
- Ngo GH, Eickhoff SB, Nguyen M, Sevinc G, Fox PT, Spreng RN, Yeo BTT. Beyond consensus:
  embracing heterogeneity in curated neuroimaging meta-analysis. *bioRxiv* 2019.
  [10.1101/149567](https://doi.org/10.1101/149567)
- Eickhoff SB, Nichols TE, Laird AR, Hoffstaedter F, Amunts K, Fox PT, Bzdok D, Eickhoff CR.
  Behavior, sensitivity, and power of activation likelihood estimation characterized by
  massive empirical simulation. *NeuroImage* 2016;137:70–85.
  [10.1016/j.neuroimage.2016.04.072](https://doi.org/10.1016/j.neuroimage.2016.04.072)
- Samartsidis P, Montagna S, Johnson TD, Nichols TE. The coordinate-based meta-analysis of
  neuroimaging data. *Stat Sci* 2017;32:580–599.
  [10.1214/17-STS624](https://doi.org/10.1214/17-STS624)
</content>
</invoke>
