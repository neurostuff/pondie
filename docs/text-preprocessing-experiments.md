# Preprocessing the paper before the extractor sees it

Ten deterministic transforms of the paper text, measured against the pipeline that does
none. The question is narrow and worth stating precisely, because most of the ways of
asking it are unanswerable with one gold record:

> Does anything you can do to the *input string* with regular expressions and a sentence
> splitter make the extractor fill the schema better?

Not "does preprocessing help" in general. Every arm here is a change to what the model
reads, on a fixed pipeline, a fixed prompt scaffold and a fixed model, so a difference is
attributable to the transform or to noise and nothing else.

Read [extraction-workflow-experiments.md](extraction-workflow-experiments.md) first. It
establishes the two facts that shape this whole design: run-to-run spread swamps most
config effects, and every failure found so far on the verified paper is *structural* rather
than a misreading. Preprocessing is an intervention on reading. That is the prior it has to
beat.

Code: [`study-schema/review/preprocess.py`](../study-schema/review/preprocess.py),
[`compare_agreement.py`](../compare_agreement.py),
[`study-schema/test_preprocess.py`](../study-schema/test_preprocess.py).

---

## 0. What was measured, and what it can support

| | |
|---|---|
| arms | 10 preprocessing strategies + `pre_control` (no preprocessing) |
| substrate | demand-driven ordering, zero-foci rule, post-condition retry — the best measured configuration in the workflow experiments |
| papers | `xevP8UDRAVh9` (the one human-verified gold record), `6oTrCJA43Jcd`, `SULKxviGFurw`, `aVGe9BmFTMDR` |
| replicates | 3 per (arm × paper), then 10 on the five arms worth resolving — 176 pipeline runs, ~360 model calls |
| model | `gpt-5.6-luna`, low effort, evidence pass off |
| gold scoring | `compare_extractions.py`, on `xevP8UDRAVh9` only; `--scope tables` as primary and `--scope all` as a check (§5.5) |
| agreement scoring | `compare_agreement.py`, on all four |

**One gold record still bounds everything.** Three of the four papers have no answer key, so
on those the only measurable quantity is whether runs agree with each other. §5.8 is about how
much that is worth, and the answer is worse than hoped: on descriptive fields agreement is a
weak positive signal fit for triage, and **on the signed cells — the fact the record exists to
carry — it anti-correlates with correctness.** So no arm ranking here rests on agreement; the
three extra papers contributed no evidence about which strategy is better, and §5.7 says so
plainly rather than dressing agreement up as a score.

---

## 1. Where preprocessing can act, and the invariant that makes it safe

`extract_record.py` builds one prompt per pass: conventions → worked models → schema →
context blocks → **the paper** → "emit the JSON object now". A transform can act on the
paper string or add a block just before it, and the two are kept apart on purpose:

| kind | what it does | arms |
|---|---|---|
| **text** | the prompt carries a *different string* — sections dropped, sections reordered, sentences selected. Nothing is added. | `sections`, `reorder`, `retrieval` |
| **digest** | the paper is untouched; a derived block is inserted ahead of it. Nothing is removed. | `abbrev`, `stats`, `contrasts`, `methods`, `cohort`, `regions` |
| both | section-scoped text plus digests | `combo` |

Separating them is what makes the result interpretable. If reduction wins and augmentation
does not, the mechanism is distraction; if augmentation wins and reduction does not, the
mechanism is retrieval difficulty; if both win, they are different mechanisms and compose.

**Nothing a strategy does moves an offset.** `build_record.py` is handed the untransformed
file, so `EvidenceSpan.start_char` and `ExtractionMetadata.source_text_hash` are computed
against the paper as the corpus stores it whatever the prompt contained. A reviewer is shown
the same text either way. This is not incidental — it is what allows an arm to delete 40% of
the paper without breaking export.

**A digest is labelled as a candidate list, never as an answer.** Every block carries:

> Derived from the paper by regular expression, not read. It over-generates… confirm every
> entry against the paper text below, and drop whatever the paper does not support. It adds
> nothing the paper does not contain, so it can never be the source for a value.

That framing is load-bearing. Regex over prose over-generates by construction, and a block
the model is told to trust imports every false positive into the record. The literature on
schema-guided extraction reaches the same conclusion from the other direction: bounding the
generator to *candidate proposal* with a separate confirmation step is what keeps recall
without paying for it in hallucination
([Efficient and Verified Research Data Extraction with LLM](https://www.mdpi.com/1999-4893/19/3/214)).

`test_preprocess.py` enforces the two invariants as tests, not as intentions: a text
strategy may not emit a prose sentence the paper does not contain (checked as a sentence
multiset, with only two named constants — the omission marker and the reordering note —
allowed through), and every coordinate table survives every strategy.

---

## 2. Why these ten

Each arm targets a failure the workflow experiments already documented, or a mechanism the
literature says is real. None of them was chosen because it was easy to write.

### The three reduction / reordering arms

**`sections`** — drop the Introduction, the Discussion and the back matter; keep front
matter, Methods, Results and tables. Aimed at the *hallucinated analysis*. On the verified
paper the entity pass invented a `Task` and two `Condition`s the study does not have, and
the Introduction of that paper is four paragraphs of other people's findings phrased exactly
like results ("the GM of prefrontal, temporal, and insular regions was negatively correlated
with the duration of heroin use"). Shi et al. showed a single irrelevant sentence measurably
degrades an LLM's reasoning on problems it otherwise solves
([Large Language Models Can Be Easily Distracted by Irrelevant Context](https://arxiv.org/abs/2302.00093));
a literature review is a whole section of them, written in the register the extractor is
looking for.

**`reorder`** — the same content, Methods and Results first, Introduction and Discussion in
the middle, tables last. Aimed at position rather than content. Attention over a long
context is U-shaped, highest at the start and end and weakest in the middle, and the effect
is large enough to change answers
([Lost in the Middle](https://arxiv.org/abs/2307.03172)).
This arm exists to separate the two mechanisms: if `reorder` wins and `sections` does not,
the problem was where the facts sat, not that the argument was present.

**`retrieval`** — BM25 over sentences against a query built from the schema's own slot names
and vocabularies, keeping ~45% of the prose in document order, with an explicit marker where
sentences were cut. Front matter and every table block are kept whole regardless of score:
the abstract states the design and every headline result in two hundred words, and the tables
are the only source of a coordinate. This is the standard priority-truncation / relevance-
filtering shape reported across LLM extraction pipelines for long documents, and it is the
most aggressive reduction here.

### The six digest arms

Each is routed to the pass it is aimed at, because a cohort digest handed to the analyses
pass is noise in a prompt that is already 120k characters.

| arm | pass | derived from | targets |
|---|---|---|---|
| `abbrev` | both | Schwartz & Hearst (2003) abbreviation detection | two passes naming one entity two ways |
| `stats` | analyses | statcheck-style APA regexes (`t(df)=`, `F(df1,df2)=`, `r=`, `χ²`, `Z`, `p`), plus coordinate triples and cluster extents | a result reported in prose and in no coordinate table |
| `contrasts` | analyses | cue-phrase sweep for comparison / correlation / factorial / null / hedge wording | analysis recall — the text side of a dual anchor |
| `methods` | entities | one regex per Methods parameter, labelled with the real schema slot | `Acquisition`, `Preprocessing`, `InferenceSettings` field accuracy |
| `cohort` | entities | sample, sex, age, criteria, arm and timepoint phrases, labelled with the real slot | `Group`, `CategoryDistribution`, `Arm`, `Timepoint` |
| `regions` | both | anatomy pattern split into ROI-context mentions and result-table labels | `Region` recall, and `Analysis.regions` having anything to point at |

Four of these are pointed at a specific documented failure:

- `stats` and `contrasts` at the **structural table ceiling**. On the verified paper, two of
  gold's six analyses are the VBM contrast between conditions, reported in one Results
  sentence and in no coordinate table, so the stage-1 parse caps analysis recall at 67%
  before extraction starts. The sentence is
  *"Comparison of the heroin and placebo conditions found no significant difference in
  either direction (heroin > placebo and heroin < placebo)."* The `contrasts` sweep finds it
  and flags it `comparison+null`. Whether the model then emits the analyses is the
  experiment.
- `regions` at the **`Region` recall 0%** failure. Gold has exactly two regions, `frontal
  lobe` and `temporal lobe`, from *"we used an explicit mask of the frontal and temporal lobe
  by WFU PickAtlas"*. Read naively that is one region with a name the paper does not have;
  the digest distributes the shared head noun and names both. It also keeps the anatomy
  labels in a result table's first column strictly apart, because those label a coordinate
  and are not `Region`s.
- `methods` and `cohort` at **field accuracy**, which is the one metric high reasoning effort
  measurably *lowered*.

The domain has prior art worth naming: `pubextract`, in this same ecosystem, already extracts
sample sizes and participant demographics from neuroimaging papers with rules
([Mining the neuroimaging literature](https://elifesciences.org/articles/94909)), and COBIDAS
enumerates the acquisition and inference parameters a paper is supposed to report
([Best Practices in Data Analysis and Sharing in Neuroimaging using MRI](https://www.humanbrainmapping.org/files/2016/COBIDASreport.pdf)).
The `methods` and `cohort` patterns are aimed at that same target list.

### Not tried, and why

- **UMLS / atlas normalisation of region names.** `Region.name` wants the paper's own
  wording; an atlas lookup would rewrite `Paracingulate Gyrus/ACC` into something the
  reviewer cannot find in the text.
- **Coreference resolution.** The entity the extractor gets wrong is usually named
  explicitly in the Methods; the failures are structural, not anaphoric.
- **Sliding-window chunking with per-chunk extraction.** It is a pipeline change, not a
  preprocessing change, and it fights the one thing the workflow experiments found is
  load-bearing — the whole-paper view the analyses pass needs to decide which analyses exist.

---

## 3. What the transforms cost

Mean over the four papers, both passes, in characters:

| arm | paper text | digest | net vs `pre_control` |
|---|---:|---:|---:|
| `pre_control` | 34,992 | — | — |
| `sections` | 21,825 | — | **−38%** |
| `retrieval` | 21,036 | — | **−40%** |
| `reorder` | 35,200 | — | +1% |
| `regions` | 34,992 | 698 | +2% |
| `cohort` | 34,992 | 832 | +2% |
| `methods` | 34,992 | 1,137 | +3% |
| `stats` | 34,992 | 1,401 | +4% |
| `abbrev` | 34,992 | 1,410 | +4% |
| `contrasts` | 34,992 | 4,066 | +12% |
| `combo` | 21,825 | 9,543 | −10% |

The paper is about a fifth of the prompt, so a 40% cut to the paper is an ~8% cut to the
prompt, and none of these arms changes cost enough to matter. **Cost is not what this
experiment is about** — every arm is affordable, and the question is purely whether any of
them changes the record.

---

## 4. The regex components, checked against a real NLP pipeline

`preprocess.py` is standard library only, which is a constraint worth defending rather than
assuming: this repo's dependency list is three packages, and a preprocessing step needing a
100 MB biomedical model is a different proposition from one needing a regex.

So the two components where a trained pipeline could plausibly do better were measured
against scispaCy — `en_core_sci_sm` for sentence boundaries, and its `AbbreviationDetector`,
which implements the same Schwartz & Hearst algorithm, for abbreviations
([ScispaCy](https://aclanthology.org/W19-5034.pdf)).
[`check_against_spacy.py`](../study-schema/review/check_against_spacy.py) does it.

Abbreviation counts are after the two-character fix described below.

| paper | sentence agreement | abbrev: ours | scispaCy | shared | same expansion |
|---|---:|---:|---:|---:|---:|
| `xevP8UDRAVh9` | 88.4% | 13 | 14 | 13 | **13** |
| `6oTrCJA43Jcd` | 80.5% | 21 | 21 | 20 | **20** |
| `SULKxviGFurw` | 87.2% | 20 | 20 | 19 | **19** |
| `aVGe9BmFTMDR` | 91.5% | 8 | 11 | 7 | **7** |

**On abbreviations the two agree on the expansion in 59 of 59 shared short forms.** The
remaining scispaCy-only hits are mostly lowercase words it treats as short forms (`control`,
`memory`, `locations`, `marker`, `spot`, `symptoms`) plus one genuine miss on our side
(`LASSO`). Sentence boundaries agree 80–92% as exact strings — the disagreements are
citation-marker and unit edge cases, and none of the digests is sensitive to a boundary
being one clause off.

The comparison also *found a bug*: a filter meant to keep `In` and `we` out of the
abbreviation table was rejecting every two-character candidate, and so throwing away `GM`
and `WM` — load-bearing abbreviations in this corpus. scispaCy had them; we did not.
Two-character candidates are now kept when both characters are upper case or digits.

**Verdict: the dependency does not pay.** For the two components a trained pipeline could
have improved, it either agrees exactly or its extra output is noise. It earned its keep as
a *test oracle*, and `check_against_spacy.py` is kept for that — it is not imported by the
pipeline.

---

## 5. Results

### 5.1 The headline metric is bimodal: a run either builds gold's contrast shape or it does not

Before any arm can be compared, the shape of the outcome has to be stated, because it
decides what a mean means. Per-run direction F1 on the gold paper takes essentially two
values, 0% and 100%:

| arm | per-run direction F1 (10 runs) |
|---|---|
| `pre_control` | 0 0 0 **100** 0 0 0 0 0 0 |
| `pre_sections` | 0 0 0 0 **100** 0 0 0 0 0 |
| `pre_reorder` | **100 100** 0 0 **100 100 100** 0 0 0 |
| `pre_retrieval` | **100 100 100 100** 0 50 **100** 0 0 **100** |
| `pre_combo` | **100 100 100 100 100** 0 0 **100** 0 **100** |

The reason is the one the workflow experiments identified: `Effect.cells` cannot be righter
than `ModelEstimation.terms`. Gold encodes each correlation as **two** cells — `held` at a
level of a categorical `perfusion condition` term, plus a signed slope on a continuous term.
A run that models the two perfusion conditions as two *continuous* covariates emits one
signed cell per analysis, and then not one of its cells can match:

| | terms built | cells | direction F1 |
|---|---|---|---|
| a losing run | `placebo-associated perfusion` (continuous), `heroin-associated perfusion` (continuous) | 4, one per analysis | 0% |
| a winning run | `perfusion condition` (categorical, 2 levels), a continuous slope | 8, held + signed per analysis | 100% |

So an arm is not making cells more accurate. **It is changing the probability that the
entity pass builds a model that can express the contrast at all.** Every number below should
be read as that probability, and the run — not the cell — is the independent unit, because
all eight cells of a run share one model.

### 5.2 Two arms move it, and both are sentence-level reduction

Thirteen runs of the control configuration were available: ten from the sweep and three from
the prompt-isolation probe of §5.4, whose prompt is byte-identical (verified programmatically,
not assumed — `--preprocess none` produces the same string). Pooling them is the honest
baseline; scoring against the sweep's ten alone would have credited the reduction arms with
about 13 points they did not earn.

Exact permutation test where the arm has 3 runs, Monte-Carlo with 200,000 draws where it has
10; one-sided, run as the unit.

| arm | n | mean direction F1 | Δ vs control | p |
|---|---:|---:|---:|---:|
| **`pre_control`** (pooled) | **13** | **23.1%** | — | — |
| `pre_combo` | 10 | 70.0% | **+46.9** | **0.033** |
| `pre_retrieval` | 10 | 65.0% | **+41.9** | **0.034** |
| `pre_reorder` | 10 | 50.0% | +26.9 | 0.18 |
| `pre_contrasts` | 3 | 33.3% | +10.3 | 0.61 |
| `pre_cohort` | 3 | 33.3% | +10.3 | 0.61 |
| `pre_abbrev` | 3 | 16.7% | −6.4 | 0.61 |
| `pre_stats` | 3 | 16.7% | −6.4 | 0.61 |
| `pre_methods` | 3 | 11.1% | −12.0 | 0.61 |
| `pre_sections` | 10 | 10.0% | −13.1 | 0.92 |
| `pre_regions` | 3 | 0.0% | −23.1 | 1.00 |

The same test on the shape indicator directly — *did this run emit ≥4 `held` cells on a
categorical term and ≥4 signed cells on a continuous one*, which is scorer-independent and
name-independent — agrees: `pre_combo` 9/10 vs control 3/10 (p = 0.010), `pre_retrieval`
8/10 (p = 0.035), `pre_reorder` 6/10 (p = 0.18), `pre_sections` 2/10 (n.s.).

**These p-values are not corrected for testing ten arms.** At Bonferroni α = 0.005 neither
survives. Two things keep them worth acting on rather than discarding: `retrieval` and
`combo` are not independent tests (combo *contains* retrieval's reduction), and the reduction
hypothesis was registered before the runs as one of three mechanisms, not selected after.
The correct status is **a signal worth a held-out replication, not an established effect.**

### 5.3 It is not the amount of text: section dropping does nothing

`pre_sections` cuts 38% of the paper and `pre_retrieval` cuts 40%. One is indistinguishable
from the control (−13.1 points, p = 0.92) and the other is the second-best arm (+41.9,
p = 0.034). Whatever is happening, **"less text" is not the mechanism.**

The difference is granularity. `sections` removes whole zones and leaves every sentence of
Methods and Results intact; `retrieval` removes low-scoring sentences from *inside* those
zones, keeping 42 of the paper's 138 prose sentences — 45% of the prose by character, fewer
than that by count because a longer sentence matches more query terms and scores higher. Both keep the sentences that name the
conditions — checked directly, both retain *"We correlated each perfusion condition (heroin
and placebo) separately…"* and *"In BPM, each perfusion condition (heroin and placebo) was
correlated separately with the VBM data."* So the win is not "retrieval kept the key sentence
and sections lost it."

A measurable difference that does track the outcome is the *ratio* of the two rival framings
in the surviving text — the factor framing (`perfusion condition`) against the
separate-maps framing (`heroin-/placebo-associated perfusion`), which is exactly the wrong
model the losing runs build:

| arm | "perfusion condition" | "…-associated perfusion" | ratio | direction F1 |
|---|---:|---:|---:|---:|
| `pre_control` | 3 | 10 | 0.30 | 23.1% |
| `pre_reorder` | 3 | 10 | 0.30 | 50.0% |
| `pre_sections` | 3 | 7 | 0.43 | 10.0% |
| `pre_retrieval` | 3 | 5 | 0.60 | 65.0% |
| `pre_combo` | 6 | 17 | 0.35 | 70.0% |

It is a hypothesis and not a result: `reorder` has the control's ratio and twice its score,
and `combo` has a middling ratio and the best score, so the counts cannot be the whole story.
Stated because it is testable — a synthetic arm that deletes only the separate-maps sentences
would settle it in ten runs — not because it is established.

### 5.4 The control does not reproduce its documented score, and the cause is *not* established

**This section previously claimed that a prompt-material edit cost ~70 points. That claim was
wrong and is retracted.** It rested on three runs of the old prompt material, all three of
which happened to succeed. Ten more runs put the same configuration at 37.5%, and the
difference from the current material is neither large nor significant. The retraction is kept
in place rather than deleted because the reasoning error is the instructive part: a 3/3 result
on an outcome that is bimodal at roughly p = 0.4 is not evidence of anything, and it was
treated as evidence.

The fact that prompted the investigation is real. The workflow experiments record 92.3%
pooled direction F1 for this configuration; the archived records from that sweep still score
**93.6%** under today's scorer, with 11 of their 12 runs building the categorical term. The
same configuration today scores **23.1%** over thirteen runs. The scorer did not change.

To test whether the prompt material explains it, two trees were built from the submodule
differing *only* in the three files the prompt renders — `extraction-readme.md`,
`representing-models.md` §5 and the extraction schema — with identical code, identical text,
identical flags. Adding `retrieval` to both turns it into a 2×2:

| prompt material | arm | n | direction F1 | per-run |
|---|---|---:|---:|---|
| `62bdd87` (old) | control | 13 | 51.9% | 100 100 100 0 0 0 0 0 0 100 100 100 75 |
| `62bdd87` (old) | `retrieval` | 10 | 35.0% | 100 0 0 100 0 0 0 0 50 100 |
| `628ad75` (current) | control | 13 | 23.1% | 0 100 100 0 0 0 100 0 0 0 0 0 0 |
| `628ad75` (current) | `retrieval` | 10 | 65.0% | 100 100 100 100 0 50 100 0 0 100 |

Two-sided permutation tests, run as the unit:

| comparison | Δ | p |
|---|---:|---:|
| prompt material, control vs control | +28.8 | 0.23 |
| `retrieval`, under the current material | +41.9 | **0.050** |
| `retrieval`, under the old material | −16.9 | 0.46 |
| current + `retrieval` vs old + control | +13.1 | 0.61 |
| **interaction** (`retrieval` × prompt material) | +58.8 | **0.041** |

Three conclusions, and the first two are corrections:

1. **The prompt material is not shown to be the cause.** +28.8 points at p = 0.23. It may
   well be a real regression — the point estimate is large and in the expected direction —
   but thirteen runs a side cannot establish it, and this report should not have claimed it.
2. **The archived 93.6% is not reproduced by either tree.** 11/12 then, 4/10 and 3/10 now, on
   the same paper with the same scorer. So the difference is not in these three files. What
   is left is the *uncommitted* code state at the time of those runs — `extract_record.py`
   changed by 369 lines across the interval, and it carries prompt text of its own in
   `MODE_NOTE`, `SATISFY_NOTE` and `DEMANDS_NOTE` that this probe did not revert — or a
   change on the model side, which nothing here can rule out. **The regression is real and
   its cause is unidentified.**
3. **`retrieval` does not transfer.** It is worth +41.9 points on the current prompt material
   and −16.9 on the old, and the interaction is the one thing in this table that clears
   p = 0.05. On this evidence `retrieval` is not a general improvement to the pipeline; it is
   an improvement *to the state the pipeline is in now*, and the mechanism §5.3 could not
   pin down is presumably the reason it is state-dependent.

That third result is the most consequential thing in this report, and it is a caution rather
than a finding to build on: an arm can win by 42 points in one configuration and lose in
another that differs only in the conventions document.

### 5.5 The six digests do nothing measurable

Every digest arm sits within ±12 points of the pooled control at n = 3, all p > 0.6. Three of
them are numerically *worse*. The digests are not failing to find their targets — the
extracted content is right, which is the point of §4 and of the tests:

- `regions` names gold's two `Region`s exactly (`frontal lobe`, `temporal lobe`) by
  distributing the shared head noun in *"an explicit mask of the frontal and temporal lobe"*,
  and keeps the eight result-table anatomy labels separate.
- `contrasts` finds the sentence carrying gold's two text-only VBM analyses and flags it
  `comparison+null`.
- `methods` recovers 3T / Magnetom Verio / Siemens / MPRAGE / TR 2000 & 3200 ms / TE 3.4 &
  12.7 ms / SPM8 / SPM5 / BPM / WFU PickAtlas / FWHM 6 & 8 mm / MNI + Talairach / FWE /
  cross-over, double-blind, within-subject — every one correct, each labelled with its real
  schema slot.
- `cohort` recovers the full sample sentence, the sex split, the age mean and SD, handedness,
  DSM-IV, SCID-II, both criteria, both arms and both occasions.

**Correct candidates delivered to the right pass changed nothing.** The most economical
reading is the one the workflow experiments already argued: these failures were never reading
failures. The model could already find `3T` and `frontal lobe`; handing them over saves it a
search it was not losing.

**And this is not an artefact of what was scored.** `--scope tables` restricts gold to the
four analyses a publication table reported — which excludes exactly the two text-only VBM
analyses `pre_contrasts` was aimed at, and would have *penalised* an arm for emitting them.
Re-scored with `--scope all`, so gold is all 6 analyses and all 12 cells, the ranking is
unchanged and the verdict on `contrasts` gets worse rather than better:

| arm | n | direction F1 | analysis recall |
|---|---:|---:|---:|
| `pre_control` | 10 | 8.0% | 66.7% |
| `pre_sections` | 10 | 10.2% | 68.3% |
| `pre_reorder` | 10 | 40.0% | 66.7% |
| `pre_retrieval` | 10 | 52.0% | 66.7% |
| `pre_combo` | 10 | 56.0% | 66.7% |
| `pre_contrasts` | 3 | 26.7% | 66.7% |
| `pre_stats` | 3 | 13.3% | 66.7% |
| `pre_regions` | 3 | 0.0% | 66.7% |

**Analysis recall is 66.7% — four of gold's six — for every arm, with zero variance at
n = 10.** `pre_sections` is the only arm that ever exceeds it, in one or two runs of ten, and
`pre_contrasts` never does. The 80% direction ceiling that every winning run hits is exactly
the four cells of the two analyses nobody finds.

So `contrasts` is the sharpest failure in the report, and the most informative. It put the
sentence carrying both missing analyses in front of the pass whose one job is to enumerate
analyses, tagged `comparison+null`, in a prompt where a `--zero-foci-rule` paragraph already
told the pass that an effect finding nothing is still an effect. The pass ignored it in all
three of its runs, and no arm at n = 10 did better. **The structural ceiling the workflow experiments identified is not a
retrieval problem** — which is worth knowing, because "surface the sentence" is the obvious
first thing to try and it demonstrably does not work.

### 5.6 On the secondary metrics, one arm improves everything

Ten replicates, properly matched by `compare_extractions.py`, gold paper, scope `tables`:

| arm | composite | entity F1 | relationship F1 | field acc |
|---|---:|---:|---:|---:|
| `pre_control` | 47.3% | 88.5% | 69.8% | 74.1% |
| `pre_sections` | +0.7 (0.0 sd) | +1.6 (0.4) | +1.2 (0.1) | +1.2 (0.3) |
| `pre_reorder` | +19.4 (0.9) | +1.7 (0.7) | +3.7 (0.4) | +2.2 (0.6) |
| `pre_retrieval` | +28.1 (1.5) | **−1.2** (−0.3) | +14.8 (1.4) | +4.1 (1.1) |
| **`pre_combo`** | **+31.6** (1.6) | **+5.8** (1.8) | **+14.6** (1.4) | **+3.4** (1.0) |

`pre_combo` is the only arm that improves every metric, and it is the only arm whose entity F1
gain clears its own spread (1.8 sd). That is the one place a digest may be pulling its weight:
combo is retrieval-like reduction *plus* the entity-side digests, and it beats retrieval on
entities by 7.0 points while matching it elsewhere. At n = 10 on one paper that is suggestive,
not established — but it is the specific comparison a follow-up should make.

`pre_retrieval` costing 1.2 points of entity F1 while gaining 41.9 of direction is the
expected trade of an aggressive cut, and on this record it is a good trade: a wrong sign
inverts the finding, a missing `Device` does not.

### 5.7 On the three papers without gold, the arms are indistinguishable

`compare_agreement.py` over 33 runs per paper, cell facts:

- **consensus holds zero cell facts on every one of the four papers.** No signed cell is
  asserted by a majority of runs, on any paper, under any arm.
- analysis facts: self = consensus = cross = **100.0%** for all eleven arms on both
  `SULKxviGFurw` and `aVGe9BmFTMDR`. Perfect agreement, zero resolution.
- entity and field agreement separate the arms by 10–30 points, but in an order that has
  little to do with the gold-scored order (§5.8).

So the honest statement about the three extra papers is that **they contributed no evidence
about which preprocessing strategy is better.** They were not wasted — they are what shows
the agreement instrument has no resolution on the metric that matters, and they surfaced the
defect below — but no arm ranking rests on them.

`6oTrCJA43Jcd` is worth naming on its own. Its stage-1 parse has exactly one entry,
*"Independent component spatial maps — Peak values distribution of the independent component
spatial maps"*, which by the stage-1 OMIT rule is a component-definition listing and not a
tested effect, so the right answer is **zero** analyses and a filled `Table.non_analysis_content`.
The arms emitted 0, 1, 2 and — once, under `retrieval` — **14**, one per component. No run
filled `non_analysis_content`. And the `demands` post-condition treats "no analyses" as a
failure, so every run that gave the likely-correct answer burned its whole retry budget and
was logged `DEGENERATE`: all six degenerate runs in the sweep are this paper, four of them in
reduction arms. **The post-condition punishes the correct answer on a paper whose only table
is not an analysis**, which makes the `degen` column uninterpretable here and is a defect in
the pipeline rather than in any arm.

### 5.8 Does convergence mean correctness? On the fact that matters, no

This was the reason for measuring agreement at all, and the answer is clean. Fifty runs of
five arms on the gold paper:

| fact family | consensus precision | consensus recall | consensus size | gold size |
|---|---:|---:|---:|---:|
| cell | — | 0.0% | **0** | 12 |
| analysis | 100.0% | 66.7% | 4 | 6 |
| entity | 60.0% | 25.0% | 10 | 24 |
| field | 66.2% | 23.0% | 77 | 222 |

And the rank correlation between an arm's agreement and its correctness:

| fact family | self-agreement vs gold | consensus-agreement vs gold |
|---|---:|---:|
| cell | **−0.47** | undefined (empty consensus) |
| analysis | +1.00 | +1.00 |
| entity | +0.70 | +0.60 |
| field | +0.70 | +0.30 |

Three conclusions, in descending order of confidence:

1. **On signed cells, agreement is not merely uninformative — it points the wrong way.**
   Spearman −0.47. `pre_sections` has the *highest* self-agreement on cells (32.6%) and a gold
   cell F1 of 0.0%; it agrees with itself because it reliably builds the same wrong model.
   Reproducibility of an error is indistinguishable from reproducibility of a fact, and on
   this record the correlation is negative. **Never use run-to-run agreement as a proxy for
   direction correctness.**
2. **The `analysis` +1.00 is an artefact of ties**, not a finding: every arm scored the same
   80.0% analysis F1, so the correlation is computed over a constant.
3. **On descriptive slots, agreement is a weak positive signal** (+0.70 for both entity and
   field, self-agreement) with a majority vote that is right 60–66% of the time and finds
   under a quarter of the truth. High precision at low recall is the profile of a **triage
   signal**: a fact all runs agree on is worth spot-checking rather than reviewing in full,
   and a fact they disagree on is worth a human. It is not a score, and a pipeline tuned to
   maximise it would be tuned to be consistent rather than right.

A caveat that cuts in the measurement's favour: `compare_agreement.py` keys facts by name, so
a run that names an entity correctly but differently from its peers counts as having invented
one. Every agreement number is therefore a lower bound, and the gap is large — the same
records score 87–94% entity F1 under the properly-matched scorer and 24–33% under the
name-keyed one. That bias cannot explain the *negative* cell correlation, which is the
conclusion that matters.

---

## 6. Is any of this worth implementing?

**One thing, conditionally: sentence-level relevance filtering (`retrieval`), or the
`combo` arm that contains it.** Everything else is a no.

| | verdict |
|---|---|
| `retrieval` / `combo` | **not adoptable yet, and weaker than §5.2 alone suggests.** +42 to +47 points of direction F1 on the current prompt material (p ≈ 0.03–0.05, uncorrected), +15 relationship, +3–4 field, +6 entity for `combo`, and 40% less text. But `retrieval` scores −16.9 on the *previous* prompt material and the interaction clears p = 0.05 (§5.4): the gain does not survive a change to the conventions document, so there is no reason to expect it to survive the next one. |
| `reorder` | **hold.** +26.9 points but p = 0.18. It is free and it is the only arm that removes nothing, so it is the cheapest thing to re-test on new gold. |
| `sections` | **no.** No effect at any n, and it removes content for nothing. |
| `abbrev`, `stats`, `contrasts`, `methods`, `cohort`, `regions` | **no.** All within ±12 points, all n.s., three numerically negative — despite each demonstrably finding what it was aimed at. |
| a scispaCy / spaCy dependency | **no.** It agreed with the stdlib implementation on 59 of 59 shared abbreviations and its extra output is mostly noise. Keep `check_against_spacy.py` as a test oracle. |

**What should actually be done first is none of the above.** Ranked by measured consequence:

1. **Find out why the control fell from 93.6% to 23.1%, and make the prompt material
   versioned input so the next one is attributable.** §5.4 establishes the regression and
   *fails* to explain it: the three rendered prompt files account for +28.8 points at
   p = 0.23, which leaves most of it unexplained. The remaining candidates are the
   uncommitted code state those runs used — `extract_record.py` moved 369 lines and carries
   its own prompt text in `MODE_NOTE`, `SATISFY_NOTE` and `DEMANDS_NOTE` — and model-side
   drift. Both are cheap to test and neither has been. Separately and regardless of the
   answer: the parent repo pins `420f799`, the working tree runs `628ad75`, and
   `ExtractionMetadata` records the model and the date but nothing about which conventions
   document produced the record. A prompt-material hash in the record, plus a control re-run
   whenever it changes, is what would have made this attributable in an hour.
2. **Fix the `demands` post-condition for papers with no analysis** (§5.7). "No analyses were
   emitted" is a legitimate answer when every table is a definition listing, and the current
   check spends the whole retry budget rejecting it and then reports `DEGENERATE`. It should
   accept zero analyses when every stage-1 entry is one the OMIT rule covers — and it should
   require `Table.non_analysis_content` instead, which no run filled.
3. **Then the held-out replication of `retrieval`/`combo`** — on gold this experiment has not
   seen, *and* on more than one prompt-material state, because §5.4 shows a single state is
   not enough to know whether the arm helps the pipeline or only helps this version of it.

If `retrieval` is eventually adopted, the mechanics are easy: the flag exists
(`--preprocess retrieval`), the transform is deterministic and standard-library,
`build_record.py` still sees the untouched file so no offset moves, and the budget is one
number to tune. What it must not be adopted with is a claim that the mechanism is understood.
§5.3 shows it is not, and §5.4 shows the practical cost of that: an unexplained mechanism is
an effect nobody can predict the transfer of.

---

## 7. What the next experiment should be

Ordered so that each step's result changes what the next one is.

1. **Two more verified gold records with the same modelling shape** — a held-level contrast
   over a factor whose levels an arm carries. Not a random sample: the effect measured here is
   entirely about whether the entity pass builds a factor or two covariates, and only papers
   that pose that choice can replicate it. This is the whole experiment's binding constraint,
   exactly as §0 of the workflow experiments said.
2. **Explain the control regression.** Bisect the interval with the *code* held variable
   rather than the three data files: `extract_record.py`'s embedded prompt text is the
   untested candidate, and the archived payloads are on disk to diff against. If nothing in
   the repo explains it, the remaining hypothesis is model-side drift, which is worth knowing
   before any further single-state measurement is trusted.
3. **The framing-count ablation** (§5.3), which is ten runs and no new gold: an arm that
   deletes only the sentences stating the two conditions as separate maps, against one that
   deletes an equal number of random Methods sentences. If the targeted arm wins, the
   mechanism is rival-framing salience and the right fix is a much smaller and more
   defensible transform than BM25 over the whole paper.
4. **A retrieval budget sweep** — 0.25 / 0.45 / 0.65 of the prose, ten runs each. The budget
   was set once, at 0.45, and never varied; if the effect is monotone in aggressiveness that
   is evidence for a distraction mechanism, and if it peaks it is evidence against.
5. **Re-run the whole matrix at a second prompt-material state.** §5.4 makes this the
   difference between "preprocessing helps" and "preprocessing helped once".

---

## 8. What this does *not* show

- **That preprocessing helps extraction.** It shows that on one paper, in one
  prompt-material state, one aggressive reduction shifted the probability of building a
  correct model from 3/13 to 9/10 at an uncorrected p ≈ 0.03 — and that the same arm loses
  16.9 points in the state that preceded it (§5.4). Six targeted digests carrying
  demonstrably correct content did nothing at all. The general result is the null one.
- **That the regex extractors are good.** They are good on the four papers they were built
  against, and two of them were explicitly tuned by reading the gold paper's output. See §9.
- **That agreement is useless.** It is a usable triage signal on descriptive fields
  (precision 60–66%) and actively misleading on signed cells (ρ = −0.47). Those are different
  verdicts for different families, and the second is the one that would have been assumed
  otherwise.

## 9. Threats to validity

- **One gold record, and one modelling shape.** Every gold-scored number is `xevP8UDRAVh9`,
  and the effect is specifically about whether the entity pass builds a two-level factor or
  two continuous covariates. A paper that does not pose that choice cannot show this effect
  either way, so "does reduction help extraction" is genuinely unanswered.
- **Uncorrected p-values across ten arms, and one of them slipped through.** §5.2 states the
  correction problem; §5.4 is a worked example of the other failure mode, a claim built on an
  n = 3 cell that ten more runs overturned. At Bonferroni α = 0.005 neither surviving arm
  survives. They are reported as a signal because
  the two are not independent and the hypothesis was registered in advance, not because 0.03
  is enough.
- **The digests were developed while reading `xevP8UDRAVh9`'s output.** The shared-head-noun
  split in `regions` and the null-flag in `contrasts` were both written after seeing what that
  paper needed. Those two arms' *content quality* (§5.5) is therefore optimistic by an unknown
  amount. It happens not to matter for the conclusion — both arms scored nothing — but a
  future arm built the same way and scoring well would need a held-out paper before anyone
  believed it.
- **The control moved under the experiment, for reasons still unknown.** §5.4: the same
  configuration scored 93.6% in the archive and 23.1% here, and the three rendered prompt
  files explain +28.8 of the ~70 points at p = 0.23. All eleven arms ran within one hour
  against one state, so the *between-arm* comparison is internally sound — but the 2×2 in
  §5.4 shows the winning arm's advantage does not survive a change of state, so no absolute
  number here should be expected to hold on a different one.
- **`compare_agreement.py` matches facts by name, not by content.** A run that names an entity
  correctly but differently from its peers counts as having invented one, so every agreement
  number is a lower bound — the same records score 87–94% entity F1 under the properly-matched
  scorer and 24–33% here. That bias inflates disagreement; it cannot explain the negative cell
  correlation in §5.8, which is the conclusion that depends on it.
- **Three replicates on six of the arms.** Enough to exclude an effect the size of
  `retrieval`'s, not enough to exclude a ten-point one. The digest arms are reported as "no
  measurable effect", not as "no effect".
- **`degen` is uninterpretable on `6oTrCJA43Jcd`** (§5.7), because the post-condition rejects
  the likely-correct answer there. Degenerate counts are reported per paper for that reason.
- **The scorer shares no code with the extractor**, and `preprocess.py` shares none with
  either. That much is in the design's favour: `compare_extractions.py` reads the schema, and
  the transforms cannot see the metric.
