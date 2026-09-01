# Fields code can fill, and the ones it cannot

> Where this sits in the pipeline: [pipeline-architecture.md](pipeline-architecture.md).

`derive_fields.py` fills extraction fields from code where code can be trusted and abstains
otherwise. This file is which fields those are, the precision measured for each, and why the
rejected candidates were rejected — the script says how.

    python -m pondie.extraction.tools.derive_fields --audit 'data/runs/<run>/records/*.extraction.json' -v
    python -m pondie.extraction.tools.derive_fields --fill  'data/runs/<run>/records/*.extraction.json' --apply

## The rule that makes this safe

**A deriver may return nothing.** Abstention is the design, not a shortfall. A wrong derived
value is worse than no derived value, because a model value at least carries evidence a
reviewer can check, and a regex that guesses launders a guess into a fact.

Two consequences, both enforced in the code:

- `--fill` writes **only into an empty field**. A model value is never overwritten.
- A disagreement is reported as a **conflict** and applied to nothing. Where the derivation
  and the model differ, one of them is wrong and neither a regex nor a schema knows which;
  that is a question for a reviewer.

The schema already carries a `deterministic` subset, and it is honoured: `Study.title`,
`doi`, `journal`, `publication_year`, `authors` and the identifiers are not asked of the
model and are absent from the extraction record. What follows is about fields currently
marked `model_extracted` that code can supply anyway.

## Measured, on the 16-paper corpus

| field | route | precision | agree | conflict | fillable | abstain |
|---|---|---:|---:|---:|---:|---:|
| `Acquisition.magnetic_field_strength_tesla` | `N T` / `N Tesla`, modal hit | **100%** | 23 | 0 | 5 | 0 |
| `Group.species` | animal-term frequency | **100%** | 32 | 0 | 0 | 3 |
| `Group.age_unit` | conditioned on species | **100%** | 22 | 0 | 10 | 3 |
| `Statistic.family` | stage-1 value kind | 93.7% | 74 | 5 | 2 | 21 |
| `Cell.direction` | contrast-name operator; statistic sign | 92.6% | 75 | 6 | 3 | 102 |

20 fields the model left empty are fillable at these precisions.

**`Cell.direction` arbitrated against reviewer gold** is the number that matters, because
"agrees with the model" is not the same as "right". Over the 101 signed gold cells: 49 both
right, 5 **deriver right and model wrong**, 1 both wrong, 46 abstained. The one both got
wrong is `kzMj26hGWacQ / Baseline > week 6`, which is an inverted gold answer — so **55/55
against corrected gold**, and the deriver never lost a cell the model had right.

Three of its five wins are the `xevP8UDRAVh9` `held → signed` cells, which were the only
confirmed extraction defect left after the instrument bugs were fixed. The name
`analysis_heroin_negative_correlation` states the slope's sign; the model put that sign on
the held term instead. Reading the name gets it right for nothing.

**`Statistic.family`'s 5 conflicts are all `f` or `chi_square`**, and they are the deriver's
fault rather than the model's. Stage 1 has no value kind for either, so an F table arrives
labelled `t-statistic`; deriving `t` from that would replace a correct `f` with a wrong
value. This is exactly why a conflict is never applied.

## Rejected candidates, and why

Recording these so they are not re-attempted.

- **`Statistic.degrees_of_freedom_denominator`** — empty in 69 of 102 analyses, so the
  incentive is real. But a `df` column appears in **1 of 57** coordinate tables. The
  degrees of freedom are in the prose, per analysis, and a paper-wide regex cannot say
  which analysis a number belongs to.
- **`InferenceSettings.multiple_comparison_method`** — 32% precision, and the failure is
  structural rather than fixable. A paper carries several `InferenceSettings` with
  different methods — a voxel threshold uncorrected, a cluster threshold FWE-corrected —
  and a paper-level match cannot scope a hit to the right one. The observed model values
  (`none; the ROI-level tests are reported uncorrected`) are also free text that no closed
  vocabulary matches.
- **`Analysis.coordinate_space`** — stage 1 supplies a normalised code (`MNI`, `TAL`) and
  the field is empty in 13 of 102, so it looks fillable. It is not the same field: the
  schema wants *the paper's own wording* here and puts the code on
  `Table.coordinate_space`, which is already deterministic. Filling this from the code
  would collapse a deliberate distinction. The wording drift that remains — `MNI` 76,
  `MNI space` 4, `Talairach` 4, `TAL` 5 — is a normalisation question for the mapper.
- **`Preprocessing.software`** — 88.9% against a closed package list, and both apparent
  failures were the *list* missing EEG software the model named correctly (`Brain Vision
  Analyzer`, `MATLAB`). A closed vocabulary cannot be shown complete, so this stays an
  audit-only check rather than a filler.

## Where the abstentions are

For `Cell.direction`, the 46 abstentions on gold cells break down as: the level matches
neither side of the contrast name (24), the name carries no comparison operator (20), and
the analysis is a slope whose statistics carry no sign (15). Those are the cells worth
asking a model about, and they are the argument for the narrow contrast schema in
[contrast-direction-rubric.md](contrast-direction-rubric.md) — ask about half as many cells,
in a shape that cannot express the errors this corpus produced.

## The order to run it in

Derive **before** the model pass, not after. A field already filled is one the prompt does
not have to ask for, and every field removed from the request is output tokens not spent —
which matters when roughly one run in four loses a pass to an empty payload. `--audit`
after the pass is then a free consistency check on what the model did supply.

---

# The systematic pass: all 171 extracted fields

The fields above were found by intuition. This section is the mechanical screen over
everything the pipeline actually fills, and the reason each field is in or out.

## The screen: within-paper invariance

A field is only reachable by a paper-wide pattern if a match can be assigned to the right
*instance*. Nothing in this schema is one-per-paper — a paper carries 1.8 acquisitions, 1.9
inference settings, 2.2 groups, 6.4 analyses on average — so the test that matters is
whether the value is **invariant across instances within a paper**. Where it is, a single
paper-wide match is unambiguous. Where it is not, the pattern finds a *set* of values and
has no way to say which entity each belongs to.

Run over the 16-paper corpus:

| class | fields | reachable by a paper-wide pattern? |
|---|---:|---|
| invariant across instances | 26 | yes |
| only ever one instance | 24 | yes |
| mostly invariant (≥75%) | 5 | with a conflict check |
| **varies within a paper** | **103** | **no** |

**That 103 is the headline.** It is not an accuracy problem and no extractor fixes it. `TR`
differs per sequence, `smoothing_fwhm_mm` per analysis, `alpha_level` between a voxel
threshold and a cluster threshold. Measured on `repetition_time_seconds`, a paper-wide modal
match scores 60% and sentence-scoping by modality reaches 75% on four answerable cases —
directionally right, and useless at that coverage.

## What was implemented, and at what precision

| field | route | precision |
|---|---|---:|
| `Acquisition.magnetic_field_strength_tesla` | modal `N T` / `N Tesla` | **100%** (23) |
| `Group.species` | animal-term frequency | **100%** (32) |
| `Group.age_unit` | conditioned on species | **100%** (22) |
| `Acquisition.mr_acquisition_type` | `2D` / `3D` | **100%** (3) |
| `StudyDesign.blinding` | `double-blind` … `unblinded` | **100%** (3) |
| `StudyDesign.assignment_structure` | `crossover` / `parallel` / … | **100%** (4) |
| `Statistic.family` | stage-1 value kind | 93.7% (79) |
| `Cell.direction` | contrast-name operator; statistic sign | 92.6% vs model, **55/55 vs gold** |

## Rejected, with the reason

**Structurally unreachable — varies within a paper (103 fields).** `repetition_time_seconds`,
`echo_time_seconds`, `smoothing_fwhm_mm`, `alpha_level`, `cluster_extent_threshold`,
`multiple_comparison_method`, every per-analysis threshold, every per-group count. The
regexes work; the assignment does not.

**Free text with a `range: string` that carries more than a code.** Rejected even where a
keyword matches reliably:

- `Group.diagnostic_system` — a keyword finds `DSM-IV`; the model wrote
  `DSM-IV-TR; NINCDS-ADRDA`. The derivation drops an edition and a second system. The schema
  also documents a false positive no keyword can avoid: an edition appearing only inside an
  instrument's title (`SCID for DSM-IV Axis II Disorders`) does not establish it.
- `ModelEstimation.hrf_model` — `canonical hemodynamic response function using a double-gamma
  function` against a derived `canonical`.

**Genuinely a reading task.** `name`, `description`, `definition`, `interpretations`,
`model_settings`, `source_definition`, `instructions`, `stimuli`,
`clinical_characteristics`, `model_representation_notes`. These are paraphrases of a paper's
prose; there is no vocabulary to match and no pattern to fit.

**No source.** `degrees_of_freedom_denominator` is empty in 69 of 102 analyses and a `df`
column appears in **1 of 57** coordinate tables.

## Non-LLM alternatives, researched

**scispaCy.** Already evaluated in this repo and rejected — but only for sentence splitting
and abbreviation detection, where it agreed on 59 of 59 shared short forms and its extra
output was noise ([text-preprocessing-experiments.md](text-preprocessing-experiments.md)
§ dependency). That verdict does not cover NER, so it was reconsidered here on its merits.

**Biomedical NER on Hugging Face.** The [OpenMed](https://huggingface.co/OpenMed) family
covers exactly the entity types this schema wants — `AnatomyDetect` (ElectraMed-560M,
ModernClinical-149M, BioPatient-108M), `DiseaseDetect` (SuperClinical-184M/434M, PubMed-v2),
`PathologyDetect`, and a zero-shot variant — trained on BC5CDR-Disease and the NCBI Disease
corpus.

**They do not address the binding constraint.** NER returns *typed spans*. The 103
unreachable fields fail at *assigning a value to an entity instance*, which is relation
extraction / slot filling, not recognition. A model that labels every `3 T` and every `2000
ms` in a paper leaves you exactly where a regex does. This is the single most important
finding of the pass: **the bottleneck is entity scoping, and no off-the-shelf recogniser
solves it.** What does solve it is a model that reads the whole methods section and attaches
values to entities — which is what the extractor already is.

Where NER is the right tool, narrowly:

- `Group.medical_condition` — 34 values, 17 distinct (50% reuse), a closed-ish disease
  vocabulary, UMLS-linkable. The one field where `DiseaseDetect` plus a linker is a better
  fit than either a regex or an LLM.
- `Group.medications` — 4 values; RxNorm or a gazetteer, but the volume does not justify a
  dependency yet.

**Brain-region lexicons.** The established route is a gazetteer from atlas label sets with
synonyms — Neuronames, BAMS, Allen — optionally with a CRF for names outside the lexicon;
this is how [large-scale connectivity extraction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4426844/)
works. It covers less of this corpus than it looks: of 45 `regions.name` values, **45 are
distinct (0% reuse)** and only 53% contain a standard anatomical term. The remainder are
*networks* — `dorsal attention network`, `executive control network`, `salience network` —
which no anatomical atlas contains, and AAL-style abbreviations (`Left cuneus (CUN.L)`)
which need the atlas's own label table rather than a general lexicon.

**The abbreviation table already in the pipeline.** `assessments.name` is 46 distinct of 47,
and nearly every value is `Long Form (ACRONYM)` — which is what `preprocess.py`'s
Schwartz & Hearst detector already finds, with no new dependency. It recovers **53%** of the
assessment names in this corpus. That is *candidate generation*, not extraction: the same
table also yields `GM`, `WM` and `fMRI`, so it needs a filter to say which long forms are
instruments. A gazetteer of psychometric instruments would be that filter.

**Small closed gazetteers, not yet built.** `devices.manufacturer` (19 values, 13 distinct,
32% reuse — Siemens / Philips / GE and their subsidiary spellings) and
`preprocessings.software` (49 values, 34 distinct, 31% reuse — SPM8, SPM12, DARTEL, FSL,
MATLAB). Both are small, closed and verifiable; both need normalisation rules more than they
need a model. `software` measured 88.9% against a hand-written list, and both apparent
failures were the *list* missing EEG packages the model named correctly — which is the
argument for maintaining the vocabulary as data rather than as a regex.
