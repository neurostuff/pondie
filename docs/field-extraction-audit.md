# Field-by-field extraction audit

> Where this sits in the pipeline: [pipeline-architecture.md](pipeline-architecture.md).

Every field the pipeline fills, what was tried on it, and the maximum precision a
non-model method can reach. Companion to [deterministic-fields.md](deterministic-fields.md),
which covers the fields that were shipped; this is the exhaustive pass behind it.

`audit_field_extraction.py` produces the table.

## What each column means

- **n** — instances across the 16-paper corpus.
- **surface** — the value is *locatable* in the paper text: verbatim, case/punctuation
  folded, or as a number in any plausible unit spelling. This is the ceiling for **every**
  surface method. String matching, a regex, spaCy and a transformer tagger all read the same
  characters; they differ in how a span is found, not in whether it is there. A value below
  ~10% here cannot be recovered by any of them, only produced by a model.
- **uniq** — located **and** matching in exactly one place, so a match is resolvable without
  further information. **This is the max precision of a surface method with no scope rule**,
  and it is the number to read as the answer for each field.
- **medC** — median candidate matches. `medC` 20 with `uniq` 0% means the value is on the
  page twenty times and nothing distinguishes the right occurrence.
- **section** — which section group the first correct match falls in, from the
  `paper_sections` offsets already in `extraction_metadata`.

## The headline

| | fields |
|---|---:|
| `uniq` ≥ 80% — a surface method can be precise | **20** |
| `uniq` 50–80% | 29 |
| `uniq` 20–50% | 31 |
| `uniq` < 20% — located but unresolvable | **78** |

| | fields |
|---|---:|
| surface ≥ 90% | 67 |
| surface 50–90% | 32 |
| surface 10–50% | 23 |
| **surface < 10% — model-only** | **36** (827 instances) |

The schema splits three ways: 20 fields a pattern can handle precisely, 36 that are not on the
page at all, and a large middle where the value is present and ambiguous. That middle is the
scoping problem, and it is why no recogniser helps — see the method notes below.

## Section scoping, measured

Restricting the search to Methods sections was tested per field. Globally it **loses**: the
unique-match rate falls from 30% to 18%, because only 47% of values are located in a Methods
section at all. Per field it can be decisive:

| field | uniq, whole document | uniq, Methods only |
|---|---:|---:|
| `groups.age_minimum` | 11% | **78%** |
| `groups.age_maximum` | 11% | **44%** |
| `preprocessings.software` | 65% | **69%** |
| `acquisitions.repetition_time_seconds` | 22% | **33%** |
| `statistic.degrees_of_freedom_denominator` | 0% | **27%** |
| `analyses.groups.n` | 2% | **26%** |
| `groups.acquired_count` | 3% | **19%** |

It is a per-field tool, not a global one.

## The full table

| field | n | surface | uniq (max precision) | medC | section | shipped | Methods-scoped | spaCy NER recall |
|---|---:|---:|---:|---:|---|---|---|---|
| `devices.model` | 17 | 100% | **100%** | 1 | methods | — | — | — |
| `measures.unit` | 6 | 100% | **100%** | 1 | results | — | — | — |
| `groups.race_distribution.category` | 6 | 100% | **100%** | 1 | methods | — | — | — |
| `analyses.details.decoded_variable` | 5 | 40% | **100%** | 1 | unsectioned | — | — | — |
| `model_estimations.hrf_model` | 4 | 50% | **100%** | 1 | methods | — | — | — |
| `regions.atlas` | 3 | 33% | **100%** | 1 | methods | — | — | — |
| `external_datasets.url` | 2 | 100% | **100%** | 1 | methods | — | — | — |
| `acquisitions.mode_of_administration` | 2 | 100% | **100%** | 1 | methods | — | — | — |
| `groups.ethnicity_distribution.category` | 2 | 100% | **100%** | 1 | methods | — | — | — |
| `acquisitions.eeg_reference` | 1 | 0% | **100%** | 1 | unsectioned | — | — | — |
| `acquisitions.eeg_placement_scheme` | 1 | 100% | **100%** | 1 | methods | — | — | — |
| `acquisitions.modality_label` | 1 | 100% | **100%** | 1 | unsectioned | — | — | — |
| `analyses.details.generalization` | 1 | 0% | **100%** | 1 | unsectioned | — | — | — |
| `analyses.effect.mediation.path` | 1 | 100% | **100%** | 1 | intro | — | — | — |
| `inference_settings.search_volume` | 1 | 100% | **100%** | 1 | methods | — | — | — |
| `groups.exclusion_criteria` | 81 | 52% | **88%** | 1 | methods | — | — | — |
| `groups.age_mean` | 22 | 95% | **86%** | 1 | methods | — | — | — |
| `acquisitions.number_of_volumes` | 7 | 100% | **86%** | 1 | methods | — | — | — |
| `groups.medications` | 13 | 100% | **85%** | 1 | results | — | — | — |
| `assessments.assessment_type` | 47 | 9% | **83%** | 1 | intro | — | — | — |
| `acquisitions.pulse_sequence_type` | 19 | 84% | **79%** | 1 | methods | — | — | — |
| `groups.inclusion_criteria` | 59 | 49% | **78%** | 1 | methods | — | — | — |
| `model_estimations.model_type` | 40 | 25% | **68%** | 1 | unsectioned | — | — | — |
| `tables.footer` | 42 | 100% | **67%** | 1 | results | — | — | — |
| `groups.age_standard_deviation` | 21 | 100% | **67%** | 1 | methods | — | — | — |
| `groups.education_summary` | 9 | 0% | **67%** | 1 | intro | — | — | — |
| `groups.handedness_distribution.category` | 6 | 100% | **67%** | 1 | methods | — | — | — |
| `external_datasets.name` | 3 | 67% | **67%** | 1 | methods | — | — | — |
| `preprocessings.software` | 49 | 94% | **65%** | 1 | methods | — | 69% | — |
| `tables.caption` | 57 | 100% | **65%** | 1 | results | — | — | — |
| `model_estimations.software` | 34 | 62% | **65%** | 1 | methods | — | — | — |
| `design.timepoints.time_from_intervention` | 14 | 21% | **64%** | 1 | unsectioned | — | — | — |
| `analyses.prespecification` | 25 | 64% | **64%** | 1 | discussion | — | — | — |
| `assessments.name` | 47 | 81% | **64%** | 1 | methods | — | — | 75% |
| `devices.manufacturer` | 19 | 95% | **63%** | 1 | methods | — | — | 73% |
| `groups.recruitment_method` | 16 | 38% | **62%** | 1 | methods | — | — | — |
| `analyses.interpretations` | 106 | 19% | **62%** | 1 | unsectioned | — | — | — |
| `analyses.definition` | 102 | 8% | **61%** | 1 | unsectioned | — | — | — |
| `analyses.details.validation_scheme` | 5 | 40% | **60%** | 1 | unsectioned | — | — | — |
| `model_estimations.estimator` | 29 | 59% | **59%** | 1 | methods | — | — | — |
| `regions.name` | 45 | 84% | **58%** | 1 | results | — | — | 50% |
| `design.timepoints.name` | 26 | 42% | **58%** | 1 | unsectioned | — | — | — |
| `groups.name` | 35 | 66% | **54%** | 1 | unsectioned | — | — | — |
| `tasks.performance_measures` | 15 | 53% | **53%** | 1 | unsectioned | — | — | — |
| `tasks.name` | 17 | 71% | **53%** | 1 | unsectioned | — | — | 38% |
| `groups.medical_condition` | 34 | 94% | **50%** | 2 | results | — | — | — |
| `acquisitions.echo_time_seconds` | 18 | 100% | **50%** | 2 | methods | — | — | — |
| `analyses.details.modulatory_input` | 4 | 0% | **50%** | 1 | unsectioned | — | — | — |
| `tasks.presentation_software` | 2 | 50% | **50%** | 2 | unsectioned | — | — | — |
| `preprocessings.steps` | 122 | 15% | **48%** | 1 | methods | — | — | — |
| `analyses.name` | 102 | 63% | **48%** | 1 | results | — | — | — |
| `groups.clinical_characteristics` | 11 | 27% | **45%** | 1 | intro | — | — | — |
| `inference_settings.correction_scope` | 16 | 69% | **44%** | 2 | methods | — | — | — |
| `groups.diagnostic_system` | 7 | 86% | **43%** | 2 | methods | — | — | — |
| `measures.specific_metric` | 24 | 50% | **42%** | 2 | methods | — | — | — |
| `model_estimations.terms.name` | 86 | 67% | **41%** | 2 | methods | — | — | — |
| `analyses.details.classes.label` | 10 | 60% | **40%** | 62 | unsectioned | — | — | — |
| `groups.sex_distribution.percentage` | 5 | 100% | **40%** | 2 | tables | — | — | — |
| `assessments.description` | 38 | 11% | **39%** | 1 | unsectioned | — | — | — |
| `groups.medication_status` | 13 | 8% | **38%** | 1 | unsectioned | — | — | — |
| `inference_settings.cluster_extent_threshold` | 13 | 100% | **38%** | 8 | methods | — | — | — |
| `inference_settings.multiple_comparison_method` | 29 | 59% | **38%** | 2 | methods | — | — | — |
| `groups.sex_distribution.category` | 24 | 83% | **38%** | 2 | methods | — | — | — |
| `tasks.instructions` | 8 | 0% | **38%** | 1 | unsectioned | — | — | — |
| `measures.source_label` | 38 | 82% | **37%** | 3 | methods | — | — | 54% |
| `design.arms.agent` | 11 | 64% | **36%** | 2 | unsectioned | — | — | — |
| `tasks.conditions.description` | 29 | 0% | **34%** | 1 | unsectioned | — | — | — |
| `groups.race_distribution.percentage` | 6 | 100% | **33%** | 9 | methods | — | — | — |
| `model_estimations.terms.source_definition` | 28 | 0% | **32%** | 1 | unsectioned | — | — | — |
| `description` | 16 | 0% | **31%** | 1 | unsectioned | — | — | — |
| `groups.description` | 34 | 0% | **29%** | 1 | unsectioned | — | — | — |
| `acquisitions.acquisition_duration_seconds` | 7 | 43% | **29%** | 1 | results | — | — | — |
| `tables.table_number` | 57 | 100% | **28%** | 2 | results | — | — | — |
| `tasks.conditions.name` | 29 | 79% | **28%** | 6 | unsectioned | — | — | — |
| `groups.species` | 35 | 69% | **26%** | 2 | intro | 100% | — | — |
| `acquisitions.mr_acquisition_type` | 4 | 75% | **25%** | 2 | methods | 100% | — | — |
| `acquisitions.repetition_time_seconds` | 18 | 100% | **22%** | 10 | methods | — | 33% | — |
| `model_estimations.model_family` | 39 | 46% | **21%** | 2 | methods | — | — | — |
| `model_estimations.stage` | 40 | 88% | **20%** | 15 | methods | — | — | — |
| `model_estimations.terms.unit` | 15 | 100% | **20%** | 5 | methods | — | — | — |
| `design.arms.name` | 16 | 94% | **19%** | 10 | unsectioned | — | — | — |
| `design.arms.description` | 16 | 0% | **19%** | 1 | unsectioned | — | — | — |
| `model_estimations.spatial_unit` | 30 | 100% | **17%** | 9 | methods | — | — | — |
| `groups.age_unit` | 25 | 100% | **16%** | 3 | methods | 100% | — | — |
| `analyses.effect.cells.level` | 158 | 89% | **15%** | 42 | unsectioned | — | — | — |
| `model_estimations.terms.levels.level` | 73 | 89% | **15%** | 12 | methods | — | — | — |
| `preprocessings.description` | 21 | 0% | **14%** | 1 | unsectioned | — | — | — |
| `tasks.stimuli` | 7 | 0% | **14%** | 1 | unsectioned | — | — | — |
| `regions.description` | 32 | 0% | **12%** | 1 | other | — | — | — |
| `design.description` | 16 | 0% | **12%** | 1 | unsectioned | — | — | — |
| `tasks.response_mode` | 16 | 12% | **12%** | 1 | methods | — | — | — |
| `model_estimations.model_settings` | 33 | 0% | **12%** | 1 | unsectioned | — | — | — |
| `model_estimations.terms.functional_form` | 42 | 93% | **12%** | 2 | methods | — | — | — |
| `groups.age_minimum` | 9 | 100% | **11%** | 15 | unsectioned | — | 78% | — |
| `groups.age_maximum` | 9 | 100% | **11%** | 2 | methods | — | 44% | — |
| `inference_settings.inference_level` | 30 | 100% | **10%** | 16 | methods | — | — | — |
| `groups.sex_distribution.count` | 20 | 100% | **10%** | 6 | methods | — | — | — |
| `analyses.effect.cells.direction` | 162 | 80% | **10%** | 8 | methods | 55/55 vs gold | — | — |
| `analyses.details.connectivity_method` | 31 | 6% | **10%** | 1 | unsectioned | — | — | — |
| `regions.definition_method` | 33 | 9% | **9%** | 1 | methods | — | — | — |
| `analyses.spatial_scope` | 102 | 31% | **9%** | 2 | intro | — | 22% | — |
| `model_estimations.terms.type` | 86 | 16% | **8%** | 2 | methods | — | 15% | — |
| `acquisitions.acquisition_voxel_size_mm` | 51 | 90% | **8%** | 6 | methods | — | — | — |
| `design.allocation` | 16 | 25% | **6%** | 5 | methods | — | — | — |
| `design.assignment_structure` | 16 | 6% | **6%** | 1 | methods | 100% | — | — |
| `inference_settings.voxelwise_threshold_value` | 16 | 100% | **6%** | 40 | methods | — | — | — |
| `regions.region_type` | 33 | 9% | **6%** | 1 | methods | — | — | — |
| `tasks.description` | 17 | 0% | **6%** | 1 | unsectioned | — | — | — |
| `tasks.design_type` | 17 | 29% | **6%** | 8 | methods | — | — | — |
| `preprocessings.smoothing_fwhm_mm` | 19 | 100% | **5%** | 18 | methods | — | 16% | — |
| `acquisitions.magnetic_field_strength_tesla` | 23 | 100% | **4%** | 18 | methods | 100% | — | — |
| `groups.acquired_count` | 31 | 100% | **3%** | 9 | methods | — | 19% | — |
| `analyses.groups.n` | 168 | 98% | **2%** | 9 | methods | — | 26% | — |
| `model_estimations.terms.variation_level` | 85 | 5% | **1%** | 4 | methods | — | — | — |
| `analyses.effect.statistic.family` | 100 | 14% | **0%** | 16 | methods | 93.7% | — | — |
| `analyses.coordinate_space` | 89 | 100% | **0%** | 16 | methods | — | — | — |
| `measures.family` | 38 | 24% | **0%** | 11 | intro | — | — | — |
| `measures.type` | 38 | 0% | **0%** | — | — | — | — | — |
| `groups.is_healthy` | 35 | 0% | **0%** | — | — | — | — | — |
| `analyses.effect.statistic.degrees_of_freedom_denominator` | 33 | 100% | **0%** | 6 | methods | — | 27% | — |
| `analyses.details.inference_target` | 31 | 3% | **0%** | 31 | unsectioned | — | — | — |
| `acquisitions.modality` | 28 | 61% | **0%** | 19 | unsectioned | — | — | — |
| `design.timepoints.relation_to_intervention` | 26 | 0% | **0%** | — | — | — | — | — |
| `groups.enrolled_count` | 25 | 100% | **0%** | 9 | methods | — | — | — |
| `design.timepoints.order` | 22 | 100% | **0%** | 24 | intro | — | — | — |
| `analyses.details.parameter_change` | 19 | 16% | **0%** | 5 | intro | — | — | — |
| `inference_settings.voxelwise_threshold_type` | 16 | 0% | **0%** | — | — | — | — | — |
| `design.arms.arm_kind` | 16 | 50% | **0%** | 22 | unsectioned | — | — | — |
| `analyses.details.edges.directionality` | 15 | 0% | **0%** | — | — | — | — | — |
| `design.blinding` | 13 | 0% | **0%** | — | — | 100% | — | — |
| `inference_settings.alpha_level` | 13 | 100% | **0%** | 21 | methods | — | — | — |
| `analyses.details.parameter_sign` | 13 | 31% | **0%** | 6 | results | — | — | — |
| `groups.excluded_count` | 12 | 100% | **0%** | 29 | methods | — | — | — |
| `model_estimations.terms.levels.order` | 11 | 100% | **0%** | 24 | methods | — | — | — |
| `inference_settings.clusterwise_threshold_value` | 11 | 100% | **0%** | 14 | methods | — | — | — |
| `groups.sex_distribution.denominator` | 11 | 100% | **0%** | 15 | methods | — | — | — |
| `analyses.model_representation_notes` | 7 | 0% | **0%** | — | — | — | — | — |
| `acquisitions.reconstructed_voxel_size_mm` | 6 | 100% | **0%** | 51 | methods | — | — | — |
| `inference_settings.cluster_forming_threshold_value` | 5 | 100% | **0%** | 37 | intro | — | — | — |
| `analyses.details.performance_metrics.name` | 5 | 100% | **0%** | 10 | unsectioned | — | — | — |
| `analyses.details.performance_metrics.reference_value` | 5 | 100% | **0%** | 18 | methods | — | — | — |
| `analyses.details.performance_metrics.relation` | 5 | 0% | **0%** | — | — | — | — | — |
| `analyses.effect.statistic.degrees_of_freedom_numerator` | 4 | 100% | **0%** | 35 | intro | — | — | — |
| `inference_settings.tfce_used` | 4 | 0% | **0%** | — | — | — | — | — |
| `tables.non_analysis_content` | 3 | 0% | **0%** | — | — | — | — | — |
| `groups.handedness_distribution.count` | 2 | 100% | **0%** | 3 | methods | — | — | — |
| `acquisitions.tracer_name` | 2 | 100% | **0%** | 10 | results | — | — | — |
| `acquisitions.scan_type` | 2 | 100% | **0%** | 7 | methods | — | — | — |
| `acquisitions.tracer_radionuclide` | 2 | 100% | **0%** | 15 | results | — | — | — |
| `inference_settings.permutation_count` | 2 | 100% | **0%** | 11 | other | — | — | — |
| `groups.socioeconomic_status_summary` | 2 | 0% | **0%** | — | — | — | — | — |
| `groups.approached_count` | 1 | 100% | **0%** | 5 | intro | — | — | — |
| `acquisitions.sampling_frequency_hz` | 1 | 100% | **0%** | 15 | methods | — | — | — |
| `acquisitions.recording_type` | 1 | 100% | **0%** | 2 | methods | — | — | — |
| `acquisitions.eeg_channel_count` | 1 | 100% | **0%** | 4 | methods | — | — | — |
| `groups.handedness_distribution.percentage` | 1 | 100% | **0%** | 3 | methods | — | — | — |
| `groups.race_distribution.count` | 1 | 100% | **0%** | 8 | intro | — | — | — |
| `groups.race_distribution.denominator` | 1 | 100% | **0%** | 2 | tables | — | — | — |

## Methods tried, and what each was worth

**String match / normalised match.** The `surface` column. Cheap, and the only way to know
whether anything else can work. 67 fields are ≥90% locatable.

**Targeted regex.** Written for the paper-scoped candidates. Eight shipped at 93.7–100%
(`deterministic-fields.md`). Attempted and rejected on numeric unit fields —
`repetition_time_seconds` 68.8%, `echo_time_seconds` 68.8%, `smoothing_fwhm_mm` 66.7%,
`alpha_level` 46.2%, `multiple_comparison_method` 32.1% — every failure a scoping failure
rather than a pattern failure.

**Section scoping.** Measured above. Helps 7 fields materially, hurts the aggregate.

**spaCy (`en_core_web_sm` 3.8.16).** Run on the five entity-name fields, the only family
where span typing is the right shape. Recall of candidate spans: `assessments.name` 75%,
`devices.manufacturer` 73%, `measures.source_label` 54%, `regions.name` 50%, `tasks.name`
38%. But the labels are general-domain — 1,557 `ORG` and 2,524 `CARDINAL` spans across eight
papers — so choosing which `ORG` is the scanner manufacturer is the same scoping problem.
Candidate generation, not extraction. scispaCy was already evaluated in this repo for
sentence splitting and abbreviations and rejected (59/59 agreement, extra output noise); this
pass adds that its NER does not change the picture, because the picture is not about
recognition.

**Hugging Face.** Researched rather than run, because the benchmark numbers settle it. The
task the 78 low-`uniq` fields actually pose is document-level N-ary relation extraction, and
on scientific articles [SciREX](https://arxiv.org/abs/2005.00512) SOTA is **F1 16.9**
(ITERX), 3.55 for template generation, 0.096 for end-to-end binary relations. The
[OpenMed](https://huggingface.co/OpenMed) biomedical NER families —
`AnatomyDetect`, `DiseaseDetect`, `PathologyDetect`, zero-shot variants — address recognition,
which is not the bottleneck. The one field where a linker is the better tool is
`groups.medical_condition`: 34 values, 17 distinct, UMLS-linkable.

**Distant supervision for salience.** Tested, and it inverts. Using the records as labels for
"instrument mentioned but not used", the features come out backwards — unused mentions are
*more* frequent (2.00 vs 1.57) and *more* often in Methods (0.52 vs 0.35) — because the
negatives are dominated by extractor recall misses and class confusion (`apnea-hypopnea
index` is a `Measure`, not an `Assessment`). Only citation proximity pointed the right way
(0.50 vs 0.29). The clean label already exists in `structure.xml`
(`entities_row` → `keep`/`drop`, `entities_verdict` → `instance_spurious` vs
`instance_missing`, which separates non-salience from a recall miss); there are **4** answered
today out of **281** rows available across 144 unreviewed entity tasks.


---

# Models actually run

Researching benchmarks was not enough, so these were installed and run. The result is that
the `surface` column predicts model performance better than any model's own benchmark does.

## spaCy `en_core_web_sm` 3.8.16 — CPU

Candidate-span recall on the five entity-name fields: `assessments.name` 75%,
`devices.manufacturer` 73%, `measures.source_label` 54%, `regions.name` 50%, `tasks.name`
38%. Labels are general-domain, so it produced **1,557 `ORG`** and **2,524 `CARDINAL`** spans
across eight papers. Recall without precision: choosing which `ORG` is the scanner is the
scoping problem again.

## GLiNER `gliner_medium-v2.1` — CPU, ~15s/paper

Given a natural-language label instead of a fixed tag set, it is far better than spaCy at
both ends:

| field | n | recall | spans predicted | of those, in the record | spaCy recall |
|---|---:|---:|---:|---:|---:|
| `regions.name` | 36 | **94%** | 562 | 14% | 50% |
| `devices.manufacturer` | 11 | **82%** | 28 | 36% | 73% |
| `assessments.name` | 32 | 56% | 63 | 43% | 75% |
| `measures.source_label` | 24 | 38% | 38 | 26% | 54% |
| `tasks.name` | 8 | 12% | 15 | 7% | 38% |

`regions.name` at 94% nearly doubles spaCy, and `devices.manufacturer` yields 28 candidates
where spaCy gave 1,557. But 562 region spans against 36 `Region` entities is the point: it
finds every region the paper mentions, and the schema wants the ones that became entities.
That is the salience problem, and it is a label problem rather than a model problem — 281
reviewer dispositions are available, 4 are answered.

## GLiNER2 `fastino/gliner2-base-v1` — CPU, ~18s/paper

Its schema-driven hierarchical extraction is the feature that matters here, and its own paper
reports it **unevaluated** for want of a zero-shot benchmark. Measured against the record on
six paper-scoped fields, 12 papers:

| field | agree | disagree | abstain | precision |
|---|---:|---:|---:|---:|
| `scanner_field_strength_tesla` | 0 | 0 | 11 | — |
| `scanner_manufacturer` | 0 | 0 | 12 | — |
| `preprocessing_software` | 0 | 1 | 11 | 0% |
| `diagnostic_system` | 0 | 7 | 0 | 0% |
| `blinding` | 1 | 0 | 7 | 100% |
| `assignment_structure` | 0 | 4 | 8 | 0% |

It abstains on the fields a regex does at 100%, and where it answers it returns a
topically-adjacent span rather than the attribute: asked for `diagnostic_system` it returned
`DTI` and `Positron Emission Tomography`; asked for `assignment_structure`,
`fronto-striato-thalamo-cortical loop`. Zero-shot schema filling does not work on this
material.

## NuExtract-2.0-2B — one RTX 3070, **2-3s/paper**

A generative model, so a cheaper LLM rather than an alternative to one — but the fastest
thing tried by an order of magnitude, and the split in its output is the finding:

| field | result | this field's `surface` |
|---|---|---|
| `magnetic_field_strength_tesla` | **5/6** (`3T`, `1.5T`, `3.0`, `3`) | high |
| `devices.manufacturer` | **~5/6** (`General Electric Medical Systems`, `Philips`) | 100% |
| `preprocessings.software` | **6/6 plausible** (`SPM8`, `SPM12`, `DPARSF`) | high |
| `groups.diagnostic_system` | **0/6** — returned `fMRI`, `DTI`, `SVM`, `C-PiB PET` | low |
| `design.blinding` | **0/6** — `anonymized`, `clinical assessment` | 0% |
| `design.assignment_structure` | **0/6** — `10-fold cross-validation`, `event-related design` | 0% |

**It succeeds exactly where `surface` is high and fails exactly where the value is a schema
code the paper never writes.** `diagnostic_system` wants `DSM-IV`; the model returns the
imaging modality, because "diagnostic system" reads like one. GLiNER2 failed the same three
fields the same way. Two unrelated architectures, one boundary, and the audit predicted it.

## Ai2 / AllenAI

Checked on request. The relevant models split cleanly, and neither half changes the picture:

- **[SciBERT](https://github.com/allenai/scibert)** (1.14M Semantic Scholar papers, 3.1B
  tokens) and **Longformer-base-4096** — encoders. Longformer's window is the right answer to
  GLiNER2's 2048-token limit, and SciBERT is the right domain. But both require task-specific
  fine-tuning: they have no zero-shot extraction mode. They are the base for the distillation
  route, which is blocked on labelled data — the same 281-versus-4 constraint.
- **OLMo 3 7B Instruct**, **Asta** — an open LLM and an agentic assistant. Both are LLMs, so
  substituting them is a cost-and-openness decision, not a capability one. (`Olmo2Config` is
  supported by the transformers build on beast, so this is easy to try.)

## What this changes

1. **`surface` is the screening tool.** It predicted, before any model ran, which fields
   NuExtract and GLiNER2 would get and which they would fail. Use it to decide what to ask a
   model for at all.
2. **GLiNER is worth adopting for candidate generation** on `regions.name` (94%) and
   `devices.manufacturer` (82%), paired with a salience filter — not as an extractor.
3. **NuExtract-2.0-2B is worth a real trial as a first pass** on the high-`surface` fields at
   2-3s/paper, with the current pipeline handling the rest.
4. **Nothing tried addresses scoping**, and the retrieve-then-extract ceiling of 6% says
   nothing will until the entity-inventory labels exist.

## NuExtract3 — and the scoping result that overturns the conclusion above

Run on beast, 4B bf16 sharded across four RTX 3070s (it OOMs on one 8 GB card),
`enable_thinking=False`, ~6s/paper.

**On the paper-scoped fields it beats NuExtract-2.0 on exactly the fields the `surface`
column said were unreachable:**

| field | NuExtract3 | NuExtract-2.0 |
|---|---|---|
| `diagnostic_system` on `84rGLhCbUJTh` | **DSM-IV** | `DTI` |
| `diagnostic_system` on `DTpwdoGbjqsq` | **DSM-IV-TR** | `C-PiB PET` |
| `diagnostic_system` on `JzsUUQbDr2bm` | **DSM-IV-TR** | `SVM` |

Two things changed between the runs — the model *and* the template (a more explicit field
name, `enum` constraints on the closed fields) — so the gain is not attributable to the model
alone. Both changes are cheap and both should be kept.

### The scoping test

The paper-scoped fields do not test scoping. This does: a **nested** template, one row per
acquisition, each carrying its own parameters.

    {"acquisitions": [{"modality": "verbatim-string",
                       "repetition_time_ms": "number",
                       "echo_time_ms": "number"}]}

`DTpwdoGbjqsq` is the case to look at:

    {"acquisitions": [
      {"modality": "magnetic resonance imaging", "repetition_time_ms": 2530, "echo_time_ms": 3.42},
      {"modality": "C-PiB PET",       "repetition_time_ms": null, "echo_time_ms": null},
      {"modality": "C-PK11195 PET",   "repetition_time_ms": null, "echo_time_ms": null}]}

Three acquisitions, the TR on the MRI, and **null on both PET scans** — which is the
discrimination the whole audit said was out of reach.

Scored against the records over 6 papers: 9 predicted acquisitions against 11 recorded,
and on non-null TR predictions

> **6/7 = 86%.** The one miss is `7HPLh5nJzmP5` fNIRS, where the record has no TR at all,
> so there is nothing to disagree with.

Against everything else tried on the same field:

| method | result on `repetition_time_seconds` |
|---|---|
| paper-wide regex | 68.8% |
| Methods-scoped regex, unique-match | 33% |
| `uniq` ceiling for any surface method | 22% whole-document |
| **NuExtract3, nested template** | **86%** |

### What this changes

The claim earlier in this document — that nothing addresses scoping — **is wrong, and this is
the counter-example.** The reasoning above was that scoping is document-level relation
extraction, that SciREX SOTA is F1 16.9, and that a retrieve-then-extract architecture tops
out at 6% here. All of that is still true of *those* architectures. What it missed is that a
template with a repeated group changes the task: the model is not asked "which of six `df =
48` is the right one", it is asked to emit one row per acquisition and fill each row's
parameters, and the structure of the answer carries the scope. That is a different problem
from retrieving a passage, and a 4B model does it at 86% where a regex does 69% and a
retriever 6%.

The honest limit: 6 papers, 7 comparable TR values. This is a promising signal on a small
sample, not a validated result. The next step is the same nested-template test over the full
corpus and against the 101-cell direction gold, which is where it would have to hold up.

## Systematic NuExtract3 sweep — 10 entity classes, 40 fields

One nested template per entity class, neutral field names throughout, 6 papers, 60 calls,
~6s each. Scored by value-set overlap per leaf, because row alignment is itself uncertain.

**35 fields had a comparable value. Pooled precision 62%. 18 fields at ≥80%.**

### Where it works (≥80%)

| field | precision | field | precision |
|---|---:|---|---:|
| `groups.enrolled_count` | 100% | `groups.age_mean` | 100% |
| `preprocessings.smoothing_fwhm_mm` | 100% | `groups.age_standard_deviation` | 100% |
| `acquisitions.magnetic_field_strength_tesla` | 100% | `groups.recruitment_method` | 100% |
| `acquisitions.repetition_time_ms` | **100%** | `acquisitions.number_of_volumes` | 100% |
| `acquisitions.echo_time_ms` | **100%** | `measures.unit` | 100% |
| `inference_settings.cluster_extent_threshold` | 100% | `inference_settings.permutation_count` | 100% |
| `groups.diagnostic_system` | **100%** | `model_estimations.hrf_model` | 100% |
| `preprocessings.software` | 93% | `model_estimations.software` | 92% |
| `groups.acquired_count` | 86% | `inference_settings.multiple_comparison_method` | 80% |

### Where it fails (<40%)

| field | precision | why |
|---|---:|---|
| `assessments.description` | 0% | free text; a paraphrase has nothing to match |
| `tasks.description` | 0% | same |
| `model_estimations.estimator` | 10% | schema vocabulary, not paper wording |
| `tasks.name` | 20% | idiosyncratic naming (`n-back`, `Stroop`) |
| `groups.medication_status` | 25% | schema vocabulary |
| `assessments.name` | 33% | present but ambiguous — the salience problem |
| `groups.medical_condition` | 33% | ditto |
| `regions.name` | 35% | 18 of 27 predictions had no gold: over-generation |
| `acquisitions.modality` | 40% | returns the sequence, not the schema's modality code |

`regions.atlas` produced **30 predictions against 0 recorded values** — pure invention, and
the clearest single argument for keeping abstention rather than accepting model output.

### Two findings that change how to use it

**1. Template context beats field naming.** The earlier ablation showed
`groups.diagnostic_system` at 1/6 with a neutral field name asked alone. Here it is **100%**
with the same neutral name — because it is nested among `medical_condition`,
`medication_status` and `recruitment_method`, and the surrounding fields say what kind of
thing is wanted. That is a better lever than the answer-hinting field name, and it does not
leak.

**2. Richer rows improve scoping.** `repetition_time_ms` and `echo_time_ms` scored 83% in the
narrow 3-field acquisition template and **100%** here, where each row also carries `modality`,
`magnetic_field_strength_tesla`, `pulse_sequence_type`, `number_of_volumes` and
`acquisition_duration_seconds`. More per-row fields give the model more to anchor a row to,
so the scope comes out of the structure rather than out of retrieval.

## Table splitting — the hypothesis tested

Can NuExtract3 do stage 1's job, splitting a coordinate table into the analyses it reports?
15 tables with existing stage-1 parses as gold, table CSV plus caption and footer as input.

| behaviour | tables |
|---|---:|
| exact split — analysis count **and** partition | 8/15 |
| every point recovered but **under-split** | 4/15 |
| points lost | 2/15 |
| **total point count exactly right** | **12/15** |

`78Wr5Hy8Eft6::tbl1` returned 87 points in one analysis where gold has 20+13+26+28 = 87.
`6kSnm3c3Jb8j::t2` returned 9 where gold has 6+3. Three of `3RuCwAHCMqQC`'s tables split
exactly, partition included (19/30, 51/23, 16/11).

**Verdict: it does not replace luna here.** It recovers coordinates almost perfectly and
under-segments, and coordinate recovery is already deterministic from the CSV, so that half is
worth nothing. It finds a clean two-way split when the table has section structure and misses
finer ones — `3NHR2KXv5akM::table-1` went 15 → 7.

## Few-shot examples: allowed, and harmful for numbers

Examples are supported (`role: 'developer'` messages, output last), and held out
leave-one-paper-out they avoid same-paper leakage. They still hurt:

| variant | TR+TE both correct | precision |
|---|---:|---:|
| zero-shot | 5/6 | **83%** |
| 2-shot, leave-one-out | 5/7 | **71%** |

The mechanism is **value copying**. `6oTrCJA43Jcd` records TE = 15 and the 2-shot run
predicted TE = 30, the examples' value; `7HPLh5nJzmP5` records no TR or TE at all and the
2-shot run invented `fMRI 2000/30` while zero-shot correctly found fNIRS. An earlier score of
86% for the few-shot variant was an artefact of checking TR alone — scoring TR and TE together
reverses the ordering. **Use examples for shape, not for fields whose values are numbers.**

## The remaining classes — grid complete

Second sweep: `analyses`, `analyses.effect.cells`, `analyses.details`,
`model_estimations.terms`, `design`, `tables`. Six more templates, 38 more fields.
**Pooled 44%, 12 fields ≥80%, 15 fields <30%.**

| works (≥80%) | | fails (<30%) | |
|---|---:|---|---:|
| `tables.table_number` | 100% | `analyses.name` | 0% |
| `model_terms.type` | 100% | `analyses.definition` | 0% |
| `tables.footer` | 100% | `analyses.interpretations` | 0% |
| `model_terms.spatial_unit` | 100% | `model_terms.name` | 0% |
| `analyses.spatial_scope` | 100% | `design.blinding` | 0% |
| `analyses.statistic_family` | 100% | `design.relation_to_intervention` | 0% |
| `analyses.degrees_of_freedom` | 100% | `analysis_details.inference_target` | 0% |
| `analyses.coordinate_space` | 100% | `design.name` (arms) | 17% |
| `tables.caption` | 93% | `analysis_details.connectivity_method` | 20% |
| `model_terms.variation_level` | 80% | `model_terms.stage` / `model_family` | 33% |

The pattern is the same one the `surface` column predicts: enums with a small closed
vocabulary and values printed on the page do well; names, definitions and interpretations —
the paraphrase fields — score zero.

### `effect.cells` — the field that decides everything

Scored against the reviewer direction gold rather than against the record:

| | |
|---|---:|
| cells in the record | 71 |
| cells NuExtract3 produced | 38 |
| of those, unsigned | 4 |
| no matching gold cell | 15 |
| scored signed-vs-signed | 19 |
| **correct** | **7** |
| flipped | 12 |

> **37%.** Against `gpt-5.6-luna` at **96.6%** and a human ceiling of **95.8%**.

The failure is structural rather than a sign error. On `JzsUUQbDr2bm` it emits
`term: "FESZ", level: "NC"` — putting the *two sides of the comparison* into the term and
level slots, where the schema wants `term` to be the factor (`term_group`) and `level` to be
which side this cell is. Every such cell then reads `negative` where gold says `positive`.
It has not understood what a `Cell` is, and no threshold or template tweak fixes that.

It also emits 38 cells where the record has 71: it under-generates by nearly half, the same
under-segmentation seen in the table-splitting test.

**Conclusion: NuExtract3 cannot do the contrast layer.** That is the one part of this schema
where being wrong is unrecoverable, and it is 60 points behind the current pipeline.

## Final grid

| method | fields tested | headline |
|---|---:|---|
| string / normalised match (`surface`) | **158** | 67 fields ≥90% locatable; 36 model-only |
| ambiguity + section scoping | **158** | 20 fields `uniq` ≥80%; 78 below 20% |
| targeted regex | 20 | 8 shipped at 93.7–100% |
| spaCy `en_core_web_sm` | 5 | 38–75% recall, 1,557 `ORG` spans — unusable precision |
| GLiNER `gliner_medium-v2.1` | 5 | `regions.name` 94%, `devices.manufacturer` 82% recall |
| GLiNER2 `gliner2-base-v1` | 6 | 0% on 4 of 6; abstains where a regex is 100% |
| NuExtract3 | **78** | 30 fields ≥80%; **37% on `Cell.direction`** |
| NuExtract3 table splitting | 15 tables | 12/15 exact point totals, 8/15 exact splits |

### The division of labour this implies

1. **Deterministic first** — the 8 shipped derivers, at 93.7–100%, on fields that are
   paper-invariant and closed-vocabulary.
2. **NuExtract3 second**, ~6s/paper, on the ~30 fields it clears 80% on: counts, ages,
   durations, field strength, TR/TE, software, thresholds, table metadata, `spatial_scope`,
   `statistic_family`, `variation_level`. Neutral field names, rich nested rows, **no
   numeric few-shot examples**.
3. **GLiNER for candidate terms** on `regions.name` and `devices.manufacturer`, as a
   shortlist for a reviewer or for luna — never as an answer.
4. **luna keeps everything else**, and in particular keeps the whole contrast layer:
   `effect.cells`, `term`, `level`, `direction`, and the analysis names and definitions.
