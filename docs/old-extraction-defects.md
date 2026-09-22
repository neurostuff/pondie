# Picking 100 papers the old extraction got wrong

`data/identifiers.json` names 100 studies to re-parse and re-extract. This file says how
they were chosen and what each flag means, so a reviewer reading a row knows what to look
for in the paper.

## The corpus scanned

`pipeline_outputs/{participant_demographics,task}/*/1.1.0/da73c01b87bf` on beast: 39,321
studies, one `results.json` each per axis, extracted June 2025 against a schema that has
since been replaced. The old run read `processed/ace/text.txt` for 24,432 studies and
`processed/pubget/text.txt` for 12,271; it never read elsevier.

## What was measured

Descriptive statistics over the fields a meta-analysis cannot run without — group size,
sex split, age, diagnosis, task name, modality — then the tails of each distribution:

| field | distribution | the tail that is wrong |
|---|---|---|
| `count` | median 22, p99 1,000 | 420 groups ≤ 0; 48 above 10,000 |
| `age_mean` | median 27, p99 75.4 | 63 above 120, to a maximum of 807,369 |
| `male_count` + `female_count` | — | 1,748 groups do not sum to `count` |
| `diagnosis` | 14,152 distinct, median 25 ch | 103 above 200 ch: criteria paragraphs, not diagnoses |
| `TaskName` | 21,330 distinct, median 25 ch | `404 Not found` (43), `N/A`, `null`, DTI, `Bombyx mori` |
| `Modality` | `fMRI-BOLD` 30,742 | 31 studies use a value outside the enum (`fMRI`, `MRI`, `ERP`) |

Two findings shaped everything after:

**The schema leaked.** 4,688 studies carry `RestingState`, `TaskDesign`, `TaskDuration` or
`RestingStateMetadata` at the top level of the task record, where they belong inside one
task. A handful of records invent group keys outright — `BMI`, `HOMA-IR`, `age_count`, and
one literal `"male_count:null,"`.

**26.4% of extracted diagnoses state an absence** — 9,551 of 36,154, by
`vocabularies.phrases.triage`. The scan reuses that function rather than re-deriving the
judgement, so a value this repo already calls `NO_CONDITION` is not counted as a disease
here either; [condition-normalization.md](condition-normalization.md) is where the rule
comes from and reports 22% on the smaller corpus it was measured against.

## Defects the 100 were drawn for

Tier A is wrong on its face. Tier B is plausible and still wrong — the class that needs a
human to look at the paper.

| tier | defect | n | what the value looks like |
|---|---|---|---|
| A | `task_is_modality` | 10 | `Diffusion Tensor Imaging (DTI)` as the task name |
| A | `dx_prose_dump` | 9 | 200+ characters of inclusion criteria in `diagnosis` |
| A | `dx_carries_counts` | 8 | `Bipolar Disorder (bipolar I [n = 37]; Bipolar II [n = 10])` |
| A | `dx_healthy_patient_group` | 8 | `group_name: patients` beside `diagnosis: Healthy` |
| A | `age_impossible` | 6 | `age_mean` 298.8; `age_minimum` above `age_maximum` |
| A | `n_impossible` | 5 | `count: 0`, or an fMRI group of 5,351 |
| A | `placeholder_value` | 4 | `Not Applicable`, `N/A`, `Unspecified` written into a slot |
| A | `offschema_value` | 3 | a group key or modality the schema does not define |
| B | `dx_criteria_prose` | 10 | `Healthy Adults With No History Of Neurological Illness` |
| B | `dx_condition_in_healthy` | 8 | a healthy group whose `diagnosis` names something |
| B | `dx_not_a_diagnosis` | 8 | `Right-handed Healthy Older Adults`, `First-time Fathers-to-be` |
| B | `task_name_is_prose` | 6 | a sentence or a title fragment where a task name goes |
| B | `resting_flag_vs_name` | 5 | `RestingState: true` on a named task, or false on `Rest` |
| B | `sex_sum_mismatch` | 5 | `20 + 18 != 36` |
| B | `task_is_questionnaire` | 5 | `Balanced Inventory of Desirable Responding` as an fMRI task |

72 of the 100 carry more than one flag (2.1 on average); `defect` is the one they were
drawn for and `all_flags` is everything the scan found on them.

## What is deliberately excluded

**Papers the model never saw.** 78 studies were extracted from a render under 5,000
characters while a full one sat beside it — 51 of them from a 203-character `404 Not
found` page rendered by ace, with a complete pubget text available. Their output is
garbage, but it says nothing about the schema or the extractor: the model had no paper.
Every study in the set was extracted from at least 17,727 characters (median 57,676), and
the render `paths.best_text` will hand the next run is a whole paper too — 17,679 to
152,476 characters, median 36,801.

That exclusion is the reason the set is not simply the worst 100 rows by flag count. The
loudest failures in this corpus are source failures, and re-running them would only
re-measure the corpus.

**Papers the pipeline cannot parse.** All 100 ship table sources and an `article.xml` or
`content.xml`, so stage 1 has tables to parse and `build_text` has an input. Group sizes
and ages live in tables; a study synced without them cannot be scored on the fields it was
selected for.

## Reproducing

The scan is not in the package — it reads a directory tree that only exists on beast, and
it answers a question asked once. `data/selection/old-extraction-defects.pmids` is the
sync input:

    python -m pondie.extraction.corpus.sync \
        --pmids data/selection/old-extraction-defects.pmids --host beast

Both that file and `data/identifiers.json` are generated, and both sit under the
gitignored `data/`. Copy them somewhere tracked if this selection needs to be pinned.
