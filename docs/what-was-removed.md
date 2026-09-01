# Five extraction pipelines, and the four that are gone

This repository carried five generations of extraction pipeline at once. Four of them had
been superseded and were kept running only by their own tests. This file is what they were
and where each one's findings live, so none of them is re-proposed as new.

The surviving pipeline is `pondie.extraction`, driven by `pondie extract`. It is described
in [pipeline-architecture.md](pipeline-architecture.md).

## The generations

| | what it was | how it ran | removed in |
|---|---|---|---|
| 1 | `information_extraction/` + `prompts/*.json` | LangExtract-style prompt-per-entity | `dcea9ea` |
| 2 | `scripts/extract_entities_gpt5_nano.py` (2,183 lines) | one script, entity discovery then confirmation | `dcea9ea` |
| 3 | `passes/run_extraction.py` | a script shelling out to one script per pass | this pass |
| 4 | `passes/pipeline/` | stages as objects, still `subprocess` underneath | this pass |
| 5 | `pondie/extraction/` | stages as functions, pydantic contracts, in process | **current** |

Generations 1 and 2 were removed before this pass and are recoverable from history.
Generations 3 and 4 are recoverable from the commit before this one.

**Why 5 and not 4.** A subprocess stage cannot be called from a test, composed, or summed:
its cost has to be scraped back out of its own logging, and its inputs are checked by
whichever script happens to read them. Generation 5 makes a stage a function of
`(paper, settings, caller) -> StageOutcome`, so the model is substituted in a test the same
way everywhere and a run total is one addition.

**What generation 5 was missing.** It was the designated pipeline but had never been
finished, and each gap was a silent one — the run reported success:

- `Build` merged the payloads and stopped. No quote was resolved to an offset, no
  `source_text_hash` was written, and `local_id` — required on `Study` — was absent. It also
  wrapped the body in `{"study": ...}`, which no reader of a record expects.
- Stages wrote `payloads/<id>/<stage>/payload.json` while `merge_payloads` globs
  `payloads/<id>/*.json`, so the builder read **nothing**: records came out with metadata
  and an empty body.
- `Evidence` collected quotes into a file of its own instead of putting an `evidence` block
  on each field. `evidence` is REQUIRED on `ExtractedValue`, so every record would have
  failed validation at the end of the run rather than at the stage.
- `Tables` *read* the `table-map.json` that the stage is supposed to *write*, and emitted
  bare scalars into `ExtractedValue` slots. Dropping this stage is the regression that
  motivated writing it down: 155 of 156 records with no tables declared while 1,076 of
  1,084 analyses referenced one.
- `SignSplit` was absent, so a table reporting both signs stayed one merged contrast.

All five are closed, and each has a test in `tests/test_stages.py` named for the failure.

## Workflows that were measured and dropped

`entity-first` and `demand-driven+recheck` were named orderings in generation 3.

`entity-first` guessed the entity inventory and then linked analyses to it.
`demand-driven` reverses that: the analyses declare the terms they need before any entity
exists. Measured in [extraction-workflow-experiments.md](extraction-workflow-experiments.md);
the finding is that a cell cannot be righter than the term it points at — asked to guess
first, the entity pass modelled a crossover's condition as a continuous covariate. Together
with the precision cascade this took direction F1 from 38.1% to 80%.

`Workflow.entity_first` is still in the enum and raises when selected, so a run recorded as
`entity_first` cannot silently mean the other thing.

`demand-driven+recheck` added a pass (`recheck_cells.py`) that re-asked the sign of each
contrast on its own. It is not in the current ordering, and the pass is removed with it.

## Removed with their findings already written up

Each of these measured an alternative that was rejected. The code is gone; the measurement
is in the document beside it.

| removed | what it measured | where the finding is |
|---|---|---|
| `benchmark/sweep_extractions.py` | pipeline variants, arm by arm | [pipeline-variant-results.md](pipeline-variant-results.md), [pipeline-variants-v13-v17.md](pipeline-variants-v13-v17.md) |
| `benchmark/eval_pipelines.py` | scoring those arms against gold | [extraction-workflow-experiments.md](extraction-workflow-experiments.md) |
| `benchmark/exp_shortlist.py` | evidence by retrieved shortlist | [evidence-union-design.md](evidence-union-design.md) — cost 21 points |
| `benchmark/build_evidence_gold.py` | assembling evidence gold from the review layer | [evidence-top1-judgements.md](evidence-top1-judgements.md) holds the 70 judgements it produced |
| `benchmark/compare_agreement.py` | reviewer-versus-reviewer agreement | [contrast-direction-rubric.md](contrast-direction-rubric.md), "The ceiling, measured" |
| `benchmark/test_derive_direction.py` | rule coverage against direction gold | [deterministic-direction.md](deterministic-direction.md) |
| `benchmark/test_direction_from_sign.py` | direction from the statistic's sign | [deterministic-direction.md](deterministic-direction.md) |
| `passes/check_against_spacy.py` | our abbreviation miner vs scispacy's | [text-preprocessing-experiments.md](text-preprocessing-experiments.md) |
| `passes/normalize_conditions.py` | conditions against MONDO/ONVOC | [normalizing-with-onvoc.md](normalizing-with-onvoc.md) |
| `passes/normalize_tasks.py` | tasks against the Cognitive Atlas | [normalizing-with-onvoc.md](normalizing-with-onvoc.md) |
| `passes/normalize_space.py` | coordinate space across a corpus | [normalizing-across-papers.md](normalizing-across-papers.md) |
| `passes/audit_field_extraction.py` | per-field surface locatability | [field-extraction-audit.md](field-extraction-audit.md) |
| `passes/audit_queryability.py` | how much of a record a query can reach | [field-extraction-audit.md](field-extraction-audit.md) |
| `passes/recheck_cells.py` | re-asking each contrast's sign alone | [extraction-workflow-experiments.md](extraction-workflow-experiments.md) |
| `passes/repair_model_graph.py` | model-assisted repair of 53 references | [interaction-simple-effects.md](interaction-simple-effects.md) |

The three normalization probes are superseded rather than merely rejected: `pondie.normalization`
is the productionised form, one module per field, reached by `pondie normalize <field>`.
[normalization-pipelines.md](normalization-pipelines.md) is why a field's shape and not the
corpus decides the method.

## Exact duplicates

`benchmark/compare_extractions.py` and `benchmark/compare.py` were 1,968-line files
differing by a single comment line. `benchmark/score_direction.py` and
`benchmark/direction.py` were the same module twice, one of them with a broken `ROOT`.
`compare.py` and `direction.py` survive — they are what `pondie benchmark` runs — and took
the two corrections the copies carried.

## The benchmark, collapsed

Five files became two. `run.py` reported only the direction number while `compare.py` computed
per-field precision, recall and F1 that nothing surfaced; `direction.py` reached into
`compare.py` for eleven symbols including three private ones, which is not two modules.
`build_evidence_gold.py` had no caller, no committed output, and read a Label Studio instance
that is not part of this repository.

| now | what it holds |
|---|---|
| `benchmark/__init__.py` | `run()`, the result models, the report. What you call and what you get |
| `benchmark/scoring.py` | everything that computes a number |

## The extraction directory, regrouped

`passes/` was a flat catch-all: corpus sync sat beside record validation, and a one-off audit
beside the builder. It is now five subpackages named for the stage of the journey a paper is
at, so where a module lives says when it runs.

Duplication removed rather than moved:

| was | now | why it mattered |
|---|---|---|
| 5 `ExtractedValue` unwrappers | `extraction/values.py` | they disagreed. `derive_fields` returned `None` for any mapping without a `value` key including non-wrappers; `validate_record` returned `None` for anything that was not a mapping, so a bare scalar that escaped repair read as a field the paper never mentioned; two others passed non-wrappers through. Three answers to one question |
| 3 `load_key_file` + `llm.load_env` | `llm.load_env` | and they disagreed too: `select_papers` overwrote an env var already set, the others deferred to it |
| 2 `same_level` | `record/direction.same_level` | same rule, same stated rationale, but only one handled stopwords -- so `ASD` reached `ASD group` in one module and not the other |
| 3 `main()` CLIs | `pondie extract --stages ...` | they existed because the old pipeline shelled out to them |

**One bug the naming found.** `build_prompt` returns `(system, user)`. `stages` unpacked it as
`(prompt, schema_name)`, and `GatewayCaller` sends `call.prompt` and ignores `schema_name` --
so the model received the instruction header and **never the paper**. Every stage reported
success. The tests substitute a `Caller` that records what it was asked without reading it,
so nothing caught it. `Prompt` is a named pair now and `ModelCall` carries both halves;
`tests/test_stages.py::test_the_paper_reaches_the_model` asserts the consequence.
