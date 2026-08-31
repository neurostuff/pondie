# pondie

Extract a queryable record of what a neuroimaging paper reports, normalize its wording onto
shared values, and select the subset a meta-analysis should pool.

Three packages, and the boundary between them is a contract rather than a convention.

```
pondie.extraction      papers -> validated records, one stage at a time
pondie.normalization   a record's own wording -> shared values, one module per field
pondie.query           records -> what a meta-analysis should pool, and what it dropped
```

## The contracts

`pondie.contracts` holds the pydantic models that cross those boundaries: `Paper`,
`Settings`, `ModelCall`, `ModelReply`, `Cost`, `StageOutcome`, `RunReport`. Every one forbids
unknown fields, so a misspelled setting is an error rather than a setting that silently does
not apply.

The record's own shape is **not** among them. That is the LinkML schema in `study_schema/`,
which generates the extraction schema, validates records and answers whether a slot is
multivalued; restating it in pydantic would be a second source of truth that drifts. Records
are read through `schema_utils.value_of`, which takes the wrapper and the declared shape from
the schema.

## Extraction

```python
from pondie.contracts import Paper, Settings
from pondie.extraction import GatewayCaller, run

report = run(papers, settings, GatewayCaller())
print(report.summary())          # a report to assert on, not text to parse
```

Five stages, and the order is the design:

| stage | model | what it does |
|---|---|---|
| `tables` | no | copies the parse manifest. `caption` and `footer` are literal strings; a model can only introduce error |
| `demands` | yes | analyses first: each declares the entities it needs, before any exist |
| `satisfy` | yes | builds exactly those entities and nothing else |
| `evidence` | yes | a supporting quote for every value — **45% of input tokens** |
| `build` | no | merge, repair, resolve quotes to offsets, validate |

`demands` precedes `satisfy` because a cell cannot be righter than the term it points at:
asked to guess an inventory first, the entity pass modelled a crossover's condition as a
continuous covariate.

A stage is a function of `(paper, settings, caller)`, not a subprocess, so it can be called
from a test with a fake `Caller` and its cost is returned rather than scraped from logging.

## Normalization

A field's shape decides its method, and three shapes recur.

| shape | fields | method |
|---|---|---|
| closed target | `coordinate_space`, `multiple_comparison_method`, `correction_scope`, `medication_status`, `sex_distribution`, `handedness_distribution` | rules over a fixed answer set |
| link | `medical_condition` | retrieval against MONDO, then UMLS |
| cluster | `task` | the corpus clustered against itself |

Two conventions every module shares. **`UNKNOWN` is not `OTHER`**: `OTHER` asserts an answer
outside the known set, `UNKNOWN` asserts we cannot tell, and they license different downstream
actions. And **nothing is bucketed silently**: an input no rule matched is reported, so a new
surface form forces a rule instead of disappearing.

```bash
python -m pondie.normalization.medication_status     # the report for one field
```

## Query

```python
from pondie.query.engine import Selection, select

result = select(Selection(measure_type={"gray_matter_volume"}))
print(result.funnel())           # where papers were lost, not just how many survived
dataset = result.to_dataset()    # one experiment per study, ready for NiMARE
```

`Selection` defaults to what a coordinate meta-analysis usually wants — whole-brain analyses,
human, primary studies — and every default is overridable, because each is a judgement rather
than a fact. Excluding `roi` matters: a region-restricted search can only report coordinates
inside that region, and pooling it with whole-brain analyses inflates convergence exactly
where studies chose to look.

## Benchmark

```bash
pondie benchmark                    # 14 papers, polarity 94.5% on 55 signed cells
```

Three sets, and confusing two of them makes the number meaningless:

| | |
|---|---|
| `benchmarks/gold/` | the reviewer's answer for a cell — the only thing scored against |
| `benchmarks/reference/` | the records the reviewer was **shown**; supplies identity, which term a row is a row of |
| `benchmarks/candidate/` | the extraction being evaluated |

All three ship, so the benchmark runs from a clean clone with no corpus and no credentials,
and `tests/test_benchmark.py` holds a floor it must not silently drop below.

Only cells both sides signed are scored; missing terms are reported as coverage rather than
penalised, so the headline cannot be a lie by omission — 94.5% covers 54% of the 101
reviewed cells, and both numbers are printed.

**Read the headline against the right thing.** Two reviewers scoring the same 239 cells agree
**78.2%** read naively. The 95.8% sometimes quoted as a "human ceiling" is that figure weighed
by provenance tier, and the narrowest defensible number is 44 cells at 95.5% where both chose
a sign — none of which shares a denominator with a polarity score over this gold. What the
doubly-reviewed set does show is that of 52 disputed cells only **2** are `positive` vs
`negative`: humans agree about polarity and argue about membership.

## Install

```bash
git clone --recurse-submodules git@github.com:neurostuff/pondie.git
pip install -e ".[normalize,meta,dev]"
python -m spacy download en_core_web_sm     # negation scope, for medication_status
pytest
```

`study_schema` is a submodule. Extraction and the record reader both need it.
