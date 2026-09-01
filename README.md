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

`pondie.extraction.models` holds the pydantic models that cross those boundaries: `Paper`,
`Settings`, `ModelCall`, `ModelReply`, `Cost`, `StageOutcome`, `RunReport`. Every one forbids
unknown fields, so a misspelled setting is an error rather than a setting that silently does
not apply.

The record's own shape is **not** among them. That is the LinkML schema under
`study_schema/`, a submodule carrying YAML and prose and no code at all; restating it in
pydantic would be a second source of truth that drifts.

Every line that *reads* that schema is here, under `pondie.schema`: `utils` (classes, slots,
ranges, values), `generate` (the extraction schema, projected from the storage schema) and
`checks` (what has to hold for the two to agree). Records are read through
`pondie.schema.reader.value_of`, which takes the wrapper and the declared shape from the
schema.

## Extraction

```python
from pondie.extraction.models import Paper, Settings
from pondie.extraction import GatewayCaller, run

report = run(papers, settings, GatewayCaller())
print(report.summary())          # a report to assert on, not text to parse
```

Or from the command line, where a run is named and everything it produces lands in one
directory under `data/runs/`:

```bash
pondie extract --pmids papers.pmids --run v3 --model <model> --env .env
pondie extract --pmids papers.pmids --run v3 --model <model> --plan   # spend nothing
```

Six stages, and the order is the design:

| stage | model | what it does |
|---|---|---|
| `tables` | no | copies the parse manifest and mints the Table ids analyses reference. `caption` and `footer` are literal strings; a model can only introduce error |
| `split` | no | a parse reporting both signs is two contrasts; the half the paper never describes is withheld and rebuilt by arithmetic |
| `demands` | yes | analyses first: each declares the entities it needs, before any exist |
| `satisfy` | yes | builds exactly those entities and nothing else |
| `evidence` | yes | a supporting quote for every value — **45% of input tokens** |
| `build` | no | merge, repair, resolve quotes to offsets, write the record |

`demands` precedes `satisfy` because a cell cannot be righter than the term it points at:
asked to guess an inventory first, the entity pass modelled a crossover's condition as a
continuous covariate.

A stage is a function of `(paper, settings, caller)`, not a subprocess, so it can be called
from a test with a fake `Caller` and its cost is returned rather than scraped from logging.

The package is seven layers, listed in dependency order — which is also reading order,
because the import graph is acyclic and each layer only knows the ones above it:

```
pondie/paths.py        where the data lives
pondie/formats/        what a record and its source text are MADE OF: the value wrapper,
                       the text normalization offsets address, the table render, the
                       parse's address space. A second implementation of any is a bug
pondie/vocabularies/   fetched term lists: ONVOC, MONDO, mined abbreviations
pondie/schema/         the LinkML schema, read through LinkML
pondie/extraction/     papers -> validated records
pondie/normalization/  a record's own wording -> shared values
pondie/query/          records -> the subset a meta-analysis should pool
pondie/benchmark/      how much of a paper an extraction got right
```

Inside `extraction/`, the directory is the journey a paper takes, so where a thing lives
says when it happens:

```
pondie/extraction/
  corpus/     getting the paper on disk. An INPUT: a run reads it, never writes it
  prompt/     what the model is asked, and what the paper looks like when it is asked
  evidence/   which characters of the paper warrant each value
  record/     turning the payloads into a record: assemble, repair, check
  tools/      things done to records afterwards; none of them runs inside a pipeline
  models.py   the pydantic contracts that cross a boundary
  values.py   the ExtractedValue wrapper: what one is, how to read one, how to make one
  stages.py driver.py llm.py parse.py usage.py
```

Every boundary is a named type, and writing them down found two bugs that were invisible
without one. `build_prompt` returns two halves; a stage unpacked them as
`(prompt, schema_name)`, so the half carrying **the paper** went into a field the caller
ignores — the model was sent instructions about a paper it had never seen, and a fake
`Caller` that records what it is asked without reading it kept every test green. Five
modules had each written their own `ExtractedValue` unwrapper, disagreeing at the edges about
whether a non-wrapper is `None` or itself.

This is the fifth extraction pipeline this repository has had and the only one it still has.
[docs/what-was-removed.md](docs/what-was-removed.md) is the other four, what each measured,
and where each finding is written up — so none of them gets re-proposed as new.

## Where the data goes

`pondie/paths.py` is the only definition, and `PONDIE_DATA_DIR` moves the whole tree.

```
data/corpus/<id>/        the synced paper. An input: fetched, never written by a run
data/runs/<name>/        one extraction: payloads/, records/, usage.jsonl
data/vocab/              fetched vocabularies, shared by every run
data/selection/          which papers to run at all
benchmarks/              gold and reference records — tracked in git, unlike data/
```

A run is a directory rather than three trees keyed by study id, because the question asked
of these files is nearly always "what did this run produce" and the older shape could only
answer "what does this paper have, from whichever run wrote last".

## Normalization

A field's shape decides its method, and three shapes recur.

| shape | fields | method |
|---|---|---|
| closed target | `coordinate_space`, `multiple_comparison_method`, `correction_scope`, `medication_status`, `sex_distribution`, `handedness_distribution` | rules over a fixed answer set |
| link | `medical_condition` | retrieval against MONDO, then UMLS |
| cluster | `task` | the corpus clustered against itself |

Two conventions every module shares. **`UNKNOWN` is not `OTHER`**: `OTHER` asserts an answer
outside the known set, `UNKNOWN` asserts we cannot tell, and they license different downstream
actions. **Nothing is bucketed silently**: an input no rule matched is reported, so a new
surface form forces a rule instead of disappearing.

```bash
python -m pondie.normalization.medication_status     # the report for one field
```

## Query

```python
from pondie.query.engine import Selection, select

result = select(Selection(measure_type={"gray_matter_volume"}))
print(result.funnel())           # where papers were lost, not just how many survived
studyset = result.to_studyset()  # one analysis per study, ready for NiMARE
```

`Selection` defaults to what a coordinate meta-analysis usually wants — whole-brain analyses,
human, primary studies — and every default is overridable, because each is a judgement rather
than a fact. Excluding `roi` matters: a region-restricted search can only report coordinates
inside that region, and pooling it with whole-brain analyses inflates convergence exactly
where studies chose to look.

## Benchmark

Two numbers, and they answer different questions:

```bash
pondie benchmark                    # per-field P/R/F1 and direction accuracy
pondie benchmark --brief            # the headline only
pondie benchmark --limit 10         # the ten worst fields
```

```
14 paper(s) · polarity 94.5% on 55 weighted cell(s) · covering 54% of 101 reviewed
1 record(s) · 76 field(s) · macro-F1 89.9%

FIELDS (76 scored, worst first)
                                        P      R     F1     acc     n
  Analysis.effect.cells[].level       0.0%   0.0%   0.0%      --     8
  Analysis.prespecification           0.0%   0.0%   0.0%      --     4
  Analysis.effect.cells[].direction 100.0%  50.0%  66.7%  100.0%     8
  ...
```

**Precision, recall and F1 are about presence** — was the field filled where the gold fills
it, and only there. **Accuracy is about the value**, over the pairs where both sides filled
it. Keeping them apart is the point: a field filled everywhere it should be, with the wrong
value in it, scores F1 1.0 and accuracy 0.0, and conflating the two would hide the commoner
defect. Fields both sides leave empty everywhere are counted separately and not scored —
with tp=fp=fn=0 the F1 formula yields 0.0, which reads as total failure when the extractor
was in fact right.

Per field and not only per entity type, because "Analysis is 94% accurate" does not say which
of its thirty fields to go and fix. Worst F1 first, so the fixable thing is at the top.

**Direction is scored only on the cells that carry weight** — a cell on one side of the
contrast or the other. A `held` level is held from both sides and has no sign to get right.
Missing terms are reported as coverage rather than penalised, so the headline cannot be a lie
by omission: 94.5% covers 54% of the 101 reviewed cells, and both numbers print.

Three sets, and confusing two of them makes the number meaningless:

| | |
|---|---|
| `benchmarks/gold/` | the reviewer's answer — a whole record for the field scores, a per-cell direction table for polarity. The only thing scored against |
| `benchmarks/reference/` | the records the reviewer was **shown**; supplies identity, which term a row is a row of, and nothing else |
| `benchmarks/candidate/` | the extraction being evaluated |

All three ship, so the benchmark runs from a clean clone with no corpus and no credentials,
and `tests/test_benchmark.py` holds a floor it must not silently drop below.

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

534 of the tests run against nothing but the repository. The rest are gated on corpus data
and say so; to lift them:

```bash
pip install -e ".[pubget]"
git clone -b enh/keep_references_option https://github.com/jdkent/pubget .tmp_repos/pubget
python -m pondie.extraction.corpus.sync    --pmids <file> --host beast \
    --root /data/alejandro/projects/ns-pond/data
python -m pondie.extraction.corpus.rebuild --pmids <file> --pubget .tmp_repos/pubget
```

That turns on the fifteen tests over `corpus.rebuild` and the coordinate-table parser,
including the one that matters most: rebuilding a paper with `keep_tables=False` reproduces
the corpus text **byte for byte**. Every offset in every existing record rests on that, and
until the checkout is present nothing checks it.

`study_schema` is a submodule and is *data*: `pondie.schema` resolves it beside the package,
or at `PONDIE_SCHEMA_DIR` when that is set. It is not installed and nothing imports from it,
so a checkout without `git submodule update --init` fails at import with the command to fix
it rather than part-way through a run.
