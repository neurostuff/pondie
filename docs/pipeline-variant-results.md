# Baseline and two variants, measured

Three configurations over the sixteen reviewer-gold papers
(`data/gold-direction-16.pmids`), scored on contrast polarity by `score_direction.py` and
priced from each stage's own token report. Run on beast in a uv-managed environment, four
papers in parallel per arm, one GPU per shard for the evidence union.

All three arms were **rebuilt with the same builder** before scoring. They were extracted
at different times and `build_record` changed underneath them, so comparing them as-built
would have confounded a pipeline difference with a repair that landed between two runs.

## Accuracy

| arm | polarity accuracy | n cells | 95% CI | sign flips | analyses that failed to align |
|---|---|---|---|---|---|
| baseline | **96.2%** (75/78) | 78 | 91.0–100% | 3 | 4 |
| `--table-rows` | 94.8% (55/58) | 58 | 87.9–100% | 3 | **28** |
| `+recheck` | **97.3%** (72/74) | 74 | 93.2–100% | 2 | 5 |

The human ceiling on this measure is 95.8% and a coin flip is 50%. All three arms sit at
or above the ceiling, and **the intervals overlap almost completely** — the arms differ by
three errors, two errors and three errors out of seventy-odd cells. Nothing here is a
distinguishable difference in accuracy, and reporting one would be reading noise.

What *is* distinguishable is coverage and behaviour:

- **`--table-rows` loses a fifth of the cells.** Its unaligned-analysis count goes from 4
  to 28 and the scored cell count falls from 78 to 58. Handing the analyses pass the
  per-analysis row detail made it restructure the analyses -- more of them, cut
  differently -- so they no longer correspond to the reference record's. A contrast the
  scorer cannot align is a contrast whose polarity is never checked, which is why its
  headline can look respectable while the arm is plainly worse.
- **`+recheck` signs more cells, in both directions.** It introduces 5 sign inventions
  where the gold is unsigned and 4 sign losses, against 0 and 0 for the baseline. Its
  headline gain comes with a willingness to commit that the headline does not price.

## Cost

Tokens per paper, from each stage's own report.

| arm | in/paper | out/paper | vs baseline (in) |
|---|---|---|---|
| baseline | 152,962 | 14,478 | — |
| `--table-rows` | 147,785 | 12,651 | −3% |
| `+recheck` | **222,356** | 15,052 | **+45%** |

By stage:

| stage | baseline | `--table-rows` | `+recheck` |
|---|---|---|---|
| demands | 39,455 → 3,732 | 40,215 → 2,754 | 39,462 → 3,151 |
| satisfy | 44,929 → 3,981 | 44,899 → 4,035 | 44,937 → 4,120 |
| recheck | — | — | **68,932 → 1,202** |
| evidence | 68,578 → 6,765 | 62,672 → 5,862 | 69,026 → 6,579 |

The recheck stage reported nothing at all until this run: it makes **one call per
analysis, each carrying the whole paper**, so its cost scales with the analysis count and
one paper with eighteen analyses spent 218,603 input tokens on that stage alone. A
workflow that included it looked free because the pass printed a line about how many
analyses it rewrote and never a token count. It is instrumented now.

## What that costs

`gpt-5.6-luna` is **$0.20 per 1M input, $0.02 per 1M cached input, $1.20 per 1M output**
([OpenAI model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna)).
Reasoning tokens are billed as output and are already inside the `completion_tokens` these
numbers come from, so `--effort low` shows up as a smaller output count rather than a
different rate.

| arm | $ input | $ output | **$ / paper** | $ / 100 papers |
|---|---|---|---|---|
| baseline | 0.0306 | 0.0174 | **0.0480** | 4.80 |
| `--table-rows` | 0.0296 | 0.0152 | **0.0447** | 4.47 |
| `+recheck` | 0.0445 | 0.0181 | **0.0625** | 6.25 |

Input is 64% of the baseline's cost and 71% of `+recheck`'s, which is where any saving has
to come from -- the 11.5:1 input-to-output ratio means output is close to free.

**The prompt cache is off**, and it is the largest lever available. Every call reports
`cache_status: DISABLED` and `cached_tokens: 0` while the stages send a near-identical
prefix. Cached input is a tenth of the price: a fully-cached baseline paper would cost
$0.0205 rather than $0.0480. That is an upper bound -- only the repeated prefix is
cacheable, not the paper text that differs per call -- but even a partial hit dominates
every difference between these three arms. It is gateway configuration, not code.

## The evidence union

Free on the LLM axis by construction -- the retriever is a local cross-encoder and adds no
call.

| arm | spans retrieved / paper | on fields the quote pass left unsupported |
|---|---|---|
| baseline | 36 | 7 |
| `--table-rows` | 32 | 7 |
| `+recheck` | 35 | 5 |

Roughly seven fields per paper get evidence they would otherwise not have, at no marginal
cost. That is the union's whole contribution and it is stable across arms.

## Two rules that did not fire

`fill_directions` filled **zero** cells across all sixteen papers, and the reason is worth
recording rather than treating as a bug. It only fills a cell the model left `absent`, and
this pipeline leaves almost none: 10 unsigned cells out of 164, of which exactly 1 sits
under an analysis whose name carries a comparison operator. The rule was measured at 98%
on 17% of the *reviewer gold*, whose records came from an earlier pipeline that gave up on
directions far more often. The gap it was built to fill has largely closed, and it now
costs nothing and earns nothing.

`mirror_withheld` also fired zero times, for a different and fixable reason: `run_extraction`
does not re-run stage 1, so the stage-1 files on disk were partitioned by the old rule and
carry no `withhold` entries. Measuring the mirror needs `parse_tables --resplit` over the
corpus first.

## Build defects

`build_record` writes the record and *then* reports a defect, so a "failure" is never a
skipped paper -- all 16 records exist in every arm and all were scored. The reference
repairs added during this work took the baseline from 5 defective records to 2.

| arm | records with a defect | repairs applied |
|---|---|---|
| baseline | 2 | 4 references repointed, 2 term ids scoped, 45 scalars, 12 wrappers, 5 listified |
| `--table-rows` | 5 | 10 references, 1 term id, 19 scalars |
| `+recheck` | 1 | 3 term ids, 39 scalars, 2 wrappers, 1 listified |

The two survivors are the cases where repair would mean guessing: a reference to
`g_all_patients` with several groups declared, and one to `mt_pib` from an analysis whose
model does not declare it.

## What to conclude

**Keep the baseline.** Neither variant earns its change.

`--table-rows` is the clearer verdict: it is marginally cheaper and it costs a fifth of
the analyses their alignment, which is a worse failure than a wrong polarity because an
unaligned analysis is silently unscored.

`+recheck` is the more tempting one and should still be declined on this evidence. A
1.1-point gain on 74 cells is one fewer error, well inside the interval, and it is bought
with 45% more input tokens and a measurable increase in signing cells the gold leaves
unsigned. If it is worth revisiting, it should be on replicates rather than one run --
this pipeline's run-to-run spread has been larger than the gap being argued about.

The two changes that did pay were not variants at all: the evidence union, which rescues
~7 fields per paper for nothing, and the reference repairs, which more than halved the
defective records.
