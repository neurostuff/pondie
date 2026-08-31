# Evidence: two locators, unioned

The evidence stage answers one question per filled field -- *which characters of the paper
warrant this value*. There are two things that can answer it: the LLM quote pass
(`review/add_evidence.py`), which reads the paper and writes a quote, and the cross-encoder
retriever (`review/evidence_retrieval.py`), which ranks sentence units against a query built
from the field and its value.

**The decision is to run both and union their spans, not to pick one.** The LLM reads the
whole paper and the retriever contributes a second span when it is confident. This file
records what that rests on, and what had to be measured twice to get right.

## What the measurement was

173 field slots across 16 papers, each one a field where a human reviewer left evidence:
drew a span in the review layer, replaced one the extractor had produced, or wrote a
supporting quote into a correction file. Built by `build_evidence_gold.py` and resolved to
fields by `prepare_evidence_eval.py`.

The gold is incomplete on purpose and the scoring has to respect that. A reviewer
highlights *a* sentence that supports the value, never every sentence that would, so a
pick matching nothing human is **unknown**, never wrong. A pick is `wrong` only when it
lands on a span the reviewer deleted *and replaced with one elsewhere* -- a bare deletion
says nothing, for the reasons in `docs/evidence-unknown-judgements.md`.

| system | correct | wrong | unknown | no pick |
|---|---|---|---|---|
| LLM evidence pass | 39.9% | 2.9% | 45.7% | 11.6% |
| retriever top-1 | 42.2% | 0.6% | 57.2% | 0% |
| retriever top-12 | 69.9% | 0.6% | 29.5% | 0% |

`correct` is the headline and it is a **floor**, not an estimate: `unknown` is dominated by
picks that support the value but were not the sentence the reviewer marked. Hand-reading 42
of them put two thirds in that category. No precision figure is quoted, because with 8
usable negatives across the whole set there is no denominator to compute one over, and the
earlier attempt at one was measuring reviewer tidying rather than error.

Top-12 is not comparable to top-1; it is reported because it is what a shortlist would
hand downstream.

## Why union, and not either one alone

At top-1 the two systems are close -- 42.2% against 39.9% correct. Read as a ranking that
says almost nothing. Read as a joint distribution it says
everything:

| | slots |
|---|---|
| both correct | 46 |
| LLM correct, retriever missed | 23 |
| retriever correct, LLM missed | 16 |
| retriever correct, LLM declined to answer | 10 |
| retriever correct, LLM wrong | 1 |

**Union correct: 96 slots, 55.5%**, against 39.9% and 42.2% for either alone. The two fail
on different fields, and the overlap (46) is smaller than the disagreement (50).

The disagreement is not random. The retriever recovers fields the LLM skipped outright --
`inference_settings.clusterwise_threshold_value`, `coordinate_space = MNI`, group names --
which are exactly the low-salience settings a model summarising a paper does not think to
quote, and exactly the fields where a value has a literal surface form to match. The LLM
wins where the supporting sentence shares no vocabulary with the value, which is 57% of
gold evidence and the case retrieval is structurally worst at.

So replacing either with the other discards a quarter of the answers. Neither alternative
survives its own numbers:

- **Retriever only** loses the 23 slots where the supporting sentence and the value have no
  words in common, and it has no way to abstain -- it always returns a top-1, so its
  `unknown` share would silently become the record's evidence.
- **LLM only** loses 26 slots, 11 of them cases where it declined to answer at all. Its 8.8%
  `no pick` rate is the honest half of that; the other half is in `unknown`.

## Shape

The retriever runs first and locally, so it costs nothing per paper beyond GPU time:

1. **Locate.** For each filled field, build the lean query (`review/evidence_retrieval.py`
   `build_query`: field leaf, up to two aliases, up to two value surface forms -- no entity
   name, which measured worse in the query than out of it), score every sentence unit and
   sentence-ified table row, and add the section prior, literal-match bonus and
   entity-mention bonus.
2. **Shortlist.** Keep the top k units.
3. **Quote.** The LLM reads the **whole paper**, as it does today. Substituting the
   shortlist was the original proposal and it was measured and rejected -- see below.
4. **Union.** The LLM's span and the retriever's top-1 both go into `evidence.sets[]` when
   they do not overlap and the retriever clears its abstention gate. The schema already
   models this: a set is one independent supporting passage, and a fact split across two
   sentences is two sets, not a contest between them.

The retriever runs locally, so step 4 costs nothing per paper. It is the whole of what
this design buys.

## The shortlist does not work

Both arms run fresh against the same model over the same 173 slots, because the evidence
stored in the records was produced by an earlier run and resolved to offsets by exact
match, which silently drops every paraphrased quote. Using it as the LLM's baseline
understated that pass by 27 points, and every comparison in the first version of this file
was wrong for that reason.

| arm | correct | unknown | no pick | unlocatable | prompt tokens | completion |
|---|---|---|---|---|---|---|
| LLM, whole paper | **67.1%** | 26.6% | 4.0% | 2.3% | 143,224 | 8,640 |
| LLM, twelve retrieved sentences | 45.7% | 50.3% | 4.0% | 0% | 79,777 | 3,607 |
| retriever top-1 alone | 42.2% | — | — | — | 0 | 0 |

**The shortlist loses 21.4 points to save 45% of the prompt.** That is not a trade worth
making, and the reason is structural rather than fixable by a better prompt: the
retriever's recall at twelve is 69.9%, so a shortlist caps the model at 69.9% before it
reads a word, and the whole-paper arm already reaches 67.1%. Handing the model a shortlist
takes away exactly the sentences it is better than the retriever at finding.

## What the union is actually worth

| | correct |
|---|---|
| LLM, whole paper | 67.1% |
| retriever top-1 | 42.2% |
| **union** | **73.4%** |
| union, plus a shortlist pass as a third voter | 76.9% |

| | slots |
|---|---|
| both correct | 62 |
| LLM correct, retriever missed | 54 |
| retriever correct, LLM missed | 11 |
| neither | 46 |

**+6.3 points for no marginal cost.** This is a smaller claim than the first version of
this file made -- it read the overlap as smaller than the disagreement, on the strength of
a baseline that was measuring stale records. The LLM dominates; the retriever adds eleven
slots out of 173. It is still worth adding, because those eleven are free, but it is a
complement and not a peer.

Adding the shortlist pass as a *third* voter is worth another 3.5 points for 45% of a
prompt. Whether that is worth paying for is a cost decision, not an accuracy one.

## What this does not fix

Roughly 46% of the retriever's remaining failures are values the paper never words --
`undirected` inferred from a correlation, `whole_brain` from a voxelwise analysis with no
mask, `non_randomized` from the absence of a statement. See
`docs/evidence-top1-judgements.md`. Union does not help: neither locator can find a
sentence that does not exist. That needs a decision about what evidence for an inferred
term means, and `value_source: generated` is already the schema's word for it.

## Open, in the order worth answering

1. **What is in `unknown`?** Half of every column. If it is mostly correct-but-unhighlighted,
   42% is an artifact of an incomplete gold and the real number is far higher; if it is
   mostly genuine misses, 42% is close to a ceiling. Nothing else here can be sized until
   this is known, and it is answerable by hand on a sample.
2. ~~**Does the shortlist preserve the LLM's accuracy?**~~ **Answered: no.** See below.
3. ~~**When should the retriever abstain?**~~ **Answered: on margin, not on score.**
   See below.
4. **How long should the shortlist be?** k=12 is not a measured choice. Recall rises from 66%
   at 12 to ~75% at 25, against a longer prompt.

## Abstention

The retriever has no `no pick`, so unioning it in imports its errors along with its wins.
Two candidate signals, swept over 226 slots and scored on confirmed-correct only --
`docs/evidence-unknown-judgements.md` establishes that deletions are not reliable
negatives, so any precision built on `correct / (correct + wrong)` is measuring the wrong
thing.

| margin cut | coverage | confirmed correct of kept |
|---|---|---|
| 0.01 | 100% | 42.2% |
| 0.92 | 61% | 61.9% |
| 2.62 | 40% | 80.0% |
| 7.01 | 20% | 80.0% |

**Margin -- the gap between the top-scored unit and the runner-up -- is a clean
abstention signal.** Confirmed-correct rises monotonically as coverage falls, and since
confirmed-correct is itself a floor, a margin cut near 2.6 keeps 40% of slots at 80%, most
of the way to twice the base rate. Past that it flattens, so there is nothing to gain from
cutting harder.

The absolute score is not a signal and must not be used as one: it peaks around 60%
coverage and then *falls* as the cut rises. Long dense sentences and the literal and
section bonuses all inflate the raw logit, so the highest-scoring slots are not the
easiest ones.

The strongest single indicator is not a threshold at all:

| | slots | confirmed correct |
|---|---|---|
| the pick contained the value verbatim | 47 | **80.9%** |
| it did not | 126 | 27.8% |

A literal hit is worth 2.9x, which says the retriever should be trusted where deterministic
matching already agrees with it and treated as a suggestion everywhere else. That is an
argument for making the literal match a gate rather than a `+4.0` bonus, and it is worth
measuring on its own.

Raw results: `data/eval/evidence-eval-rows.json`, scores in `data/eval/evidence-scores.json`, gold in `data/eval/evidence-gold.json`,
resolved slots in `data/eval/evidence-jobs.json`.
