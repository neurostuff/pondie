# Normalising task and condition names for finding contrasts

Measured over the 1,817 committed records. The conclusion is not the obvious one: **neither
task names nor condition names should be normalised as flat vocabularies**, and the unit worth
curating is the contrast, decomposed into a target and a baseline.

## First, the join, because no naming scheme survives without it

**2,580 of 12,098 cells reach a `Condition` at all — 21%.** The other 9,518 name a level as a
string and resolve to no entity, so they are invisible to any query about conditions however
well the names are normalised.

That is `record-defects.md` finding 2 from the task side, and it bounds everything below. Of
6,860 `FactorLevel`s, 23.3% carry `conditions`; 54% of the 3,010 declared `Condition` entities
are referenced by nothing at all. **Normalising a vocabulary that four cells in five cannot
reach is optimising the wrong stage.** The 715 recoverable links `name_links` writes are the
first thing to do, and 595 of them are conditions.

## Task names: a long tail, and keying on them makes things worse

| | mentions | raw forms | token bags | bags in ≥2 papers | coverage |
|---|---|---|---|---|---|
| `tasks[].name` | 1,449 | 1,075 | 887 | **17%** | 49% of paper-mentions |
| `conditions[].name` | 3,013 | 1,655 | 1,415 | 23% | 63% |

The heads are real -- `emotion regulation` 107 papers, `resting state` 60, `cue reactivity` 36
-- and the tail is not spelling. 200 curated task bags reach 52% of paper-mentions, which is
the ceiling `NORMALIZATION.md` already measured: **task names are minted per paper**, and
Cognitive Atlas is incomplete for this literature for the same reason.

Worse, the task is a bad key for contrasts. Keyed as `(task, contrast)` there are **763
distinct combinations of which 25 recur — 3%.** Multiplying two sparse keys multiplies the
sparsity. A query for "the food-cue contrast" should not have to agree with the record about
what the task was called.

## Condition names: the head is `neutral`, which means nothing on its own

The single most common condition in the corpus is **`neutral`, 168 mentions**, then `alcohol`
40, `reappraise` 38, `fixation` 34, `rest` 32, `look neutral` 31. A flat condition vocabulary
therefore spends its head on terms that are uninterpretable without the thing they are
contrasted against: `neutral` in a smoking study and `neutral` in an emotion study are
different conditions with one name.

Normalising the *pair* does not fix it either. Recovering the contrasts directly -- cells with
a direction, on levels reaching conditions -- gives **1,061 contrasts over 619 distinct pairs,
of which 11% recur, covering 26% of paper-mentions.** Worse than the condition names they are
built from, because a pair inherits the variance of both sides.

## What works: a target against a baseline

The recurring pairs are the same few contrasts written differently -- `smoking > neutral`,
`cue smoking > cue neutral`, `alcohol > neutral`, `cocaine cues > cues neutral`,
`negative > neutral`, `look negative > look neutral`, `negative pictures > neutral pictures`.
The variance is on the **baseline** side and in the modifiers, and the baseline is a small
closed set: a condition defined by the *absence* of the manipulation.

Classifying each contrast by whether a baseline sits on exactly one side:

| shape | contrasts | |
|---|---|---|
| target > baseline | 494 | 47% |
| baseline > target (reversed) | 109 | 10% |
| neither side is a baseline | 405 | 38% |
| both sides look like a baseline | 53 | 5% |

**57% have a baseline on exactly one side**, and once it is lifted off, the target side alone
is a far smaller vocabulary:

| | distinct bags | top 25 | top 50 | top 100 |
|---|---|---|---|---|
| condition names, flat | 1,415 | 26% | 34% | 44% |
| **target side of a baseline-anchored contrast** | **205** | **46%** | **62%** | **81%** |

**Seven times fewer terms, and 100 of them reach 81% of the contrasts** where the flat
condition vocabulary needs more than 200 to reach 55%. The heads are exactly what a query
asks for: `smoking` 29, `reappraise` 26, `food` 17, `negative` 17, `reappraisal` 17,
`decrease` 16, `smoke` 15, `regulate` 15, `alcohol` 14, `increase` 11.

## The recommendation

**1. Write the join before touching the vocabulary.** 79% of cells reach no condition.
`name_links` recovers 595 condition links on an exact name match; the rest is
`record-defects.md` finding 7 and needs the extraction pass, not a normaliser.

**2. Add a small closed `Condition` role: `target` or `baseline`.** This is the whole
mechanism. A baseline is a condition defined by the absence of the manipulation --
`neutral`, `fixation`, `rest`, `baseline`, `control`, `look`, `maintain`, `view`, `scrambled`,
`non-X`. It is a closed enum, it is asked of the model once per condition rather than inferred
from a contrast, and it is what makes the target side a curatable vocabulary.

**3. Curate the target side, not the condition side.** 205 bags, ~100 terms for 81% coverage.
That is a tractable list, unlike the 1,415-bag condition vocabulary or the 887-bag task
vocabulary.

**4. Lemmatise inside the target side, and nothing more.** `reappraise` (26) and
`reappraisal` (17), `smoking` (29) and `smoke` (15) are morphology and merge with no judgement.
That is worth about a fifth of the head and is the cheapest available gain.

**5. Link targets to a content vocabulary, not a task ontology.** The targets split cleanly
into *stimulus content* -- `smoking`, `alcohol`, `food`, `cocaine` -- which resolves against
the same substance terms `medical_condition` uses, and *process* -- `reappraise`, `regulate`,
`increase`, `decrease` -- which is where Cognitive Atlas does reach. Task-level Cognitive Atlas
alignment was measured as incomplete precisely because it was being asked to cover the
stimulus half.

**6. Leave `FactorLevel.level` and `Cell.level` literal.** They are the join key, and
`spans.fold_label` states the rule: `AD` and `AD group` are not the same level and calling
them equal hides a join failure. Normalise a *derived* slot beside them -- the condition's
role and the target's canonical term -- and keep the literal for the join.

**7. Do not key a contrast on the task.** 3% recurrence against 11% for the pair alone. The
task is worth recording and is not worth joining on.
