# What is systematically wrong with the extraction records

Measured over the 1,817 records committed at `autonima-results
experiments/record_arms/records/`, by `scripts/audit_records.py`. Every number here comes
from that script; rerun it to reproduce them.

```
python scripts/audit_records.py --records '<records>/*/*.extraction.json'
```

**All five deterministic fixes are implemented, tested and measured.** Over the 1,817
records: **errors 3,200 → 1,828 (−43%)**, warnings 44,603 → 33,249 (−25%), and the
queryability figures in finding 2 realised in full. `tests/test_record_defect_fixes.py`
carries 28 cases, each a shape observed in the corpus with its count.

| repair | stage | changes written |
|---|---|---|
| `unwrap_singletons` | `shape` | 21,701 |
| `conclusions` | `merged` | 11,154 |
| `name_links` | `merged` | 876 |
| `redundant_levels` | `merged` | 728 |
| `table_purpose` exemption | check | 799 errors removed |

Two things the implementation added that the report did not propose. `check_all` now
catches a raising rule and reports it as a finding, because unguarded it took every rule
after it silently — a record with no findings and a record whose checks never ran look
identical. And `validate.check_value_cardinality` asserts a wrapper's `value` against its
own declared cardinality, which is the half of finding 3 that keeps the next slot from
going the same way.

Seven findings. They are ordered by volume times confidence, and each is labelled by what
it needs, because that is the decision:

| | | volume | fix |
|---|---|---|---|
| 1 | `check_table_purpose` errors on the repair's own output | 799 errors, 461 papers | **deterministic**, one line |
| 2 | A level names an entity the record declares and does not point at it | 876 links | **deterministic**, prototyped |
| 2b | A wrapper with no `extraction_status` aborts the rule set | 151 fields, 3 crashes | **deterministic** |
| 3 | A scalar enum slot holds a one-item list | 21,701 fields | **deterministic** |
| 4 | Derived values are labelled `reported` | ~10,300 fields | **deterministic** |
| 5 | A cell names a level on a continuous term | 1,204 errors, 527 papers | 62% deterministic |
| 5b | **A cell's level and its direction state opposite signs** | 26 | **check only** |
| 6 | Entities nothing points at | 1,624 Conditions, 528 Timepoints | mostly finding 2 |
| 7 | 823 levels name nothing the record declares | 823 | **prompt** |
| 9 | `parallel` on a study with no arms — the enum has no value for observational cohorts | 956 records | **schema** + check |

Finding 5b is the smallest and the one to do first: it is the only one where a wrong answer
changes a meta-analytic map rather than costing a join.

## The state of play

The existing 19 rules already **error on 1,053 of 1,817 papers (58%)**, and warn 44,565
times. That is not a call for more checks. Findings 1, 3 and 4 say the opposite: a large
part of what is reported is the checks disagreeing with the pipeline's own deliberate
output, and a large part of what is *not* reported is invisible because one warning drowns
it. Fixing 1 and 4 removes about 44,500 findings without touching a record, which is what
makes the remaining ones readable.

---

## 1. `check_table_purpose` errors on `derive_table_effects`'s own output

**799 errors across 461 papers — 25% of every error in the corpus.** All 799 have the same
value:

```
Table.purpose = "reported_effect",  value_source = "generated"   (799 of 799)
```

`derive_table_effects` deliberately writes `reported_effect` on every table an analysis
cites — that is its whole job, and the join settles it without a model. `check_table_purpose`
then errors on any table that is `marked` and cited:

```python
marked = values.read(table.get("purpose"))
if marked and local_id in referenced:
    findings.error(..., f"says this table reports {marked!r} rather than an effect, ...")
```

`reported_effect` is truthy, so the check reads the repair's answer as a contradiction of
itself. The message is self-refuting when printed: *"says this table reports
'reported_effect' rather than an effect"*.

**Fix.** Exempt the one value that means "it reports an effect":

```python
if marked and marked != "reported_effect" and local_id in referenced:
```

The check's stated purpose — "a table marked as non-analysis that an `Analysis.tables`
nonetheless names is a contradiction" — is unaffected, because `reported_effect` is not a
non-analysis marking. Its second branch, the missed-analysis warning, is untouched.

## 2. A level names an entity the record declares, and does not point at it

`FactorLevel` is the join from an analysis to the thing it compared. Over 6,860 declared
levels:

| slot | carried | rate |
|---|---|---|
| `groups` | 3,256 | 47.5% |
| `conditions` | 1,598 | 23.3% |
| `timepoints` | 186 | 2.7% |
| `arms` | 163 | 2.4% |
| `regions` | 21 | 0.3% |

**1,713 levels (25%) are on a categorical term and reach no entity at all** — a bare string.
`check_cell_terms` says why that matters: "the mapper joins these on the string". This is the
corpus-scale confirmation of `normalization-pipelines.md`'s "91% of analyses cannot say which
arm they belong to".

**715 of the 1,713 (42%) fold to the exact name of an entity the same record already
declares**, with a `local_id` sitting right there:

```
conditions   level 'high-calorie foods'        == cond_high_calorie_foods
conditions   level 'smoking cue'               == cond_smoking_cue
timepoints   level 'predose'                   == tp_predose
arms         level 'Exercise'                  == arm_exercise
groups       level 'women'                     == grp_women
```

595 conditions, 64 timepoints, 29 groups, 26 arms, 1 region. Both halves of the join are in
the record; nothing wrote it down.

### The fix, investigated

`link_by_name` in `scripts/audit_records.py` is the working prototype, and **every decision
it makes comes from the schema**. Per record it indexes each declared entity by folded name
*and class*; for an empty reference slot it takes the slot's declared `range` as the set of
things that may be connected, and writes the link on an exact fold match to exactly one
candidate. That is `align_cell_levels`' existing rule — repair only where the choice is not a
choice — applied one level up.

Three things the schema supplies that a hand-written table got wrong:

**Which classes may connect.** `attribute.range`, resolved through `Schema.resolves_to` so
subclasses satisfy a supertype — a slot declaring `Acquisition` is satisfied by an `MRI`.
A first version carried its own slot→kind table and missed 27 links the schema finds:
`Analysis.defines_regions` (16), `DecodingClass.condition` (4), `Analysis.tasks` (3),
`ModelTerm.assessment` (2), `ModelTerm.region` (2).

**Which references are identity and which are relations.** A name says what a thing *is*; it
says nothing about how two things *relate*. The schema separates them exactly: **a reference
whose `range` is the owner's own kind is a relation.** Of 39 reference slots, three are, and
they are precisely the three:

| self-ranged slot | what it means |
|---|---|
| `Analysis.mirror_of` → `Analysis` | sign-reversed twin of |
| `ModelEstimation.inputs_from` → `ModelEstimation` | fitted on the output of |
| `ModelTerm.interaction_with` → `ModelTerm` | crossed with |

No list is needed, and that matters because a list is what I would have got wrong. Matched on
a name these three propose **9,151 self-links** — an entity's name trivially matches itself —
and after excluding self, `interaction_with` still proposes 708 links of which **97% are a
term named `group` in one model matching a term named `group` in another** (`age` 44 times,
`diagnosis` 18, `sex` 10). Two models each having a group factor is not a crossing, and the
link would fabricate interactions that `check_crossings` then reports as unrecorded.

**What shape to write.** `attribute.multivalued`: a bare id string, or a bare list of them.
Never an `ExtractedValue` — a reference is an address inside the record, not a claim about
the paper, which is why `IDENTIFIERS` exempts these slots from the evidence checks. The first
prototype wrapped them, which would have written 838 malformed fields.

One guard the schema cannot supply: **exact fold match, never substring.** A further 966
links are reachable on a single substring match and are not taken — 403 of them
`Analysis.defines_regions`, 101 `Analysis.regions`. `normalize_open_fields.py` measured what
containment does to a hierarchy: it merged `emotion regulation`, the corpus's most frequent
task term, into a rarer variant, with 38 candidate hosts. A substring is a hint; report it.

**One decision, not a guard.** 122 matched levels name two kinds of entity at once, and 112
are `arms` + `groups` — a level named "Exercise" matching both the arm and the cohort
allocated to it. Writing both is correct; that is the parallel-group design `Group.arm` exists
for. The other 10 should be reported. No name matched two entities of the *same* kind
anywhere in the corpus, so the ambiguity guard never had to fire.

### What it buys, in queries

**876 links over 265 records**: 572 `FactorLevel.conditions`, 164 `.arms`, 72 `.timepoints`,
29 `.groups`, 1 `.regions`, 16 `Analysis.defines_regions`, 11 `Group.arm`, 4
`DecodingClass.condition`, 3 `Analysis.tasks`, 2 `ModelTerm.assessment`, 2 `ModelTerm.region`.

Re-validating those 265 records: **415 rule errors before, 415 after.** No new error of any
kind, which is the claim that matters for a fixer that writes into the record.

Queryability, measured **per (analysis, level) pair** because that is what a query traverses —
one unwritten link costs every analysis whose model reaches the term, which is why 838
FactorLevel links move 2,184 pairs:

| | before | after |
|---|---|---|
| levels resolved to an entity | 72.3% | **83.8%** |
| analyses whose contrast is fully resolvable from the entity graph | 67.3% | **77.2%** |
| analyses that can say which condition | 37.5% | **49.7%** |
| analyses that can say which arm | 3.6% | **7.8%** |
| analyses that can say which occasion | 5.0% | **7.1%** |

**+454 analyses become fully resolvable** — every categorical level in their contrast reaches a
declared entity, so the comparison can be reconstructed from the entity graph instead of from
string matching.

The honest limit is the arm row. It more than doubles, and 92% of analyses still cannot say
which arm they belong to. But **that number is not a defect rate**, and reading it as one was
my error — see finding 9.

## 2b. A wrapper with no `extraction_status` crashes the rule set

**151 wrappers in 12 papers carry `value` and no `extraction_status`** — `Cell.direction` as
`{"value": "positive"}`, and the same on `Analysis.prespecification` (35),
`Analysis.spatial_scope` (25), `Effect.kind` (24), `Statistic.family` (22).

`values.read` reads a mapping without a status as a nested entity and returns **the dict
itself**, which is how it is supposed to tell a wrapper from an entity —
`_records._descend` documents exactly this. So a consumer asking for a direction gets
`{"value": "positive"}`, and `check_crossings` does `direction in {"positive", "negative"}`
and raises `TypeError: unhashable type: 'dict'` on 3 records.

Two fixes, and the second is the one that generalises:

- A `shape` repair setting `extraction_status: "extracted"` on a wrapper that carries a
  `value` and no status. 151 fields, no judgment.
- **`check_all` must not let one rule's exception abort the rest.** It iterates `RULES` with
  no guard, so those 3 records lose every check registered after `crossings` — silently, and
  the loss is invisible in the output because a record with no findings and a record whose
  checks never ran look identical. `validate.py` propagates the exception to the caller.

## 3. A scalar enum slot holds a one-item list

**22,431 fields hold a list where both schemas declare a scalar, and 21,701 (96.7%) hold
exactly one item.**

```json
"spatial_scope": {"value": ["whole_brain"], "value_source": "generated", ...}
"coordinate_space": {"value": "Talairach", "value_source": "reported", ...}
```

The second is right and the first is not, and the difference is that `spatial_scope`'s range
is an enum. `ExtractedSpatialScope.slot_usage.value` declares `any_of: [SpatialScope, string]`
and is not multivalued, so the record violates the extraction schema as well as storage.

It concentrates on exactly the slots a query filters on:

| slot | instances |
|---|---|
| `Analysis.spatial_scope` | 4,470 |
| `Analysis.prespecification` | 4,328 |
| `Region.definition_method` | 2,360 |
| `Group.species` | 2,207 |
| `Region.region_type` | 2,151 |
| `ModelEstimation.spatial_unit` | 1,897 |
| `ModelEstimation.model_family` | 1,633 |
| `InferenceSettings.correction_scope` | 1,199 |

`values.read()` returns the list as it stands, so a consumer testing
`spatial_scope == "whole_brain"` matches nothing on 4,470 analyses. This is the slot that
decided a paper's inclusion in the record-arms experiment: 17133391 was rejected on
`spatial scope: roi`.

**Two fixes, and the check is the important one.**

- A `shape` repair unwrapping a one-item list on a scalar slot. 21,701 fields, no judgment.
- `validate.py` should check the wrapper's `value` against the declared cardinality. It
  does not today, which is why 22,431 violations passed validation in silence. That is the
  reason this went unseen, and it will catch the next slot rather than this one.

**730 are genuinely multi-valued and are a decision, not a fix.** Some are contradictions to
surface (`spatial_scope: ['whole_brain', 'roi']`, 31 times — the two exclude each other).
Others say the storage slot has the wrong cardinality: `Region.region_type` holds
`['anatomical', 'atlas_parcel']` 198 times and `Region.definition_method` holds
`['atlas', 'anatomical_a_priori']` 143 times, which is a region defined two ways rather than
an extraction error.

## 4. Derived values are labelled `reported`

`check_value_source_honesty` warns **43,772 times in 1,792 of 1,817 papers (98.6%)** — 98% of
all warnings in the corpus. A warning that fires on 98.6% of papers is not read, so this
finding is really about the other 793 warnings it buries.

The premise is sound: 21% of `reported` values carry no sentence. But it is not spread evenly.
Four slots are near-total, and every one of them is a value the **pipeline derives**:

| slot | reported | no sentence | rate |
|---|---|---|---|
| `ModelTerm.type` | 3,041 | 3,037 | **100%** |
| `Effect.kind` | 467 | 464 | **99%** |
| `Cell.direction` | 3,364 | 3,095 | **92%** |
| `FactorLevel.order` | 5,575 | 3,694 | 66% |
| `sex_distribution.percentage` | 1,571 | 1,050 | 67% |
| `sex_distribution.denominator` | 1,551 | 675 | 44% |
| `InferenceSettings.tfce_used` | 1,111 | 638 | 57% |

`direction.py` computes direction. `derive_effect_kind` computes kind. `derive_denominators`
computes the denominator. `order` is an index. `type` is a classification — no paper writes
"this term is continuous", so `reported` cannot be true of it even in principle.

**Fix.** Two halves, and both are needed:

- Where the pipeline writes one of these, it must write `value_source: generated`. Some code
  paths already do; these slots show that not all do.
- A `shape` repair relabelling `reported` → `generated` on a closed list of slots whose value
  is a conclusion rather than a quotation, and only when `evidence.status` is `not_found`.
  A value with a sentence keeps `reported`, because then the paper did say it.

This removes roughly 10,300 warnings from four slots, and leaves the shape the check was
written for — `family = electrophysiology` on a BOLD study, `spatial_scope = roi` nobody
stated — visible for the first time.

## 5. A cell names a level on a continuous term

**1,204 errors across 527 papers (29% of the corpus)**, the largest error class. The reading
is not what the message suggests: 1,185 of 1,205 are on terms typed **continuous**, which
legitimately declare no levels. The defect is the cell, not the missing level. Three shapes:

| | | share |
|---|---|---|
| the level restates the term's own name | 547 | 46% |
| the level is a direction word | 214 | 18% |
| the level is categorical, so the `type` is wrong | 424 | 36% |

**Restating the term's name** is content-free: `BMI` on term `BMI`, `age` on `age`,
`pack-years` on `pack-years`, `AUDIT score` on `AUDIT score`. A continuous regressor has no
level, and naming it after the term says nothing a reader did not already have.
*Deterministic: drop the level where it folds to the term's own name.*

**A direction word** — `positive` (127), `negative` (57), `higher` (18) — belongs in
`Cell.direction`, not in `Cell.level`. Comparing the two by polarity rather than by string
(`positive`/`higher`/`greater`/`increase` against `negative`/`lower`/`less`/`decrease`):

| | | share |
|---|---|---|
| same polarity — the level duplicates `direction` | 185 | 86% |
| **opposite polarity — the record contradicts itself on the sign** | **26** | 12% |
| `direction` empty — the level is the only sign there is | 3 | 1% |

*Deterministic for 185: drop a level whose polarity its own `direction` already carries.
Move it for the 3 where `direction` is empty.*

**The 26 are the most consequential defect in this report** and are not a cleanup. A cell
reading `direction='negative'` with `level='positive'` states both signs of the same effect,
and `direction` is what decides whether a coordinate enters an increase map or a decrease
map. Silently dropping the level would pick `direction` by default and bury a 50/50
question. These want an error naming both slots:

```
   9  direction='negative'   level='positive'
   5  direction='negative'   level='higher'
   3  direction='positive'   level='decrease'
   3  direction='positive'   level='lower'
   3  direction='negative'   level='increase'
   2  direction='negative'   level='activation'
   1  direction='positive'   level='negative'
```

`direction.py` and the sign-split pass already own polarity, so this check belongs beside
them rather than in the cell rule: it is the one place a wrong answer changes a map.

Findings deterministic here: 547 + 185 + 3 = 735 of 1,185 (62%).

**The remaining 424** are categorical levels on a mistyped term — `bvFTD` (25), `AD` (18),
diagnostic groups typed continuous. 154 of the 424 (36%) fold to a Group or condition the
record declares, which makes the mistyping demonstrable rather than suspected. Flipping
`type` also requires synthesising the `levels` the term should have declared, so this wants
an error naming both slots, not a silent repair.

## 6. Entities nothing points at

| class | declared | orphaned | rate |
|---|---|---|---|
| `Condition` | 3,010 | 1,624 | **54%** |
| `Timepoint` | 696 | 528 | **76%** |
| `Region` | 3,238 | 717 | 22% |
| `Assessment` | 4,359 | 732 | 17% |
| `ModelTerm` | 4,895 | 563 | 12% |
| `Group` | 4,227 | 396 | 9% |

(`Analysis` and `Study` are roots; nothing referencing them is correct.)

`Condition` and `Timepoint` are mostly finding 2 seen from the other end — the level that
should have pointed at them exists and is a bare string. Fixing finding 2 reclaims 595
Conditions and 64 Timepoints directly.

The others are a different question and want a check rather than a fix: a Region defined and
never analysed may be a real ROI definition the paper states and no analysis used, which is
exactly what `Table.purpose`'s non-effect kinds exist to record. An orphan is worth a
warning with the class named, so the two can be told apart — the same argument that made
`Table.purpose` worth having.

## 7. What is left is a prompt problem

**823 of the 1,713 unjoined levels (48%) name nothing the record declares**, and 270 of
finding 5's 424 mistyped levels likewise. No deterministic rule reaches these: the entity is
absent, so there is nothing to join to and nothing to check against.

The shape is the same one `Demands`/`Satisfy` was ordered to prevent — "asked to guess an
inventory first, the entity pass modelled a crossover's condition as a continuous covariate".
The analyses pass declares the levels it needs and the entity pass does not create all of
them. `Demands` already emits a demand list; the measurable question is whether `Satisfy` is
dropping demands or whether the demands never name these levels. That is one join away from
the existing payloads and worth measuring before any prompt is rewritten.

## 9. Why arms and occasions look so bad, and what is actually wrong

The 3.6% arm and 5.0% occasion rates in finding 2 are over *all* analyses, and most analyses
are in studies that have neither. Conditioning on the entity existing to be referenced:

| | analyses | reference one | rate |
|---|---|---|---|
| in a record declaring ≥1 `Arm` | 706 | 157 | **22.2%** (not 2.6%) |
| in a record declaring ≥1 `Timepoint` | 1,186 | 222 | **18.7%** (not 3.7%) |

Only **241 of 1,817 records declare an Arm (13%)** and 363 a Timepoint (20%). So the first
answer is yes: mostly the analyses are not testing arms or occasions, and the low
unconditional rate is correct rather than a failure.

Of the analyses that *could* link and do not, the prose says which is which:

| | | |
|---|---|---|
| arm declared, no link, and the analysis names one of the declared arms | 206 | **recoverable** |
| arm declared, no link, and the prose names no arm | 343 | correct — pooled or baseline contrast |
| occasion declared, no link, prose names a change over time | 109 | **recoverable** |
| occasion declared, no link, prose names no change over time | 855 | correct |

So roughly **half the arm gap and a tenth of the occasion gap is a missing join**, and the
rest is analyses that genuinely do not test it. These 315 are not reachable by finding 2's
fixer: it matches a *level's* name, and here the arm is named in the analysis's own prose.
`check_arm_reachability` already warns on the 206 and nothing acts on it.

### The reason arms look like they should be there

**1,120 records (61.6%) say `assignment_structure: parallel`, and 956 of them declare no Arm
at all.** The enum's own description makes that a contradiction:

> `parallel`: Each arm is a separate cohort, and no participant is in more than one, so the
> allocation is a property of a Group: **set `Group.arm`**.

Those 956 are not treatment studies:

| their `allocation` | |
|---|---|
| `non_randomized` | 569 (59.5%) |
| `not_applicable` | 318 (33.3%) |
| `randomized` | 35 (3.7%) |

93% have no randomisation, 93% declare two or more Groups, and their groups are
`healthy controls` (132), `bvFTD` (80), `controls` (67), `AD` (57), `smokers` (36) —
**diagnostic cohorts, not treatment arms.**

This is a **schema gap, not a model error.** The vocabulary offers `parallel`, `crossover`,
`within_subject`, `single_group`, and the commonest design in this literature — several
naturally-occurring cohorts, scanned once, no intervention — fits none of them.
`single_group` is "One cohort measured once", so a two-cohort observational study cannot use
it. The enum's top-level description frames the whole choice as being about arms — "Whether
the **arms** of the study are separate cohorts or the same participants at different times" —
which presupposes arms exist. Asked which kind of arms a case-control study has, a model
answers `parallel`, because two cohorts are in parallel.

**Proposals:**

- **Add a permissible value** for the observational multi-cohort design: several cohorts that
  were not allocated to anything, `Group.arm` stays empty, and the cohorts reach an analysis
  through `FactorLevel.groups` — which is what these records already do, correctly, in the
  47.5% of levels that carry a group. 956 records (53% of the corpus) would stop asserting an
  allocation that never happened. **Done and tested — see below.**
- **A check**: `assignment_structure` in `{parallel, crossover}` with zero `Arm`s declared is
  a contradiction of the enum's own text, and nothing reports it today.
  `check_arm_reachability` fires only once an Arm exists, so this whole class is invisible to
  it.
- The 206 + 109 recoverable joins want an **error naming both slots** rather than a repair.
  Matching an analysis's prose against an arm name is substring matching on prose, which is
  the operation finding 2 declines for the reason `normalize_open_fields.py` measured.

### `observational_cohorts`, added and tested

`AssignmentStructure` now carries `observational_cohorts`, `parallel` requires that something
was administered, `single_group` requires both of its halves, and
`Allocation.non_randomized` no longer offers "or an existing exposure" — the clause that let
569 records call a diagnosis an assignment.

Tested with `scripts/ab_enum_wording.py`: same input, same model, same temperature, the enum
block the only difference. Not a re-extraction — the paper texts are not on this host, so the
input is the design evidence each record already quotes, which is the text the original answer
was read off. That buys a sample large enough to see past run-to-run variation, which three
rounds of `population_characteristics` rewrites never had.

**66 papers to tune on, then 66 disjoint papers to check** — the same seeded shuffle, second
slice, overlap zero. Predictions were written down before the held-out run.

| | tuning | held-out | predicted |
|---|---|---|---|
| target (`parallel`, no arms, ≥2 cohorts) → `observational_cohorts` | 29/30 | **28/30 (93%)** | ≥90% ✓ |
| genuine randomized trials stay `parallel` | 5/5 | **8/8 (100%)** | 100% ✓ |
| `crossover` unchanged | 6/6 | **6/6** | ≥90% ✓ |
| `within_subject` unchanged | 5/6 | **5/6** | ≥80% ✓ |
| `single_group` unchanged | 7/12 | **5/12 (42%)** | ≥70% ✗ |

The `single_group` prediction failed, and it is the prediction that was wrong rather than the
change: the **old** enum also scores 5/12 on that draw. `single_group` is an unstable label in
both arms, because "one cohort measured once" has two conditions and papers routinely satisfy
one. It is worth its own look and is not a regression from this change.

**One iteration, and only on the tuning set.** The first wording fixed 30/30 of the target but
pulled two *mixed* designs to `observational_cohorts` — a crossover with a between-subject
group factor, and a repeated-measures stress study comparing men and women. So
`observational_cohorts` now states that a within-subject structure takes priority, with the
reason: the cohort comparison is already recorded in `FactorLevel.groups`, and the
repeated-measures structure has no other slot. That recovered both, cost one target paper
(itself a two-session counterbalanced design, so arguably the right answer), and eliminated
three arguable `within_subject` → `single_group` flips.

**What the held-out set also confirmed, unasked:** 7 of the 12 `parallel`-*with*-arms papers in
the tuning set and 2 of 12 in the held-out set flipped, and adjudicating every one shows the
flips are corrections — their "arms" were `['maltreated', 'non-maltreated']`,
`['bvFTD', 'controls']`, `['no intervention']`, `['prenatal cannabis exposure', ...]`. The
same schema gap was also producing **invented Arm entities**, not just a wrong label: across
the corpus **47 of the 164 `parallel`-with-arms records have every arm name identical to a
cohort name**, and 28 of those are `non_randomized` or `not_applicable`. So the change reaches
roughly 989 records, not 956.

**Known limit, named rather than tuned away.** The priority rule moves a controlled trial with
pre/post scanning to `within_subject` (`27091455`: DBT vs usual care, before and after 12
weeks). The rule's own justification holds — the arms survive on `Group.arm` — but `parallel`
is arguable there, and `AssignmentStructure` being single-valued means a mixed design cannot
state both halves. I did not iterate on this, because iterating against the held-out set would
turn it into a second tuning set and there would be nothing left to check the wording against.

## 4b. What is behind the 32,500 warnings that remain

`conclusions` removed 11,312 of the 43,810. The rest is not the same defect thinned out, and
diagnosing it corrected two things I had said.

**Not a template default.** The prompt hands the model `{"value_source": "reported"}` in its
worked skeleton, so I assumed it was copying. It is not: `generated` values are unevidenced at
**48.9%** against `reported` at **16.4%**, and per slot the separation is wide —
`assessments.name` 3% against 100%, `regions.name` 10% against 99%,
`groups.enrolled_count` 7% against 94%. The label discriminates threefold.

**It is the evidence pass failing to place a quote**, which its own docstring allows for:
"what it cannot place is left `not_found` for `repair` to go looking for." The repair locator
recovers 2,616 spans, **1.6% of all located spans**, so that fallback is barely working — and
its gate is part of why, since `edit.py` refuses any quote under 20 characters without trying
while 80,022 of the reported values are three words or fewer.

Three readings are ruled out by measurement:

- **Not a handful of bad texts.** Median paper 11.6% unevidenced, p90 30.2%, none above 80%,
  and the papers at ≥50% are only 7% of the warnings.
- **Not verbatim-ness.** Flat by value shape: 17.0% lists, 17.0% strings of 4+ words, 17.3%
  strings of ≤3 words, 11.8% numbers. A number printed in the text fails nearly as often as a
  phrase. Booleans alone stand out at 33.9%, which is `tfce_used` at 57%.
- **Not one bad locator.** 80.4% of found spans are `model_quote`, 17.5% `literal_match`,
  1.6% `repair_pass`.

What is left is a **gradient of the same thing finding 4 fixed**. Against the 16.4% base:
`sex_distribution.percentage` 67%, `tfce_used` 57%, `cells.level` 39%,
`terms.source_definition` 35%, `analyses.interpretations` 34%, `statistic.family` 31%,
`analyses.name` 26% — all the record's own encoding rather than the paper's words. The low end
is verbatim material: `assessments.name` 3%, `groups.name` 6%. So the five slots relabelled
were the absolute end of a continuum, not a separate kind, which is why the fix took 11,312
and left 32,500.

### The split, and the matcher

`status: not_found` collapsed two opposite faults: **no quote was offered** (recall) and **a
quote was offered and rejected** (fidelity). `build` counted both and wrote neither into the
record, so the committed corpus cannot be asked which dominates. Both are now recorded:

- `Evidence.unlocated_quotes` — how many proposed quotes no locator could place, omitted
  where none were, so its presence is the claim. The same argument `EvidenceSource` was added
  for: a distinction the pipeline already knows and the record did not carry.
- `BuildReport.fields_quote_unlocated` and `resolved_cased`, both in the build notes.
- `check_value_source_honesty` now says which fault it found, so the warning names an owner.

And `spans.resolve` gained a **third pass**. It does no fuzzy matching and should not — the
integrity gate asserts `normalized[start:end] == span.text`, and `fold` is a 1:1
length-preserving translation because "folding has to preserve string length, or the offsets
it produces would not address the original document". But its tolerance was only *between*
tokens, since each token is escaped whole:

| perturbation | before |
|---|---|
| newline or double space between tokens | resolves |
| curly quote, en dash, non-breaking space | resolves |
| trailing punctuation that is in the text | resolves |
| **lowercased first letter, or Title Case** | **fails** |
| dropped hyphen, dropped comma, paraphrase, ellipsis | fails |

Case is the one of those a model makes mechanically. It is now a third pass, tried only after
the case-sensitive one finds nothing, so it is **strictly additive** — a case-sensitive hit at
a later offset can never be displaced by a case-different one earlier. Offsets are unaffected
because the match is against `fold(normalized)`, which is length-equal to the document.

**What this buys is not yet measured, deliberately.** A built corpus cannot be asked: a
resolved span keeps the document's text, not the quote that located it. `resolved_cased` is
the instrument, and it reads on the next run. Persisting the split before changing the matcher
is the whole point — otherwise the change ships with no way to tell whether it worked.

## Checked and ruled out

Two things that look like defects in this corpus and are not, recorded so they are not
re-reported:

- **`is_healthy` and `response_mode` are "not declared on Group/Task".** The records predate
  the schema changes that derived `is_healthy` and renamed `response_mode` to
  `response_modality`. This is the records being older than the schema, not a record defect.
- **48,131 "scalar slot holds a list" against the *extraction* schema.** The generator
  projects a multivalued scalar to one wrapper holding a list, so `multivalued` is False on
  the wrapper while the list is correct. Cardinality has to be read from the storage schema,
  which is what finding 3 does — it is 22,431 there, not 48,131.
