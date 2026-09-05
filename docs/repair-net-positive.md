# Making the repair pass net positive

## Why this exists

Repair ran on 18823721 (opioid cue-exposure fMRI, 40,737 chars) and left the record worse.
Measured by diffing `unrepaired/18823721.extraction.json` against `records/18823721.extraction.json`:

| what repair did | count |
|---|---|
| fields written | 156 |
| **spans destroyed** (`evidence: present` -> `not_found`) | **26** |
| spans gained (`absent`/`not_found` -> `present`) | 8 |
| validation errors introduced | 5 |
| values changed | 4 (1 neutral, 3 damaging) |

Net evidence delta **-18 spans**. The pass is currently a liability on this paper.

## Diagnosis

### D1. Re-writing an unchanged value destroys its warrant

    grp_opioid_patients.age_mean
      BEFORE  value 44.5  source "reported"   evidence present
              span "mean age = 44.5 years" (11346-11367)
      AFTER   value 44.5  source "generated"  evidence not_found

The value did not change. The proposer re-proposed what was already there, `edit.apply`
wrote it, and `_wrap` (`record/edit.py:610`) rebuilt the wrapper from scratch. `_wrap`
only looks for a span when `len(str(value)) >= 20`, so every short value -- every numeric
-- lands on `{"status": "not_found"}` and `value_source: "generated"`.

`refuses_losing_the_warrant` (`record/edit.py:123`) is supposed to prevent exactly this.
It returns `None` (allow) when an existing span still contains the new value, and its
docstring says "The old spans are kept when they still contain the new value". They are
not kept: the guard allows the edit and the writer then discards the evidence.

Two defects, one symptom:
  * **D1a** an edit whose value is unchanged should not be an edit at all
  * **D1b** an edit the guard allowed *because* the old span supports it must inherit that
    span, not drop it

### D2. `_wrap`'s 20-character floor makes numerics ungroundable

`str(12)` is two characters, so `acquired_count` can never carry a span written by repair,
however well the paper supports it. The floor exists to stop enum tokens being "found" as
substrings; it also guarantees that any numeric rewrite loses its warrant.

### D3. `evidence: not_found` is being read as "unsupported" -- by me, and possibly by code

Of 114 writes marked `generated`/`not_found` on this paper, the ones checked against the
article (`Philips`, `Intera`, `SPM2`, `FDR`, `DSM-IV`, and all four assessment
instruments) are **all present in the text**. `not_found` means the locator failed, not
that the value is wrong. Any rule of the form "drop what cannot be grounded" would discard
correct data. I proposed exactly that rule before checking, and it was wrong.

### D4. Misattribution, not fabrication

`grp_opioid_patients.medications` became `"haloperidol"`. The word is in the paper --
"One patient was excluded from data analysis because of haloperidol use" -- so it is not
invented. It is attached to the wrong entity: that patient was excluded, so the analysed
group of 12 contains nobody taking it. Meanwhile `grp_controls.medications` was correctly
filled with "no current psychotropic medication" *and* grounded.

The failure class is scope, not invention. A grounding check cannot catch it, because the
token really is in the document.

### D5. Type regressions

4 of the 5 introduced validation errors are a list slot written with a bare string:
`medications`, `medical_condition`, `response_mode`, `preprocessings[].software`. The 5th
is `tables[].non_analysis_content = "connectivity_seeds"` on a table whose own caption is
"Brain regions that show co-activation with the subthalamic nucleus (STN)" -- and "seed"
appears **0 times** in the article. That one is genuinely unsupported.

## Proposed metrics

A repair pass is net positive on a record when it does not lose warranted facts and adds
correct ones. Four measures, computed by diffing pre- and post-repair records:

**M1 — span delta (primary, hard gate).**
`spans_gained - spans_destroyed`, where destroyed means a field went `present` ->
`not_found`/`absent`. **Must be >= 0 on every record.** Today: -18 on 18823721.

**M2 — provenance monotonicity (hard gate).**
Count of fields whose `value_source` went `reported` -> `generated` while the value was
unchanged. **Must be 0.** Today: 26.

**M3 — introduced validation errors (hard gate).**
From the record validator, post minus pre. **Must be 0.** Today: 5.

**M4 — value accuracy on writes (the quality measure).**
Against a hand-built ground truth, for each written field: correct / wrong / unverifiable.
Reported as precision over writes that changed or created a value. Needs manual reading;
see below. This is the only measure that can say repair *helped*, as opposed to did no
harm.

**M5 — fill yield (secondary).**
Fields that went absent -> present-with-a-correct-value, per record. Guards against a
"fix" that achieves M1-M3 by doing nothing.

M1-M3 are deterministic and free. M4 needs ground truth.

## Ground truth

Hand-read papers and record the correct value for a fixed field set, so M4 is scored
against something other than the model's own output. Proposed set, chosen because repair
touched them and they are checkable in a few minutes per paper:

    groups[].acquired_count, age_mean, age_standard_deviation, medications,
    medical_condition, diagnostic_system, is_healthy
    tasks[].response_mode, instructions
    acquisitions[].magnetic_field_strength_tesla, modality
    preprocessings[].software
    model_estimations[].software, model_family
    inference_settings[].multiple_comparison_method, correction_scope, inference_level
    tables[].non_analysis_content

Start with 18823721 (already read closely) plus papers drawn from the cue_reactivity
cohort. Record as `benchmarks/repair_truth/<pmid>.json`, one file per paper, with a
`quote` beside each value so a disagreement can be adjudicated against the text.

## Corrections from review (round 1)

The reviewer checked every claim above. Five were wrong or mis-scoped:

* **"156 fields written" is 83 paths written twice.** `iterations=2` re-writes everything,
  so the break at `repair.py:341` is dead code and every write count in this doc was 2x.
* **M2 was 27, not 26**, and is **22** under R2, which counts a downgrade only where there
  was a warrant to withdraw. `build` can emit `reported` with `not_found`; a pass
  relabelling that to `generated` corrects the record. `M2 == 0` would have forbidden it.
* **My textual evidence for D3 was itself the grep trap.** `grep -ci intera` returns 9
  because it matches "interaction"; case-sensitively it is 1, the scanner. `grep -ci ROI`
  returns 20, all inside "heroin". The conclusion survives -- the values are supported --
  but not by the counts I quoted. `seed` = 0 is confirmed.
* **D4's example was scored backwards.** `grp_controls.medications = "no current
  psychotropic medication"` is not a correct fill: the slot asks for the drugs a cohort
  was taking, names only, and the criterion is worded identically for both groups. The
  right value is null for both. I had counted a wrong write as the pass's best moment.
* **D5's root cause is not in `edit.py`.** `repair.run` is handed the *extraction* schema,
  where multiplicity lives in the range name (`ExtractedStringList`) and the attribute's
  own `multivalued` is False. `values.shape` read it raw. Fixed by `Schema.is_multivalued`.

Two things the metrics could not see at all:

* **Reference slots.** Repair made 6 on 18823721 -- both groups' `diagnostic_instrument`
  and both `correction_regions` -- of which at least 4 are wrong, and none is visible to
  M1/M2/M3/M5 because a reference carries no wrapper and the walk yields wrappers only.
  This needs its own gate (R4).
* **M1 is gameable upward.** 7 of the 10 spans gained on 18823721 are table captions and
  titles, which are trivially locatable because they *are* the text being searched. A
  netted span count rewards that. R1 replaces it with per-field warrant preservation.

## Open questions for review

1. Is M1 the right primary? It is gameable by a pass that writes nothing. M5 is the
   counterweight, but the pair needs a stated trade-off.
2. Should M2 be a gate or a metric? A genuine correction *should* be able to downgrade
   `reported` to `generated` when the old citation was wrong.
3. How many papers of ground truth are enough to distinguish a real improvement from
   noise, given repair touches ~150 fields per record?
4. D4 (misattribution) is invisible to every automatic check proposed here. Is there a
   deterministic guard, or does it need the ground truth to catch at all?

## Results

Two arms over the same pristine records, production `pondie.extraction.repair`.
Deterministic measures, 13 papers whose pre-repair record is byte-identical in both arms:

| | before | after |
|---|---:|---:|
| spans destroyed | 227 | **2** |
| provenance downgrades | 141 | **0** |
| findings introduced | 54 | 1 |
| fields filled | 316 | 257 |
| papers failing a gate | 11 of 13 | 1 of 13 |

**The pass no longer subtracts.** That was the whole of the first three rounds and it is done.

### It does not pass the gate

`yield >= wrong + invented`, scored against hand-read truth, **fails on all four papers in
the after arm**. On 18823721 -- the one controlled paper with enough writes to read
anything into -- bad writes fall 9 -> 5 with yield unchanged at 4, and correct fields rise
40 -> 44 while filling ten fewer. Real movement, and still short by one write.

### Four corrections to numbers this document previously carried

* **Introduced findings are 1, not 0.** `tasks[].conditions ... got dict` on 12860777
  survives in the artifact, because the fix postdates the run. `M3 == 0` is a claim no run
  has demonstrated, and it should not be repeated until one has.
* **The before arm is 54 findings, not 61.** The earlier figure was measured with a
  different validator, so 61 -> 0 compared two rulers.
* **The fill drop is 59, not 16.** 285 -> 269 compared twelve papers against fifteen.
* **Only two of the four truth papers are controlled.** `16038771` and `21118656` start
  from different pre-repair records in the two arms, so a delta on those is the
  extractor's, not repair's.

### What the 59 fewer fills cost

67 given up, 8 gained. Eighteen are `non_analysis_content`, the most-invented slot in the
corpus and the source of ten of the before arm's findings -- refusing those is the point.
The rest divides into three classes, and only one is invention: **paraphrase**
(`model_settings` on 18823721 is in the paper, reworded, and the document test looks for
the words), **unit conversion** (`450.0` seconds for "7.5 minutes"), and inference.
So yes: damage was traded for yield, and the trade is not free.

Rule A's cost is smaller than the fill drop suggests. Of what it refused, every
adjudicable case on 18823721 is correct, and the only possible losses across all four
papers are four values the truth set marks `support: inferred`. **It refused nothing the
paper states.**

### Residual risk

The record is now honest about what it does not know and no longer destroys what it knew.
What it still does is state, with a citation, a fact belonging to a different entity in
the same paper -- `haloperidol` from an excluded patient, an age from a subgroup, a
criterion read as an observation. That is about one changed write in two, and **no gate in
this document can see it**. Catching it needs the candidate span recorded at write time,
which is separate work.

## A rule this review learned the hard way, four times

Four separate findings in this work were the same mistake: **a verbatim test applied to a
value the paper was never supposed to have printed.**

* `Intera` "absent" -- `grep -ci` matched it inside "interaction"; case-sensitively it is
  the scanner, once.
* three Luna writes flagged as fabrication -- the scorer tested a reference target by the
  entity's *label*, and the model paraphrases: `sadomasochistic preference screening and
  questionnaire` for a paper that says "sadomasochistic preferences".
* the link quote rule -- refused 3 of 6 correct links, and at corpus scale is inapplicable
  to 82% of them, because `Measure`, `ModelEstimation`, `Acquisition` and `Table` have no
  name for a sentence to contain.
* `mirror_of` at 10% "never named" -- 83 of 83 mirrored analyses have an unnamed target,
  which is not a defect: a mirror is the direction the paper did NOT report, synthesised
  from the original with the sign flipped. If its label were in the paper it would not be
  a mirror.

**A verbatim test is only valid against a value the paper was supposed to have printed.**
That is why `Rule A` is asked of the value and not of the slot: a vocabulary term is chosen
from a list and is either in the source or invented; a duration, a definition, a minted id,
a derived label and a synthesised mirror are none of them quotations, and the same test on
them measures the record's vocabulary rather than the paper's content.

Both findings that survived this review were established against hand-read truth rather
than against string presence -- that the pass is wrong about most of what it changes, and
that every link it chose between candidates was wrong.
