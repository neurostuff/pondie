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

## Open questions for review

1. Is M1 the right primary? It is gameable by a pass that writes nothing. M5 is the
   counterweight, but the pair needs a stated trade-off.
2. Should M2 be a gate or a metric? A genuine correction *should* be able to downgrade
   `reported` to `generated` when the old citation was wrong.
3. How many papers of ground truth are enough to distinguish a real improvement from
   noise, given repair touches ~150 fields per record?
4. D4 (misattribution) is invisible to every automatic check proposed here. Is there a
   deterministic guard, or does it need the ground truth to catch at all?
