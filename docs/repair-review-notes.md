# Adversarial review of `repair-net-positive`

Reviewer's notes on `docs/repair-net-positive.md` (commit 6a774dd) and the fix in 2a7638c.
Everything here was re-derived from the records and the article text; nothing is taken from
the diagnosis doc on trust.

## What I measured, and on what

Four runs have both an `unrepaired/` and a `records/` copy AND a pass that actually wrote
something. All four are from 4 Sep 2026, 16:31-19:48, i.e. essentially the current code
(`edit.py` mtime 16:28, `repair.py` 17:11):

| run | pmid | span delta | destroyed | gained | M2 | introduced | filled |
|---|---|---|---|---|---|---|---|
| pondie-prose-18823721 | 18823721 | **-16** | 26 | 10 | 26 | 6 | 31 |
| repair-baseline | 11058476 | **-12** | 20 | 8 | 17 | 7 | 13 |
| pondie-21118656 | 21118656 | **-10** | 12 | 2 | 11 | 0 | 11 |
| pondie-newtpl | 16038771 | **-1** | 15 | 14 | 22 | 5 | 16 |

The diagnosis is right that repair is net negative, and it is right on more than one paper.
Two corrections to the doc's framing:

* **"fields written 156" is 83 unique paths written twice.** `repair.run` loops
  `iterations=2` (repair.py:337) and the break at repair.py:341 fires only when a pass
  writes nothing. Before 2a7638c nothing stopped pass 2 re-writing everything pass 1 wrote,
  so the break was dead code and every repair paid for a second proposer sweep that could
  not change the record. Every "written" count in the doc is 2x. (After 2a7638c the no-op
  refusal makes the break reachable, which is a real cost saving worth claiming.)
* **The other 30 papers under `pondie-cue-flex` are not evidence.** All 30 have
  `written: []` -- the pass ran with no proposer. Any "repair ran on N papers" claim from
  that directory is a claim about N null runs.

So the whole evidence base for the diagnosis is 4 papers. That is enough to establish the
mechanism and not enough to set a threshold.

## D1 -- confirmed, and the causal story is right

Independently reproduced: 26 fields went `present` -> `not_found`; **26 of 26 appear in
`repairs/18823721.json`'s `written` list**, and 25 of 26 kept their value byte-for-byte.
There is no second mechanism. `relocate` touched 2 fields and both gained a span.

The path is the one the doc names. `apply` skips a reference write whose value is unchanged
(edit.py:527) and has no such skip on the value side, so `entity[name] = _wrap(value, text)`
ran for a value that had not changed, and `_wrap`'s 20-character floor (edit.py:657) put
`not_found` on every numeric. `refuses_losing_the_warrant` allowed the edit precisely
because the old span supported it, then the writer discarded the span it had just been
shown.

One number to fix: **M2 as defined is 27, not 26**, and `reported -> generated` overall is
29. The doc reuses "26" for two different sets.

## D2 -- confirmed but mis-scoped

The 20-character floor does make numerics ungroundable through `_wrap`. But the floor is not
the reason the record lost spans -- the reason is that `_wrap` was called at all. Raising or
removing the floor is the more dangerous of the two fixes: `_bare("12") in _bare(span)` is
true of most sentences containing a number, and `_bare("p")` is true of nearly every
sentence in the paper. Leave the floor where it is; the inheritance path in 2a7638c is the
right answer and does not need it changed.

## D3 -- confirmed, with one term that needed re-checking

`grep -c -i` counts lines, and case-insensitivity is doing damage in this diagnosis:

* `grep -ci intera 18823721` returns **9**, because it matches "interaction", "inter-block"
  and "inter-stimulus". Case-sensitively there is exactly one, and it is the scanner:
  "Imaging was performed using a 3T MRI scanner (Philips Intera, Best, The Netherlands)".
  The claim survives; the evidence for it did not.
* `grep -ci ROI 18823721` returns **20**, all inside "heroin". Word-boundary count is 0.

The substantive claim -- `not_found` means the locator failed, not that the value is wrong --
holds. Verified present: `haloperidol` 1, `Philips` 1, `Intera` 1 (case-sensitive), `SPM2` 1,
`FDR` 1, `DSM-IV` 1.

## D4 -- confirmed as a class, but the doc's own example is scored wrong

The haloperidol sentence is verbatim what the doc says it is:

> One patient was excluded from data analysis because of haloperidol use the day before the
> fMRI scanning procedure took place.

and the diagnosis "misattribution, not fabrication" is right.

**Where I disagree:** the doc says `grp_controls.medications` "was correctly filled with 'no
current psychotropic medication' *and* grounded". It was grounded and it is not correct.
`Group.medications` is documented as "The drugs or other agents this cohort was taking, as
the source names them, one agent per entry. Names only." An inclusion criterion is not an
agent. And the criterion is worded identically for both groups:

> (h) no current psychotropic medication            [patients]
> (f) no current psychotropic medication            [controls]

So repair read one criterion list and produced a wrong answer for the controls and a
catastrophic one for the patients. Treating the control write as the pass's success case is
the wrong lesson: the correct value for both groups is null.

**D4 is much bigger than the doc says, because reference slots are where it lives.** Repair
made 6 reference-slot writes on 18823721:

    groups/grp_controls.diagnostic_instrument        <- 4 assessments
    groups/grp_opioid_patients.diagnostic_instrument <- the same 4 assessments
    inference_settings/inf_interaction.correction_regions <- [reg_stn]
    inference_settings/inf_stn.correction_regions        <- [reg_stn]
    model_estimations/mod_fmri_glm.preprocessing         <- [prp_fmri]
    tasks/tsk_cue.acquisitions                           <- [acq_fmri]

By the ground truth in `benchmarks/repair_truth/18823721.json`, the first four are wrong.
`diagnostic_instrument` is "The study assessment that established this group's defining
condition"; ASI, OCDUS, DDQ and SHAPS measure drug-use history, craving and anhedonia, and
two of them were given to the patients only ("Two measures of craving were used in all
opioid-dependent subjects"). The STN is a *result* of the interaction contrast, not a
restriction on it; "region of interest", "small volume" and word-boundary "ROI" all occur
zero times in the paper.

**None of these six writes is visible to M1, M2, M3 or M5.** A reference slot holds a bare
list of local_ids with no `ExtractedValue` wrapper, and `values.iter_fields` -- which
`scripts/repair_delta.py` walks -- yields wrappers only. The measurement harness cannot see
the half of the pass where the worst errors are.

## D5 -- confirmed, and the root cause is not in `edit.py`

The four type regressions have one cause, and it is a schema mismatch, not a missing
coercion. `values.shape` (values.py:345) reads `attribute.multivalued` directly:

    if attribute is not None and attribute.multivalued and not isinstance(result, list):
        return [result]

`stages.py:900` hands `repair.run` the **extraction** schema. Under it, multiplicity lives
in the range name, not in `multivalued`:

    Group.medications      extraction  multivalued=None  range=ExtractedStringList
    Group.medications      storage     multivalued=True  range=string

so the branch never fires and a scalar lands in a list slot. `values.cast` gets this right
in the same file, because it goes through `sch.value_ranges(attribute)`, which unwraps the
`ExtractedValue`; `shape` does not. `ExtractedStringList.value` does declare
`multivalued: true`, so the fix is a wrapper-aware multiplicity read used by both `shape`
and `edit._multivalued` (edit.py:557, which has the same bug and is currently saved only
because reference slots keep `multivalued: true` in both schemas).

This also means `repair.run`'s docstring -- "the pass itself reasons with storage, because
that is where `required`, `multivalued` and the vocabularies live" -- is false in
production. It reasons with extraction, and `multivalued` is the one of the three that
breaks silently.

The fifth introduced finding, `tables[].non_analysis_content = "connectivity_seeds"`, is
genuinely unsupported: `seed` occurs 0 times, `connectiv` 0 times, `co-activation` once (in
the caption). But `connectivity_seeds` **is** a permissible value of `TableContent`, so
`cast` passes it and no vocabulary check can catch it. It is a D4 misattribution wearing a
D5 costume.

## The fix in 2a7638c: three attacks, all landing

Run against beast's venv with the committed code (`scratchpad/attack_fix.py`, reproduced in
the commit trail):

**A. The no-op guard misses ints in float slots.** `str(values.read(current)) == str(value)`
compares an unnormalised stored value with a cast proposal. The record holds
`grp_controls.age_mean = 40` (int); `cast` returns `40.0`; `"40" != "40.0"`, so the write
proceeds. Verified in the real run: that field is `40` before and `40.0` after. Harmless on
this paper because `_inherited` rescues it, but the comparison should round-trip both sides:
`values.shape(sch, class_name, name, values.read(current)) == value`.

**B. The fix gives the type regression a warrant it did not have.** Record holds
`software: ["SPM2"]` with a verified span; proposal `"SPM2"`; `shape` returns the scalar (D5
above); `str(["SPM2"]) != str("SPM2")`, so the no-op guard does not fire;
`refuses_losing_the_warrant` allows it because the span contains "SPM2"; `_inherited` then
hands the new value the old evidence and `reported`. Result:

    software: value='SPM2'  src=reported  evidence=present     <- invalid type

Before the fix this field was `generated`/`not_found` and showed up in M1 and M2. After the
fix it is invisible to both. **Land the `shape` multiplicity fix before or with this one**,
or the D1 fix makes four validation errors harder to see rather than fixing anything.

**C. This is the one most likely to make your fix wrong.** `medical_condition` is the single
destroyed field on 18823721 whose *value changed*:

    ['DSM-IV heroin dependence']  ->  'heroin dependence'

`refuses_truncation` does not fire because `old` is a list, not a str.
`refuses_shortening_a_list` does not fire because it requires `len(old) > 1`.
`refuses_losing_the_warrant` allows it because the span contains the new value. So
`_inherited` hands the shortened value the old span and `reported`:

    medical_condition: value='heroin dependence'  src=reported  evidence=present

Before the fix: M1 -1, M2 +1 -- the damage was visible. After the fix: both gates green, the
record strictly worse, and the field is *also* still a bare string in a list slot. **The fix
converts a measured regression into an unmeasured one, and both proposed hard gates score it
as an improvement.** Two guard changes close it:

* `refuses_truncation`: unwrap a single-element list before comparing. Then
  `_bare("heroin dependence") in _bare("DSM-IV heroin dependence")` fires the refusal.
* `refuses_shortening_a_list`: `len(old) > 1` -> `len(old) >= 1`, so replacing a list with a
  scalar is refused whatever its length. (This overlaps with the `shape` fix and is worth
  having anyway: it is the guard that states the rule.)

The suite is green on all three: `824 passed, 17 skipped` on beast at 2a7638c. A, B and C
are all uncovered. `tests/test_edit_warrant.py` tests the case where the fix works.

**D. `_inherited` and `refuses_losing_the_warrant` are the same predicate written twice.**
Same loop, same `_bare` containment test, in two functions that must never disagree -- if
they drift, an edit refused by one can inherit through the other. Define the guard in terms
of the helper: `return None if _inherited(edit.current, edit.value) else Refusal(...)`.

**E. A no-op is not a refusal.** "already recorded with this value" goes into `log.refused`,
which is documented as "Why a write did not happen, in terms a reviewer can act on". It
inflates `refused` (76 on 18823721 already) with entries nobody can act on and mixes
"declined a bad write" with "had nothing to do". A separate counter costs one field on
`EditLog`.

## Attack on the metrics

### M1 (span delta) as primary gate: no

The doc worries M1 is gameable by writing nothing. The worse problem is the other direction.
Here are all 10 gained spans on 18823721:

    analyses[6].coordinate_space   (relocated citation)
    groups[1].exclusion_criteria   (relocated citation)
    groups[1].medications          <- wrong value, see D4
    assessments[2].name  assessments[3].name
    tables[0].caption  tables[0].title  tables[0].description
    tables[1].caption  tables[1].title

Seven of ten are table captions and titles -- strings that are long, verbatim, and trivially
locatable, because they *are* the text being searched. Two are relocations of citations that
already existed. Exactly one is a new fact, and it is wrong. A pass that copies table
captions into `title`, `caption` and `description` scores +N on M1 for free. Netting a
caption copy against a destroyed age citation is not a measurement.

### Counter-proposal

**R1 (gate, replaces M1 and M2). Warrant preservation, per field, not netted.**
A violation is a field where either:
  a. the value is unchanged (compared after round-tripping both sides through
     `values.shape`) and evidence went `present` -> not-present; or
  b. `value_source` went `reported` -> `generated` while evidence stayed `present`.
Gate: **0 violations on every record.** Not a sum, so nothing can offset it. Today: 22 on
18823721 (see R2 for why not 26).

**R2. M2 must not count corrections.** 4 of the 26 downgrades on 18823721 are fields whose
pre-state was `reported` with evidence NOT present -- a dishonest pairing `build` produced,
which repair corrected to `generated`. `M2 == 0` as written forbids the honesty fix that
`_wrap`'s own docstring exists to make. Count a downgrade only when the pre-evidence was
`present`. That answers open question 2: gate, but only on the restricted definition.

**R3. Span delta becomes a reported triple, not a gate:** (destroyed-on-unchanged-values,
destroyed-on-changed-values, gained). The first is R1 and gates at 0. The second is allowed
but each instance should be nameable. The third is informational and should be split by slot
so caption-copying is visible as caption-copying.

**R4 (new gate, the hole). Reference-slot accounting.** Count reference writes separately and
score them against ground truth; today they are 6 writes and >=4 errors on 18823721 and
**zero of them appear in any proposed measure**. One deterministic guard is cheap and would
have caught 4 of the 6: refuse a reference write when the identical target list is being
written to more than one entity of the same class in one sweep. "The same four instruments
diagnosed both groups" and "the same region corrected both contrasts" are the shape of the
error, and it is detectable without reading the paper.

**R5 (replaces M4). Four verdicts, two headline numbers.** Verdicts: correct / wrong /
**invented** / unverifiable, where *invented* means a value was written where the ground
truth says `"support": "absent"` with `value: null`. Headlines:
  * **damage rate** = (wrong + invented) / writes that changed or created a value.
  * **yield** = fields that went absent -> a correct value.
Precision over all writes is meaningless because most writes re-write what was already
there. `also_acceptable` in the truth files is what stops the scorer manufacturing false
"wrong" on ambiguities the paper genuinely leaves open.

**R6. M5 without ground truth is a fabrication counter.** `filled` is 31 on 18823721 and
includes both `diagnostic_instrument` writes. Report it, never gate on it, and never quote
it as evidence of value.

**R7. Answering open question 3.** Repair changes or creates a value on roughly 10-25 fields
per paper (18823721: 3 changed + ~29 new wrappers with a value). To separate a damage rate
of 40% from one of 20% at any useful power you need ~80-100 changed-value writes per arm,
i.e. **8-10 papers**. But R1/R3/R4 are per-field deterministic and a single paper can fail
them, so gate on every paper available and reserve the 8-10 for the damage rate.

**R8. Answering open question 4 (is D4 catchable deterministically?).** Partly, and not by
grounding. Three cheap refusals, each of which catches a case in the ground-truth set:
  * *span sharing.* Refuse a write to `entity_B.slot` whose only warrant is the span already
    cited for `entity_A.slot`. Catches 21118656, where one sentence carries three groups'
    means and SDs, and 16038771, where the only age in the paper (28.2) belongs to the 19
    participants who were students, across both groups.
  * *identical reference lists* (R4).
  * *negation proximity.* `haloperidol` sits inside "One patient was **excluded** ... because
    of haloperidol use". A value whose span contains an exclusion cue within a clause of it
    is the specific shape of this failure and is worth a refusal even at some false-positive
    cost, because the alternative is a record that says the analysed cohort took an
    antipsychotic.
None of the three needs a model. What is left after them does need the ground truth.

### On the measurement harness (`scripts/repair_delta.py`)

Its arithmetic reproduces exactly (I re-derived -16 / 26 / 10 / 26 / 31 without importing
pondie). Three changes:

1. **Key by `local_id`, not by list index.** `values.iter_fields` emits `groups[0].age_mean`.
   `create` appends, so order is stable today and the metric is correct today. The day
   anything sorts, dedupes or removes an entity, every path after it shifts and the diff
   reports a wall of destroyed-and-gained that is entirely artefact. Fall back to index and
   report how many paths matched by index only.
2. Walk reference slots too (R4). `iter_fields` structurally cannot see them.
3. Restrict the M2 count as in R2, and stop gating on the netted M1 as in R1/R3.

## Ground truth

`benchmarks/repair_truth/{18823721,11058476,16038771,21118656}.json` -- 4 papers, 196 fields,
every one carrying a verbatim quote. `benchmarks/repair_truth/verify_quotes.py` checks that
every quote appears in the article text; it passes 196/196. Run it after any edit.

Method and judgement calls:

* **Read from the article only.** No pondie record was opened for any value. I did look at
  *which fields* repair touched, to choose the field set; I did not look at what it wrote
  until the values were recorded.
* **Papers chosen for scorability, not cohort.** The brief said to draw from cue_reactivity;
  I did not, because the cue_reactivity repair run (`pondie-cue-flex`) wrote nothing on all
  30 papers, so ground truth there would score an empty pass. 18823721, 11058476, 16038771
  and 21118656 are the four papers in the whole data directory where the current code
  actually repaired something.
* **`support: stated | inferred | absent`.** A third category the doc's field set lacks and
  needs. `model_family: general_linear_model` is right on 16038771 (the paper says "based on
  the general linear model (GLM) approach") and right-but-unstated on 18823721 ("general
  linear" occurs 0 times) and **wrong** on 11058476, which fits beta distributions by
  nonlinear regression and never mentions SPM. A pass should be free to write an inferred
  value; it must mark it `generated`. Scoring that cannot tell the three apart will either
  punish correct inference or reward invention.
* **`also_acceptable`.** Where the paper genuinely admits more than one reading -- 44.5 (the
  analysed 12) vs 42.8 (the enrolled 16) for `age_mean` on 18823721, seconds vs milliseconds
  for `echo_time_seconds` -- the alternatives are listed. Without this the truth set
  manufactures disagreements that are the scorer's fault, not the pass's.
* **`must_not_exist`.** 21118656 is a structural VBM study with no in-scanner task; "fMRI"
  occurs 0 times. Any Task, `response_mode`, `instructions` or fMRI Acquisition on that
  record is invented, and no field-by-field comparison would report it, because there is no
  field to compare against.
* **Fields added to the doc's set:** `enrolled_count`, `excluded_count`,
  `diagnostic_instrument`, `correction_regions`, `height_threshold_type/value`,
  `repetition_time_seconds`, `echo_time_seconds`, `smoothing_fwhm_mm`, `coordinate_space`,
  `hrf_model`, `spatial_unit`, Device `manufacturer`/`model`. The two reference slots
  (`diagnostic_instrument`, `correction_regions`) are the point: they are where the errors
  are and where no current measure looks.
* **53 of the 196 fields are `null`.** That is deliberate. A pass that fills them is
  inventing, and a metric that only scores filled fields cannot see it.
* **Traps recorded on purpose,** each a grounded-but-wrong value a locator will happily cite:
  - 18823721: `haloperidol` (the patient it names was excluded); `connectivity_seeds` on a
    table whose paper never says "seed"; the STN as a correction region.
  - 11058476: `general_linear_model` for a nonlinear beta-fit; `benzocaine` (a film prop) as
    a medication; Talairach/MNI as `coordinate_space` when the paper names the space only by
    citation; the working-memory instruction sentence attached to the film task.
  - 16038771: `age_mean = 28.2`, which belongs to the 19 student participants across both
    groups; `TA = 100 ms` read as TR; `Bonferroni`, which in this paper corrects the
    behavioural post-hocs and not the imaging; `EPSON EMP-7250` (the projector) as the
    device.
  - 21118656: `SD 7.7`, which one sentence supplies for three groups; `Hommel`, which
    corrects the behavioural post-hocs; Talairach (FreeSurfer stream) vs MNI (VBM stream) as
    one `coordinate_space`; `fMRI` on a study that has none.

## The single thing most likely to make the fix wrong

Attack C. `_inherited` gives a *changed* value the old value's span and provenance whenever
the old span happens to contain the new value as a substring. That is the same predicate the
guard uses to *allow* the edit, so every edit the guard lets through on a truncation now
comes out `reported` + `present`. On 18823721 that turns the one genuine value regression --
`['DSM-IV heroin dependence']` -> `'heroin dependence'` -- from a field that failed both
proposed gates into a field that passes both. Add the two guard changes under C before
measuring anything, or the harness will report the fix as a clean win on exactly the record
where it made a real fact worse.

## Appendix: reproducing attacks A, B and C

    ssh beast-proxy 'cd /home/james/pondie && .venv/bin/python - <<"PY"
    from pondie.schema import reader
    from pondie.extraction.record.validate import EXTRACTION_SCHEMA
    from pondie.extraction.record import edit as E
    from pondie.extraction.record import spans as st

    sch = reader.load(EXTRACTION_SCHEMA)
    TEXT = ("and 17 healthy controls (mean age = 40.0 years, S.D. = 10.1, range = 23-54). "
            "Inclusion criteria were (d) diagnosed with DSM-IV heroin dependence. "
            "Imaging data were analyzed using SPM2 (Statistical Parametric Mapping).")

    def wrapper(value, quote):
        span = st.resolve(TEXT, quote).as_record()
        return {"extraction_status": "extracted", "value": value, "value_source": "reported",
                "evidence": {"status": "present",
                             "sets": [{"source": "extraction", "spans": [span]}]}}

    def run(name, cls, container, entity, proposal):
        E.apply(sch, {container: [entity]}, cls, entity, proposal, TEXT)
        for slot in proposal:
            n = entity.get(slot)
            if isinstance(n, dict):
                print(name, slot, repr(n.get("value")), n.get("value_source"),
                      (n.get("evidence") or {}).get("status"))

    run("A", "Group", "groups",
        {"local_id": "g", "age_mean": wrapper(40, "mean age = 40.0 years")},
        {"local_id": "g", "age_mean": "40"})
    run("B", "Preprocessing", "preprocessings",
        {"local_id": "p", "software": wrapper(["SPM2"], "Imaging data were analyzed using SPM2")},
        {"local_id": "p", "software": "SPM2"})
    run("C", "Group", "groups",
        {"local_id": "g2", "medical_condition":
            wrapper(["DSM-IV heroin dependence"], "diagnosed with DSM-IV heroin dependence")},
        {"local_id": "g2", "medical_condition": "heroin dependence"})
    PY'

Output at 2a7638c:

    A age_mean 40.0 reported present
    B software 'SPM2' reported present
    C medical_condition 'heroin dependence' reported present

B and C should be refusals. All three should be tests.
