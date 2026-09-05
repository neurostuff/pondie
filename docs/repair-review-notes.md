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

---

# Round 2: review of 8e22ad2, and R4 implemented

## Attacks on the new code

A, B and C are closed. Five new probes; three land.

**F. Wrong numbers can still inherit a warrant — the F2 case.** `_inherited` tests
`_bare(value) in _bare(span_text)`, which for a short value is substring matching on
digits. Verified against the committed code:

    acquired_count: 12 -> 1     span "consisted of 12 opioid-dependent patients"
    result: written, value=1, src=reported, evidence=present

"1" is inside "12", so the corrupted count inherits the correct value's citation and comes
out `reported` + `present`. It passes R1, R2 and R3 and the record now says one patient. The
float case survives by luck (`_bare(39.0)` is "390", which the span happens not to contain);
every integer and every one- or two-character enum token is exposed.

**G. The `is_multivalued` fix disabled `refuses_shortening_a_list`.** The guard reads
`not isinstance(edit.value, list)`. `shape` now returns a list for these slots, so the
proposal is always a list and the guard can never fire:

    software: ["SPM2", "FSL"] -> ["SPM2"]
    result: written, FSL dropped, src=reported, evidence=present

That is 16701903's two echo times, the case the guard's own docstring was written for, and
`is_multivalued` newly makes `MRI.echo_time_seconds` and `Preprocessing.smoothing_fwhm_mm`
multivalued under the extraction schema — so the guard was switched off precisely on the
slots it was for. Compare lengths, not shapes: `len(new) < len(old)`.

**H. A legitimate list extension is refused.** `["SPM2"] -> ["SPM2", "FSL"]`, both named in
the paper, is refused as "loses the span that warranted the value it replaces", because
`_bare(["SPM2","FSL"])` is "spm2fsl" and no span contains that. So on a grounded list slot
the pass can now *shorten* a list and cannot *extend* one. Both come from `_bare` being
applied to a list's `repr`.

**One change fixes F, G's laundering half and H: make `_inherited` match by value shape.**

    list    every element must match
    number  numerically equal to some numeric token in the span text
    string  `_bare` containment at >= 4 characters, word-boundary match below that

Measured on every destroyed field in all four repaired runs available — **218 fields over 14
papers**: `_bare` containment inherits 217, the shape-aware match inherits 216. The single
difference is `spatial_scope: 'roi'` on 14667419, where `_bare` found "roi" inside a longer
word. So the tightening costs **nothing real** and closes F and H. It does not close G;
`refuses_shortening_a_list` has to compare lengths.

**Is `Schema.is_multivalued` right?** Yes, and I checked rather than reasoned. Over every
class in both schemas: **0 disagreements** between extraction and storage after the change,
**0 slots with `any_of`** (so the multi-range case does not arise), and 23 extraction slots
where it correctly overrides the raw flag. The storage path is unchanged because
`attribute.multivalued` short-circuits first.

**But the fix is half-landed.** `recall.template_for` (recall.py:235) still reads
`slot.multivalued` raw, so the proposer template offers `"medications": "verbatim-string"`
rather than a list on all 23 slots. The pass cannot propose two medications, two software
packages or two echo times; `shape` now wraps whatever single value it gets. Same raw read
at builder.py:70/75/595. `is_multivalued` should be used there too, or list slots stay
capped at one element by construction.

**Does `_same` silently suppress a real edit?** Not in the reachable paths, and I looked for
the `None == None` case specifically: `apply` refuses a proposal whose `shape` is `None`
before `_same` runs, so the new side is never `None`, and `None == value` is False. `shape`
is idempotent on every branch (`float`, `int`, enum, list-wrap), so double-shaping the
stored side is safe. The one collapse it does perform is deliberate: in an integer slot
`12` and `12.7` are one value, and declining to rewrite one with the other is right.

## R4 implemented

`scripts/repair_references.py` (mine; `repair_delta.py` untouched). Reference accounting,
the shared-target signal, and truth-based scoring of the reference slots.

**Your instinct to make the shared-target rule a refusal in `edit.py` is wrong as a blanket
rule, and the data says so.** Over `runs/repair-baseline`, 12 papers, the pass made **15
shared-target writes and all 15 are correct**:

    11296095 analyses.assessments -> [asm_scid]              6 analyses, one interview
    12860777 analyses.tasks -> [tsk_alcoholic_beverage_...]  2 analyses, one task
    14667419 analyses.tasks -> [tsk_alcohol_cue_reactivity]  3 analyses, one task
    14667419 analyses.tables -> [tbl1, tbl2, tbl4]           2 analyses, three tables
    14679386 model_estimations.preprocessing -> [prp_fmri]   2 models, one stream

A paper with one task and six contrasts is the normal case. Refusing on the pattern would
have blocked every one of those and caught nothing. What discriminates is the **slot**: the
script therefore splits shared-target writes into `EXCLUSIVE` (gated), `SUSPECT` (reported)
and normal (reported). `EXCLUSIVE` holds one slot today, `groups.diagnostic_instrument`,
because the schema describes it as the assessment that established *this* group's condition.
On 18823721 that gate fires on both groups; on the 12 baseline papers it fires zero times.

So: **put it in `edit.py`, for `EXCLUSIVE` slots only.** It cannot be a `GUARDS` entry — a
`Check` sees one edit and this is a property of the sweep, and it is the *second* write that
must be refused. `_sweep` already iterates one class at a time; the home is a per-class set
of `(slot, tuple(sorted(targets)))` already written, passed to `apply`. `--explain` prints
this argument with the counts.

R4 results:

| run | ref slots changed | targets + | shared-target | exclusive | suspect |
|---|---|---|---|---|---|
| 18823721 (pre-fix) | 6 | 12 | 4 | **2** | **2** |
| repair-baseline (12) | 27 | 44 | 15 | 0 | 0 |
| repair-fixed (6 done) | 15 | 16 | 8 | 0 | 0 |

## R5 implemented, and the A/B result you will not like

`scripts/repair_score.py`. Five verdicts — I added **`inferred`** between correct and
invented, for a field the paper is silent on filled with a value the truth lists as
defensible (`correction_scope: whole_brain` on a whole-brain acquisition). Scoring that as
invention punishes the pass for being right; scoring it as correct hides that it must be
stamped `generated`.

Pre-fix, on the four truth papers:

| pmid | scored | correct | inferred | wrong | invented | missed | changed | damage | yield |
|---|---|---|---|---|---|---|---|---|---|
| 18823721 | 55 | 40 | 1 | 0 | 9 | 5 | 13 | 69% | 4 |
| 11058476 | 50 | 37 | 0 | 6 | 1 | 6 | 3 | 67% | 1 |
| 16038771 | 44 | 38 | 0 | 0 | 2 | 4 | 6 | 33% | 4 |
| 21118656 | 52 | 36 | 1 | 8 | 5 | 2 | 2 | 50% | 1 |

`wrong` counts the record's state, `changed` counts what this pass did, and only the
intersection is repair's fault — most of 21118656's 8 wrongs are the extractor's. Damage
rate is computed on `changed` only.

**The A/B on 11058476, the one truth paper the fixed run has reached, is identical:**
correct 37, wrong 6, invented 1, changed 3, damage 67%, yield 1 — in both arms. The warrant
fix changes how writes are *stamped*, not what they *say*. It makes the pass non-destructive;
it does not make it net positive.

The deterministic half of the same A/B, on the 6 papers `runs/repair-fixed` has finished:

| | destroyed | M2 | M3 | M5 filled |
|---|---|---|---|---|
| baseline (same 6) | 92 | 59 | 29 | 156 |
| fixed | **1** | **0** | 10 | **156** |

Yield is bit-identical, which is the answer to "did the fix cost anything": no. And the M3
residue has changed *class* — 12 kinds of introduced finding became 2, and every list-type
one is gone:

    baseline  tables[].non_analysis_content 10, medical_condition 7, inclusion_criteria 6,
              exclusion_criteria 5, response_mode 5, performance_measures 5, software 5,
              medications 2, +4 more
    fixed     tables[].non_analysis_content 3, tasks[].conditions 1

Everything `is_multivalued` was meant to fix is fixed. What is left is the misattribution
class -- a table declared non-analysis while an analysis names it -- which is D4, which is
what R4 and R5 exist for.

## Q1 and Q3, with numbers

**Q1 — the M5 floor that stops a do-nothing pass.** Do not pick a floor; state the
break-even, because the data now supplies both sides. Over the four truth papers the pass
made **24 changed-value writes**, of which 10 were correct or defensibly inferred and **14
were wrong or invented**. Per paper: **yield 2.5, damage 3.5.** A pass that writes nothing
scores 0 and 0, so on this evidence **the do-nothing pass is ahead**. The honest gate is

    R1 == 0  and  R3 == 0  and  R4(exclusive) == 0  and  yield >= wrong + invented

per record, which is one stated trade-off rather than two thresholds pulled from the air. On
the truth papers repair fails the fourth clause today on three of four (4 vs 9, 1 vs 7, 4 vs
2 — 16038771 is the one that passes).

**Q3 — how many papers.** Measured, not guessed: repair changes or creates **6 values per
paper** within the truth field set (13, 3, 6, 2). Separating a damage rate of 60% from 30%
at 80% power and α = 0.05 needs ~43 changed writes per arm, so **8 papers per arm** at the
current 50-field truth set. Two caveats. The set covers ~50 of the 120-240 fields in a
record, so it sees roughly a quarter of what the pass touches; widening the field set on the
same 8 papers buys the same power more cheaply than adding papers. And the four papers here
range from 33% to 69% damage, so a 4-paper read cannot distinguish a real improvement from
which papers you picked.

## What I still need from you

* `runs/repair-fixed` covers 1 of my 4 truth papers. For a real R5 A/B, run the fix over
  **18823721, 16038771 and 21118656** as well -- those are where the truth set is densest
  and where three of the four known failure cases live.
* The three code changes above: shape-aware `_inherited`, length-based
  `refuses_shortening_a_list`, and `is_multivalued` in `recall.template_for`.

---

# Round 3: review of ec3db6d, and the `EXCLUSIVE` refusal

## The warrant test, measured over the whole corpus

Not on the four papers -- on every grounded value in every pre-repair record available,
**7,664 of them**, old `_bare` containment against the committed `_warrants`:

| value kind | n | old `_bare` | new | old only |
|---|---|---|---|---|
| bool | 135 | 22 | **0** | 22 |
| list | 962 | 447 | **555** | 4 |
| number | 1098 | 923 | 867 | 56 |
| short string (<4) | 583 | 449 | 360 | 89 |
| string | 4886 | 3022 | 3022 | 0 |
| **total** | **7664** | **4863** | **4804** | **171** |

171 warrant claims removed, 112 added, and the composition is the whole argument. The
additions are all in `list`, which is H. The removals are concentrated exactly where the
false positives were, and I checked rather than assumed:

* **All 22 boolean matches were false.** `is_healthy: False` matched "corrected for multiple
  comparisons using the **false** discovery rate"; `is_healthy: True` matched "This was
  **true** even if there was a similar...". Every one is a discussion sentence or a method
  sentence that says nothing about the cohort. The blanket `False` for booleans is right,
  and it costs nothing.
* The 56 numbers and 89 short strings are the `12`-contains-`1` and `heroin`-contains-`roi`
  classes.
* Strings of four characters or more are untouched, which is why nothing real moved.

**One correction: the commit says 216 of 218; the committed code gives 215.** The third loss
is `education_summary` on 21118656, a pipeline-written summary sentence longer than the span
it cites. All three losses are cases where the harness re-proposes the *identical* value,
which `_same` now intercepts before `_inherited` is reached, so the production consequence is
still zero -- but the number in the message should be 215.

## Attacks on F/G/H

F, G and H are closed; I re-ran all three against the committed code and they behave as
advertised. Four probes at the new edges; one lands.

**Your word-boundary question: it reintroduces nothing, and I looked for it specifically.**
Case is handled (`text.lower()` both sides). Punctuation and unicode dashes never reach the
boundary path, because anything with a dash in it is four or more bare characters and takes
the `_bare` route -- `DSM-IV`, `MPRage`, `3-D` all still match. The one hazard I could
construct is this corpus's own tokenisation: it prints `T 1 -weighted`, `T 2 *`, `3 T`, so
`\bt1\b` cannot match `T 1`. **But it does not occur.** Of the 583 grounded short values in
the corpus, allowing interior whitespace (`\bt\s*1\b`) recovers **zero** additional ones. The
223 the boundary rejects are vocabulary tokens the extractor derived rather than quoted --
`roi`, `glm`, `TAL` -- where the paper writes "region of interest", "general linear model",
"Talairach". `_bare` did not match those either. So there is nothing to fix here.

**K2 (lands). A partial list extension is refused wholesale.** `["SPM2"] -> ["SPM2",
"MarsBaR"]`, where MarsBaR is in the paper but not in *this field's* span, is refused as
"loses the span that warranted the value it replaces". `_inherited` requires every element to
be warranted by the existing spans, and `refuses_losing_the_warrant` refuses when it is not.

This matters more now than it did an hour ago, because you have just made the proposer able
to return lists (`is_multivalued` in `template_for`). Most extensions it proposes will name a
second value from a second sentence, and this refuses all of them -- so the template fix
cannot show up as yield. The edit is a strict superset of a warranted value, which is not a
loss of anything: it should be written with the old spans kept and `value_source` dropped to
`generated`. `refuses_shortening_a_list` and `refuses_truncation` already cover the case
where content is actually removed, so `refuses_losing_the_warrant` does not need to.

**Not defects, but caps worth knowing.** A unit-converted numeric can never inherit --
`echo_time_seconds: 0.028` against "TE = 28 ms" compares 0.028 to 28. And a grounded boolean
can never be corrected, only left alone; `_same` keeps the no-op case safe, so this costs
nothing today.

**builder.py:70/75/595 -- you are right, I withdraw it.** Those read the attribute on a
container or a nested/reference list, where multiplicity is on the attribute in both schemas
and no `Extracted*` wrapper intervenes. `is_multivalued` would return the same answer more
slowly. The one that mattered was `template_for`, and that is landed.

## `EXCLUSIVE` implemented

In `edit.py` and `repair._sweep`, as specified.

* `EXCLUSIVE_REFERENCES: frozenset[tuple[str, str]]` holds `("Group",
  "diagnostic_instrument")` and nothing else, with the fifteen-correct-writes evidence in
  the comment so the next reader does not generalise it.
* `apply` takes `claimed`, a caller's dict keyed `(class_name, slot, sorted added targets)`
  and valued with the `local_id` that took them first. `_refuses_a_claimed_target` refuses a
  later, different entity. It is not a `Check` and cannot be one: a `Check` judges one edit
  against the record, this judges an edit against what the same sweep already wrote, and the
  first write is legitimate.
* `_sweep` builds one `claimed` per class sweep and threads it through. Across the two
  iterations nothing double-fires, because pass two sees pass one's write as `existing` and
  `added` is then empty.
* Keyed on what is **added**, not on the whole slot: an entity that already held a target is
  not claiming it again.
* Two tests: the 18823721 shape is refused, and two model estimations sharing one
  preprocessing are still written.

`831 passed, 17 skipped` on a clean copy of the branch (829 plus the two new).

## Q4: the smallest change with the best effect on yield-vs-damage

Measured, not argued. Every value repair changed or created on the four truth papers,
cross-tabulated against whether the pass could ground it. **21 changed writes, of which 18
are ungrounded, and 72% of those are wrong or invented:**

| | correct | inferred | wrong | invented | total | bad |
|---|---|---|---|---|---|---|
| grounded | 2 | 0 | 0 | 1 | 3 | 33% |
| ungrounded | 4 | 1 | 2 | 11 | 18 | **72%** |

Restricting to value slots -- reference slots are the `EXCLUSIVE` guard's job now -- there
are 16 changed writes, 7 good and 9 bad, and two candidate rules:

| rule | kept (good/bad) | refused (good/bad) | yield | damage | passes `yield >= damage` |
|---|---|---|---|---|---|
| today | 16 (7/9) | 0 | 7 | 9 | no |
| **A** refuse every ungrounded write | 3 (2/1) | 13 (5/8) | 2 | 1 | yes |
| **B** refuse an ungrounded write whose value is not in the document | 14 (7/7) | 2 (0/2) | 7 | 7 | exactly |

**A is the lever and it is cheaper than D3 feared.** Your D3 warning -- "any rule of the form
drop what cannot be grounded would discard correct data" -- is right in kind and wrong in
size. It costs exactly five values across four papers, and all five are recognisable:

    SPM2 / SPM99 / SPM5   preprocessings/model_estimations.software -- named verbatim in
                          every one of those papers; the locator simply missed them
    excluded_count = 7    24 enrolled minus 17 analysed, both already in the record
    correction_scope      whole_brain, inferred from a whole-brain acquisition

The first two have obvious carve-outs -- allow an ungrounded write whose value occurs in the
document, and allow a count the record's own arithmetic produces -- and with those A costs
one value and removes eight errors. **That is the smallest change with the best effect: A
plus the document-presence carve-out.** B on its own is nearly free and nearly useless: it
removes 2 of 9, because most of the wrong values *are* in the document.

**What neither rule touches is the residue, and it is all D4.** Every one is a correct value
in the wrong place:

    haloperidol                          a patient who was excluded
    no current psychotropic medication   a criterion, and the other group's too
    age_mean 28.2                        the 19 students, across both groups
    enrolled_count 17                    the analysed count in the enrolled slot
    beta distribution as hrf_model       a model that has no HRF

The rule I would expect to catch these is the span-sharing refusal (R8i): refuse a write
whose warrant is a sentence already cited by another field -- the same slot on a sibling
entity (28.2, medications) or a different slot on the same entity (enrolled vs acquired).
**I have not measured it**, because only 3 of the 16 writes are grounded and the candidate
span of an ungrounded write cannot be reconstructed after the fact. It needs instrumenting at
write time, and I would do that before building it.

## Still outstanding

* **`runs/repair-final` has not started** (0 of 15). Task 3 -- R5 damage and yield, fixed
  versus baseline, on all four truth papers -- is not done and I will not estimate it. Say
  the word when it lands and I will run `repair_delta`, `repair_references` and
  `repair_score` across both arms.
* `refuses_losing_the_warrant` should allow a strict superset (K2 above).
* The 216 in ec3db6d's message should be 215.

---

# Round 4: the final measurement

`runs/repair-final` (15) against `runs/repair-baseline` (12) and
`runs/pondie-prose-18823721`. Everything below is measured with one build of the code, on
the artifacts as they stand.

## First: what is actually a controlled comparison

**13 of the 15 papers start from a byte-identical pre-repair record. Two do not.**
`repair-final/unrepaired/16038771` and `.../21118656` differ from the copies in
`pondie-newtpl` and `pondie-21118656` (55,216 vs 59,979 and 68,659 vs 69,114 bytes), so any
delta on those two is confounded by the extractor and is not attributable to repair. That
leaves **two** truth papers in a controlled A/B -- 18823721 and 11058476 -- and 13 papers for
the deterministic half.

## Deterministic half, on the 13 controlled papers

| | pre | post |
|---|---|---|
| spans destroyed | 227 | **2** |
| provenance downgrades (R1/M2) | 141 | **0** |
| findings introduced (M3) | 54 | **1** |
| fields filled (M5) | 316 | 257 |
| papers failing a deterministic gate | 11 of 13 | **1 of 13** |

That is the result. Three corrections to the numbers in your message.

**M3 is 1, not 0.** The survivor is `Study.tasks[].conditions: 'conditions' is multivalued
but got dict` on 12860777 -- the nested-slot bug f8f0998 fixes. f8f0998 fixes the *code*; the
run predates it, and the record still carries the finding. Re-measuring with current code
still reports it, because the defect is in the artifact. **M3 == 0 is a claim about code that
no run has yet demonstrated.** Re-run 12860777 and it is earned.

**The baseline M3 is 49, not 61.** 61 was measured before `repaired_by` was declared on
`ExtractionMetadata`; with the current validator the same baseline records score 49. Quoting
61 -> 0 compares two validators. The honest pair over the 13 controlled papers is 54 -> 1.

**The fill drop is 59, not 16.** 285 -> 269 compares twelve papers to fifteen. On the 13
controlled papers it is 316 -> 257: 67 fills given up, 8 newly gained, net **-59 (-19%)**.

## R4

The gate `exclusive_shared == 0` passes on all 15, and the guard did work: on 18823721
`grp_controls.diagnostic_instrument` is gone, taking that paper's invented reference writes
from 4 to 3.

**But 21118656 exposes a hole in the guard, and it is mine.** Three groups took *overlapping
subsets* of the same three interviews:

    grp_ptsd                     [CAPS, MINI, vivo Checklist]
    grp_traumatized_controls     [CAPS, MINI]
    grp_nontraumatized_controls  [MINI]

Keyed on the whole target tuple, those are three different lists, so nothing collided and all
three were written -- two of them onto control groups that have no condition for an
instrument to have established. The copy does not arrive as a copy. Fixed here: the claim is
made **per target**, and the part of a later write that is still free is written rather than
the whole write refused, so an unlucky sweep order cannot cost a correct link. Two tests.
**The effect on 21118656 is predicted, not measured** -- it needs a re-run.

## R5, and the gate clause

| paper | controlled | pre changed / bad / yield | post changed / bad / yield | gate pre | gate post |
|---|---|---|---|---|---|
| 18823721 | yes | 13 / 9 / 4 | 9 / **5** / 4 | fail | fail |
| 11058476 | yes | 3 / 2 / 1 | 5 / 3 / 2 | fail | fail |
| 16038771 | no | 6 / 2 / 4 | 4 / 3 / 1 | pass | fail |
| 21118656 | no | 2 / 1 / 1 | 1 / 1 / 0 | pass | fail |

**`yield >= wrong + invented` fails on all four papers in the post arm.** Plainly: repair is
not yet net positive by the gate we agreed.

The two rows that mean anything are the controlled ones, and only 18823721 has enough writes
to carry an inference. There it is real progress: **bad writes 9 -> 5 with yield unchanged at
4, and correct fields 40 -> 44 while filling ten fewer.** It fails 4 < 5 by one write. On
11058476 both sides rose by one. The two uncontrolled rows moved from pass to fail, and I
will not attribute that to the fix, because their starting records changed.

## Q2: did Rule A cost anything real?

139 refusals reading "nothing in the paper places this value", across 13 papers. Against the
truth set:

* **18823721 -- all five adjudicable refusals are right.** Three `non_analysis_content`
  (including `connectivity_seeds`, the flagship fabrication) and two `correction_scope: roi`,
  where the truth records the paper as silent and "region of interest", "small volume" and
  word-boundary ROI occur zero times.
* **11058476 -- two `correction_scope`** and **16038771 -- two `is_healthy`**, where the
  truth carries a value with `support: inferred`.

So the measured cost is **at most four values, and all four are inferences rather than
statements**: `whole_brain` reasoned from a whole-brain acquisition, `true` reasoned from the
absence of any clinical claim. Rule A refusing those is the rule working, not failing. **It
refused nothing the paper states.** My round-3 prediction of "one value" was low by three,
and the direction was right.

## Q3: the 59, and whether damage was traded for yield

Yes, materially, and the 16 hid it. The 67 given-up fills by slot:

    18  non_analysis_content        the most-invented slot in the corpus, and the source of
                                    10 of the baseline's 49 introduced findings
    18  free text                   description 9, stimuli 2, model_settings 2,
                                    inclusion_criteria 2, exclusion_criteria 2,
                                    recruitment_method 1
    31  everything else             correction_scope 4, species 2, is_healthy 2,
                                    model_family 2, coordinate_space 2, medications 2, ...

Of the 15 I can adjudicate against the truth set, roughly half are right refusals and half
are values a reader would keep -- and **the losses fall into three recognisable classes**:

    paraphrase        model_estimations.model_settings = "Contrasts: (1) neutral vs.
                      low-level baseline; ..." -- the sentence IS in 18823721, reworded, so
                      the document test misses it. Same for two assessment `description`s
                      and 11058476's `stimuli`.
    unit conversion   acquisitions.acquisition_duration_seconds = 450.0, which the paper
                      reports as "7.5 minutes". 450 is not a token in the document.
    inference         is_healthy, correction_scope: whole_brain (Q2 above).

So no, the 59 is not all inventions. Two of the three classes have cheap carve-outs; the
third is Rule A doing its job.

## Q4: are we done?

**No, but non-destructiveness is done and that is the larger half.** The record repair now
produces destroys no warrant, downgrades no provenance, introduces no type error and (once
the per-target claim lands in a run) duplicates no exclusive reference. Everything the first
diagnosis was about is closed.

What remains is that the pass still writes roughly one wrong-or-invented value per two
changed writes, and the gate fails on every truth paper. Three things, in order of expected
effect per unit of work:

1. **Re-run.** Two of the four claims above are about code that no artifact demonstrates:
   M3 == 0 and the per-target exclusive claim. And a re-run of the *pre* records for
   16038771 and 21118656 would take the controlled truth set from two papers to four, which
   is the difference between a result and an anecdote.
2. **Two carve-outs to Rule A**, both small, both recovering measured losses without
   weakening it: allow a numeric that equals a document token under a unit factor
   (`450 s` <-> `7.5 minutes`), and allow a free-text value -- honestly `generated` -- when a
   majority of its content words occur in the document. Together those address 20 of the 67
   given-up fills, which is the cheapest available movement on the `yield` side of the gate.
3. **The span-sharing refusal for D4**, which is still where the residue lives: `haloperidol`
   on a patient who was excluded, `age_mean 28.2` belonging to the 19 students across both
   groups, `enrolled_count 17` which is the analysed count. **Separate piece of work.** It
   needs the candidate span recorded at write time before it can be measured at all, which is
   an instrumentation change to `apply` plus a run, and I would not fold it into this branch.

**Residual risk if this ships as is:** the record is honest about what it does not know and
no longer destroys what it knew, so a reviewer reading provenance can trust it. What it still
does is state, with a citation, facts that belong to a different entity in the same paper --
and no gate in this document can see that. The four truth papers say that is about one write
in two. Anyone consuming repaired fields at face value should know that number.

---

# Round 5: the controlled measurement, and what it settles

All four truth papers now start from byte-identical pre-repair records; I verified each pair
with `cmp` rather than taking it on report. `repair-pre2` is genuinely pre-fix -- zero
occurrences of "already recorded with this value", "nothing in the paper places this value"
or "drops values the record already held" in its provenance, against 127/28/n in `post2` for
the same papers. **This section supersedes round 4's R5 table, which had two uncontrolled
rows.**

## Deterministic half, 15 controlled papers

| | pre | post |
|---|---|---|
| spans destroyed | 241 | **2** |
| provenance downgrades | 151 | **0** |
| findings introduced | 58 | **0** |
| fields filled | 341 | 268 |
| papers failing a deterministic gate | **13 of 15** | **0 of 15** |

M3 == 0 now has an artifact behind it; in round 4 it did not. Fill drop is **-73 (-21%)** on
the controlled set.

## R5: the controlled A/B

| paper | pre changed / bad / yield | gate | post changed / bad / yield | gate |
|---|---|---|---|---|
| 18823721 | 13 / **9** / 4 | fail | 9 / **5** / 4 | fail |
| 11058476 | 3 / 2 / 1 | fail | 5 / 3 / 2 | fail |
| 16038771 | 6 / 3 / **3** | **pass** | 4 / 3 / **1** | **fail** |
| 21118656 | 0 / 0 / 0 | pass (vacuous) | 1 / 0 / 0 | pass (vacuous) |
| **total** | **22 / 14 / 8** | 2 of 4 | **19 / 11 / 7** | 1 of 4 |

Damage rate **64% -> 58%**. Yield 8 -> 7. `yield >= wrong + invented` is **not passed**, and
the branch does not make it pass.

Paper by paper, because the total hides three different stories:

* **18823721 is the clean win: 9 bad writes -> 5, yield unchanged at 4.** Four inventions
  removed at no cost -- `grp_controls.medications`, `grp_controls.diagnostic_instrument`,
  `inf_stn.correction_scope: roi`, and `tables[tbl2].non_analysis_content:
  connectivity_seeds`, the fabrication this whole review opened on.
* **16038771 got worse, and it is the fix's doing.** Bad unchanged at 3; yield 3 -> 1,
  because Rule A refused two correct `is_healthy: True` writes. Nothing was gained on that
  paper and the gate went from pass to fail. This is exactly the cost I flagged in round 4's
  Q2 as "at most four values, all inferences"; two of the four are now measured, on a
  controlled pair.
* **11058476** does more of both: bad 2 -> 3, yield 1 -> 2. The new bad write is
  `grp_comparison_subjects.enrolled_count = 14`, the same shape as the one already there.
* **21118656** is neutral. Its one changed write replaces
  `['posttraumatic stress disorder', 'major depression']` with `['PTSD', 'major depression']`
  -- a correct value for a correct value.

**A correction to my own ground truth, found while doing this.** That last write scored
`wrong` until I looked at it: my `also_acceptable` list did not contain the abbreviated
spelling, and the scorer matches on normalised substrings, under which "PTSD" and
"posttraumatic stress disorder" do not match. Fixed in `21118656.json`, which moves the post
total from 12 bad to 11 and that paper from fail to pass. A truth set is a measuring
instrument and this one was reading one write short.

## The residual, measured

For every wrong or invented write, I asked whether the value it states occurs in the paper at
all -- the same `_warrants` test the pass uses, against the whole document:

| | bad writes | value IS in the paper | value is not |
|---|---|---|---|
| pre | 14 | 12 | 2 |
| **post** | **11** | **11** | **0** |

**Every remaining wrong or invented value in the repaired records is a value the paper
genuinely contains, attached to the wrong entity or the wrong slot.** The residue is D4 and
nothing else. The two writes that stated something the paper does not contain are precisely
the two Rule A removed. So the headline of this exercise, as a measurement rather than an
estimate:

> On the fields it changes, repair is wrong or invents about **58%** of the time
> (11 of 19 changed writes over four hand-read papers), and **100% of those errors are a real
> fact from the paper put in the wrong place** -- an excluded patient's drug recorded as the
> cohort's medication, a students' mean age recorded as a group's, an analysed count recorded
> as the enrolled count, a result region recorded as a correction region.

No grounding check can catch any of it, because every one of them is grounded.

## Did the per-target `EXCLUSIVE` fix work? No -- and I was wrong to claim it was needed

`post2` has zero "already belongs to" refusals on 21118656 and the three overlapping
`diagnostic_instrument` lists are unchanged. The reason is that **repair never wrote them**:
all three are already in `repair-post2/unrepaired/21118656`, so the extractor wrote them and
the guard correctly did nothing.

Round 4 said "21118656 exposes a hole in the guard, and it is mine". That was wrong. I read
the post record, saw three overlapping lists, and attributed them to the pass without
checking the pre record -- the same mistake in kind as the grep that started this review. The
overlapping-subset hole is real *as code* and the unit tests demonstrate the tuple key would
miss it, but **no run demonstrates it**, and the tuple key already handled the only case
repair actually produces in this corpus.

The guard itself is measured working, on 18823721: `grp_controls.diagnostic_instrument` went
from a four-item copy of the patients' list to absent. One invented reference write removed,
under a controlled comparison.

## Rule A, honestly

On the truth field set Rule A is a **wash: it removed 2 bad writes and cost 2 good ones.**
Its demonstrated value is elsewhere and the field set cannot see it -- 46 of its 139 refusals
are `non_analysis_content`, the slot that produced 10 of the pre arm's 58 introduced
findings, and M3 going to 0 is largely its doing. Judge it on M3, not on M4.

## The next lever, for follow-up rather than now

Two carve-outs to Rule A, both recovering measured losses without weakening it:

* a numeric equal to a document token under a unit factor -- `acquisition_duration_seconds =
  450.0` where the paper says "7.5 minutes";
* a free-text value, honestly `generated`, when a majority of its content words occur in the
  document -- `model_settings`, `description`, `stimuli`, which the pass rewords so the
  verbatim test misses text that is in the paper.

Together they address roughly **20 of the 67 given-up fills**. They will not move the gate on
their own, because the gate fails on the damage side and the damage is all D4.

After those, the span-sharing refusal, which is a separate piece of work: it needs the
candidate span recorded at write time before it can be measured, and every one of the 11
residual errors would be a test case for it.

## Recommendation

**Ship it.**

The branch is a large, controlled, unambiguous win on destructiveness -- 241 destroyed spans
to 2, 151 downgrades to 0, 58 introduced findings to 0, 13 of 15 papers failing a gate to 0 --
and roughly a wash on content: damage 64% to 58%, yield 8 to 7, the gate still failed. The
alternative is leaving in production a pass that provably subtracts from the record, which is
worse than one that is honest and unhelpful.

Two things must go in the summary rather than be discovered later:

1. **`yield >= wrong + invented` is not met**, on any controlled paper with writes to score.
   Repair is non-destructive, not net positive.
2. **58% of what repair changes is wrong, and all of it is grounded.** Anything downstream
   that reads a repaired field at face value should know that number. `repaired_by` in
   `extraction_metadata` is what makes those fields findable.

The one measured regression -- Rule A costing two correct inferences on 16038771 -- is worth
accepting for the M3 collapse it bought, but it should be named, not netted away.

---

# Round 6: links, and template arms for the residue

## Link ground truth, and your prediction falsified

`benchmarks/repair_truth/*.json` now carry a `links` array: for each entity, which targets
the slot may legitimately hold, judged **against the slot's own schema description** rather
than against whether the paper mentions both ends. `verify_quotes.py` checks link quotes too;
209 quotes, all verbatim.

That distinction changed an answer. 16038771 writes `asm_sadomasochistic_preferences` onto
four Table 3 analyses, and read against the paper alone I would have called it wrong -- the
questionnaire assigns group membership, it is not a covariate. Read against the slot --
"Assessments whose measurements were used in **or selected the sample for**this analysis" --
it is right, and Table 3 is panelled `nonSM` / `SM`, so the sample was selected by it. Four
writes flipped from wrong to correct on the strength of one clause.

Scored:

| arm | link writes | writes wrong | targets | targets wrong |
|---|---|---|---|---|
| pre | 14 | 6 (**43%**) | 28 | 20 (**71%**) |
| post | 11 | 3 (**27%**) | 14 | 6 (**43%**) |

**Your prediction -- "the link damage rate is worse than 58%" -- is falsified by write in
both arms and by target in the post arm.** It survives only as pre-arm targets, 71%.

## But the rate is an artifact, and this is the finding

For every scored link I counted how many candidates of the target class the record actually
held. A link to the only entity of its class is not a choice.

| | correct | wrong |
|---|---|---|
| pre: the record held one candidate | **8** | 2 |
| pre: the record held several | **0** | 4 |
| post: the record held one candidate | **8** | 2 |
| post: the record held several | **0** | 1 |

**Every correct link write in both arms is to the only entity of its class in the record.
The pass has never been observed to choose a link correctly.** Three `ModelEstimation.
preprocessing`, one `Task.acquisitions` and four copies of one `Analysis.assessments`
decision -- eight writes, zero choices. And "forced" does not mean safe: the two
`correction_regions <- reg_stn` writes on 18823721 are forced *and* wrong, because the record
held one Region and the right answer was to write nothing.

Counting only the writes that required a judgement -- five choices among candidates, plus
four forced writes whose correct answer was "no link" -- the pass scored **0 of 9 across both
arms**. The aggregate 27% looks respectable because forced links dilute it by 8.

So your prediction was wrong as stated and right underneath it: links are not 58% wrong, they
are 100% wrong wherever a decision is involved, and the schema's forced links are carrying
the average.

## Junk targets: one signal fails, one works, and neither is a link guard

15127179 mints twelve `Assessment` entities -- the unrepaired record has zero -- and four are
not assessments: `statistical parametric mapping`, `STATISTICA`, `Pearson's linear
correlation coefficient`, `Spearman's correlation coefficient`. `asm_statistical_parametric_
mappi` carries `evidence: present`, so grounding does not see it either.

**The signal I expected does not work.** I hypothesised that a junk Assessment's name would
collide with a `software` value elsewhere in the same record. Measured over all 2,019
`Assessment` entities in every run: **5 collisions, and the only non-trivial one is a
substring artefact** -- `asm_r` ("R") inside "matlabr2006a". The paper writes "statistical
parametric mapping" in prose and "SPM99" in the software slot, so they never match. Do not
build it.

**The signal that works is the record telling you itself.** The junk entities announce what
they are in `assessment_type`:

    asm_statistical_parametric_mappi   type='fMRI analysis'
    asm_statistica                     type='regression analysis'
    asm_pearson_s_linear               type='correlation'
    asm_spearman_s_correlation         type='correlation'

against `questionnaire`, `clinical scale`, `diagnostic interview` for the eight real ones.
Across the corpus `assessment_type` is an **open string with 211 distinct values over 2,019
entities, 118 of them singletons**, while five values cover 66%. Roughly 31 entities declare
themselves software or statistics.

My call: **not a link guard.** Refusing the link leaves the junk entity in the record for
something else to link to, and it is the second error, not the first. Two options at the
right place, which is minting:

* **now, cheap:** `create` refuses an `Assessment` whose `assessment_type` is one of about
  eight strings -- `statistical software`, `statistical package`, `fmri analysis`,
  `regression analysis`, `correlation`. Deterministic, ~1.5% of entities, and it is a
  deny-list, which is the brittleness you flagged;
* **durable:** close the vocabulary. The machinery exists -- `values.cast` already refuses a
  value outside an enum's permissible values and `create` already refuses an entity it cannot
  build validly -- and the distribution supports it. The risk is the one `region_type` is
  written for: a closed vocabulary drops a legitimate long tail, so it wants `any_of: [Enum,
  string]` and a rule that the open branch is not a licence for "fMRI analysis".

## Quote-carrying helps links *more* than values, and the cases say why

This is the one place I think structure clearly wins, and it is a stronger case than the
value case -- your point 3 is right and it is worth more than you claimed.

A link's quote is checkable by a rule that needs no schema knowledge: **the quote must name
the target.** Against the three wrong post-arm link writes:

    diagnostic_instrument <- [DDQ, SHAPS, ASI, OCDUS]   needs a sentence saying these four
                                                        established the diagnosis. None
                                                        exists; `spans.verify` kills it.
    correction_regions <- reg_stn  (x2)                 needs a sentence saying the
                                                        correction was restricted to the
                                                        STN. None exists.

All three die. Whereas for *values* the same rule does almost nothing: `haloperidol` has a
real quote naming it, and so does `age_mean 28.2`. For a value the quote only helps through a
second rule -- does the sentence name this entity, is there an exclusion cue in it -- and for
a link the target-naming rule is direct and complete.

## The template arms, and my argument with your ordering

Implemented in `recall.py` behind `PONDIE_TEMPLATE`, a comma-separated set of `described`,
`quoted`, `scoped`. Empty is the shape every measurement in this document was taken with. An
environment variable and not a `Settings` field on purpose: these are arms, and the winner
should become the only shape rather than a fifth setting. 845 tests pass.

**I disagree with the ordering, and with the premise under it.** You wrote that "every one of
your 11 residual errors is a value belonging to a different entity in the same paper". I said
"the wrong entity **or the wrong slot**", and the split matters:

    wrong entity   3 of 11   haloperidol, age_mean 28.2 (x2)
    wrong slot     8 of 11   enrolled vs acquired count (x2), diagnostic_instrument,
                             correction_regions (x2), hrf_model on a model with no HRF,
                             spatial_unit, inference_level

Entity-scoping addresses the minority. So my ordering is:

1. **`described` -- ship the schema's own slot descriptions.** Free, no extra calls, and it
   targets the 8 wrong-slot errors head on. **Seven of the eleven residual errors have their
   fix written in the schema already and never shown to the model:**

       enrolled_count         Number enrolled after screening and before acquisition
       acquired_count         Number for whom data were acquired or who were scanned
       diagnostic_instrument  The study assessment that established this group's defining
                              condition
       correction_regions     The regions correction was restricted to
       hrf_model              The haemodynamic response basis the design matrix was built
                              with
       medications            The drugs or other agents this cohort was taking

   The template says `{"enrolled_count": "integer", "acquired_count": "integer"}` and nothing
   else. `vocabulary` already ships enum descriptions for exactly this reason; this ships the
   slots'. Adds 2,814 characters on `Group`, the largest class, against a premise of tens of
   thousands.
2. **`quoted`** -- for the link argument above, and because it unblocks everything else: the
   span-sharing refusal, Rule A's paraphrase and unit-conversion carve-outs, and a
   negation-cue refusal all need the candidate sentence at write time. Costs a doubled
   template (1,785 -> 3,774 characters on `Group`), which is a real risk for a 3B model and
   is itself worth measuring.
3. **`scoped`** -- the rule that a value belongs to the entity its `local_id` names. Targets
   the remaining 3.
4. **single-slot** -- only if 1-3 do not move it. It is the same information at N times the
   cost, and 1 is the cheap version of the same idea.

**`quoted` does nothing until `edit.py` changes, and `edit.py` is yours.** `recall.unquote`
parks citations under `proposal["_quotes"]`, a key no class declares, so `apply` skips it
today. What it needs:

    apply(...)          quotes = proposal.get(recall.QUOTES) or {}
    the value branch    written = _wrap(value, text, quote=quotes.get(name))
    _wrap(...)          when `quote` is given, resolve it with `spans.resolve`/`verify`
                        against `text` and use that span, rather than searching for the
                        value; keep the 20-character search as the fallback

That is the change that retires the 20-character floor and the failed locator both: the span
becomes the model's own sentence, verified, exactly as `adjudicate` already does it.

## A confound in the Luna arm

`ModelProposer.SHAPE` tells the network model "A value belongs to the entity named by its
`local_id` and to no other. A number stated for a subgroup, an excluded participant or
another cohort is not this entity's value". NuExtract is told none of that. **So the two arms
differ in prompt as well as in model, and a lower Luna damage rate would not separate them.**
That instruction is now `recall.SCOPED`, assembled in `_Proposes.propose` so both arms get
it under `PONDIE_TEMPLATE=scoped` -- run the Luna arm with it, or take the sentence out of
`SHAPE`, but do not run one arm with it and one without.

## The sixth thing we are fooling ourselves about

It is mine, and it is that I have been quoting **58%** as though it were a measurement rather
than a fraction.

It is 11 of 19. The Wilson 95% interval on 11/19 is **36% to 77%**. Every damage rate in this
document has that shape: the link rate is 3 of 11, the per-paper rates are 5 of 9 and 3 of 5
and 3 of 4 and 1 of 1. I wrote "58%" into a docstring in `recall_llm.py` and into the
`Settings` comment for `proposer_kind`, and it will be read as a property of the pass.

What the evidence supports is the *direction* and the *kind*: repair is wrong on more of what
it changes than it is right, and the errors are real facts in the wrong place rather than
inventions. What it does not support is any comparison of two arms that differ by less than
about twenty points -- which includes the Luna comparison as designed. **If Luna comes back
at 45%, that is not a fall; it is inside the interval.** Four papers cannot separate 58% from
45%, and by my own round-3 estimate we need eight papers per arm for a 30-point difference
and more than that for anything smaller.

So: run Luna over the fifteen papers, not the four. It is the same code, the arms already
share inputs, and the cost is linear where the statistics are not. If the budget only covers
four, then the honest output of that arm is a direction and not a rate, and it should be
written down as one.

---

# Round 7: four proposers, one question, and an answer the damage rate cannot show

Four arms, four papers, byte-identical inputs (verified with `cmp` on all sixteen pairs).

## R5, per paper and per arm

| arm | 18823721 | 11058476 | 16038771 | 21118656 | gate |
|---|---|---|---|---|---|
| pre (no fix) | 13/9/4 fail | 3/2/1 fail | 6/3/3 **pass** | 0/0/0 pass | 2 of 4 |
| NuExtract per class | 9/5/4 fail | 5/3/2 fail | 4/3/1 fail | 1/0/0 pass | 1 of 4 |
| Luna per class | 3/1/2 **pass** | 4/1/3 **pass** | 7/4/3 fail | 6/6/0 fail | 2 of 4 |
| Luna 2 batches | 2/0/2 **pass** | 1/0/1 **pass** | 5/4/1 fail | 5/5/0 fail | 2 of 4 |
| Luna whole record | 2/0/2 **pass** | 2/0/2 **pass** | 6/4/2 fail | 5/5/0 fail | 2 of 4 |

(changed / wrong+invented / yield)

| arm | changed | bad | yield | damage | 95% Wilson |
|---|---|---|---|---|---|
| pre | 22 | 14 | 8 | 64% | [43%, 80%] |
| NuExtract per class | 19 | 11 | 7 | **58%** | [36%, 77%] |
| Luna per class | 20 | 12 | 8 | 60% | [39%, 78%] |
| Luna 2 batches | 13 | 9 | 4 | 69% | [42%, 87%] |
| Luna whole record | 15 | 9 | 6 | 60% | [36%, 80%] |

**Every pairwise comparison is indistinguishable.** Ten comparisons, largest |z| = 0.65.
NuExtract against whole-record Luna is z = 0.12. No arm beats the *pre* arm on the gate.

## Your hypothesis, plainly: the damage rate does not fall

**It does not.** 58% -> 60%, z = 0.12. On the metric you asked me to use, giving the model
the whole record in one call changes nothing, and neither does batching, and neither does
changing the model. If that is the whole question, the answer is no.

**But the metric cannot see what happened, and something large did.** I took the ten fields
NuExtract got wrong and asked what each arm put in them:

| arm | fixed | left empty | still wrong |
|---|---|---|---|
| NuExtract per class | 0 | 0 | 10 |
| Luna per class | 1 | 7 | 2 |
| Luna 2 batches | **1** | **9** | **0** |
| Luna whole record | **1** | **9** | **0** |

Not one of the ten survives into the batched or whole-record arms. `haloperidol`, the four
craving questionnaires as a diagnostic instrument, the STN as a correction region twice, the
analysed count in the enrolled slot twice, the students' 28.2 as a group's mean age, `beta
distribution` as an HRF -- every misattribution this review has been chasing since round one
stops.

**And it stops by abstention, not by attribution.** Nine of the ten are left empty; exactly
one is corrected (`mod_glm.spatial_unit: 'roi' -> ['voxel']`). The model did not see that the
number belonged to another cohort and pick the right one. It declined to answer.

That is worth having -- an empty field is honest and a wrong one is not, and `missed` rises
from 19 to 23 while `wrong + invented` falls -- but it is not what the hypothesis predicted,
and it should not be written up as though it were. **Cross-entity context is a lever for
silence, not for correct attribution.**

Paired on those ten fields, the effect is not marginal: ten fields move from wrong to
not-wrong and none moves the other way, sign-test p ≈ 0.002 on four papers. **The abstention
result is established; the rate difference is not.** That is a fact about the two
measurements rather than about the arms -- a paired comparison on the same fields is
enormously more efficient than comparing two marginal rates, and I should have proposed it
three rounds ago.

## What the Luna arms acquired instead

The residue does not shrink, it moves. By slot:

    NuExtract per class   correction_regions 2, enrolled_count 2, age_mean 2, medications 1,
                          diagnostic_instrument 1, inference_level 1, hrf_model 1,
                          spatial_unit 1                                  -- 8 slots
    Luna whole record     diagnostic_instrument 6, diagnostic_system 2,
                          medical_condition 1                             -- 3 slots

Eight of the whole-record arm's nine errors are the pass attaching a diagnostic instrument or
a diagnostic system to a group that does not have one -- on 21118656, where NuExtract changed
one field and got it right, Luna changes five and gets all five wrong. **That is the
link-choice failure from round 6, and it is not a context problem.** Round 6 measured the
pass at 0 for 9 on link writes that required a judgement; giving it more context made it
willing to make more of those judgements, and they are still wrong.

`EXCLUSIVE` fired once in each Luna arm, on 16038771 (`already belongs to grp_nonsm`) -- the
first production evidence for it outside 18823721. It does not touch the six remaining
errors, because those give *different* instruments to different groups, so no target is
claimed twice.

## The link comparison is not usable, and I will not report it as though it were

| arm | reference slots changed | targets added | scored against truth | unverifiable |
|---|---|---|---|---|
| NuExtract per class | 11 | 14 | 11 | 0 |
| Luna per class | 61 | 117 | 15 | 42 |
| Luna 2 batches | 61 | 87 | 15 | 44 |
| Luna whole record | 43 | 43 | 15 | 26 |

**Luna writes four to six times as many links as NuExtract on the same four papers**, and my
link ground truth covers 15 of them. The Luna arms score 0% wrong on that 15 -- and quoting
that as "Luna links are perfect" would be reporting 25-35% coverage as a result. The papers
where the Luna link errors concentrate (21118656) have no link truth at all; the field-side
scorer catches them because `diagnostic_instrument` is in `entities[].fields`, and the
link scorer cannot see them because that file's `links` array is empty. **Extending link
truth to 21118656 and 11058476 is the cheapest thing that would make the next comparison
mean something.**

## What the intervals permit

With 13 to 22 changed writes per arm, the narrowest 95% interval is ±20 points. Concretely:

* four papers cannot separate 58% from 40%. They cannot separate 58% from 69%.
* the whole-record arm produces 3.75 changed writes per paper, so **15 papers gives about 56
  changed writes** -- enough to detect a fall from 58% to 28% at 80% power, not enough for
  58% to 43%.
* so: **run the whole-record arm over the fifteen, and expect it to settle a 30-point
  question and nothing finer.** At 3.8 calls and $0.05 per four papers that is about $0.19,
  which is not a reason to hesitate.
* and run the paired comparison alongside it. The ten-field result above needed four papers
  and reached p ≈ 0.002. Ask "what did this arm put in the fields the other arm got wrong"
  rather than "what are the two rates".

## A correction to myself, the seventh

I nearly reported that the Luna arms reintroduce invention -- three writes flagged
`NOT-IN-PAPER` where every NuExtract error was a real fact misplaced. They do not. My scorer
tests reference targets by the entity's **label**, and Luna composes labels: the instrument
is called `sadomasochistic preference screening and questionnaire`, which is not verbatim
anywhere, while "sadomasochistic preferences" appears four times. I was measuring the model's
paraphrase and reading it as a fabrication. Fixed to test content words; with that fix **every
bad write in every arm is a real fact from the paper in the wrong place, 9 or 12 of 9 or 12,
in all five arms.** The finding this whole exercise turns on survives the correction, and it
is the second time a naive string test has nearly made me claim a fabrication that was not
one -- the first was `Intera` in round 4.

## Where I would put the next effort

Not on the proposer. Four proposers and three context regimes moved the damage rate by less
than the noise, and the one large effect they produced -- abstention -- is available more
cheaply and more predictably from a refusal than from a model's reticence.

The residue is now one slot family. Six of nine remaining errors are `diagnostic_instrument`,
which the schema already rules out in words the model is given and ignores: *"Naming one here
claims it classified this cohort, which is narrower than having been administered to it."*
That is not a context problem, not a model problem, and not a template problem -- it is a
slot whose correct answer is "nothing" far more often than the pass believes, and the check
for it is the quote rule from round 6: **a link needs a sentence naming both the group and
the instrument in the act of diagnosing, and if the model cannot produce one, the link is
refused.** That kills all six.
