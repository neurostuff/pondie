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
