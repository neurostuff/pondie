# Five more pipelines, from what the evidence work measured

P1–P12 are in [pipeline-hypotheses.md](pipeline-hypotheses.md) and
[pipeline-architecture.md](pipeline-architecture.md). These five are grounded in
measurements that did not exist when those were written, and each names the number it
would move and the number that would falsify it.

What changed the ground:

- The quote pass reaches **67.1%** of human-marked evidence on the whole paper. Handing
  it a twelve-sentence shortlist instead costs **21.4 points** to save 45% of the prompt,
  so retrieval cannot substitute for reading ([evidence-union-design.md](evidence-union-design.md)).
- The retriever adds **+6.3 points** as a second voter, free, and is now in
  `evidence/quote.py`.
- A pick containing the value verbatim is **80.9%** confirmed-correct against 27.8%
  without — deterministic matching, where it applies, beats the learned scorer by 2.9x.
- **61%** of reviewer evidence is in Methods; Discussion holds 0.7%.
- Direction from a contrast's name is **98%** on the 17% of cells it answers, and
  recovers cells the model marked `absent` ([deterministic-direction.md](deterministic-direction.md)).
- The corpus texts the pipeline reads **contain no tables**: 2 pipe characters against 54
  in the same paper's table-bearing build. One table-row candidate out of 2,076.
- **41** gold slots are entities a reviewer created that the extractor never produced.
- One run in four loses the entity pass, and that variance is larger than any config
  delta measured so far.

## V13 — Table-bearing text

**Change.** Build the pipeline's text with the result tables rendered inline, as the
ns-pond build already does, and drop the separate table injection that compensates for
their absence.

**Why it should win.** Every value read off a table — group sizes, demographics, peak
statistics, coordinates — currently has no locatable evidence, because the sentence it
would be quoted from is not in the text the locator searches. The retriever's table-row
handling exists, is tested, and fired **once in 2,076 candidates**. The 46 slots where
neither locator found anything are concentrated on exactly these fields.

**Measured by.** Evidence `correct` on the 173-slot human set, split by whether the value
appears in a table. **Falsified if** the added tokens cost more on the extraction passes
than the evidence gain is worth, or if `correct` on non-table fields drops — a longer
prompt can crowd out the values it is meant to support, which is why evidence was split
into its own pass to begin with.

## V14 — Derive first, then ask for the remainder

**Change.** Run every deterministic filler before the extraction passes —
`derive_fields.py`'s eight derivers, `coordinate_space`, and `direction_of` — and remove
the fields they answer from what the model is asked to emit.

**Why it should win.** It is the only change here that reduces *output* tokens, which is
where the cost is: evidence was 57% of output before it was split out, and stripping it
moved analysis recall from 94% to 98%. A field the model is not asked about cannot be
answered wrongly, and the derivers are at or above parity where they fire. Today
`fill_directions` runs *after* extraction and can only repair; run first, it also saves
the asking.

**Measured by.** Composite and output tokens per paper. **Falsified if** removing a field
from the prompt degrades a *neighbouring* field — the prompt is one context and the
fields are not independent, which is the risk that makes this worth measuring rather than
assuming.

## V15 — An entity-recall pass

**Change.** After the record is assembled, one small call: here are the entities you
found, here is the paper, which entities of these classes does the paper describe that
are missing? Emit only names and the sentence that introduces each.

**Why it should win.** Entities carry **0.20** of the composite and every downstream term
is conditional on the entity map — `extraction-comparison-metrics.md` says so outright.
The evidence work turned up 41 entities a reviewer had to create by hand, which is a
recall failure the current pipeline has no pass aimed at. Output is tiny: names and one
quote each.

**Measured by.** Entity F1 against the reviewer-created set, which already exists as a
by-product of the gold build. **Falsified if** it invents entities — precision matters
more than recall here, because a spurious entity gives every analysis a wrong term to
point at.

## V16 — Repeat the demands pass, union the declarations

**Change.** Run the demands pass three times at low effort and take the union of declared
entities; keep one sampled record for everything else.

**Why it should win.** One run in four loses the entity pass entirely, and that variance
is larger than every prompt delta measured to date — it is why config comparisons on this
pipeline have needed replicates to say anything. Two extra cheap calls buy a floor under
the failure mode that costs the most, and the demands pass is the cheapest of the four.

**Measured by.** Standard deviation of the composite across replicates, not its mean.
**Falsified if** the union admits entities that no single run would have declared and the
satisfy pass then has to invent evidence for them.

## V17 — The retriever as a checker, not a contributor

**Change.** Keep the union, and add a targeted re-ask: where the model's quote and the
retriever's confident top-1 land in different sections, or where the value appears
verbatim in the paper but not inside the quoted span, re-ask that field alone with the
disagreement named.

**Why it should win.** It aims at the two failures the union does not fix. 26.6% of the
quote pass's answers are `unknown` and 2.3% are unlocatable, and a literal match is
80.9%-reliable — so "the value is in the paper but not in your quote" is a strong,
cheap signal that something is wrong, not a guess. The re-ask is one field at a time and
only for flagged fields.

**Measured by.** Evidence `correct`, and the count of unlocatable quotes. **Falsified if**
the flag fires on fields that were already right — the disagreement rate is not the error
rate, and the abstention curve says the retriever is only 42% correct at full coverage.

## The measurement problem all five share

`extraction-comparison-metrics.md` states plainly that **evidence spans are not
compared**. V13 and V17 improve only evidence, so the composite cannot see them at all,
and V14's saving is in tokens rather than score. Before running any of these, the
composite needs an evidence term or these have to be scored on the human evidence set as
a separate axis — otherwise a real improvement reports as no change, which is the same
mistake the first version of the union design made by scoring against stale records.

Order worth running: **V13** first, because it is the only one that changes what every
other stage can see; then **V16**, because it makes the rest measurable; then V14, V15,
V17.
