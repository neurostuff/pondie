# Pipeline fixes, August 2026

What changed, what it was measured against, and what is still open. The code says what each
fix does; this says why it was worth doing and how much it bought.

The corpus throughout is the 300 schizophrenia papers staged on beast at
`/home/james/nsv-runs/schiz`, of which 299 build. The bench is the sixteen reviewer-gold
papers in `data/gold-direction-16.pmids`, scored by `pondie.benchmark`.

## The repair chain was never wired in on beast

`unwrap_plain_slots` was defined but never called in the copy of `record/builder.py` the
schizophrenia run used, and every paper logged `repairs: none fired`. Rebuilding the same
payloads locally, with no new model calls, moved:

| | original | rebuilt |
|---|---|---|
| records with no validator error | 91/298 (31%) | 201/299 (67%) |
| validator errors | 2,022 | 808 |
| analyses with `source_table_analysis` | 26.9% | 82.5% |

Most of that was one shape error with a long tail. 98 of 345 acquisitions carried
`acquisition_type` as an `ExtractedValue` wrapper rather than the bare string the schema
declares. The slot is the type designator, so it never resolved to `MRI`, and all seven
MRI-specific parameters the model *had* extracted were reported undeclared on the base
class: roughly 570 findings across 80 papers, none of them extraction mistakes.

The lesson is about deployment rather than code: a stale checkout fails silently, because
every stage still succeeds and the record still validates against a weaker class.

## One parse-key numbering, not three

`Analysis.source_table_analysis` holds `<table id>#<ordinal>`, and it only works if the
prompt and the builder number identically. They did not. `extract_record.stage1_block`
numbered over the entries it showed the model -- withheld halves excluded -- while
`build_record.resolve_source_table_analysis` numbered over the whole parse. On any paper
with a sign split the model was told `t1#2` and the builder resolved `t1#2` to a different
row group.

A wrong key that exists is worse than a missing one: it passes the join and attaches an
analysis to another contrast's coordinates. 26 of 300 papers were exposed.

Both now call `parse_tables.parse_keys`, which numbers over the full parse so hiding an
entry cannot renumber its siblings. `extract_sz_contrasts.py` had a fourth inline copy;
folding it onto the same function left every count unchanged, which is the evidence that
it had agreed by luck.

## The mirror was named after the half it is not

`mirror_analysis` deep-copied the described half and overwrote `local_id`, `mirror_of` and
`source_table_analysis` -- but not `name`. The result was an analysis called `FESZ > NC`
whose cells say NC > FESZ, sitting on the same table as the real `FESZ > NC`.

All 36 mirrors in the corpus carried a colliding name. The cost was not cosmetic: on
`JzsUUQbDr2bm` the direction bench scored a *correct* extraction as a sign flip, because
two candidates shared a name on tbl3 and the entity matcher took the reversed one. Those
flips appear in every replicate of every arm, baseline included, which is what a
deterministic matcher does with a tie.

The fix takes the withheld parse entry's own label, `"<described name> (reversed)"`.
Inverting the operator was rejected: only 4 of 50 described names contain a bare `>` or
`<`, the rest being labels like `GM Spatial Map` or `Seed: Right anterior cingulate
cortex`, which have nothing to invert. Collisions fell from 36 to 2.

A second fix was considered and abandoned. 19 of the 36 mirrors duplicate the cells of an
analysis already in the record, which looked like grounds for suppressing them -- until
measurement showed all 19 address row groups carrying coordinates their twin does not.
Suppression would have silently dropped foci.

## `points` and `mirror_of` were undeclared

Both were written onto analyses by the builder and declared by neither schema, so the
validator reported the pipeline's own output as undeclared attributes on 21 of 299 records.

`points` is gone: the schema stores no coordinates by design, and the mirror now reaches
its rows through `source_table_analysis` like every other analysis. `mirror_of` is real
content and is now declared -- via an `add_slot` deviation, because storage marks it
`deterministic` and the projection drops such slots, yet here the code that fills it runs
on the extraction side. Without it the two halves of a sign-split contrast look like
independent findings, and a meta-analysis counting both counts one sample twice.

## Two ways a rebuild lost papers

the runner's text-flavour list listed two text flavours where `Flavour`
lists four, so a paper whose only text is `ace` built fine under the pipeline driver and
failed here with "no text". It now derives from the one list.

`build_tables_payload` read `processed/pubget/tables.jsonl` unconditionally. One paper
without a manifest raised `FileNotFoundError` and took its whole shard down: 7 of 300
papers rebuilt. A missing manifest is now a legal answer meaning "no coordinate tables".

## The tables stage cannot be retrofitted

The stage had never run for this corpus -- no `tables.json` payload exists, 1 of 298
records had a non-empty `tables` list, and 98.7% of `analyses[].tables` references dangled.
Running it afterwards made things worse, not better: 0/299 clean records and 3,853 errors,
because the analyses' table references were produced when the model was shown a different
(empty) table map, and the deterministic Table fields carry no evidence blocks.

The stage has to run *before* the extraction passes. The corpus was left without it, and
the coordinate join is unaffected because coordinates are reached through the parse key,
not through `tables`.

## What the fixes bought the query

Same 687 case-control contrasts detected either way; the difference is how many join.

| | before | after |
|---|---|---|
| joined by `source_table_analysis` | 121 | 451 |
| skipped, name matched several row groups | 194 | 58 |
| SZ > Control | 85 analyses / 64 studies | 127 / 83 |
| Control > SZ | 248 analyses / 118 studies | 324 / 145 |

The name-match fallback now fires zero times: every included contrast joins by exact key.

## Two defects a reviewer logged, and what closed them

Both were logged per paper in a `known-gaps.yaml` and are now gone. Recorded here rather
than there because that file had turned into an allowlist a module read to *suppress*
findings, which was never the intent of writing them down. Re-measured on the seventeen
records under `benchmarks/`: all five papers report **zero** cross-reference findings of
either kind.

**A seed region named rather than referenced.** `MODE_NOTE["entities"]` never mentioned
`Region`, so eleven of sixteen papers emitted none — and an analysis whose seed had no
Region to point at wrote the region's *name* into the reference slot. Five papers carried
it: `7HPLh5nJzmP5` (`left dlPFC parcel`, `right dlPFC parcel`), `aVGe9BmFTMDR` (seeds named
by coordinates — `responder Rlocation (-42, 34, 38)`), `QQCjAAT6SwwQ` (seven salience-network
seeds), `kzMj26hGWacQ`, `TgcHKMRfrVog` (nine references to three rsFC seeds).

It was invisible twice over: `check_local_ids` could not see it at all until it learned to
follow the type designator into `ConnectivityDetails`, and even then nothing ran
`check_local_ids` — `BuildReport.dangling` was declared and never assigned, so the count the
build reports could not print. *Closed by* the Region paragraph now in `MODE_NOTE`, which
says a Region is an entity in the sense a Group is and that pass is the only place one can
be created.

**Two sibling models minting the same column ids.** On `QQCjAAT6SwwQ`, two linear
mixed-effects models each declared `term_age`, `term_sex`, `term_education` and `term_group`,
so a `Cell.term` naming one of them could not say which model's column it meant. Both
declarations resolve, so nothing but a duplicate-id check notices. *Closed by* the
`ModelEstimation` description deviation stating that a model fitted over a different sample
is a different record.

The shape both share is worth keeping: a reference that **resolves to the wrong thing** is
harder to find than one that resolves to nothing, and neither was reachable by the
validator — both are `check_local_ids` findings, which is a different stream.

## Open

- **`FactorLevel.arms` is unreliable.** The prompt block filled it 13%, 0% and 19.5% across
  three replicates. Filling nothing in one run of three means it is not firing dependably,
  which was the whole point of adding it.
- **12 interaction-named contrasts** are included in the case-control query and should not
  be; an interaction is not a simple group contrast.
- **2 contrasts were captured with the direction inverted** relative to their own row-group
  name -- a precision error, and the kind that does most damage per instance.
- **Query recall is unmeasured from above.** Against row-group names it is 82% on
  coordinate-carrying groups, but a name reference only finds misses it can read.
