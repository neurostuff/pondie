# How a paper becomes a record

The pipeline is easy to misread as nine steps that each do a bit more. It is not. It is
**four changes of representation**, and every stage belongs to exactly one of them. Knowing
which one a stage is in tells you what it may touch, what it may assume, and what breaks if
it is moved.

```
  a document          an address space         a draft record        a checked record
 ───────────────     ──────────────────       ────────────────      ─────────────────
  text + tables  ->  parse entries with   ->  payloads of wrapped ->  one record, spans
                     ids and coordinates      values, per pass        resolved, repaired
```

## 1. The document, and why it is frozen

`data/corpus/<id>/` is an **input**. A run reads it and never writes it, and
`text_index.normalize` does one thing — canonicalise line endings — because every
`start_char` in every record already written addresses that exact byte sequence. This is the
constraint that shapes everything downstream: `spans.fold` is a 1:1 length-preserving
translation and not a normaliser, for the same reason.

## 2. The address space: the parse is where coordinates live

**The schema stores no coordinates.** That single fact explains three stages.

An `Analysis` reaches its foci through `source_table_analysis`, a key into the stage-1 parse.
So anything that is going to carry coordinates must exist *as a parse entry* before the
extraction passes run, and three no-model stages build that space:

| stage | what it adds to the address space |
|---|---|
| `tables` | a `Table` per parsed table, with the id every `Analysis.tables` reference will hold |
| `prose` | a parse entry for coordinates the paper states in prose and no table reports — otherwise the analysis can be extracted and its location cannot be stored |
| `split` | a table reporting both signs is two contrasts; the reversed half is withheld here and rebuilt arithmetically at the end |

All three run before any model sees the paper, and all three are **deterministic**. If this
layer is wrong, no later stage can recover: a paper with no parse entry has no analysis to
extract, which is why 8 of `vbm_of_ptsd`'s 22 gold studies contribute nothing to its map.

## 3. The draft: payloads, not a record

The model passes do not write a record. They write **payloads** — one JSON file per pass,
under `payloads/<study>/` — and the order is the design:

```
demands   analyses first: each declares the entities it needs, before any exist
satisfy   build exactly those entities and nothing else
fill      ask for the slots still open, round after round, until none are
evidence  a supporting quote for every value  (45% of input tokens)
```

`demands` precedes `satisfy` because **a cell cannot be righter than the term it points at**:
asked to guess an inventory first, the entity pass modelled a crossover's condition as a
continuous covariate. `fill` exists because `satisfy`'s shape — render a class, ask for
records — is right for deciding what exists and wrong for finishing what already does.

A payload is not yet a record, and the difference matters: entities refer to each other by
document-local `local_id`, nothing is resolved, and a value may be in the wrong shape. Every
value is an `ExtractedValue` wrapper, and **reading one is not `node["value"]`** — a wrapper
distinguishes absent, `not_reported` and reported-empty, and conflating any two is a silent
wrong answer. `values.value_of` is the only reader.

## 4. The record: merge, repair, warrant, check

`build` is one stage doing four things in a fixed order, and they are four different kinds of
operation:

```
merge     payloads -> one body           (merge_payloads, load_aliases)
repair    22 deterministic fixes         (repairs.build_sequence, in order)
warrant   quotes -> spans on the document (_resolve_field, via spans.resolve)
check     19 rules + LinkML structure    (rules.check_all, validate)
```

**Repair and check are not the same thing and the boundary is the whole design.** A repair
runs only where the choice is not a choice — a dangling id that folds to exactly one declared
id is a transcription slip and nothing is being decided. Everything else is *reported*:
`align_cell_levels` repairs a level that folds, `check_cell_terms` reports one that does not,
and choosing between two candidates would be a claim about the paper. The registry in
`repairs.py` is data for this reason — each repair carries what it does, why it sits where it
does, and what it changed.

`warrant` is where the document comes back. Nothing between stage 2 and here has needed the
text; this step alone resolves each quote against it and is why a record's spans can be
asserted to satisfy `normalized[start:end] == span.text`.

## The five invariants worth knowing before changing anything

1. **Offsets address the frozen document.** Any transformation of the text must preserve
   length, or every existing record's spans become wrong.
2. **The parse is the only route to coordinates.** A thing that should carry foci must be a
   parse entry before the model passes run.
3. **A wrapper is read through `values.value_of`.** Five modules once each wrote their own
   unwrapper and disagreed at the edges.
4. **A repair decides nothing.** If two answers are possible, it reports.
5. **A `local_id` is an address.** The review layer keys answers on
   `paper|value|<Class>|<local_id>|<path>`, so an id that changes between extractions of the
   same paper orphans every answer a reviewer gave.

---

# The architectural review, and what it changed

## Found: ten hand-rolled walks of the same tree

`values.iter_fields` finds wrappers by their marker and knows nothing about the schema, which
is right for jobs that need only the value. Every job needing the *declared* shape — is this
slot multivalued, what range, reference or nested — walked the record itself, and **ten
functions in `builder` did**. Nine opened with the same
`if not isinstance(node, dict) or values.is_field(node): return`; eleven with the same
`designated_type` line. One passed no path. One followed lists where another did not.

**`record/walk.py`** is that walk, once, as generators — `fields`, `references`, `slots`,
`entities`, `declared_ids` — so a repair reads as a loop over the thing it cares about rather
than a recursion with the loop body buried inside it. Six repairs converted:
`unwrap_singleton_lists`, `listify_scalars`, `relabel_conclusions`, `coerce_numeric_values`,
`unwrap_plain_slots`, `repair_references`.

Verified byte-for-byte against the pre-refactor implementations over all 1,817 committed
records: **0 differences**, with `unwrap_singleton_lists` firing 21,701 times and
`relabel_conclusions` 11,154 times identically in both.

Two walks stay hand-written and the reason is worth stating: `listify_nested` and
`apply_aliases` run at the `shape` stage, **before the record matches the schema**, and a
schema-guided traversal cannot be used by the repairs that exist to make the schema
applicable. `align_cell_levels` and `drop_redundant_cell_levels` also stay: they walk
analyses and model terms together, which is a join rather than a traversal.

## Found: the duplication hid a divergence that silently disabled a repair

`repair_references` built its index of declared ids by sweeping the **top-level lists**, so it
never saw a `ModelTerm` under `model_estimations[].terms` or a `Condition` under
`tasks[].conditions`. Measured: **it misses 10,867 declared ids** across the corpus — 4,895
ModelTerms, 3,010 Conditions, 696 Timepoints, 448 Arms. `validate.index_ids` descends and has
disagreed with it all along, which is exactly what two implementations of one idea produce.

Consequence: every `Cell.term` and every `FactorLevel.conditions` reference looked dangling to
it, and `pool_for` looked for a Study-level list that does not exist for them — so it repaired
**none of the two slots that dangle most**.

**And fixing it was reverted, which is the more useful finding.** Given the schema-guided index
the `sole` rule reaches slots it was only safe never to reach, and it was wrong on most of what
it then touched: `asm_mini` repointed to `asm_ftnd` — a psychiatric interview onto a
nicotine-dependence scale — and `trm_three_way_interaction` and `trm_quitting_motivation_main`
both collapsed onto `trm_cue_1`, on a record declaring one term whose cells name three. A guard
refusing any target two distinct names reach removed the collapses, kept
`asm_diagnostic_interview -> asm_heroin_craving_questionnaire`, and refused 11 repairs that
looked right. **88 of the 90 repairs this function makes come from `sole`**, so it cannot
simply go.

I first wrote that this needed a labelled ground truth. **It does not, and the reason is
structural.** `ids.mint` builds a local_id from "the shortest thing the *paper* fixes", so a
dangling id carries the name the model meant — adjudicating a repoint is comparing two names,
not judging a paper. Over the 120 firings: **70% decided by token overlap, 8% by initialism**
(`asm_scid` → "Structured Clinical Interview for DSM-V", `tsk_midt` → "Monetary Incentive
Delay Task", using the Schwartz & Hearst matcher already in `vocabularies.abbreviations`),
**18% rejected because the names share nothing**, 4% needing a look at the record. **96%
without a model**, and the 21 rejections are inspectably right.

**Applied.** `repair_references` now repairs a transcription slip outright, and anything else
only when all three hold: the slot's kind has exactly one declared entity, its name agrees
with the dangling id by shared word or initialism, and no differently-named reference wants
the same target. `declared` is read schema-guided, which is what the name test makes safe.

Measured against the previous version over the 1,817 records: **90 repairs → 89. 81 kept, 8
gained, 9 dropped.**

All 8 gained are unambiguous: `trm_film_condition` onto a term *named* "film condition",
`asm_scid` onto "Structured Clinical Interview for DSM-V", `no_intervention` onto
`arm_no_intervention`, `r_nucleus_accumbens` onto `reg_nucleus_accumbens`,
`reg_amygdala_right` onto "Amygdala".

6 of the 9 dropped are the wrong repoints the rule exists to stop — `reg_vs`, `reg_caudate`,
`reg_insula` and `reg_thalamus` each onto one "Gain versus nongain reward-processing regions",
and `tsk_resting_state` onto `tsk_fear_conditioning_task`.

The other 3 are a conservative loss and worth naming: `tsk_esom` and `tsk_esom_nf` both want
the one task, named "Emotion Self-Other Morph Neurofeedback (ESOM_NF)", and the collapse guard
refuses both because they are differently named. Exempting competing names that are variants
of each other would recover them and would also re-admit
`trm_smoking_opportunity_cue`/`trm_quitting_motivation_cue`, which likewise share a word — so
the exemption is unsafe and 3 lines in 90 is the right price for the guard.

## Found: the prompt renderer imported the record builder for a schema question

`render.ENTITY_LISTS` was `builder._entity_lists()` — a private function, reached across a
layer boundary, for a fact neither module owns. And because the result was cached in a
module-level constant, **importing `builder` parsed the LinkML schema as a side effect.**

`reader.entity_lists(schema)` now owns it, `builder` caches it lazily, and `render` no longer
imports `builder` at all.

## Done: warranting has a home

`extraction/evidence/warrant.py` is the third arrow of the model, and the package's own
`__init__` already described the job — "which characters of the paper say so" — with `quote`
asking for a sentence and `record.spans` locating one string. What was missing was the step
that applies the second to the output of the first.

It owned **thirteen of `BuildReport`'s sixteen counters** while the build owned three, so the
counters moved with it into a `Warrant` the report holds. The names lost a prefix they no
longer need:

| was | is |
|---|---|
| `report.resolved_exact` | `report.warrant.exact` |
| `report.resolved_cased` | `report.warrant.case_insensitive` |
| `report.failures` | `report.warrant.unresolved` |
| `report.fields_quote_unlocated` | `report.warrant.unlocated` |

`BuildReport` is now three fields and a `Warrant`, `summary()` delegates, and `build` reads
`report.warrant = evidence.warrant(body, normalized)` — one line where the text re-enters the
pipeline, which is the fact the model exists to make visible.

`_STATUSES` was duplicated in the move and is now `values.STATUSES`, beside the wrapper
contract that owns it: both `repair_wrappers` and `warrant` ask.

## Where the repairs live

A repair is a function from a record to a changed record plus a note. The sequence that
orders them is data, in `record/repairs.py`. The implementations are in
`record/fix/`, split by what they do to the record:

| module | what its repairs do | examples |
| --- | --- | --- |
| `fix/shape.py` | make a payload match the wrapper contract | `repair_wrappers`, `unwrap_singleton_lists`, `coerce_numeric_values` |
| `fix/derive.py` | compute a value the paper fixes but no pass wrote down | `derive_denominators`, `derive_table_effects`, `derive_coordinate_spaces` |
| `fix/link.py` | make a reference point at the entity it names | `repair_references`, `link_entities_by_name`, `scope_duplicate_terms` |

The split follows the changes of representation above: `shape` works in the payload
representation, `derive` and `link` in the record. A reader asking "why is this field
wrong" knows which of the three to open from the shape of the wrongness.

This is where the `builder ↔ repairs` cycle was. `builder.py` held the implementations and
`repairs.py` lazily imported `builder` to reach them, with a comment explaining the
deferral. Now `repairs` imports `fix` at module level, `builder` imports both, and nothing
imports `builder`. `builder.py` is 385 lines and orchestrates: merge payloads, apply
aliases, run the sequence, warrant, assemble.

The move was verified over the 1,817 records in `record_arms`: the `AT_MERGE` group run
under both trees produced byte-identical bodies and identical repair logs on every record.

## Found: five places knew what a stage-1 parse is

Invariant 2 says the parse is the only route to coordinates, so "what a parse holds and how
to address it" is one of the load-bearing facts in the package. It was in five places:

| where | what it knew |
| --- | --- |
| `extraction/parse.py` | `TableParse`, `ParsedAnalysis` — the document and one entry |
| `formats/parse_keys.py` | `<table_id>#<ordinal>`, moved here to stop `query` importing `extraction` |
| `stages._parsed_points` | every coordinate the parse holds |
| `stages._parsed_tables` | every table the parse read |
| `render.PROSE_TABLE_ID` + two bare `"prose"` literals | the id a prose entry carries |

Two readers of the document lived in the 1,272-line file that runs the pipeline, reached by
whichever stage needed them first. They are now `TableParse.coordinates` and
`TableParse.source_tables()`, beside the class they read, and `ParsedAnalysis` gained the
`coordinates` and `is_prose` its callers were computing from `.raw`.

`PROSE_TABLE_ID` went to `formats/parse_keys.py` rather than to `extraction/parse.py`,
because `benchmark` and `query` both compare against it and neither imports `extraction` —
the same argument that put `parse_keys` there.

## Found: the `tables` stage hand-rolled the manifest reader that `formats` already had

`stages._manifest_tables` read `<study>/processed/<flavour>/tables.jsonl` line by line and
lifted `metadata.table_label` — which is `formats.table_parse.read_manifest`, twelve lines
further down the same path. This is the failure mode the `formats` docstring is written
about, and the two had already drifted: `read_manifest` returns `""` for an absent caption
and the stage's copy returned `None`, which the stage's own `_manifest_value` then wrapped
as a blank string rather than `not_reported`.

`_manifest_value` now treats blank and absent alike — an empty caption and a missing one
make the same claim, which is the argument `Tables.run` already makes for a manifest with
no rows — and the stage calls `read_manifest`. Verified over the 79 manifests and 210 table
rows on disk: **0 differences** on the five fields the stage reads.

`Paper` gained `study_dir`, since `read_manifest` takes the study rather than the file, and
`Paper`'s docstring already claims to be "the only filesystem knowledge a stage needs".

## Found: `benchmark.backfill.entry_id` was `parse_keys` again, already drifted

It computed `<table>#<n>` for one position, defaulting a missing `table_id` to `prose` where
`parse_keys` leaves it empty — so the two disagreed on any entry without one, and its own
docstring says the join has to match what candidates write. None of the 4,280
`source_table_analysis` values in the corpus has an empty prefix, so the divergence never
fired. It delegates now.

`stages.py` is 1,209 lines and holds a Protocol, a base, nine stage classes in pipeline
order, and `sequence()`. It is not split further: the order is the thing the file
communicates, and nine files would hide it.

## Found: `repair` named two opposite things

`record/repairs.py` held the deterministic sequence run inside `build`. `extraction/repair.py`
is the stage that runs after `build` and asks a model to settle contradictions. Invariant 4 —
"a repair decides nothing" — is true of the first and is precisely the opposite of the second's
job, and all four import sites of the stage aliased it (`import repair as repair_pass`) to get
a name they could read.

The stage keeps the name: it is `StageName.repair`, `--stages repair`, its output directory,
and `EvidenceSource: repair_pass` in every record already written. The inner one moves, to
`record/fix/sequence.py`, so `fix/` is now the whole of the deterministic half — what each
fix does and the order they run in — and `repair` means one thing in the package.

The sequence now names each fix by kind (`shape.repair_wrappers`, `derive.derive_denominators`,
`link.repair_references`) rather than reaching through the flat re-export, so the one place a
reader wants to know which kind a fix is says so in the call. The flat `fix.<name>` re-export
stays for everyone else.

Verified over the 1,817 records: byte-identical bodies and identical logs.

## Found: the guard layer was in the package whose own map excluded it

`record/__init__.py` lists the modules that turn payloads into a record and says
"`builder.build` is the whole of it: everything else is called from there." `record/edit.py`
was 923 lines in that directory, absent from the list, and never called from `build` — its
only production callers were the `repair` stage and one `label_of` in `recall`. Its docstring
opens "every write a repair pass makes goes past the same refusals," and the stage's own
docstring names its three steps "propose, guard, adjudicate."

So `extraction/repair/` is a package now: `__init__` proposes and adjudicates, `guard.py`
refuses. A reader tracing the stage finds all three steps in one directory.

Two pieces did not go with it. `from_local_id` is `mint` run backwards and reached
`ids.PREFIX` through a deferred import to do its job; `label_of` is what falls back to it.
Both are in `record/ids.py` now, beside the convention they read, which is also how `recall`
stops importing another stage's guard for a display name.

`record/__init__.py` gained the two lines it was missing — `walk` and `ids` were as absent
from the map as `edit` was, and unlike `edit` they belong there.

The guard's docstring had its second paragraph twice, from an earlier edit. Now it says which
of the three steps it is.

## Found: `onvoc` and `abbreviations` reached into each other

A deferred import inside a function body is the signal all of the moves above were found by,
so it is worth running deliberately. One mutual cycle turned up: `abbreviations.disagreements`
deferred an import of `onvoc.stems` to tell two spellings of one expansion from two different
expansions, while `onvoc` deferred one back — twice — for the paper's own abbreviations. Three
`# noqa: PLC0415` suppressions, one per call site, and a fourth `# noqa: E402` on a mid-file
`import fold`.

What both wanted is neither module's. `folding` already claims the language-general job and
says where the line is: "`use disorder -> dependence` and `affective -> mood` are claims about
psychiatry and belong to a vocabulary." But `group`, `scale`, `questionnaire` and `disorder`
carrying no identity is a claim about clinical writing rather than about English or about any
one ontology — a third thing, and `vocabularies/labels.py` is it: `_WEAK`, `tokens`, `content`,
`stem`, `stems`, `acronym`.

`surface_forms` and its two regexes stayed in `onvoc`, and the reason is the test that the cut
is in the right place: `surface_forms` takes the paper's abbreviations, so putting it in
`labels` would make `labels` import `abbreviations` and rebuild the cycle one layer down.

    onvoc -> abbreviations -> labels -> folding

Four suppressions gone, and no function-body imports left in the package.
