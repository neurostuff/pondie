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

So the rule to apply is: a pool of one, **and** the names agree by token or initialism, **and**
no two distinct names reach the same target — the last because an interaction term shares a
token with the main effect it contains. A model is needed only for genuine synonymy with no
shared token and no initialism, which none of the 120 exhibits.

Not applied in this pass, which was a refactor: the function uses the shared walk and keeps
its narrow index, 90 repairs before and after, 0 differences, with the whole measurement in
its docstring.

## Found: the prompt renderer imported the record builder for a schema question

`render.ENTITY_LISTS` was `builder._entity_lists()` — a private function, reached across a
layer boundary, for a fact neither module owns. And because the result was cached in a
module-level constant, **importing `builder` parsed the LinkML schema as a side effect.**

`reader.entity_lists(schema)` now owns it, `builder` caches it lazily, and `render` no longer
imports `builder` at all.

## Named, not done: `builder.py` is three modules

At 1,954 lines it holds three unrelated responsibilities, and the line that separates them is
already visible in the layer list above:

| what | lines | belongs |
|---|---|---|
| merge and orchestration | ~250 | stays: this is `build` |
| **warranting** — `_resolve_field`, `_walk`, `_iter_sets` | ~150 | `extraction/evidence/`, whose `__init__` already says "which characters of the paper say so" |
| ~20 repair implementations | ~1,400 | beside `repairs.py`, which holds the sequence and lazily imports `builder` to get the functions — a cycle papered over with a deferred import |

The warranting move is the clean one and it is not free: those three functions own **twelve of
`BuildReport`'s fields**, so they should take an `EvidenceReport` that `BuildReport` holds, and
that ripples into `summary()` and the stage notes. Worth doing as its own pass with its own
verification rather than appended to this one.
