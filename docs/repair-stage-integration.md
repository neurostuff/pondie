# Folding the repair pass into pondie

## What exists now, and why it cannot stay

`scripts/pondie_arm/` holds ~2,600 lines written to answer one question — does a pondie
record substitute for full text in screening — and it grew a second job on the way: a repair
pass that finds and fixes defects in a built record. The second job is worth keeping. The
form it is in is not.

Concretely, `repair_loop.py` is a single module that owns a MiniCheck client, a NuExtract
client, the schema tables, twelve guards, the sweep controller, a differential validator and
a CLI. Its state lives in six module-level globals (`REF_SLOTS`, `DECLARED`, `REQUIRED`,
`RANGES`, `ENUMS`, `ABBREV`) filled by `main` and read from functions three call-levels
down, which is why its tests need an autouse fixture that reproduces `main`'s setup. Nothing
outside the script can call any of it.

**What the last day established, which is the argument for keeping it:** the pass finds real
defects. Differential validation — validating the repaired record against its own input —
found 665 introduced violations across 15 records that nobody had seen, and then, once the
guards were written, 21 across 37. Every one traced to a specific, nameable mistake:
`correction_scope` written onto the wrong class, references overwritten instead of unioned,
a value coerced past its declared range, an entity pruned on a checker's bad day.

**And the two models earn their place.** NuExtract links entities the extractor left
unlinked — a scanner, a diagnostic instrument, four ROIs on an olfactory-cortex study — and
its proposals are gated by MiniCheck, which rejects a bad citation cleanly: 0.041 for an
acknowledgements sentence offered as a warrant, 0.919 for a real one.

## What was built

Seven modules, all inside `extraction/`, and the placements are what the review of the first
draft argued for rather than what the first draft did.

```
extraction/repair.py               the four steps, and the contradictions worth asking about
extraction/recall.py               Proposer, NuExtract 3, templates, sweep order, candidates
extraction/evidence/grounding.py   Checker, MiniCheck, and the policy: what can be grounded,
                                   which proposals the paper supports, which spans do not
extraction/record/edit.py          the single guarded write path, and the six refusals
extraction/record/ids.py           the address convention, in one place
extraction/record/validate.py      `Validator.diff`
extraction/record/rules.py         `check_one_protocol_per_acquisition`
formats/values.py                  `cast` and `shape`
extraction/stages.py               `Repair`, seventh in DEMAND_DRIVEN, on by default
```

Four things changed from the plan above, each because the code disagreed with it:

**`guards.py` did not survive as a module.** It had one caller, its `Edit` existed to carry
that caller's locals, and its `Refusal` was that caller's return payload -- so `edit.py`
imported it for its own type. `Guard.after` was copied from `repairs.Repair.after` without
the `check_order` that gives it meaning, which is how a field described as "the mechanism I
needed twice and did not have" was still not had. The six checks and their docstrings moved
into `edit.py` unchanged; the ceremony did not.

**The grounding policy moved out of `repair.py`.** Grading a proposal and dropping a span
that supports something else are the same judgement the `Checker` makes, so they sit beside
it. `repair.py` keeps the orchestration and the adjudication.

**`CONTAINER` became `Schema.containers()`.** The plan listed it for deletion and the first
draft rebuilt it by hand -- and the hand-kept version was already missing `ExternalDataset`,
which is the failure a hand-kept table has.

**The models are built once per process, not per stage call.** `Stage.run` is per paper and
under a thread pool, so constructing them there loaded ~10 GB of weights for each of 52
papers and put two proposers on one card.

## What the first draft got wrong

Worth recording, because the pattern repeated and is the thing to watch for next time.

**The stage had never run.** `paper.text()` raised `TypeError` on its first line of real work
-- `Paper.text` is a property returning a Path -- and the adjudication read `reply.body`,
which is an attribute of the `MalformedReply` exception rather than of a reply. Both were
hidden by test stubs that duck-typed the contracts wrongly, which is the failure `Prompt`'s
own docstring already memorialises.

**Four things were plumbed and unused**: the checker was threaded through the sweep and never
called; `--checker-device` was accepted and ignored; `pattern` was a rule condition the
validator silently skipped; `same_entity` was only consulted on a path that does not mint
duplicates, and no caller ever supplied the abbreviations it needs. Each read correctly and
did nothing. Unit tests do not catch this shape; running the thing does.

**Two silent data faults.** `cast` began with `str(value)`, so a multivalued slot given
`["a", "b"]` took the single string `"['a', 'b']"` -- legal enough that the validator passed
it. And the pass validated extraction-shaped records against the storage schema, reporting
every `ExtractedValue` wrapper as `must be a string, got dict`: twenty findings on one
record, none of them real.

## First: delete, do not move

Most of the script's "infrastructure" is a second implementation of something pondie already
exports. Written under time pressure against a schema I was still learning, and each one is
a place the two can now disagree. These are deletions, not migrations:

| script today | already in pondie | why mine is worse |
|---|---|---|
| `reference_slots()` | `Schema.classify()` -> `"reference"`, and `Schema.iter_slots()` | mine re-derives "range is a class and `inlined` is False", which is exactly `classify`'s body -- and `classify` documents why it reads the schema's own `inlined` rather than LinkML's inference. `tests/test_schema_reader.py` pins the reference set; mine is pinned by nothing |
| `declared_slots()` | `Schema.attributes()` | a one-line wrapper |
| `RANGES` table | `Schema.ranges(slot)` | mine reads `slot.range` directly and so misses `any_of` ranges, which is why `groups.n` came back `None` |
| `ENUMS` table | `Schema.enums` | a dict comprehension over it |
| `study_keys()` | `Schema.attributes("Study")` + `classify` | same derivation, done twice |
| `text_of()` | `values.read()` / `values.read_scalar()` | mine handles fewer wrapper shapes |
| `is_field` (in render) | `values.is_field()` | mine tests a different key |
| my leaf walkers (4 of them) | `values.iter_fields(node, path)` | four hand-rolled walks with four different path conventions |
| `wrap()` | `values.wrap(value, source=, evidence=)` | mine defaults `evidence`, which `values.wrap` deliberately refuses to do because `extracted` + `not_applicable` is a hard schema error |
| `id_index()` | `Validator.index_ids()` | mine walks blindly; pondie's is schema-guided and follows type designators |
| `sections()` | `evidence.retrieval.sectionize()` | mine already calls it and then re-implements the joining |
| `acronyms()` + stoplist | `vocabularies.abbreviations` | already replaced; the stoplist is gone |

**What is genuinely new, and is what actually moves:**

| script today | becomes | note |
|---|---|---|
| the twelve guards | `repair/guards.py` | pure `(target, slot, value, record, schema) -> Refusal \| None` |
| `typed()` | `formats/values.py` | a cast to a slot's declared range and vocabulary belongs beside `wrap` |
| `carried_evidence` | `repair/edit.py` | uses `record/spans.py` to resolve, as it already does |
| `sweep_order`, `reference_block` | `repair/sweep.py` | both derive from `iter_slots` once it replaces `REF_SLOTS` |
| `orphans_cell_terms`, `model_terms` | `record/rules.py` | `_chain_terms` is already there and does most of it |
| `introduced()` | `Validator.diff(before, after)` | it is a validator concern |
| `bind_foci`, `stage1_analyses` | `repair/recall.py` | the necessary condition for an analysis |
| `strip_provenance` | `repair/report.py` | the audit belongs in the report, not the record |
| `mint_id` | `repair/edit.py` | the prefixes come from the extraction prompt's rule 6 and should be read from one place, not two |
| `QuoteIndex` (render_record) | `formats/` | a record rendering, not a repair concern |
| `expand_spans.py` | `record/spans.py` | deterministic, no models, belongs with the span tools |
| the CLI | `pondie repair` | alongside `pondie extract` |

## Style

Matching what is already there rather than inventing:

* **A docstring says why, not what.** Every non-obvious decision in pondie carries the
  measurement or the failure that forced it -- `classify` explains why it does not trust
  `SchemaView.is_inlined`, `values.wrap` explains why `evidence` has no default. The guards
  each have a paper behind them and should say so.
* **Failures are counted, not printed.** `StageOutcome` and `RepairLog` in `record/repairs.py`
  are the pattern: a log the caller can query, not `print` at three call depths.
* **No module-level mutable state.** `repair_loop`'s six globals become one frozen dataclass
  passed explicitly, which is what makes the guards pure and the tests fixture-free.
* **Protocols for the heavy dependencies**, as `Caller` already is for the LLM.
* **`from __future__ import annotations`**, `Strict` pydantic models for payloads, `Path`
  everywhere, and the `#:` comment form for field documentation.

## Guards, and the tests that hold them

Thirty-three tests exist (`test_repair_guards.py`), one per failure mode, each naming the
paper it was found on. They move to `tests/test_repair_guards.py` unchanged in substance;
what changes is the fixture — the autouse block that reproduces `main`'s globals disappears
once the tables are a value passed in.

The guards, and what each stopped:

| guard | found on |
|---|---|
| an edit keeps its warrant when the span still supports it | 23021615 |
| an edit that only shortens is not a correction | 22952599 |
| a value is cast to its declared range, or refused | 28416565, 16701903 |
| a value outside a closed vocabulary is refused | 28888350, 21418787 |
| a slot is written only on a class that declares it | 23021615 |
| scope and the regions beside it must agree, both ways | 11950456, 19538748, 19996042 |
| a multivalued reference unions, and holds each target once | 12853571, 23021615 |
| nothing references itself | 27082610, 19942229 |
| repointing may not orphan the terms a cell names | 19942229 |
| an entity something points at is never pruned | 22952599 |
| pruning is skipped when the checker fails most of the record | 16038682 |
| a class is swept after what it points at | 16508348 |
| foci bind on a row group, never positionally | 22952599 |
| a null row group is still a row group | corpus-wide, 142 of 1,822 |

## Open questions that remain

**A created `Analysis` still cannot be valid.** It requires nine slots; a flat template
carries `name`, `definition`, `spatial_scope`, `prespecification` and two references.
`effect` needs a Cell with a ModelTerm to point at, and `groups` needs AnalysisGroup objects
-- both nested, so `nu_type` never asks for them and a proposal cannot carry them. The pass
refuses by name, which turns "should it write analyses" into "have the proposer return cells
and groups, and it will". Nothing is invented in the meantime.

**`REASONED` is a hand-written list.** The durable answer is `value_source: generated` --
which `groundable` already honours -- or a LinkML subset on the slot. Matching by bare slot
name catches `Measure.type` and `ModelEstimation.stage` along with the ones meant.

**The port has not been measured on a corpus.** `repair.run` has been exercised end to end on
one paper and the suite covers the guards, but the 52-record comparison that motivated all of
this was produced by the prototype scripts. Whether the ported pass reproduces those results
is untested, and is the next thing worth doing.

**Two prototype capabilities were not ported**, deliberately and worth restating: foci
binding, and pruning entities. The second is the larger omission and the better decision --
every serious loss measured came from the prune, and pondie's repair is purely additive.
