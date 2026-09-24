# Replacing the frontier model, stage by stage

> Where this sits: [pipeline-architecture.md](pipeline-architecture.md) is the pipeline and
> the alternatives measured against it; [field-extraction-audit.md](field-extraction-audit.md)
> is every non-frontier method run against this corpus so far. This file asks a narrower
> question of both: **which stages could be served by a small local model, which could be
> served by one fine-tuned on this pipeline's own output, and which must not be.**

The worry that motivates the question is the right one: extraction here is not slot-filling
over independent fields, it is binding a fact to the entity it belongs to across a whole
paper. This file argues that the binding problem is **already solved by the workflow rather
than by the model**, that this is what makes substitution tractable, and that it is also what
bounds it — one stage holds the whole paper at once, it is the cheapest stage in the pipeline,
and it is the one that must stay.

---

## 0. The short answer

| stage | model today | can it be replaced | by what |
|---|---|---|---|
| `tables` `prose` `split` `build` | none | — | already deterministic, 4 of 9 stages |
| `demands` | frontier, 1 call | **no** | the only stage that must hold the paper, the table parse and the contrast set at once. ~12% of spend, ~100% of the cascade risk |
| `satisfy` | frontier, 1 call | partly | bounded by the shopping list; failure is a dangling reference and a post-condition already catches it |
| `fill` | frontier, ≤6 calls | **yes, largely** | slots arrive path-addressed with their owner named. This is the shape NuExtract3 already clears 80% on for ~30 fields |
| `evidence` | frontier, 4–7 calls, **45% of input** | **yes, and first** | extractive, abundantly labelled, and byte-exactly verifiable. The best distillation target in the pipeline by a wide margin |
| `repair` | frontier, 2 calls | optional today | its deterministic half is most of its value; its model half is wrong about 11 of 19 changed fields |

Two things worth doing **before** any of it, because each is cheaper than a local model and
neither costs accuracy:

1. **Turn the prompt cache on.** Every call reports `cache_write_tokens: 36423,
   cached_tokens: 0, cache_status: DISABLED`. Five stages send a near-identical 36k prefix and
   each pays full input price. That is gateway configuration, not code
   ([pipeline-architecture.md](pipeline-architecture.md) D2), and it is worth more than any
   substitution below.
2. **Run the derivers before the model, not after.**
   [deterministic-fields.md](deterministic-fields.md) already says so and it is still not the
   order: a field already filled is a slot `fill` never asks about.

---

## 1. Where the money actually goes

~310k input and ~27k output tokens per paper, an 11.5:1 ratio. The output is small; the input
is the whole cost, and **most of the input is not the paper**:

| prompt component | chars | share |
|---|---:|---:|
| `extraction-readme.md` (conventions) | 46,982 | 32% |
| `representing-models.md` §5 (worked models) | 32,472 | 22% |
| rendered schema, entity side | 62,075 | 42% |
| **the paper itself** | 27,567 | **19%** |

That table (from
[extraction-workflow-experiments.md](extraction-workflow-experiments.md) §2) is the single
most important fact for this question, and it cuts two ways.

**It is why a local model looks infeasible and is not.** A 147,644-character prompt does not
fit comfortably anywhere cheap. But four fifths of it is *the schema and its conventions
restated in English on every call* — which is exactly what fine-tuning replaces. A student
that has the schema in its weights, or a grammar over the projected JSON schema, is sent the
paper and nothing else: ~9k tokens, which fits a 32k-context 4B model with room for the
answer. **The context-window objection to local models is an artefact of prompting, not of
the task.**

**It is also why per-call decomposition is cheap and why it is not the lever.** All 23
entity-mode classes rendered one at a time sum to 57,523 chars — *less* than the 62,075 of the
monolithic render — so splitting a pass into per-class calls does not cost 23× against a
shared prefix. What it costs is measured in §3, and it is not tokens.

---

## 2. What has already been run against this corpus

Consolidated from [field-extraction-audit.md](field-extraction-audit.md); nothing here is new
measurement.

| method | scope | headline |
|---|---|---|
| string / normalised match (`surface`) | 158 fields | 67 fields ≥90% locatable; **36 model-only** |
| ambiguity + section scoping (`uniq`) | 158 fields | 20 fields ≥80%; **78 below 20%** |
| targeted regex | 20 fields | 8 shipped at 93.7–100% |
| spaCy `en_core_web_sm` | 5 fields | 38–75% recall, 1,557 `ORG` spans per 8 papers |
| GLiNER `medium-v2.1` | 5 fields | `regions.name` **94%** recall, `devices.manufacturer` 82% — and 562 spans for 36 entities |
| GLiNER2 `base-v1` | 6 fields | 0% on 4 of 6; abstains where a regex is 100% |
| NuExtract-2.0-2B | 6 fields | right where `surface` is high, 0/6 where the value is a schema code |
| **NuExtract3 4B** | **78 fields** | **30 fields ≥80%**; pooled 62% then 44% on two sweeps; **`Cell.direction` 37%** |
| NuExtract3, table splitting | 15 tables | 12/15 exact point totals, 8/15 exact splits |
| cross-encoder retriever, evidence | 173 slots | top-1 42.2% against the model's 67.1%; **union 73.4%** |
| MiniCheck grounding | repair proposals | rejects a bad citation cleanly, and cannot see the error class that actually occurs (§7) |

**The boundary this establishes is sharp and it is not about model size.** A small model
succeeds where the value is printed on the page and fails where the value is a schema code the
paper never writes (`diagnostic_system` wanting `DSM-IV`, `blinding`, `assignment_structure`)
or a paraphrase of prose (`name`, `definition`, `description`, `interpretations` — all 0%).
Two unrelated architectures hit the same boundary on the same fields, and the `surface` column
predicted it before either ran.

**The missing measurement is the middle of the size range.** Every local arm tried here is
2–4B, and the only comparison is against a frontier reasoning model. Nothing between has been
run — `OLMo 3 7B` is noted as easy to try and was not tried, and no 30B-class open-weights
model appears anywhere in these docs. That gap matters because `Caller` is an OpenAI-compatible
protocol and `--model` is a string: **a whole-pipeline open-weights arm costs zero lines of
code.** Serve Qwen3-32B or a gpt-oss-class model with vLLM, point `OPENAI_API_GATEWAY` at it,
run `pondie extract --run oss`. That is the cheapest information available in this whole
document and it should be bought before anyone fine-tunes anything.

---

## 3. Three ways to make a task smaller, and which ones survived contact

This is the part that answers the long-range-context worry directly, because all three have
been tried here and they did not fare alike.

**Slicing the input loses.** Handing the evidence pass twelve retrieved sentences instead of
the paper cost **21.4 points** (67.1% → 45.7%) to save 45% of the prompt, and the cause is
structural: the retriever's recall at twelve is 69.9%, so a shortlist caps the model below
where the whole-paper arm already sits
([evidence-union-design.md](evidence-union-design.md)). Dropping sections did nothing at all
([text-preprocessing-experiments.md](text-preprocessing-experiments.md) §5.3). Sliding-window
chunking was rejected without running, on the grounds that it fights the whole-paper view the
analyses pass needs.

**Slicing the question misattributes.** The repair sweep originally asked per class — 28 calls
a paper, the same 40k-character premise each time. Every error it made was a real fact from
the paper written onto the wrong entity: an excluded patient's drug as the cohort's medication,
a subgroup's mean age as a group's. `propose_with_extractor.py` says why in its own comment:
*asked only about `Group`, the model cannot see that the number in front of it was given for
an excluded participant*. The fix was to ask about the whole record in one call. **Narrowing
the question removes exactly the context that would have prevented the error.**

**Structuring the output wins.** The one thing that moved scoping is a template with a
repeated group. Asked for `{"acquisitions": [{"modality", "repetition_time_ms",
"echo_time_ms", ...}]}`, a 4B model returns one row per acquisition with the TR on the MRI row
and `null` on both PET rows — **86%** on comparable TR values, against 68.8% for a paper-wide
regex, 33% for a Methods-scoped one and a 22% `uniq` ceiling for any surface method. And
richer rows scored better than narrow ones: TR/TE went 83% → 100% when each row also carried
modality, field strength, pulse sequence and volume count. The model is not being asked *which
of six `df = 48` is the right one*; it is being asked to emit one row per acquisition, and
**the scope comes out of the shape of the answer.**

The same principle is the pipeline's own largest measured win. Demand-driven ordering — the
analyses declaring the entities they need before any entity exists — took direction F1 from
38.1% to 80%, because a cell cannot be righter than the term it points at. That is an
output-structure change, not an input or a prompt change.

> **The rule for any substitution: decompose by output structure, never by input slicing, and
> never by narrowing the question below the entity that carries the ambiguity.**

---

## 4. Where the long-range binding actually lives

Given that rule, the useful audit is not "how big is the paper" but **how much of the binding
each stage has to discover for itself, and how much arrives as input.**

| stage | what it must discover | what arrives already bound | verdict |
|---|---|---|---|
| `demands` | which analyses exist, which entities they need, whether a term is categorical or continuous, and what its levels are | the stage-1 table parse, grouped by table, with foci counts and parse keys | **irreducible.** Nothing tells it; it decides, and everything downstream inherits the decision |
| `satisfy` | each declared entity's own attributes | the shopping list: every `local_id`, `kind`, `label`, `term_type` and `levels` it must emit | bounded. The contract is checkable and a post-condition already checks it |
| `fill` | one value, or a reason there is none | the full path `groups[grp_ptsd].age_mean`, the owner's label, the declared range, the vocabulary and a gloss of each term | **already a per-slot question with the entity named.** No binding left to discover |
| `evidence` | which characters support this value | the path, the entity and the value itself | **retrieval, not reasoning** |
| `repair` | entities and links nobody asked for | the whole record and the paper | needs the widest view of the four, and it is the one measured to misattribute |

The reason substitution is possible at all is in the third and fourth rows. `fill` addresses a
slot as `path + owner label + type + vocabulary`, because `_label` exists precisely so the
model knows which entity it is answering about — a bare `local_id` printed "Cell ?" and the
model returned the right set of directions assigned to the wrong cells, *which is what guessing
looks like when the guesser knows the shape*. That fix means the binding is carried in the
question, and a student model never has to rediscover it. Likewise `evidence` is handed
`path = value` per line.

So the honest framing is: **the long-range work is concentrated in one call.** `demands` is
one call per paper, of nine stages, and everything else is downstream of a contract it wrote.
Keeping a frontier model there and nowhere else is both the safest and the cheapest split
available, and it is roughly the inverse of where the money currently goes.

---

## 5. The training data that already exists, counted

Distillation needs the teacher's output, not reviewer gold, and that distinction is what makes
this different from the fine-tuning route
[pipeline-architecture.md](pipeline-architecture.md) §6 records as blocked. The blocker there
is real and specific — 4 of 281 entity-salience dispositions answered — and it blocks *learning
salience*, which is a judgement no record carries. It does not block learning to reproduce a
record the pipeline already produced.

Counted over the 16 records in `benchmarks/candidate/`, which are ordinary pipeline output:

| | per paper | × 1,817 records |
|---|---:|---:|
| `ExtractedValue` fields | 221 | ~400,000 |
| filled (`extracted`) | 198 | ~360,000 |
| **fields carrying a resolved span with character offsets** | **155** | **~280,000** |
| analyses | 5.7 | ~10,000 |
| `Effect.cells` | 11.0 | **~20,000** |
| distinct field paths in the schema | 153 | — |

78% of extracted fields carry an offset-resolved span. Those spans are not an annotation
someone would have to make: `warrant.py` resolved each one against the normalised text and the
integrity gate asserts `text == source[start:end]` on every one of them. **That is a
quarter-million span-supervision examples sitting in the corpus.**

The per-stage payloads are also training data in the form a student wants: `payloads/<id>/
demands.json`, `satisfy.json` and the filled copies are literally `(paper [+ context]) →
(payload)` pairs for each pass, one per paper.

Three cautions about using any of it:

- **The teacher is not uniformly right.** `audit_records.py` counts 3,200 errors over the
  corpus falling to 1,828 after deterministic fixes; `record-defects.md` names seven classes.
  Training on unfiltered output trains the defects in.
- **A free quality filter already exists.** Keep only records that validate clean, whose
  post-conditions passed on every pass, whose quotes resolved, and which the audit sweeps do
  not flag. That is a large, cheap, entirely deterministic filter on label quality, and using
  it costs nothing but corpus.
- **The ceiling is the teacher.** A student distilled from luna cannot beat luna, and on the
  two things this pipeline is measured on — direction at ~96.6%, and 55/55 for the
  deterministic deriver against corrected gold — the teacher is already at or near the gold's
  own noise floor. Distillation here buys cost and independence, never accuracy.

---

## 6. Four students, in the order their failures are cheapest

The ordering criterion is deliberately not "how much does it save". It is **how localized and
how detectable is the failure**, because that is what decides whether a substitution can be
adopted without a gold set that does not exist (§8).

### S1 — the evidence student. Do this one first.

*Task:* given the paper, a field path and its value, return the span that warrants it.

*Why it is first:*

- **Biggest single cost.** 45% of input tokens, 4–7 calls a paper, and the pass is the one
  stage whose output the benchmark does not score at all.
- **Labels are abundant and free.** ~280,000 resolved spans, each with offsets.
- **Verification is deterministic and already implemented.** A proposed quote either resolves
  byte-exactly against the normalised text or it is dropped, and the field falls back to
  `not_found` — which is a status the record already carries 689 times in 16 papers. **A wrong
  student answer degrades to the current failure mode rather than to a new one.** No other
  stage has that property.
- **The task is extractive.** No paraphrase, no schema code, no invention — the failure classes
  that kill small models on this material.
- **It fits the hardware.** Formulated as ranking sentence units against a field query, a
  training example is ~256 tokens, not 12k. That trains on one 8 GB card; a long-context
  generative SFT does not (NuExtract3 4B in bf16 needs four of the 3070s, and only the W4A16
  checkpoint with `--language-model-only` fits one).

*Two forms, and the cheap one is the honest first try:*

| | form | baseline it must beat |
|---|---|---|
| **S1a** | a trained re-ranker over sentence units — ModernBERT/SciBERT/Longformer, query = field leaf + aliases + value surface forms, trained on the 280k teacher spans | the **untrained** cross-encoder at top-1 42.2% and recall@12 69.9%. Those numbers are an off-the-shelf model with hand-written bonuses, not a trained one |
| **S1b** | a small generative model emitting the quote, grammar-constrained, gated by the byte-exact resolver | the model pass at 67.1% confirmed-correct, union 73.4% |

*What would falsify it:* S1a failing to clear ~70% top-1 on held-out teacher spans would say
the query formulation, not the training, is the limit — the retriever's known weakness is
precisely the 57% of gold evidence whose supporting sentence shares no vocabulary with the
value, and training cannot conjure a lexical overlap that is not there. In that case S1a
becomes a *first voter* (it already earns +6.3 points as one) and S1b is the real candidate.

*What it is not:* a shortlist for the frontier model. That was measured and it costs 21.4
points. The student must answer, not narrow.

### S2 — the fill student.

*Task:* given the paper and a batch of path-addressed open slots with owner, range and
vocabulary, answer each with a value or one of five reasons.

*Why it is second:* the binding already arrives in the question (§4), abstention is a
first-class answer the schema has a token for, and an unsettled slot is the pipeline's normal
state rather than a defect. The measured fit is direct — NuExtract3 clears 80% on ~30 of these
fields zero-shot, and the fields it fails are enumerable in advance from the `surface` column.

*The one change worth making while doing it:* ask per entity with a nested template rather than
as a flat list of 250 lines. That is the output-structuring result of §3 applied to the stage it
most obviously fits, and it is testable with the frontier model first, which separates the
decomposition from the substitution.

*Route, do not replace wholesale.* Deterministic derivers → student on the ≥80% set → frontier
on the remainder. `apply_fill` already discards an answer under an id it did not ask about and
refuses to overwrite a settled slot, so the routing has a guard rail before anything is written.

### S3 — the satisfy student.

*Task:* given the paper and the shopping list, emit exactly the declared entities with their
attributes filled.

*Why it is third:* the contract is explicit and `postcondition_failures` already checks it —
a declared entity the pass did not emit is named, and the retry names the fault. So the
dominant failure is *detected*, and detection is what makes a substitution adoptable. What is
not detected is a filled-but-wrong attribute, and that is the same misattribution class §3
warns about, so this arm needs the whole-record view kept and the per-class temptation
resisted.

*Expect to keep paraphrase fields on the teacher.* `name`, `definition`, `description`,
`interpretations`, `model_settings`, `source_definition` measure 0% for every small model
tried. [pipeline-architecture.md](pipeline-architecture.md) P8 already argues that fields no
analysis reaches do not deserve budget on the pass that decides cells; the same argument says
they do not deserve a frontier call either, and the cheapest resolution may be to defer them
rather than to distil them.

### S4 — the contrast student. The interesting one, and the one to attempt last.

*Task:* the cells — which term, which level, which sign.

*The case against:* NuExtract3 scores **37%** here against the pipeline's 96.6%, and the failure
is not a sign error. It emits `term: "FESZ", level: "NC"` — both sides of one comparison stuffed
into the two slots — because it has not understood what a `Cell` is. It also under-generates by
half, 38 cells where the record has 71. No threshold or template tweak fixes a category error.

*The case for trying anyway, once:*

1. **~20,000 teacher-labelled cells exist**, which is the one place in this corpus where
   distillation has a large, structured, homogeneous target. The 37% was zero-shot; nothing has
   ever fine-tuned on this shape.
2. **The shape the student fails at is the shape the schema is being asked to change anyway.**
   [pipeline-architecture.md](pipeline-architecture.md) D1 proposes replacing nested
   `Effect.cells[]` with a flat `{parse_index, axis_term, plus[], minus[], within[],
   sign_status}`, in which over-celling, `held`→signed, malformed direction, level paraphrase
   and dangling term references are all *unrepresentable* — and it says in as many words that
   this "gives a small model a chance at the contrast layer, which the nested shape
   demonstrably does not". D1 and S4 are the same experiment, and D1 should be run with the
   frontier model first so the schema change and the model change are not confounded.
3. **Half the cells never need a model.** The contrast-name deriver resolves 55 of 101 signed
   gold cells at 100% and abstains on the rest. The student's real job is the 46 abstentions.

*And the reason to attempt it last:* direction carries 0.45 of the composite, is the one fact a
synthesis cannot recover from anywhere else, and is currently at or above the reviewer gold's
own error rate — of 17 sign flips investigated across four runs, **three were real and all three
were errors in the gold**, not the extractor. There is no headroom to buy and a great deal to
lose.

---

## 7. What no student and no small model can do here

Recording these so they are not rediscovered.

- **Salience.** GLiNER finds 562 region spans for 36 `Region` entities and 94% recall is not
  the problem; deciding which mention became an entity is. That needs the 281 reviewer
  dispositions, of which 4 are answered, and no amount of teacher output substitutes — the
  records contain the decisions, not the reasons.
- **Misattribution.** The dominant error class in the repair pass is a real fact from the paper
  attached to the wrong entity. A grounding model cannot see it, because the token genuinely is
  in the document; `.agent/repair/investigation.md` states that no gate proposed there catches
  it, and it is about one changed write in two.
- **Values the paper never writes.** 36 fields are model-only at `surface` < 10%. `is_healthy`
  is the clearest: asked directly, 168 of 1,817 records called a cohort healthy beside a real
  diagnosis, 132 citing the paper's own wording — which is why it is derived from
  `medical_condition` in code instead.
- **A verbatim test as a correctness check.** Four separate findings in this repo were the same
  mistake. A duration, a definition, a minted id, a derived label and a synthesised mirror are
  none of them quotations, and testing them against the document measures the record's
  vocabulary rather than the paper's content. Any acceptance gate a student is put behind
  inherits this rule.
- **Few-shot examples on numeric fields.** Measured: 83% → 71%, by value copying from the
  examples. Use examples for shape, never for fields whose values are numbers.

---

## 8. The measurement problem, which is the real blocker

Every substitution above is a bet that the student is not worse. The corpus cannot currently
settle that bet:

| gold that exists | size |
|---|---|
| verified whole records | **1** (`xevP8UDRAVh9`) |
| reviewer direction tables | 14 papers, 101 cells |
| hand-judged evidence slots | 173 |
| hand-read repair truth papers | 4 |
| entity-salience dispositions answered | 4 of 281 |

And the substrate is noisy where it is thin: at n=1 the same configuration run four times had
a direction F1 standard deviation of **20 points on a mean of 10**. (The later 0.1% replicate
spread is a different measurement — the 101-cell polarity metric after the scorer's operator
bug was fixed — and it is the one to use.)

**So do not try to prove a student is as good. Prove it does not diverge, and measure that at
corpus scale.** Everything needed already exists:

| instrument | what it answers | where |
|---|---|---|
| `benchmark.scoring.compare(teacher, student)` | per-field, per-entity and per-relationship agreement, via optimal bipartite assignment — no gold required | `pondie/benchmark/scoring.py` |
| `audit_records.py` | six deterministic defect sweeps over any corpus of records | `scripts/audit_records.py` |
| `Validator` + `Validator.diff` | findings introduced relative to a reference record | `record/validate.py` |
| the warrant counters | spans placed, unresolved, unlocated, downgraded | `evidence/warrant.py` |
| post-condition failures per pass | degenerate and partial payloads | `render.postcondition_failures` |
| the 101-cell polarity gold | the one number with a real answer, sd 0.1% | `pondie benchmark` |

Run the student arm over the same 1,817 papers and report: agreement against the teacher per
field, defect counts from the audit, span-resolution rate, post-condition failure rate, and
polarity on the 101 cells. A student that agrees 95% per field, introduces no findings, resolves
spans at the same rate and holds polarity is adoptable on this evidence. One that quietly drops
`Region` entities shows up in the entity F1 of the comparison, on 1,817 papers, without a single
new label.

The repair pass's own gates are the model for this — M1/M2/M3 are deterministic, free, and
caught a pass that was destroying 227 spans and downgrading 141 provenances before anyone
scored a value.

---

## 9. The order to actually do this in

Each step is cheap, each has a falsifier, and each changes what the next one is.

1. **Turn caching on.** Zero accuracy risk. Measure the input-token drop per paper. *Falsified
   if* `cached_tokens` stays 0, in which case it is a gateway limitation and should be written
   down as one.
2. **Run the whole pipeline on a 30B-class open-weights model.** No code. One run over the
   benchmark papers plus a corpus-scale agreement run. This is the largest single unknown in
   this document. *Falsified if* polarity falls below ~90% on the 101 cells or the post-condition
   failure rate rises materially — at which point the answer is "not the whole pipeline", and
   §6's ordering is how to spend the rest.
3. **Build the corpus-scale non-regression harness** (§8). It is a script over instruments that
   already exist, and nothing after this point is interpretable without it.
4. **Train S1a, the evidence re-ranker**, on the 280k resolved spans. Score against held-out
   teacher spans and against the 173 hand-judged slots. *Falsified if* it cannot beat the
   untrained cross-encoder's 42.2% top-1 by a wide margin; that would say the query, not the
   training, is the limit.
5. **Re-shape `fill` to nested per-entity templates, with the frontier model**, and measure. This
   separates the decomposition from the substitution, and §3 predicts it is worth something on
   its own. *Falsified if* field agreement drops or the open-slot count stops falling.
6. **Then S2**, routed: derivers → student on the measured ≥80% set → frontier on the rest.
7. **Run D1, the flat contrast schema, with the frontier model.** Only if it lands does S4
   become a sensible question.
8. **Never substitute `demands`** without a stratified gold set that does not exist — ~5 papers
   each across ~8 analysis shapes, which
   [extraction-workflow-experiments.md](extraction-workflow-experiments.md) already names as the
   binding constraint on everything else.

---

## 10. Do not re-propose

| idea | why not | where it was measured |
|---|---|---|
| retrieve-then-extract for evidence | −21.4 points for −45% prompt; the retriever's recall@12 caps the model below where it already is | [evidence-union-design.md](evidence-union-design.md) |
| per-class sweeps to shrink a call | removes the context that prevents misattribution | `repair/propose_with_extractor.py` |
| dropping sections from the paper | no effect at any n, and it removes content for nothing | [text-preprocessing-experiments.md](text-preprocessing-experiments.md) §5.3 |
| an entailment model to gate proposals | the failure class is scope, and the token really is in the document | `.agent/repair/investigation.md` D4 |
| off-the-shelf biomedical NER | returns typed spans; the bottleneck is assigning a value to an entity instance | [deterministic-fields.md](deterministic-fields.md) |
| coreference resolution | 37% appended, 27% substituted, no gain | [pipeline-architecture.md](pipeline-architecture.md) §3 |
| NuExtract3 for stage 1 table splitting | recovers coordinates that are already deterministic from the CSV, and under-segments 7 of 15 | [field-extraction-audit.md](field-extraction-audit.md) |
| pruning unreferenced entities | all six unreferenced entities on the one verified paper are correct; unreferencedness is a symptom of a missing analysis | [extraction-workflow-experiments.md](extraction-workflow-experiments.md) §3 |

---

## 11. What this document does not establish

- **No arm here has been run.** Every number is drawn from measurements already in this
  repository; the synthesis, the ordering and the cost attribution are arguments, not results.
- **The 280k span count is an extrapolation** from 16 records in `benchmarks/candidate/` (2,478
  resolved spans, 155 per paper, 78% of extracted fields) to the 1,817 committed records. It
  assumes those records carry evidence at a comparable rate, which has not been checked
  corpus-wide.
- **The cost split is the documented one** (~310k in / ~27k out, evidence 45% of input) and is
  not re-derived from `usage.jsonl` here. A per-stage token table from a real run would sharpen
  every priority in §9 and is half an hour of work against a corpus this document did not have.
- **`docs/serving-the-proposer.md` describes code that is no longer in the package.** The
  in-process NuExtract proposer, the vLLM `NuExtractServer`, `Settings.proposer_url`, the
  MiniCheck grounding step and the cross-encoder ranking all went; `evidence/retrieval.py` keeps
  only `sectionize`, and `README.md`'s account of `repair` having a local half is stale. That
  serving document remains the best record of how to run a quantised model on these cards and
  should be kept, but it should say that it documents a removed arm.
