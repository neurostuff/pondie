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

## 6a. The two small-model classes, and what each is actually for

"Is a 4B model, or a BERT, too small for this work" is the wrong axis, and the audit table
says why. The axis that predicts every result in this repository is:

> **Choosing among candidates the pipeline already has is a small model's job. Deciding what
> the candidates are is not.**

An encoder cannot generate, so it can never say which analyses exist. It can say which of nine
occurrences of `n = 24` is this group's, which sentence warrants this value, and whether a
proposed span supports its claim — and every one of those is a place this pipeline currently
spends a frontier call or gives up.

### Encoders are already here, and both of the ones shipped are untrained

`normalization/_embedding.py` runs SapBERT (a PubMedBERT fine-tune, ~110M) for entity strings
and MiniLM for prose, and [normalization-rationale.md](normalization-rationale.md) measures
them inverting completely by input length: R@1 66.3% and 50.6% on short entity strings, 24.5%
and 58.5% on task descriptions. So the dependency class is already accepted and already earns
its keep on this material.

What has never been tried is a **trained** one. Both of those are off-the-shelf embedders used
zero-shot; so was the cross-encoder that scored 42.2% top-1 on evidence. Every encoder number
in this repository is a zero-shot number, and this corpus holds ~280,000 labelled examples
(§5).

### Five encoder jobs, with the labels that already exist

| | task | labels on disk | window | baseline it must beat |
|---|---|---|---|---|
| **E1** | evidence re-ranking — score (field query, sentence unit) | ~280,000 resolved spans | ~256 tok | untrained cross-encoder, 42.2% top-1 / 69.9% recall@12 |
| **E2** | **occurrence disambiguation** — the value is on the page *k* times; which one is this entity's | the teacher's resolved span picks one of the *k* | **not a window — see §6b, where this is corrected into an assignment** | the `uniq` column, which is this task scored for a string matcher |
| **E3** | warrant abstention — does this span support this value, yes or no | positives = spans that resolved; negatives = `warrant.downgraded`, plus high-ranked sentences the teacher passed over | ~256 tok | the hand-tuned margin cut: 40% coverage at 80% confirmed-correct |
| **E4** | cell signing — label a cell that already exists | ~20,000 teacher cells | **the bound row group, not a retrieved passage — see §6b** | the deriver, which is 55/55 on gold and **abstains on 46 of 101** |
| **E5** | salience — did this mentioned region become a `Region` | the records | ~256 tok | GLiNER at 94% recall and 562 spans for 36 entities |

**E2 is the one worth being excited about, and it is new.** The `uniq` column has been read
throughout this repository as a ceiling — "the value is on the page twenty times and nothing
distinguishes the right occurrence". It is a ceiling *for a string matcher*. It is a
description of a ranking problem for anything that reads the words around each candidate, and
the label is free, because the teacher's span already says which occurrence was right.

Counted off the audit table:

| band | fields | instances (16-paper corpus) |
|---|---:|---:|
| `uniq` ≥ 80 — a string match is already right | 20 | 223 |
| **`surface` ≥ 90 and `uniq` < 20 — on the page, wrong occurrence** | **40** | **758** |
| `surface` < 10 — model-only, no span to rank | 36 | 827 |

Median candidate counts across those 40 fields run 2 to 51: `analyses.groups.n` is 168
instances at 98% surface, 2% uniq, 9 candidates; `groups.enrolled_count` is 9 candidates;
`magnetic_field_strength_tesla` is 18; `smoothing_fwhm_mm` is 18. Pick-one-of-nine with the
entity label in the query is exactly what a cross-encoder does, and 758 instances over 16
papers extrapolates to of order 10⁴–10⁵ across the corpus.

And the training set is drawn from the **highest-precision slice of the teacher's evidence**,
which is the part that makes this better-founded than E1: where the pick contained the value
verbatim, confirmed-correct was 80.9% against 27.8% where it did not
([evidence-union-design.md](evidence-union-design.md)). E2's labels are by construction all in
the first group.

E5 carries a warning the others do not. Distant supervision for salience was tried here and
**inverted** — unused mentions came out more frequent and more often in Methods, because the
negatives are dominated by extractor recall misses and class confusion. The teacher's silence
is not a reliable negative. E5 needs the reviewer dispositions the rest of this document says
are unavailable, and should be attempted last or not at all.

### What fine-tuning NuExtract3 would and would not fix

NuExtract3 is itself a template-filling fine-tune, so further tuning it on this schema is the
natural move rather than an exotic one. Read against its measured failures:

| its zero-shot failure | would a fine-tune fix it |
|---|---|
| schema codes the paper never writes — `diagnostic_system` answered `DTI`, `SVM`, `C-PiB PET` | **yes, and this is the strongest case.** It is a vocabulary-grounding failure, which is exactly what labels teach. The jump from 0/6 (NuExtract-2.0) to 3/3 DSM-IV (NuExtract3 + explicit field names + `enum` constraints) already came from constraining the vocabulary; a fine-tune is the strongest form of the same move |
| `regions.atlas`, 30 predictions against 0 recorded values | **yes** — it is over-generation, and abstention is learnable from a corpus where the slot is usually empty |
| paraphrase fields at 0% (`name`, `definition`, `description`) | **partly.** The teacher's house style is consistent, which is what makes it learnable at all, but 0% is a long way to come and these are the fields P8 argues should be deferred rather than bought |
| `Cell.direction` 37%, `term: "FESZ", level: "NC"` | **probably not**, and D1 changes the target shape anyway. Attempt it after D1, not before |
| under-segmentation — 38 cells where the record has 71, 8/15 exact table splits | unknown, and it is the failure mode least addressed by more of the same labels |

The practical constraint is sequence length, not parameters. A LoRA over 4B at the 10–12k
tokens a paper takes does not fit an 8 GB card; the truncation that makes it fit already exists
as `repair.stage._premise`, which cuts to Methods and Results through `sectionize` for exactly
this reason. Train on the premise, not the paper.

### Where neither class reaches

`demands`. It has to hold the paper, the table parse and the contrast set at once and decide an
inventory — an encoder cannot generate one and the 4B under-segments both tables and cells.
That is §4's conclusion arrived at from the other direction, and it is the same one call.

---

## 6b. Finding the context: join to it, do not retrieve it

The E2 and E4 rows above were first written as "score the candidate against a window of text
around it", and that is wrong for a reason worth stating, because it is the same mistake §3
says loses.

**A window around the candidate usually does not contain the discriminator.** A Methods
sentence reads "twenty-four participants were scanned"; it is about a sample size and says
nothing about *whose*. The thing that makes it this group's is somewhere else — the group was
named two paragraphs earlier, or the number is a cell in a demographics row, or the other
number in the same sentence belongs to the other cohort. A pointwise cross-encoder over ±256
tokens is asked to find information that is not in front of it, which is input slicing wearing
a different hat.

And for a cell's direction it is worse: what settles it is the table's own rows. The rubric's
error investigation is explicit — reviewers sign correctly when the rows are in front of them,
and the one genuine attribution error came from the grid highlighting **another contrast's
rows** while the reviewer read a plausible sentence instead.

### Three kinds of fact, and only one of them is a retrieval problem

| | the value is | how you get its context | examples |
|---|---|---|---|
| **printed** | a string in the document | match it, then *assign* it (below) | `age_mean`, `magnetic_field_strength_tesla`, `software` |
| **bound** | determined by a structure the record already points at | **join. Do not search** | `Cell.direction` from its row group, `Statistic.family` from the parse's value kind, a level's identity from the entity it links to |
| **inferred** | never worded by the paper | there is nothing to find. Derive it and mark it `generated` | `spatial_scope: whole_brain` from no mask, `undirected` from a correlation, `held` from being on neither side, `is_healthy` |

Most of the difficulty in asking "how do I find the supporting text" comes from treating the
second and third kinds as the first. **And the `surface` column already says which is which** —
that is what it was measuring all along.

### The teacher already tells you which fields are not retrieval problems

Free diagnostic, run over the 16 records in `benchmarks/candidate/`: group `evidence.status`
by field path and read off the fields the extraction model itself could not cite.

| uncitable | field |
|---:|---|
| **92.3%** (84/91) | `analyses.prespecification` |
| 73.3% | `design.blinding` |
| 69.2% | `tasks.response_mode` |
| 68.8% | `design.allocation`, `design.assignment_structure` |
| 64.3% | `tasks.design_type` |
| **54.9%** (50/91) | `analyses.source_table_analysis` |
| 42.9% (39/91) | `analyses.spatial_scope` |
| 41.4% | `statistic.degrees_of_freedom_denominator` |
| 40.9% | `design.timepoints.relation_to_intervention` |
| 40.7% | `groups.species` |

Pooled over the 101 field paths with ≥8 instances: **678 of 3,048 filled values, 22.2%, carry
no citation from the model that wrote them.** Every field at the top of that list is an
inferred fact, and a locator aimed at it — trained, retrieved or frontier — is hunting a
sentence that was never written. `groups.species` is the purest case: 41% uncitable, and it is
already derived in code at 100%.

**One of those rows is a defect, not a category.** `analyses.source_table_analysis` is a parse
key the pipeline minted; it is an address, not a claim about the paper, and `mirror_analysis`
already writes its own copies as `generated` / `not_applicable` for exactly that reason. The
extraction pass's copies go through `apply_evidence` like any other value, fail to match, and
land on `not_found` — ~50 per 16 papers, so of order 5,000 spurious `not_found`s corpus-wide,
inflating the 32,500 `warrant.py` counts. This is the rule
`.agent/repair/investigation.md` says was learned the hard way four times: **a verbatim test is
only valid against a value the paper was supposed to have printed.** Fixing it is a one-line
change in how that slot is wrapped.

### E2, corrected: an assignment, not a ranking

The discriminator for "which `24` is this group's" is overwhelmingly *co-location with where
that group is being described*, and **the record already knows where that is**: the group's own
`name`, `medical_condition` and other filled slots carry resolved `start_char`/`end_char`. So
the feature is arithmetic on offsets the record holds, not a semantic judgement about a
sentence.

Two consequences:

1. **Anchor the window on the entity, not on the value.** Score a candidate by its position
   relative to the entity's known spans, whether it shares a sentence or a table row with the
   entity's name or one of its abbreviations, and whether it sits in the section this field's
   warrants usually land in — a prior that is *measurable per field path* from the corpus
   rather than hand-written.
2. **Score the whole matrix, not one cell of it.** "24 patients and 19 controls" is two
   candidates and two groups, and the real constraint is that they are one-to-one. A pointwise
   model cannot express that; an assignment over an entity × candidate matrix can, and it is
   the same optimal bipartite assignment `benchmark/scoring.py` already implements for entity
   matching.

That reframing is §3's rule applied to its own proposal: the scope comes out of the *shape of
the answer* — one number per group, each used once — rather than out of a retrieved passage.
It is also why the nested acquisition template reached 86%: it forced one row per acquisition.
The encoder version of that template is an assignment problem.

The text model in it can then be small, because most of the signal is structural: offset
distance, shared row, unit agreement, section prior. The words break ties.

### E4, corrected: the table is bound, so most of it is not retrieval at all

`Analysis.source_table_analysis` carries the parse key of the listing entry the analysis was
emitted for, and `Analysis.tables` carries the Table's `local_id`. So the row group, its
coordinates, its statistic values *with their signs*, and the table's caption and footer are a
dictionary lookup — the pipeline built that address space precisely because "`tables` cannot do
it, a table usually reports several contrasts and several analyses usually cite the same
table".

Against the deriver's abstentions, that suggests three deterministic widenings before any model
is trained:

| the abstention | what would close it | is it retrieval |
|---|---|---|
| the level matches neither side of the contrast name | expand both sides through **the paper's own abbreviation table**. `direction.same_level` compares word sets and never consults `vocabularies.abbreviations`, so `FESZ` cannot reach `first-episode schizophrenia` even though the paper defines it on first use and the miner already found it | no |
| ditto, where the level is a bare string | match through the entity the `FactorLevel` **links to** — `groups`, `arms`, `timepoints`, `conditions` — rather than only through the level's own text | no |
| the name carries no operator | `fill_directions` already concatenates `name` and `definition`, and the rubric says the definition is what states the ordering. What is missing is the **bound row group's own signs**, which the parse holds and this path never reads | no |
| a slope whose statistics carry no sign | genuinely prose, and this is the residue | yes, but narrowly |

That last one is the only search, and it is a well-posed one rather than an open one: the
sentence names the term or the region and carries a cue word from a small closed set —
increase, decrease, positive/negative correlation, greater, reduced — and it is in Results. A
cue-word-plus-entity-mention query over one section is a different problem from "find the
sentence supporting this abstract fact", and it is the shape E1 is good at.

And where no sentence exists, the correct answer is not to find one. `mirror_analysis` is the
worked precedent in this package: it flips a sign, marks the value `generated`, and **drops the
span**, with a comment saying why — keeping the described half's quote would ship a verified
span supporting the opposite claim, which is a false citation. A cell whose direction was
inferred rather than read should say so the same way.

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
   down as one — and §9a is then a second route to the same saving, because a block-level cache
   makes the reordering `render.py` ruled out pay after all.
2. **Run the whole pipeline on a 30B-class open-weights model** (§9a for which, and on what
   card). No code. One run over the benchmark papers plus a corpus-scale agreement run. This is
   the largest single unknown in this document, and the two numbers that decide it are polarity
   on the 101 cells and the post-condition failure rate. *Falsified if* polarity falls below ~90% on the 101 cells or the post-condition
   failure rate rises materially — at which point the answer is "not the whole pipeline", and
   §6's ordering is how to spend the rest.
3. **Build the corpus-scale non-regression harness** (§8). It is a script over instruments that
   already exist, and nothing after this point is interpretable without it.
4. **Run the uncitable diagnostic over the corpus** (§6b) — `evidence.status` grouped by field
   path over all 1,817 records, which is a script and no model. It partitions the schema into
   printed, bound and inferred, and every later step is aimed with it. It also sizes the
   `source_table_analysis` defect. *Falsified if* nothing.
5. **Close the deterministic widenings on direction** (§6b): abbreviations into
   `direction.same_level`, level matching through the linked entity, and the bound row group's
   signs. Score against the 101 cells. *Falsified if* the deriver's abstention count does not
   fall, which would say the abstentions are the slope case and not the alias case.
6. **Then E2, as an assignment** over the 40 fields where `surface` ≥ 90 and `uniq` < 20,
   anchored on the entity's resolved spans rather than on a window round the candidate. Still
   the cheapest trained model here — one 3070, labels already on disk. *Falsified if* accuracy
   at *k* candidates does not clear the `uniq` column by a wide margin on held-out papers,
   which would say the structural features do not carry it either.
7. **Train S1a/E1, the evidence re-ranker**, on the 280k resolved spans. Score against held-out
   teacher spans and against the 173 hand-judged slots. *Falsified if* it cannot beat the
   untrained cross-encoder's 42.2% top-1 by a wide margin; that would say the query, not the
   training, is the limit.
8. **Re-shape `fill` to nested per-entity templates, with the frontier model**, and measure. This
   separates the decomposition from the substitution, and §3 predicts it is worth something on
   its own. *Falsified if* field agreement drops or the open-slot count stops falling.
9. **Then S2**, routed: derivers → E2 and the student on the measured ≥80% set → frontier on
   the rest.
10. **Run D1, the flat contrast schema, with the frontier model.** Only if it lands does S4
   become a sensible question.
11. **Never substitute `demands`** without a stratified gold set that does not exist — ~5 papers
   each across ~8 analysis shapes, which
   [extraction-workflow-experiments.md](extraction-workflow-experiments.md) already names as the
   binding constraint on everything else.

---

## 9a. Which open-weights model, and on what card

Step 2 above says "a 30B-class open-weights model" and that is the size band, not a product.
What decides the choice here is not a leaderboard; it is five properties this pipeline
specifically exercises.

| property | why this pipeline cares |
|---|---|
| **grammar-constrained decoding under vLLM** | the strongest reason to serve locally at all, and it is not a cost reason. 57% of runs need at least one shape repair from `build`, and the variants it cannot repair crashed the scorer (`Cell.term` as a dict) and lost the shopping list. A real JSON-schema grammar over the projected schema removes that class structurally, which is [extraction-workflow-experiments.md](extraction-workflow-experiments.md) §6 recommendation 5 |
| **block-level prefix caching** | see below. It is worth more than the model choice |
| **an 11.5:1 input-to-output ratio** | this is a prefill-bound workload, so a sparse model with few active parameters gives large-model behaviour at small-model throughput. It is the right architecture for this shape of call |
| **a cheap non-thinking mode** | measured here: low effort suffices, and high effort *lowers* field accuracy by 1.3 sd while tripling wall clock. A model whose reasoning can be turned down, or off, matches what the pipeline actually wants |
| **a permissive licence and, ideally, open data** | the records are a research artefact; a model whose training data cannot be described weakens the provenance claim the rest of this repository is careful about |

### The prefix-caching correction, which is the largest single number here

`prompt/render.py` says the ordering "is not a cache optimisation, and an attempt to make it
one failed": a byte-identical prompt caches 100% on the gateway and a prompt sharing a 3.6k
prefix with a different suffix caches **zero**, so "no reordering can help".

**That is true of the gateway and false of vLLM.** The gateway's cache is whole-prompt exact
match; vLLM's automatic prefix caching is block-level over the KV cache and reuses any shared
leading blocks. The 29,152 tokens of conventions, worked models and paper that every pass
sends — 29k of the 36–42k a call carries — become a real shared prefix the moment the serving
stack changes, and [extraction-workflow-experiments.md](extraction-workflow-experiments.md) §2
already wrote down the reordering that exposes it: conventions → worked models → paper → then
the stage-specific ask.

One thing that write-up does not mention and that has to be got right: the *system* message is
first in the token stream, and `build_prompt` puts the mode-specific note in it
(`SYSTEM_HEAD.format(lists=...) + MODE_NOTE[mode]`). Two passes therefore diverge at the very
first block and share nothing, however the user turn is ordered. The invariant material has to
lead the whole stream, not just the user half.

Done properly, every call after the first for a given paper reuses ~75% of its input. That
changes the economics of a local arm more than any choice of weights, and it is also the
control that makes a like-for-like cost comparison against the gateway honest.

### Candidates, as of this writing

Named by property rather than ranked, because the point is to run one, not to argue about
which. Sizes and context lengths below are approximate and should be checked against the
current model card — the line moves fast enough that anything here is a starting point.

| | why it is on the list | the catch |
|---|---|---|
| **Qwen3-30B-A3B-Instruct** (~30B total, ~3B active, Apache 2.0) | the best fit for the five properties: sparse so prefill is cheap, thinking separable from instruct, strong structured-output behaviour, and 4-bit community quants exist — which is what makes it runnable on the cards this project has | a MoE at 4-bit is the least predictable thing to serve; expect the vLLM flags to need the same care `serving-the-proposer.md` documents |
| **gpt-oss-120b** (~117B total, ~5.1B active, Apache 2.0) | the closest open analogue of what the pipeline uses today, and the only one whose *reasoning effort* maps one-to-one onto `Settings.effort`. ~5B active means it serves cheaply for its quality | needs a single 80 GB card. Its native MXFP4 format wants Hopper or newer, so it is a rental, not a beast job |
| **Qwen3-32B** (dense, Apache 2.0) | the control. If a dense 32B and a sparse 30B disagree, the disagreement is informative; if they agree, the MoE is free | dense 32B at 4-bit leaves little room for a 40k-token prompt across four 8 GB cards |
| **gpt-oss-20b** (~21B total, ~3.6B active) | the cheapest thing that might clear the bar, and the right first call if renting a card is friction | same MXFP4/Ampere problem, so it is a community 4-bit quant or nothing on these cards |
| **an OLMo 3 instruct model at the largest size available** | the only line where the training data is published, which is the provenance argument the rest of this repository would make. [field-extraction-audit.md](field-extraction-audit.md) already notes `Olmo2Config` works on the beast transformers build | the 7B named there is below the band this needs; whether a 30B-class sibling exists should be checked |

Deliberately not on the list: a domain-tuned biomedical LLM. The failures measured here are
structural rather than lexical — a model that has not understood what a `Cell` is does not fix
it by knowing more neuroanatomy — and the one place domain knowledge would pay,
`Group.medical_condition`, is already served by a MONDO/UMLS linker.

### The hardware, which is the binding constraint

`serving-the-proposer.md` establishes it on the way past: a 4B model in bf16 **OOMs on one
8 GB card** and needs all four of the 3070s; only the W4A16 checkpoint fits one, and only with
`--language-model-only`. Ampere has no native FP8 or MXFP4, so every candidate above is a 4-bit
W4A16/AWQ/GPTQ build on this machine or it does not run at all.

Four 8 GB cards is 32 GB total. A 30B model at 4 bits is ~17 GB of weights before any KV cache,
and a call here carries 36–42k tokens. That fits, barely, at low concurrency, and it will spend
the screening run fighting for KV rather than measuring anything.

**So rent one 80 GB card for step 2.** A screening run over the benchmark papers plus a
corpus-scale agreement pass is hours, not weeks, and it removes the quantisation and the memory
pressure from a measurement whose whole purpose is to find out whether the *model* is good
enough. Beast is the right home for whatever is adopted afterwards, and for every fine-tune in
§6 — S1a in particular is a 256-token-per-example job that one 3070 handles comfortably.

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
- **The candidates in §9a are a starting point with a shelf life.** Their parameter counts,
  context lengths and quantisation formats should be read off the current model cards rather
  than from this table, and a release since it was written may well displace all of them. The
  five properties the table is built from are the durable part; the names are not.
- **The prefix-caching correction in §9a is reasoning, not a measurement.** That vLLM's
  automatic prefix caching is block-level rather than whole-prompt is a property of the serving
  stack; that it would recover ~75% of input on this pipeline's prompts follows from the 29,152
  shared tokens `render.py` already counts. Neither has been run here, and the system-message
  ordering is the part most likely to make it silently not happen.
- **§6b's uncitable table is 16 papers and one run's records**, and `not_found` conflates two
  things `warrant.py` is careful to separate elsewhere: a value nobody quoted, and a value
  whose quote no locator could place. Read as "the model did not warrant this", which is what
  the partition needs, and not as "no sentence exists" — corpus-wide, per field path, is what
  would establish the second, and that is step 4 of §9.
- **The three deterministic widenings in §6b are diagnoses, not measurements.** That
  `same_level` never consults the abbreviation table is a fact about the code; that this is
  what the 24 level-mismatch abstentions are made of is a hypothesis, and the abstention
  breakdown it rests on comes from [deterministic-fields.md](deterministic-fields.md) over
  101 gold cells.
- **§6a's band is counted off the audit table, which is 16 papers.** 40 fields and 758
  instances at `surface` ≥ 90 / `uniq` < 20 are that corpus's numbers; the extrapolation to
  10⁴–10⁵ assumes the field mix holds across 1,817 records, which nothing here checks. The
  claim that those instances are *rankable* is a hypothesis — `uniq` establishes only that a
  string matcher cannot do it.
- **The 280k span count is an extrapolation** from 16 records in `benchmarks/candidate/` (2,478
  resolved spans, 155 per paper, 78% of extracted fields) to the 1,817 committed records. It
  assumes those records carry evidence at a comparable rate, which has not been checked
  corpus-wide.
- **The cost split is the documented one** (~310k in / ~27k out, evidence 45% of input) and is
  not re-derived from `usage.jsonl` here. A per-stage token table from a real run would sharpen
  every priority in §9 and is half an hour of work against a corpus this document did not have.
- **The local arm is gone from the package, and two documents said otherwise.** The in-process
  NuExtract proposer, the vLLM `NuExtractServer`, `Settings.proposer_url`, `repair.POISON`, the
  MiniCheck grounding step and the cross-encoder ranking all went; `evidence/retrieval.py` keeps
  only `sectionize`. `serving-the-proposer.md` now says in its first lines that it documents a
  removed arm and is kept for what it measures about running a quantised model on an 8 GB card,
  and `README.md`'s `repair` paragraph now describes the three steps that exist. Neither was
  deleted: every measurement in the serving document applies again the moment any local model
  is adopted.
