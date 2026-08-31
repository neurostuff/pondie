# The pipeline, its data, and the alternatives that were measured against it

Read this before changing the extraction path. It says where the data comes from, what each
stage does, which pipeline shapes were tried, and what each one measured. It stops at the
built record: normalizing the record's fields against a vocabulary, or against the corpus
itself, is [normalization-pipelines.md](normalization-pipelines.md).

The goal bounds every decision here: **a queryable reconstruction of the analyses that
coordinate tables report, with every value warranted by a span.** Direction carries 0.45 of
the composite and is the one fact a synthesis cannot recover from anywhere else, so a change
that saves cost and moves direction is not a saving.

---

## 1. Where the data comes from

Nothing below is in git. All of it is fetched.

### The corpus: ns-pond, on `beast`

```
beast:/data/alejandro/projects/ns-pond/data/<study_id>/
├── identifiers.json                     pmid, doi
├── source/pubget/article.xml            the article, for text rebuilding
├── source/pubget/tables/<n>.csv         one CSV per table
├── source/ace/<pmid>.html               the ace render, when there is no pubget one
└── processed/{pubget,ace}/
    ├── text.txt                         plain text
    ├── tables.jsonl                     table manifest
    ├── analyses.jsonl                   the pond's own stage-1 parse (diffed, not used)
    └── coordinates.csv
```

**39,273 studies.** Only ~12,390 have a pubget render; the rest are ace. The two layouts hold
the same five artifacts under different names, so which sync script to use is decided per
study, not per corpus:

| render | script | note |
|---|---|---|
| pubget (`processed/pubget/text.txt` non-empty) | `review/sync_texts.py` | then `review/build_text.py` to inline tables |
| ace only | `review/sync_texts_ace.py` | writes the pubget paths itself; no `build_text` step |

Both take `--host beast --root /data/alejandro/projects/ns-pond/data`. **The default host is
`beast-proxy`, which does not resolve from this machine** -- pass `--host beast`.

Checking `analyses.jsonl` exists is not enough to select a paper: a study can have a parse
and no text render. Check `text.txt` size and the table CSV count.

### Local layout, after sync

```
data/texts/<id>/
├── stage1/analyses.json          the analysis inventory -- see §2
├── stage1/table-map.json         pubget table_id -> record Table local_id
├── processed/local/text.tables.txt   THE text every offset addresses
└── source/pubget/tables/*.csv
data/records/<id>.extraction.json     one record per paper
data/gold/direction/<id>.direction.json   reviewer direction table
data/gold/direction-prediction-history.json   pre-fills as reviewers saw them
```

### The review layer

Label Studio at `validate.neurostore.xyz`, five projects. Reachable only through the
host's nginx-proxy: **`localhost:8080` will not work even when the container is up.**
`.env`'s `LABELSTUDIO_TOKEN` is a JWT, so `lsapi.Client` exchanges it for a bearer token.

### The model gateway

Portkey, at `OPENAI_API_GATEWAY`. `OPENAI_API_KEY` is a Portkey key and works against
`/v1/analytics/*` as well as the gateway -- but those endpoints sit behind a Cloudflare
check that rejects a default Python user agent, and want an explicit time range.

---

### The contract at each seam

Each handoff below has a format the next stage assumes and does not check. Two of them fail
*silently* — the run reports success and produces nothing — so they are worth stating as
contracts rather than leaving to be rediscovered.

| seam | contract | what a violation looks like |
|---|---|---|
| ns-pond → `data/texts/` | a study is usable when `processed/*/text.txt` is non-empty **and** the table CSVs are present | a study with `analyses.jsonl` and no render passes a naive check and yields nothing |
| `--pmids` file | TAB-separated, three columns: `pmid \t study_id \t source` | a bare id per line logs `skipping unparseable line`, then **`all stages clean` with zero records written** |
| `--texts <root>` | `<root>/<id>/processed/<flavour>/text.txt` and `<root>/<id>/stage1/{analyses.json,table-map.json}` | stage 1 is **not** regenerated here; a missing `stage1/` skips the paper |
| `--examples <dir>` | the directory built records are **written to**, despite the name | its default is `review/examples`, the suite's fixture — accepting the default overwrites test data |
| record → any consumer | read it with `schema_utils.value_of` / `slot_value`, which take the shape from the schema; **`pipeline/repairs.py` has already normalized the structure at build time** (`wrappers`, `unwrapped`, `listified_scalars`, `coordinate_space`, 16 in all) | a hand-rolled unwrapper returns the wrapper for `not_reported` and drops every entry but the first from a multivalued slot, both without erroring |
| `data/vocab/` | `onvoc.json`, `cognitiveatlas-*.json`, `abbreviations.json`, `mondo.json`; fetched, none in git | absent files degrade to no matches rather than an error |

The `not_reported` shape is the one that bites repeatedly: `value` missing is a *positive
assertion that the paper did not report it*, and is not the same as the field being absent.

**Do not hand-roll the unwrap.** `schema_utils.value_of(node, multivalued)` reads the wrapper
and the slot's declared shape; `slot_value(classes, class_name, entity, slot)` takes
`multivalued` from the schema so a caller never guesses it. The three outcomes it keeps apart
are the three a hand-written unwrapper conflates:

| record holds | `value_of` returns | the claim |
|---|---|---|
| slot absent | `None`, or `[]` when multivalued | nothing was asked |
| `{"extraction_status": "not_reported", ...}` | `NOT_REPORTED` — falsy, `str()` empty, iterates empty | the paper did not say |
| `{"value": [...]}` on a multivalued slot | the list, always | the paper said this |

`Group.medical_condition` is the field that shows why: it is multivalued because a cohort has
comorbidities — 161 groups carry two conditions, 39 carry three, one carries eleven — so a
reader taking the first element answers a different question from the one the schema records.
Measured across the three corpora its shapes are **clean**: 931 single-item lists, 237 with two
or more, 246 reported-empty, 311 absent, 51 `not_reported`. A hand-written unwrapper that
returns the wrapper when there is no `value` key reports those last 51 as malformed scalars;
they are not.

One genuine inconsistency remains, and it is semantic rather than structural: **246 groups hold
`{"extraction_status": "extracted", "value": [], "evidence": {"status": "not_found"}}`** —
asserted as extracted, with nothing in it and evidence marked not-found. That is a different
claim from `not_reported`, and the two are being used for what looks like the same situation.
A validator rule, not a repair.

---

## 2. The current pipeline (P0), stage by stage

```
  ns-pond ──sync_texts[_ace]──▶ data/texts/<id>/
                                     │
  parse_tables.py ◀───────────────────┘        ONE LLM CALL PER TABLE (autonima)
      │  + split_opposite_signs()               deterministic post-pass
      ▼
  stage1/analyses.json          the analysis inventory: names, points, statistics
      │
      ├──▶ run_extraction.py --workflow demand-driven
      │        tables    copy the pubget manifest        deterministic
      │        demands   analyses first; each declares   LLM
      │                  the entities it needs
      │        satisfy   fill that shopping list          LLM
      │        evidence  attach spans to values           LLM   ← 45% of input
      │        build     build_record.py + validate       deterministic
      ▼
  data/records/<id>.extraction.json
      │
      ├──▶ derive_fields.py --fill      8 fields at 93.7-100%
      ├──▶ ls.py export                 tasks per project
      └──▶ ls.py sync --apply           to Label Studio
```

**Measured cost, from `usage.jsonl`:** ~310k input and ~27k output tokens per paper, an
**11.5:1** input-to-output ratio. `evidence` alone is 45% of input.

**Measured accuracy:** direction 96.6% on the 101-cell reviewer gold.

Do not read that against "a 95.8% human ceiling" — the two are not on the same denominator,
and the 95.8% is not raw agreement. Of 239 doubly-reviewed cells, two reviewers **agree 78.2%
read naively**; 95.8% is that figure weighed by provenance tier, and the narrowest defensible
number is **44 cells, 42 agreed — 95.5%**, restricted to cells where both reviewers chose a
sign. The derivation is contrast-direction-rubric.md, "The ceiling, measured".

What the doubly-reviewed set does support is a shape: of 52 disputed cells only **2** are
`positive` vs `negative`. The rest argue about whether a term is in the contrast at all.
Humans agree about polarity and disagree about membership, so a polarity score compared
against a membership-inclusive agreement figure is comparing different questions.

**The cache is off.** Every call reports `cache_write_tokens: 36423, cached_tokens: 0,
cache_status: DISABLED`. Five stages send a near-identical prefix and each pays full input
price. Nothing in the code causes this; it is gateway configuration.

---

## 3. Component measurements the alternatives are built from

| component | what it does well | what it cannot do |
|---|---|---|
| **contrast-name deriver** | 55/101 signed gold cells, **100%** correct, abstains on the rest | names without an operator (49%) |
| **field derivers** (8) | 93.7-100% on paper-invariant closed-vocabulary fields | anything varying within a paper |
| **`surface`/`uniq` screen** | predicts which fields any method can reach | -- |
| **NuExtract3** (4B, ~6s/paper) | 30 fields ≥80%; evidence in-gold **50%** over 50 fields | `Cell.direction` **37%**; under-segments tables |
| **GLiNER** (CPU, ~15s/paper) | `regions.name` **94%** recall, `devices` 82% | no calibration; 562 spans for 36 entities |
| **NLI rerank** | entity-scoped counts: `sex_distribution.count` 5→58% | hurts unique-string fields badly |
| **naive string search** | 25 fields at ≥80% | ambiguous numbers (`groups.n` 16%) |
| **Maverick coref** | resolves `the group` → `healthy controls` | **no gain**: 37% appended, 27% substituted |
| **spaCy NER** | -- | 1,557 `ORG` spans per 8 papers |

---

## 4. Pipelines already hypothesised and measured (P1-P7)

Recorded so they are not re-proposed. Detail in
[pipeline-hypotheses.md](pipeline-hypotheses.md).

| | shape | direction | output tok/paper | verdict |
|---|---|---:|---:|---|
| **P0** | current | **96.6%** | 15,429 | baseline |
| P1 | deterministic direction first, model on abstention | 98% on 50 cells | 15,078 | accuracy holds, saves 2% |
| P2 | flat contrast schema `{parse, axis, plus[], minus[], sign}` | untested | untested | needs a luna run |
| P3 | deterministic + NuExtract3 pre-fill | untested | 13,103 | 15% saving |
| P4 | agreement cascade (deriver vs model) | 98% on 50 cells | -- | identical to P1 here |
| P5 | level shortlist as closed vocabulary | untested | -- | levels 32% recall |
| P6 | cells-only pass | untested | 3,252 | not a whole pipeline |
| P7 | + deterministic evidence | -- | 8,965 | evidence share was overstated |

**Why P7 was overstated.** "61% of evidence is a string search" was locatability, not
correctness: only 44% of located values land on the sentence the extractor chose, and the
cheap unique-string fields carry that 44%. Realistic share is ~25 of 86 fields.

---

## 5. Five that build on the demand-driven design

P0's discovery, stated in `run_extraction.py`:

> the entity pass cannot know what the contrasts will need. Asked to guess, it modelled a
> crossover's condition as a continuous covariate, and **a cell cannot be righter than the
> term it points at**.

Two ideas. **Dependency ordering** -- the consumer declares its needs before the producer
supplies, because the producer cannot guess. And a **precision cascade** -- error flows
terms → cells → direction, so accuracy upstream is worth more than accuracy anywhere else.
Together, 38.1% → 80% direction F1.

All five below keep that loop and push it further. Each must **match 96.6% direction first**;
cost is read only once accuracy holds.

### P8 — Demand at field granularity, not entity granularity

`demands` declares *which entities* an analysis needs; `satisfy` then fills each entity's
whole schema. The principle stops one level short of where it applies.

```
  demands  →  {"required_entities": [{"kind": "Group", "label": "...",
                                      "needed_for": ["cells.level"],
                                      "fields": ["name", "species"]}]}
  satisfy  →  fills ONLY the declared fields
  derive   →  the paper-invariant closed-vocabulary ones (8 fields, 93.7-100%)
  defer    →  fields no analysis reaches
```

P0's own justification: a field no contrast reaches cannot make a cell wrong, so it does not
deserve budget on the pass that decides cells. The audit says which those are -- 36 fields
are model-only, 78 have `uniq` below 20% -- and the adjustment set is *derived* (model terms
minus celled terms), so a large part of the record is never read by a query.

*Predicted:* direction unchanged or better, the pass having less to hold; output falls with
the deferred set. *Falsified if:* narrowing the ask degrades terms, which shows up in
direction immediately.

### P9 — More parse rules, in the zero-foci mould

The zero-foci rule is one deterministic table fact -- an entry with no coordinates is a
tested effect that found nothing -- injected as a prompt rule. **+16 points** paired with
demand-driven; **-25 alone**. That asymmetry is the lesson: a parse fact pays when the pass
it informs is ordered to use it.

The parse knows more facts of that shape and tells the model none:

| parse fact | the rule it becomes |
|---|---|
| the name carries `>` / `<` | the ordering is stated; the plus side is the left operand |
| the rows carry both signs | this entry is two contrasts, already split -- do not merge |
| the value kind is `correlation` | the term is continuous; its cell names no level |
| rows grouped under headings | the headings are the levels, not the columns |
| a `df` column exists | the denominator degrees of freedom are on the page |

The first is measured: the contrast-name deriver resolves **55 of 101** signed cells at
**100%**. P1 used that to *override* the model afterwards -- but a sign patched after the
fact still points at whatever term the model chose. As a rule it keeps the model's own
reasoning consistent with it, which is how zero-foci worked.

*Predicted:* direction ≥ P0, by the same mechanism. *Falsified if:* rules interact the way
zero-foci did alone -- so each is measured paired, never in a batch.

### P10 — Split `satisfy`: terms, verified, then cells

If a cell cannot be righter than its term, terms deserve their own pass and their own
verification, and cells should be asked only once terms are fixed.

```
  parse   →  level vocabulary from table headers
             (NuExtract3: 100% where headers encode them, 0% where prose does)
  terms   →  a dedicated pass choosing from that vocabulary
  verify  →  every declared level appears in the parse or the text; every term a cell
             names exists in a reachable stage
  cells   →  asked last, over fixed terms, choosing sides only
```

This is exactly where NuExtract3's contrast failure came from -- `term: "FESZ", level: "NC"`,
both sides of one comparison stuffed into two slots, because it had no fixed term to point
at. It is also where both bugs found this session lived: the exporter and the scorer were
each level-matching failures.

*Predicted:* the largest accuracy headroom here, because it attacks the upstream of the
cascade. *Falsified if:* term accuracy is already near-perfect -- **which is unmeasured.**
There is no term-level gold, only the direction table, so this pipeline is gated on the
`structure` project's entity dispositions (281 rows available, 4 answered).

### P11 — Demand-driven evidence

`evidence` is the one stage P0 never applied its own principle to: it sweeps the whole
record, and costs **45% of input**. Make it demand-driven -- the analyses declare which
values carry a claim a query will read, and only those get warrant.

```
  demands  →  also declares which values need evidence
  evidence →  runs on that list, routed by the measured table:
                naive string search   25 fields ≥80%
                tuned NLI             ambiguous counts (sex_dist.count 5→58, groups.n 12→50)
                NuExtract3            50 fields, 50% in-gold, 74% verbatim
                luna                  schema-code fields whose ceiling is ~0
```

*Predicted:* input 310k → ~170k per paper. *Falsified if:* reviewers reject the substituted
spans -- the real test, since "agrees with luna" is not "is correct". A different-but-correct
span is fine; a wrong one is not.

### P12 — Demand-driven retry

`--max-attempts` re-asks the **whole pass** when a post-condition fails, and roughly one run
in four loses a pass. The same principle applies to failure: re-ask only the demands that
went unsatisfied.

```
  post-condition fails  →  diff what was asked against what came back
                        →  re-ask only the missing entities/fields, with the fault named
```

The retry already names the fault rather than resampling blindly; this narrows *what* is
re-asked as well as *how*. On a degenerate empty pass the two are the same, so the win is on
partial failures, which are the common case.

*Predicted:* retry cost falls sharply; direction unchanged. *Falsified if:* a partial re-ask
produces an incoherent payload -- the entity pass may need the whole contrast set in view to
be consistent, which is the demand-driven premise working against a narrowed re-ask.

---

### Deviations, where the evidence is strong enough to leave the design

**D1 — Flat contrast schema.** Replace nested `Effect.cells[]` with
`{parse_index, axis_term, plus[], minus[], within[], sign_status}` and rebuild the schema
shape in code. This deviates because it changes what the model is asked for, not the order it
is asked in. The hypothesis is strong: over-celling, `held`→signed, malformed `direction`,
level paraphrase and dangling term references are all **unrepresentable** in that shape, and
every one of them is an error class observed in this corpus. It also gives a small model a
chance at the contrast layer, which the nested shape demonstrably does not.

**D2 — One cached prefix.** Every call reports `cache_write_tokens: 36423, cached_tokens: 0,
cache_status: DISABLED`. Five stages send a near-identical 36k prefix and each pays full
input price. Deviates only in prompt assembly -- the ask is unchanged, so direction should be
untouched, which makes it both the cheapest to run and the cheapest to falsify. Gateway
configuration, not code.

## 6. What has not been tried, and why not

- **Fine-tuning SciBERT or Longformer** -- the right base models, no zero-shot mode, blocked
  on labels: 4 entity-salience dispositions answered of 281 available.
- **BioCoref domain adaptation** -- coref gained nothing (37% appended, 27% substituted),
  so adapting it would optimise a component that is not the lever.
- **Larger GLiNER / scispaCy NER** -- the bottleneck is scoping, not recognition, and the
  `surface` ceiling bounds any recogniser.

---
