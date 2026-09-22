# Normalizing a field depends on the field's shape, not on the corpus

> Where this sits: [pipeline-architecture.md](pipeline-architecture.md) covers extraction, up
> to `data/runs/<run>/records/<id>.extraction.json`. This file starts there. The vocabulary-matching
> layer common to several of these pipelines is [normalizing-across-papers.md](normalizing-across-papers.md);
> its measured coverage is [normalizing-with-onvoc.md](normalizing-with-onvoc.md).

No single normalization pipeline fits every field. A field's shape determines the method,
and three shapes recur. Using the wrong shape wastes effort: task
descriptions have no target vocabulary to link to, medical conditions have one and should not
be clustered, and cohort role has four values and needs neither.

| shape | example fields | method | where |
|---|---|---|---|
| closed enum | cohort role, task epoch, coordinate space | schema enum + extraction instruction | the schema |
| open, vocabulary exists | `medical_condition`, `arms.agent`, `regions.name` | **link** to the vocabulary | `normalize_conditions.py` |
| open, no vocabulary | `tasks.name` + description | **cluster** the corpus against itself | `normalize_tasks.py` |

## The encoder is chosen by input length, not by domain

Measured on this corpus, the same two models invert completely:

| | short entity strings (median 17-30 chars) | task descriptions (~400 words) |
|---|---|---|
| SapBERT | **R@1 66.3%** (ONVOC), 62.9% (MONDO, 32k candidates) | R@1 24.5% — last, below TF-IDF |
| all-MiniLM-L6-v2 | R@1 50.6% | **R@1 58.5%**, R@10 86.4% |
| char 2-4gram TF-IDF | R@1 52.3% | ARI 0.008 — merges nothing |

SapBERT is trained on UMLS synonym pairs; a paragraph is off-distribution for it and a
30-character string is off-distribution for a sentence encoder. Neither is "the biomedical
model". Pick by the length of what you are embedding.

Two corollaries that also do not transfer between the two regimes:

- **Sparse and dense views help on prose and hurt on entity strings.** Adding a TF-IDF channel
  over the task description lifted pair AP from 0.301 to 0.407; every SapBERT + char-n-gram
  hybrid scored *below* SapBERT alone on MONDO retrieval, monotonically worse as the char
  weight rose. IDF preserves a rare exact phrase a mean-pooled embedding discards, and there
  are no rare phrases left in a three-word disease name.
- **Concatenation dilutes.** Folding `performance_measures` into one task signature *shrank*
  the stop-signal / go-no-go margin from +0.040 to +0.031, because a sentence embedding is a
  mean over the passage and the one discriminating token is averaged away by the shared
  vocabulary around it. Fields are separate channels, combined by a model.

## Shape 1 — closed enum: do not build a pipeline

`arm_kind` and `relation_to_intervention` are enums in the schema and are queryable today with
no normalization work at all. `cohort_role()` in ``pondie.normalization.contrasts`` is the counterexample:
a hand-written regex over group names, which fails on **13% of schizophrenia group names, 57%
of MID and 60% of depression**. It looks fine because it was written against schizophrenia.

A four-value field does not want a clusterer or a vocabulary. It wants an enum and a schema
description carrying the decision rule. `docs/deterministic-fields.md` is the other half of
this: where code can fill the field outright, it should, and abstain when it cannot.

## Shape 2 — link: `normalize_conditions.py`

Short strings, an existing target, and a long tail that is mostly not rare diseases.

```
0. triage      negation to a sentinel, compounds split, qualifiers lifted off the head term
1. expand      the corpus abbreviation store (scispacy Schwartz-Hearst), acronyms only
2. lexical     fold-exact against MONDO labels and exact synonyms
3. embedding   SapBERT retrieval, routed three ways by cosine
4. rollup      MONDO is_a to the nearest ancestor THE CORPUS uses >= --min-support times
5. report      what could not be placed, with support, as vocabulary evidence
```

**Triage carries most of the value, and runs before any lookup.** Of 1565 `medical_condition`
values, **315 (20%) are negations** — "no neurological or psychiatric disorder" — recording the
*absence* of a condition. Matched against a disease ontology every one of them retrieves
something at plausible similarity.

The gate order stated here — `Group.is_healthy` first, the regex as fallback — is **wrong, and
[normalization-layer.md](normalization-layer.md) has the measurement**. On the full 2,115-record
corpus the two agree on 93% of the mentions where both are known, but where `is_healthy` is
true and the string carries no negation the string is usually naming a trait the cohort was
*selected for*: `obesity`, `cannabis use`, `postmenopause`. The string decides; the flag fills
the mentions whose string says nothing either way.

**The accept threshold cannot be a single cut.** On 1500 held-out MONDO synonym queries the
score distributions overlap: correct matches have p10 = 0.807 while *wrong* top-1s have a
median of 0.820. Precision/recall is 72%/91% at 0.80, 79%/82% at 0.85, 86%/70% at 0.90. The
stage routes three ways — auto-accept, a review queue carrying top-5 candidates, and reject
with the nearest miss recorded — rather than pretending a threshold separates them.

**The rollup stops at ancestors the corpus itself uses**, not at a fixed ontology depth, so the
target is queryable by construction: a one-paper `remitted anorexia nervosa` lands beside the
anorexia nervosa other papers use.

Result on the three corpora: 620 distinct forms reduce to **72 MONDO terms, 69 of them
carrying a UMLS CUI (96%)**.

## Shape 3 — cluster: `normalize_tasks.py`

No target vocabulary is usable — ONVOC has no task branch at all, and Cognitive Atlas
retrieval from the description alone tops out at R@1 62.9% with no threshold that separates
covered from new (81% of unmatched task signatures score above the 10th percentile of the
known-covered set). The corpus is therefore clustered against itself.

The method is written out in full, with the changes the re-measurement below asks for, in
[task-clustering-method.md](task-clustering-method.md).

```
1. name ladder   folded equality + bidirectional containment -> MUST-LINK,
                 and the weak labels stage 2 trains on
2. pair model    logistic regression over per-channel similarities
                 (name, prose, setting, measures, conditions, prose_lex)
3. clustering    agglomerative on 1 - P(same task) with must-link enforced, a rescue pass
                 for singletons, then families from prose geometry over identity centroids
```

Measured: name ladder alone resolves 50%; the pair model reaches **AUC 0.941** (0.888 with the
name channel held out); clustering gives **167 identities / ~130 families, ARI 0.619, V 0.783**
against name-derived gold.

> **Re-measured 2026-09 on the full 2,115-record corpus, and the clustering numbers do not
> hold.** Nothing in the checkout reproduced them -- there was no evaluation code. One was
> written to get the numbers below and then removed with the rest of the superseded
> clustering work; it scored `pondie.normalization.task`, which
> [task-clustering-method.md](task-clustering-method.md) argues should be replaced. On 1,672 tasks: pair model
> AUC **0.916** under a grouped 5-fold split (0.868 without the name channel), which is the
> quoted figure within noise; but clustering gives **121 identities, ARI 0.474 / V 0.727**.
>
> More to the point, the gold those are scored against is the name-ladder components, and the
> ladder is *also written into the distance matrix as must-link pairs at d=0* -- so the score
> partly confirms that the constraints were applied.
>
> A second gold was tried and **is not independent either**. Scoring against the Cognitive
> Atlas paradigm each name resolves to gives ARI 0.524 over the 493 tasks that carry one --
> but measured against the ladder directly, the two golds agree at ARI 0.584 / V 0.841, and
> **93% of the pairs the Cognitive Atlas links the ladder had already linked**. Only 959 of
> its 15,842 same-paradigm pairs are new. It corrects the ladder at the margin -- `Go-NoGo
> task` and `alcohol go/no-go task` share no token subsequence and do share a Cognitive Atlas
> term -- and it is otherwise the same object. **There is no gold for task identity in this
> repository, and no number here is scored against one.**
>
> What survives that is the part checkable by inspection rather than by a metric: **23 of the
> clusterer's 52 paradigm-bearing clusters mix two or more paradigms**. One holds ten,
> including `go/no-go task`, `Stroop task` and `Emotion Regulation Task` together; another
> merges `go/no-go task`, `monetary incentive delay task` and `temporal discounting task`.
> Those are different paradigms under any reading, and no choice of gold makes them one.
>
> The cause is the name ladder rather than the model, and it is one line. Containment over
> **two-token** spans chains a 288-task component out of 120 distinct names -- `emotion
> regulation` is a topic, not a task name, so containing it links `Emotion Regulation Task`
> to `Facial Emotion Recognition Test` to `Decision-Making Task`. Requiring three tokens cuts
> the largest component from 288 to 112:
>
> | containment minimum | components | largest |
> |---|---|---|
> | exact fold only | 1,057 | 87 |
> | **>=2 tokens (shipped)** | 657 | **288** |
> | >=3 tokens | 874 | 112 |
> | >=4 tokens | 1,010 | 87 |
>
> And the shipped identity threshold of 0.50 may not be the best one: 0.30 scores ARI 0.611 /
> V 0.834 against the Cognitive Atlas gold, against 0.524 / 0.793 at 0.50. Weak evidence,
> given what that gold turned out to be.
>
> Neither change is made here, and the threshold one should not be made on this evidence:
> tuning a parameter against a gold that is 93% the name ladder selects for agreeing with the
> name ladder. The chaining fix stands on its own -- it is a structural fact about the input,
> not a score -- but it too wants a real gold before and after.
>
> The replacement workflow is `pondie/normalization/task.py`.
>
> **What a real gold would be**: a sample of papers whose tasks a reader labels from the
> Methods section, blind to the record. `benchmarks/gold/direction/` is the template and the
> proof it is affordable here -- per-cell answers from numbered annotators, with `tier` and
> `disputed` recorded so agreement is visible. That exists for contrast direction on five
> papers and for nothing else. Two hundred hand-labelled tasks would settle every number
> above, and until they exist these are machine-versus-machine agreement rates.

Three design points that were arrived at by measurement and are easy to get wrong:

- **Distant supervision, then a grouped split.** Positives are pairs inside a name-ladder
  component; the split is by component so no task appears in train and test. The honest number
  is the one with the name channel excluded, because the pairs the model exists to judge are
  exactly those the name cannot resolve.
- **The model decides identity; geometry decides families.** A logistic probability saturates
  near 0 for non-matches, so it is a good decision score and a bad metric — using it for the
  coarse cut piled every distance at 1.0.
- **Hand-written discriminators were tried and removed.** A marker table of paradigm regexes
  (SSRT for stop-signal, and so on) was built to separate structurally similar tasks, and
  ablation showed it made clustering *worse* — 243 clusters / ARI 0.600 with it, 185 / 0.619
  without, and stop-signal stayed separate from go/no-go either way. Giving the name its own
  weighted channel had already solved it.

## Shape 4 — partition: `population_characteristics.py`

A fourth shape, and the first where the target is not a vocabulary but a *second field*.
`Group.population_characteristics` holds what a study chose its cohort for — habitual
exposure, training, occupation, lifestyle, atypical body habitus. Its whole value is that a
query can filter on it, and that value survives only while every entry is discriminative.

Asking the model for discrimination failed, three times. Each round rewrote the slot
description and re-extracted the same ten papers:

| round | entries | selective | non-selective | restating a typed slot | duplicate |
|---|---|---|---|---|---|
| v1 — asked, loose wording | 33 | 61% | 6% | 24% | 9% |
| v2 — `is_healthy` derived, tight wording | 37 | 86% | 0% | 14% | 0% |
| v3 — wording built around deviation | 30 | 87% | 7% | 7% | 0% |

The trend reads as progress and is not measurable as any. The v2→v3 differences are two and
three entries out of thirty; re-running one annotation stage under an unchanged config moves
mean map R² by 0.034, and nothing suggests extraction is quieter than that. Three rounds of
description edits produced changes that cannot be attributed to the edits, and v3 put back
the two values the round was written to remove. This is the `is_healthy` finding again: a
description cannot outvote the source's own wording.

So the question is left alone and the answer is partitioned afterwards. Non-selective values
move to `Group.other_characteristics`, which is `deterministic` in the storage schema and
therefore never projected into the extraction schema — the model is not asked about it and
cannot fill it. Moved rather than dropped: the values are not wrong, and a reader auditing a
cohort wants to see "normal weight" and "right-handed". They just cannot share a field with
the trait a filter selects on.

### Full-string on a reduced core, never a substring

`search(r"healthy")` moves "Otherwise healthy adult smokers" and loses that cohort's
defining trait. So each value is folded, stripped of generic person nouns and hedges at both
ends, and matched **in full**: "Otherwise healthy adult smokers" reduces to `healthy adult
smokers`, which no pattern matches, and stays. "Healthy weight children" reduces to `healthy
weight`, which does.

### Two asymmetries, both of which a blunter rule inverts

**Handedness.** "right-handed" is normative; "left-handed" and "mixed-handed" are selective,
because a study recruiting left-handers recruited for that.

**Negation.** A negated *condition* is normative — every control cohort in the corpus carries
"no neurological or psychiatric disorder". A negated *exposure* is selective: "no history of
smoking" *is* the control arm of a smoking study, and "cannabis use less than 50 times" is how
a cue-reactivity paper defines its comparison group. An `EXPOSURE` lexicon is therefore
decisive against every rule, which is what lets the negation rule be stated broadly.

Two narrowings of the negation rule each removed a measured false positive:

- Bare qualifiers are not illness heads. With `significant` among them, "no significant
  re-experiencing, avoidance, or hyperarousal symptoms" moved — a PTSD study's comparison
  group, and the clearest wrong answer in the sweep.
- Named diagnoses never trigger it, which is why `disorder` and `diagnosis` are not
  domain-generic words. With them there, "No PTSD diagnosis" and "no post-traumatic stress
  disorder" moved. They reach a separate `BARE` branch that admits a bare illness noun only
  when nothing but a qualifier stands between it and the negation — so "no chronic
  conditions" moves and "no anxiety disorder" does not.

### Measured on the corpus

The committed 1,817-record corpus predates the slot, so `report()` falls back to the fields
these traits were landing in instead — every value of at most eight words, prose excluded.
**2,583 of 25,077 values (10%) move**:

| rule | moved | distinct forms | heaviest form |
|---|---|---|---|
| `no_condition` | 934 | 558 | "No history of neurological or psychiatric disorders" |
| `handedness` | 603 | 19 | "Right-handed" (294) |
| `health` | 519 | 74 | "healthy" (73) |
| `senses` | 171 | 22 | "Normal or corrected-to-normal vision" (83) |
| `language` | 121 | 31 | "Native English speaker" (20) |
| `mri_eligibility` | 87 | 21 | "Not pregnant" (13) |
| `cognition` | 65 | 11 | "Cognitively normal" (24) |
| `consent` | 43 | 8 | "Provided written informed consent" (14) |
| `weight` | 25 | 7 | "normal weight" (13) |
| `development` | 15 | 8 | "Typically developing" (5) |

The per-field rates are the evidence that it is aimed at the right class of statement:

| field | values | moved |
|---|---|---|
| `inclusion_criteria` | 8,642 | **23.1%** |
| `medical_condition` | 4,333 | 11.7% |
| `clinical_characteristics` | 503 | 10.3% |
| `exclusion_criteria` | 11,599 | **0.3%** |

Inclusion criteria are where prerequisites live, and they are what this rule is about.
Exclusion criteria name conditions rather than negate them, so almost nothing matches — the
rule is reading the negation, not the disease word.

Precision, on the two curated vocabularies: **0 of 157** task terms move, and 72 distinct
condition forms do — all of them health assertions or generic-condition negations, which is
`medical_condition` holding the absence of a condition, a defect that field's own audit had
already found at 5%.

### What it does not do

6.7% of the values it keeps restate a slot that already exists — "male" (128), "age 18-65
years" (36), "caucasian" (15), "left-handed" (13). Those belong in `sex_distribution`,
`age_minimum`/`age_maximum`, `race_distribution` and `handedness_distribution`, and routing
them there is a different mechanism from this one: a partition decides whether a value is
selective, not which typed slot carries it. This rule leaves them where the model put them.

## The long tail is a promotion rule, not a matching problem

70% of `medical_condition` forms and 62% of task identities occur exactly once. Measured, that
tail is real rather than under-merging: all 150 singleton task clusters have a name that is
unique in the corpus, and their median nearest-neighbour P(same task) is 0.317 against 0.984
for clustered tasks.

A cluster earns a vocabulary term through **cross-paper support**, never a per-item score. A
term used once is a paper's phrasing; a term used in ten is a term the vocabulary lacks, and
the unplaced report is the evidence for it. The same rule governs both shapes.

## What is upstream of all of it

None of this is queryable if the analysis cannot be joined to the entity. Measured on the
depression corpus, of 391 model term levels only **31 carry `arms` (8%) and 73 carry
`timepoints` (19%)**, so 91% of analyses cannot say which arm they belong to however well the
arms themselves are normalized. `audit_queryability.py` counts these joins; fix them before
normalizing the entities they point at.
