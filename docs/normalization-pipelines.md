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
something at plausible similarity. `Group.is_healthy` agrees with the negation regex on
**1070/1106 (97%)** of groups and should be the primary gate, with the regex as fallback for
the 62 where it is unset.

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
