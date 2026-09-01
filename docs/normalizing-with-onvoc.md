# Normalizing tasks and conditions against ONVOC

Measured on 328 records — the 299 schizophrenia papers and the first 29 monetary-incentive-delay
papers — using ``pondie.normalization._onvoc``.

## ONVOC is already the level we want to settle at

752 classes, and the hierarchy is **two deep**: a branch and its leaves. Only 54 classes have
children of their own, and those are branches themselves (`Tests`, `Executive Function`,
`Substance Abuse`). There is no subtype tree — ONVOC has `Schizophrenia` and nothing below it.

Rolling up to "a step above the subtype" is not tree navigation. ONVOC's leaf *is* that
step, and the whole problem is getting a paper's surface form onto it.

Tested directly: stripping subtype qualifiers (`first-episode`, `chronic`, `paranoid`,
`treatment-resistant`, `remitted`, `drug-naive`) from the 562 unmatched group names recovered
**zero**. Papers in this corpus do not write "paranoid schizophrenia"; they write
"schizophrenia patients", which already matches, or "healthy controls", which cannot.

## ONVOC has no task vocabulary

Searching all 752 labels for paradigm words returns five: `Continuous Performance Task`,
`Stroop Test`, `Wisconsin Card Sorting Task`, `Lexical Decision Task`, `Hooper Visual
Organisation Task`. All five sit under `Tests` and are clinical instruments. There is no
n-back, no monetary incentive delay, no flanker, oddball, go/no-go, or face-processing task.

Tasks therefore route to Cognitive Atlas, and that works: 62.5% of task names matched in the
schizophrenia corpus and 88.7% in the MID corpus. Using ONVOC for tasks is not a matter of
tuning; the terms are absent.

## Measured coverage

| route | schizophrenia | MID |
|---|---|---|
| `tasks.name` (Cognitive Atlas) | 62.5% | 88.7% |
| `tasks.conditions.name` | 29.4% | 50.8% |
| `regions.name` | 44.0% | 37.0% |
| `groups.name` | 35.0% | 24.8% |
| `measures.source_label` | 26.2% | 43.1% |
| `assessments.name` | **2.2%** | **4.0%** |

## The gaps are three different problems

**A missing concept, not a matching failure.** Of 562 unmatched group names, 233 are
phrasings of *healthy control*: `healthy controls` (133), `healthy control subjects` (25),
`healthy volunteers` (16), `controls` (12), `healthy participants` (11), `HC` (6), and more.
ONVOC has no such term. `Population Groups` is ethnicities; the nearest concept is `Health →
Typical Health`, which is a health state rather than a study role. One term would recover
41% of the unmatched group names, and no string manipulation recovers any of them.

**A missing branch.** `Tests` holds 53 neuropsychological batteries; the corpus asks it about
clinical rating scales. 41 instruments appear in three or more papers and none are present:

| instrument | papers |
|---|---|
| Positive and Negative Syndrome Scale (PANSS) | 118 |
| Structured Clinical Interview for DSM-IV (SCID) | 68 |
| Scale for the Assessment of Positive Symptoms (SAPS) | 34 |
| Scale for the Assessment of Negative Symptoms (SANS) | 33 |
| Edinburgh Handedness Inventory | 26 |
| Brief Psychiatric Rating Scale (BPRS) | 21 |
| Mini-International Neuropsychiatric Interview (MINI) | 14 |
| National Adult Reading Test | 8 |
| Beck Depression Inventory (BDI) | 6 |

These should not be lumped into an existing category: PANSS is not a Wechsler scale, and a
query for "studies that measured negative symptoms" needs SANS and PANSS to be distinct terms.

**Wrong vocabulary.** `BOLD signal`/`activity`/`response` (33 papers), `functional
connectivity` (22), `fractional anisotropy` (16), `gray matter volume` (18) are unmatched
because ONVOC has no measure or modality branch. The storage schema already has `MeasureType`
and `MeasureFamily` enums for exactly these, so the route is wrong, not the vocabulary.

Surface-form fixes are worth having but small: mapping adjectival diagnosis forms
(`schizophrenic` → `Schizophrenia`, `depressed` → `Depressive Disorder`) and stripping study-role
head nouns recovered 31 of 562 values, 6%.

## The design

1. **Expand abbreviations from the paper first.** Table headers and cell levels carry `HC`,
   `SZ`, `PANSS`. Schwartz-Hearst detection over the paper's own text is what turns those into
   matchable phrases, and it must run before any vocabulary is consulted.
2. **Route by field before matching.** `Wechsler Abbreviated Scale of Intelligence` contains
   the word `Intelligence`, and an unrouted match returns that concept confidently and wrongly.
   Which branches a field may draw from is part of the mapping, not a filter afterwards.
3. **Match through an ordered ladder** — exact, synonym, variant, acronym, contains, stem,
   overlap — and keep the layer on the result. The layer is the confidence; collapsing it to a
   score loses the distinction between an exact label and a token overlap.
4. **Roll up through MONDO, not through ONVOC.** ONVOC is flat, but `onvoc-mappings/mondo.tsv`
   ties `Schizophrenia` to `MONDO:0005090`, and MONDO *does* carry the subtype hierarchy. A
   phrase that matches a MONDO descendant with no ONVOC crosswalk should walk up MONDO's
   ancestors and stop at the first one that has a crosswalk. That is how a subtype reaches the
   general term without ONVOC needing to enumerate subtypes — and it uses the ties ONVOC
   already publishes. The crosswalk's own `mapped_term_label` column adds nothing lexically:
   it repeats the ONVOC label, and Schizophrenia, Depressive Disorder and Bipolar Disorder
   each receive zero synonyms from it.
5. **Record that a rollup happened.** A mapping reached by dropping a qualifier is a weaker
   claim than an exact one, and a query that cannot tell them apart will silently treat
   first-episode and chronic cohorts as the same population.
6. **Require the record to corroborate an acronym.** ONVOC contains exactly one label whose
   initials are MDD, and it is Mood Dysregulation Disorder, while every paper writing MDD
   means Major Depressive Disorder. A paper that means the expansion writes it somewhere.
7. **Treat a citation as evidence of a proper noun.** An instrument is nearly always
   introduced with one — "PANSS (Kay et al., 1987)". A phrase followed by a citation is a
   named scale rather than a description, which is a cheap signal for ranking candidate terms
   and for telling `Brief Psychiatric Rating Scale` apart from `brief rating of symptoms`.
8. **Emit what could not be placed.** The useful output of a normalization layer is not only
   what it mapped. A term used once is a paper's idiosyncrasy; a term used in ten is a term
   the vocabulary lacks, and the list above is what that produces.

## What to propose to ONVOC

Ranked by how much of this corpus they unlock:

- **`Healthy Control`** under a study-role branch — 233 unmatched values, and nothing else
  recovers them.
- **A clinical rating scale branch** distinct from `Tests`: PANSS, SANS, SAPS, BPRS, SCID,
  MINI, BDI, HAM-D, MADRS, YMRS, Edinburgh Handedness, NART.
- Nothing for tasks. That belongs to Cognitive Atlas, which already covers them.
