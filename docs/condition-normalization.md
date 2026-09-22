# Normalizing `Group.medical_condition`

Why this field is linked the way it is. The code states the rules; this file states the
reasons and carries the measurements they rest on.

Companion to [normalization-layer.md](normalization-layer.md), which is the contract for
the layer as a whole. Where the two disagree, this file is later.

> **The corpus and the schema have diverged.** Every corpus count below was measured over
> records extracted before `is_healthy` became derived and before cohort traits were split
> into `population_characteristics` and `other_characteristics`. The corpus therefore holds
> shapes the current slot no longer asks for — `healthy smokers`, where the trait belongs
> in `population_characteristics` and only the dependence belongs here; `not reported` as a
> value, where `extraction_status` is the place to say it. Those are slot defects. The rule
> is to *read* them correctly, not to encode them: recover the condition out of a
> trait-shaped value rather than swallowing the value whole, and treat a non-answer as
> unread rather than as a well cohort. Nothing in the code is tuned to a shape that a
> re-extraction under the current schema would stop producing.

## 1. What a value asserts, before any vocabulary

`vocabularies.phrases.triage` decides this, and both the MONDO route and the ONVOC route
read it from there so the two cannot drift.

Four layers, in order:

| | |
|---|---|
| `UNREAD` | a non-answer is not an answer. First, so "unknown" never reaches the wellness test |
| `HEALTHY` | a bare assertion of wellness, anchored at both ends |
| a parse | negation scope over the **whole** value, before it is split |
| a cue window | the residue of the parse, and the whole layer where no parser is installed |

### The gate was anchored, tested once, and ran before the split

So it could only ever see a cue in the first position — and the cues that matter are not
there. Every one of these was a wrong answer in production:

```
Parkinson's disease; no dementia                     -> Parkinson's Disease AND Dementia
major depressive disorder / no comorbid anxiety …    -> Depressive Disorder AND Anxiety
healthy adults without severe neurologic disorders   -> a sick cohort
absence of major depressive disorder                 -> Depressive Disorder
HIV-negative, drug-free                              -> the thing they deny
```

`absence of X` is the sharpest of them: there is no `neg` dependency anywhere in it and no
negating adjective. The negation *is* the head noun and the disorder is its `of`-complement,
so both the regex and the dependency scan missed it. `absence` and `lack` are now negators
in their own right.

The errors are not scattered. 22% of `medical_condition` values state an absence and they
are overwhelmingly the **control** cohorts, so the mistake lands on exactly the groups a
patients-versus-controls query has to tell apart.

### Parse the whole value, then split

A negation scopes over a coordination. "no history of major depression or other mental
illness" denies both conjuncts; split first and the second arrives at the vocabulary with
its cue left behind in the first. So: one parse of the whole value, drop the characters in
scope, then split what survives.

**By character offset, not by rejoining tokens.** A tokenizer does not round-trip. Rejoining
spaCy's tokens turned `treatment-resistant` into `treatment - resistant` — which the
qualifier pattern no longer matches — and `Parkinson's` into `Parkinson 's`, which is a
vocabulary key that cannot exist. The parse decides which characters are in scope and
changes nothing else.

**Then a cue window over the residue.** A parse drops what it can prove and sometimes leaves
the cue standing: in "physically healthy adults without severe neurologic disorders",
`disorders` parses outside `without`'s subtree, so the survivor still reads "…without
disorders". NegEx's forward scope (Chapman et al. 2001) catches it. Scope runs to the end of
the part rather than over a fixed token window, because the caller has already split on the
separators that terminate one. This is also the entire negation layer on a host with no
parser, and it is strictly better than the leading anchor it replaces.

Where the scope came from is recorded on the row (`parse`, `cue`, `assertion`), for the same
reason `Mapping.method` is recorded: a scope a parse proved and one a forward-window regex
guessed are different claims.

### `healthy` was unanchored, and that broke the derived field

`healthy` was one alternative of the leading negation pattern, so `healthy smokers` and
`healthy obese adults` triaged as absences, the condition in them was never looked up, and
`Group.is_healthy` — derived through the same call — came back True for both. The storage
schema calls that a contradiction in as many words: "healthy male smokers" with nicotine
dependence is `false`, because the addiction and obesity literature writes "healthy" to mean
free of *comorbidity*.

`is_healthy.py` already held a careful pattern — anchored at both ends, with a study-role
allowlist — and it could never help, because it was consulted *first* and the loose one
caught what fell past it. One rule now, `phrases.HEALTHY`: a value made of nothing but
wellness words, study-role nouns and punctuation. `healthy older adults` and
`Healthy, non-clinical population` match; `healthy smokers` and `healthy obese adults` do
not, because `smokers` and `obese` are in neither list and a word outside both ends the
match.

The test is re-applied to whatever survives the negation pass, so "physically healthy adults
without severe neurologic disorders" — which survives as "physically healthy adults" — does
not reach a disease vocabulary. An embedding retrieval returns *something* for any input.

### Three states, and the third is not the second

`Group.is_healthy` is "unset when `medical_condition` was never extracted — that is not a
healthy cohort, it is an unread one". `extraction_status` carries that when the extractor
sets it; a model that writes the same claim into the *value* (`unknown`, `not reported`) used
to be read as a cohort with no conditions. `NOT_READ` is now distinct from `NO_CONDITION`.

### Where `is_healthy` gets a vote

`phrases.absent(text, is_healthy)` is the rule, and before this work it had **no callers** —
the documented three-way behaviour was implemented nowhere. The MONDO route read the flag
into a `Counter` that nothing returned.

The order is measured and is the reverse of what
[normalization-pipelines.md](normalization-pipelines.md) recommended. The two agree on 93% of
the 3,624 mentions where both are known, but of the 237 where `is_healthy` is true and the
string carries no negation, most name a real trait the cohort was *selected* for: `obesity`,
`cannabis use`, `postmenopause`, `nicotine dependence`. Letting the flag overrule the string
threw those away.

So the string decides; the flag fills only the mentions whose string says nothing either way.
The asymmetry matters: **the flag can add an absence and never remove one**, because it is
itself derived from this field and a flag that could overrule its own source would be a loop.

## 2. MONDO, not SNOMED

Three reasons, none of them close:

| | |
|---|---|
| **it cannot be shipped** | SNOMED CT needs a UMLS/SNOMED affiliate licence per user. There is no URL this repository can fetch, so `data/vocab/` could not hold it and a checkout could not reproduce a run |
| **it is not disease-only** | SNOMED is a full clinical terminology — procedures, body structures, organisms, findings. `medical_condition` asks for a diagnosis, and every non-disease branch is surface area for a confident wrong match. MONDO is diseases and nothing else |
| **MONDO carries it anyway** | of 32,109 live MONDO classes, 8,881 (28%) xref a SNOMED concept id and 21,660 (67%) a UMLS CUI |

Picking MONDO gives up only the SNOMED concepts that no disease maps to. Each linked row
carries `mondo`, `umls` and `sctid`, so a clinical system can be asked in its own
identifiers.

Release 2026-09-01, CC-BY, fetched by `python -m pondie.vocabularies.fetch mondo`. That
fetcher is new: `paths.py` documented `mondo.json`, `pipeline-architecture.md` listed it and
`load_mondo` read it, but **nothing in the repository had ever written it**. The module's
measurements were taken on a machine where someone had put the file there by hand, so they
could not be reproduced anywhere else, and a fresh checkout got `FileNotFoundError` from
inside a JSON parse.

## 3. Retrieval over surface forms, not labels

The encoder is `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` — and yes, that is a model
trained on UMLS synonym pairs, which is the right distribution for this input. The choice is
already measured in `normalization/_embedding.py`: on short entity strings SapBERT reaches
R@1 66.3% against MiniLM's 50.6%, and on 400-word task descriptions the two invert
completely. Nothing about the model needed changing.

What needed changing is what it was asked about. Retrieval ran over `vocab.labels` (32,109)
while the exact index ran over `vocab.surface` (102,615), so the ~70,000 exact synonyms MONDO
publishes were reachable only by a literal string match and invisible to the encoder. That is
backwards: a synonym is precisely the form a paper is likely to have written, and the label
is the one form an ontology editor chose. Scoring against every form and resolving to the
node behind the best one is the same retrieval with the vocabulary's own evidence added.

### The thresholds, and the distributions they are read off

`calibrate()` runs the measurement rather than asserting it: take MONDO nodes carrying two or
more exact forms, hold one form out of the index, retrieve it against everything else, and
record the top-1 cosine separately for the queries that resolve to the right node and the
ones that do not.

Measured over 1,500 held-out forms, R@1 = **0.769**:

| | n | p10 | median | p90 |
|---|---|---|---|---|
| correct top-1 | 1,154 | 0.923 | 0.990 | 1.000 |
| wrong top-1 | 346 | 0.737 | 0.879 | 0.969 |

**Indexing the synonyms is what made the score usable.** The previous docstring reported
correct matches at p10 0.807 against wrong top-1s with a median of 0.820 — an overlap so
complete that no cut separates them — and cut at 0.90/0.80 anyway. Against all surface forms
the correct median moves to 0.990 and the two distributions come apart.

They still overlap, so there are two cuts and not one:

| cut | accepted | right | wrong | precision | of all correct |
|---|---|---|---|---|---|
| 0.80 | 1,398 | 1,137 | 261 | 0.813 | 0.985 |
| 0.85 | 1,323 | 1,119 | 204 | 0.846 | 0.970 |
| 0.90 | 1,221 | 1,074 | 147 | 0.880 | 0.931 |
| 0.93 | 1,117 | 1,017 | 100 | 0.910 | 0.881 |
| **0.95** | **1,026** | **959** | **67** | **0.935** | **0.831** |
| 0.97 | 882 | 851 | 31 | 0.965 | 0.737 |
| 0.98 | 781 | 762 | 19 | 0.976 | 0.660 |

`ACCEPT = 0.95`, `REVIEW = 0.85`. The first cut was set at 0.97 on the argument that the
errors are asymmetric — a wrong mapping is queried across a corpus and believed, a queued one
is looked at. Real records moved it down: the 0.96–0.97 band turns out to hold correct
matches that differ from MONDO's label by orthography, not near misses.

```
0.968  insomnia disorder                          -> insomnia
0.966  behavioural variant frontotemporal dementia -> behavioral variant of frontotemporal dementia
```

One is en-GB spelling plus an inserted `of`, the other a dropped head noun. Queuing those for
a human buys nothing, and nothing wrong was observed between 0.95 and 0.97.

**The proxy is optimistic.** A held-out MONDO synonym is written in ontology style and is
often a near-duplicate of a sibling synonym, which is why the correct p90 is 1.000. A corpus
phrase is not. Treat the table as the shape of the trade-off rather than as the precision
this will show on records, and re-run the sweep if the index or the model changes:

```
python -m pondie.normalization.medical_condition --calibrate
```

### What the review band actually holds

Run over the 16 benchmark records, everything the retrieval could not accept:

```
0.968  insomnia disorder        -> insomnia                                    right, accepted
0.936  substance dependencies   -> substance dependence                        right, queued
0.914  ADHD                     -> attention deficit-hyperactivity disorder,   WRONG, queued
                                   susceptibility to, 1
0.908  depressive episode       -> depressive disorder                         coarser, queued
0.897  neuropathic pain         -> neuropathy, painful                         plausible, queued
0.739  ongoing depressive episode         -> depressive disorder               rejected
0.726  compression of the dorsal root …   -> nerve compression syndrome        rejected
```

This is the argument for two cuts stated in one table. The band below the accept is genuinely
mixed and the mix is not ordered by score: the *wrong* answer at 0.914 outscores a plausible
one at 0.897. A single cut anywhere between them buys a wrong mapping to save a queue entry.

Two things the `ADHD` row shows. First, MONDO carries OMIM susceptibility loci —
"attention deficit-hyperactivity disorder, susceptibility to, 1" — which score high against
a short query and are not the clinical diagnosis; that is a standing precision hazard in this
vocabulary. Second, it is an abbreviation, and the run had no `--texts`. Given the paper's own
definition it expands to `attention deficit hyperactivity disorder` and matches exactly at
the lexical layer, never reaching retrieval at all. The abbreviation layer is a precision
mechanism, not only a recall one.

### Abbreviations, expanded per paper

The MONDO route had **no expansion layer at all** — `link` went from the raw head straight to
`folding.variants`, which is orthography and cannot bridge `bvFTD` to four words. So the
acronyms failed on this route for a reason that has nothing to do with MONDO's coverage, and
any comparison against the ONVOC route, which does expand, was measuring that asymmetry
instead of the two vocabularies. Both routes now expand, and both expand against the paper's
own definitions only.

## 4. The bridge: MONDO identity, ONVOC grain

MONDO is the grain the literature writes a diagnosis in. ONVOC is the grain a query filters
on. Neither is canonical, and which one a meta-analysis needs is the meta-analysis's
decision — so a row carries both, plus the relation between them.

The relation is derived, not compared. Matching a value twice and comparing the two labels
cannot tell "the same concept at two grains" from "one of these is wrong", which is the only
thing worth learning. An identifier relation with a path behind it can:

```
behavioral variant of frontotemporal dementia   MONDO:0017160   SCTID 716994006
  -> dementia (MONDO:0001627)                   ancestor, 7 steps up
  -> Dementia (ONVOC:0000190)                   crosswalk
```

Four layers, in falling order of how much they claim:

| `onvoc_via` | |
|---|---|
| `crosswalk` | the matched node is tied to an ONVOC term by ONVOC's own file |
| `ancestor` | a node above it is — so the ONVOC term is a **parent** of this one, and the row is a grain gap |
| `name` | ONVOC has the term and the crosswalk simply does not say so |
| `ancestor-name` | the same, for a node above it |
| *(empty)* | ONVOC cannot name this at any grain — the row is a **proposal** |

### Why the ancestor walk is load-bearing

The published crosswalk (`data/vocab/onvoc-mappings/mondo.tsv`) is 90 rows over 69 ONVOC ids,
reaching 66 of the 205 disorder-branch concepts. A direct-lookup-only bridge would be two
thirds empty, and the terms it misses are not obscure: `Alcohol Use`, `Smoking`, `Insomnia`,
`Chronic Pain`, `Huntington's Disease`.

Walking up recovers more than the flat route ever did.
[normalization-layer.md](normalization-layer.md) lists `alcohol dependence` (106 studies) and
`nicotine dependence` (101) as ONVOC **coverage gaps** — but that was measured by matching the
corpus string against ONVOC's labels, which measures ONVOC's flat surface rather than what it
can name. Both are MONDO terms whose ancestors reach `Substance Dependence`. ONVOC can name
those cohorts. What it lacks is the *substance*, which is a narrower and far more actionable
proposal.

### Why the name layers exist

The crosswalk is incomplete in the other direction too. ONVOC carries `Post-Traumatic Stress
Disorder` and `Huntington's Disease` and the file ties neither to MONDO, so an
identifier-only bridge reports them as terms ONVOC lacks. A proposal is the one output of
this layer that somebody acts on, so proposing a term the vocabulary already has is worse
than proposing nothing. The name check runs before a row is allowed to become a proposal, and
only on the tight end of the ladder — `exact`, `synonym`, `stem`. `contains` and `overlap`
are the layers that produced defect 2 in normalization-layer.md, and the input here is an
ontology label rather than a paper's prose.

### An ambiguous crosswalk row decides nothing

`MONDO:0005148` is listed under both Type 1 and Type 2 Diabetes Mellitus, so at least one of
those rows is wrong. Taking the first row seen made every type 2 cohort in the corpus a type 1
cohort — a wrong answer sourced entirely from the crosswalk, arriving with the ontology
author's authority. A MONDO node claimed by two ONVOC terms is dropped, and the name layer
then resolves both correctly. Two of the 88 ties go this way.

## 5. What a row is, and why not an aggregate

One `Link` per (study, group, head). The corpus-wide `Counter` it replaces could say how
often a form occurred and nothing else: no study was kept beside the mapping, so the result
could not be joined to the record it came from, could not be compared with the ONVOC route
row for row, and could not be audited back to a paper. Aggregation is a view over rows; rows
are not recoverable from an aggregate.

The row keeps the **finest** MONDO node that matched, never the rolled-up one. A route that
emitted only the rolled node — which this did — makes "are the two vocabularies at different
levels of the hierarchy" a question its own output cannot answer.

`denied` is carried rather than discarded. A cohort described as "schizophrenia; no substance
abuse" has said something true about substance abuse, and dropping it cannot be told apart
from a paper that never mentioned substances — the same conflation `is_healthy` keeps three
states to avoid.

`Result.grain_gaps()` and `Result.proposals()` are the two views that answer the original
question. A grain gap **is** agreement: the MONDO node is a descendant of the node ONVOC
names, so the two are one concept at two grains and the finer one is a term ONVOC could
carry. A proposal is a real gap. Both count distinct **studies**, not mentions: a paper
naming its diagnosis once per group is one piece of evidence that the term exists, not four.

## 6. Abbreviations must name a paper

An expansion is a fact about the paper that wrote it. 21.4% of short forms are expanded
differently by different papers and only 0.1% two ways inside one paper, so the paper is the
unit at which an abbreviation is unambiguous.

The store already enforced that in its lookup and lost it at the call sites:

- `corpus.py` called `for_paper(text)` **without the study id it had in scope**, so the
  store's own per-paper rows were never layered and only freshly re-mined text answered
  anything — the 1.4 MB `abbreviations.json` contributed nothing to a run.
- When a paper's text was missing, the `except` handed the mapper the **corpus-wide** store,
  which expands nothing for anybody. Silent: the run finished, every acronym in the corpus
  went unexpanded, and the output read as a vocabulary-coverage result rather than a bug.
- `repair.guard._words` calls `expand(token)` with one argument, and `paper` was a required
  *argument*, so every store that reached it raised `TypeError`. The expansion pass in repair
  was dead rather than strict — the opposite of what requiring it was for.

Now: the store carries the paper it is scoped to; `for_paper` refuses to scope without
naming one; `expansions_in` raises when neither a paper nor a scoped store is given, and
raises again if the two disagree; and a paper whose text cannot be read gets `None` and is
counted in `unexpanded_papers` rather than silently downgraded.

Raising is right here because the alternative failure is invisible. An unexpanded acronym
looks exactly like a term the vocabulary lacks, and the candidate list is downstream.

## Still open

- Risk phrasings (`high risk for depression`, `familial risk for bipolar disorder`, 32
  mentions) are read as diagnoses by both routes. Risk is neither diagnosis nor absence and
  needs a third sentinel of its own.
- `regions.name` resolves `anterior cingulate cortex` to the caudal parcel by containment,
  where it should refuse.
- The ONVOC route still emits its own `Mapping` rows and this route emits `Link` rows. They
  now share triage, the abbreviation rule and the study/group key, so a row-for-row
  comparison is possible; nothing yet runs it.
