# Grouping tasks by multichannel similarity

There is no vocabulary for tasks and no gold for them either. `docs/task-terms.md` shows the
ceiling on the vocabulary route: the Cognitive Atlas names a paradigm for 29% of mentions,
ONVOC for none, and the head of this corpus — `resting state` (144 studies), `cue reactivity`
(50), `cue exposure` (26) — is in neither. So the corpus is grouped against itself, and the
question is how.

This file is the method. Most of it is implemented in `pondie/normalization/task.py` and
`_clustering.py`; the changes in [§4](#4-three-changes-the-measurements-demand) are not, and
each one carries the measurement that asks for it.

## 1. The grain

One group per **paradigm**, not per variant. `letter n-back`, `emotional faces n-back` and
`N-back working memory task` are one group; `n-back` and `Sternberg` are two. The properties
that distinguish variants — stimulus content, stimulus and response modality, load level — are
*attributes of the member*, carried alongside, not reasons to split.

That is a choice about what a query needs, and it is the reason the clusterer has a second
level. `identity` is the paradigm; `family` groups identities that share a construct
(`go/no-go` with `stop-signal`, as inhibition). Two thresholds, not one.

## 2. Six channels, kept separate

Each is a similarity in [0,1] between two tasks, computed independently. They are **not**
concatenated into one vector before comparison, and that is measured rather than stylistic:
folding `performance_measures` into a single signature *shrank* the stop-signal / go-no-go
margin from +0.040 to +0.031, because a sentence embedding is a mean over its passage and the
one discriminating token gets averaged away by the shared vocabulary around it.

| channel | input | encoder | what it carries |
|---|---|---|---|
| `name` | `tasks[].name` | MiniLM, cosine | the paradigm when the paper names it |
| `prose` | `description` + `instructions` | MiniLM, cosine | what participants actually did |
| `setting` | `stimuli` + `design_type` + `response_modality` | MiniLM, cosine | the apparatus: block vs event, button vs speech |
| `measures` | `performance_measures` | MiniLM, cosine | RT/accuracy/SSRT — often the sharpest paradigm tell |
| `conditions` | `conditions[].name`, as a **set** | MiniLM, soft overlap | the factorial structure |
| `prose_lex` | `description` + `instructions` | TF-IDF 1–3gram, cosine | the rare exact phrase mean-pooling discards |

Two of these deserve their rationale stated.

**`conditions` is a set, not a paragraph.** Joining the condition names into one string would
let a task with nine conditions dominate one with two. The overlap is the mean of the two
directional best-matches — for condition sets *A* and *B*, `(mean_a max_b cos(a,b) + mean_b
max_a cos(a,b)) / 2` — which keeps `win` beside `gain` without penalising a short list. This
is the channel that does the work you asked it to: two go/no-go papers that named the task
differently still both have a `go` and a `no-go` condition.

**`prose` and `prose_lex` are the same text twice, and both carry weight.** Adding the sparse
view lifted pair AP from 0.301 to 0.407. IDF preserves `SSRT` or `Cyberball`; a mean-pooled
embedding does not.

### 2.1 The encoder, and a choice nobody has checked

**One pretrained model does all the semantic work**: `sentence-transformers/all-MiniLM-L6-v2`
— 6 layers, 384 dimensions, 22.7M parameters, general-domain. Nothing is fine-tuned and, since
the pair model went ([§4.1](#41-drop-the-ladder-from-training-keep-only-its-equality-rule)),
nothing is fitted at all. `prose_lex` is sklearn's TF-IDF and the clustering is sklearn's
agglomerative; neither learns anything from this corpus.

The rule in `_embedding.py` is to pick by input length: `for_phrases` (SapBERT, UMLS-trained)
for entity strings, `for_prose` (MiniLM) for sentences, measured at R@1 66.3% vs 50.6% on
short strings and 58.5% vs 24.5% on task descriptions. **The task channels do not follow it.**
All six call `for_prose`, including `name` and `conditions` — and `for_phrases`'s own docstring
names its use case as "a disease name, a group label, **a condition**".

The premise has also drifted from the corpus. MiniLM was chosen for prose described as "~400
words". Measured here, the prose channel is a median of **46 wordpieces** and a p90 of 82 —
about 35 words, which is nearer the short-string regime where SapBERT won than the paragraph
regime where it lost. (Truncation is a non-issue either way: 1 task of 1,672 exceeds the
256-wordpiece limit.)

Swapping the entity channels to SapBERT is not a cosmetic change:

| | clusters | largest | categories | in no category |
|---|---|---|---|---|
| MiniLM everywhere (what produced `task-categories.md`) | 377 | 226 | 173 | 204 |
| SapBERT for `conditions` only | 325 | 256 | 156 | 169 |
| SapBERT for `name` + `conditions` | 258 | 272 | 132 | 126 |

MiniLM vs SapBERT-for-both agree at **ARI 0.731**. So roughly a quarter of the partition turns
on an encoder choice that contradicts the module's own contract, rests on a description of the
input that does not hold here, and cannot be adjudicated because there is no gold. SapBERT
merges more; whether that is better is exactly the question [§5.1](#51-label-the-304-pairs-the-ladder-cannot-settle)
exists to answer, and it is now the second thing those 304 labels would settle.

The structural diff is [task-categories-compared.md](task-categories-compared.md); the two
full listings it was drawn from were outputs of a superseded pipeline stage and have been
removed. 36 SapBERT categories merge what
MiniLM split and 37 MiniLM categories are split by SapBERT, so this is not one model being
uniformly coarser; they disagree in both directions.

### 2.2 How often a name really is (stimulus × paradigm)

The two-dimension reading — `emotional Stroop` and `cocaine Stroop` as one paradigm crossed
with two stimuli — is the premise the facet design rests on, so it is worth knowing how much
of the corpus it describes. Measured over the 1,696 task mentions, with the paradigm's own
span removed before looking for a stimulus (otherwise `monetary incentive delay` counts
`monetary` as a stimulus and the rate inflates from 14% to 23%):

| | mentions | |
|---|---|---|
| named paradigm × stimulus | 229 | **13.5%** |
| stimulus × bare presentation frame (`visual food cues`) | 212 | 12.5% |
| paradigm, no stimulus | 608 | 35.8% |
| stimulus, no frame | 268 | 15.8% |
| frame only | 47 | 2.8% |
| neither | 332 | 19.6% |

**42% of names carry a stimulus term and 49% carry a paradigm term, but only 14% carry both
explicitly** — 26% if a bare presentation frame (`cues`, `images`, `stimuli`, `viewing`) is
read as naming the paradigm, which for cue reactivity it effectively does: `visual food cues`
*is* a cue-reactivity task whose name is only its stimulus.

**The pattern is productive, which is the part that matters.** 60 distinct combinations are
used; **13 of 25 paradigms appear with two or more stimuli, and 11 of 13 stimuli appear with
two or more paradigms.** It is a real cross-product, not a coincidence of wording:

| paradigm | tasks | stimuli it crosses |
|---|---|---|
| cue exposure | 26 | alcohol, cannabis, cocaine, drug, food, opioid, social, tobacco |
| cue reactivity | 59 | alcohol, cannabis, cocaine, drug, food, opioid, tobacco |
| Stroop | 25 | cocaine, drug, emotion, opioid, social, tobacco, words |
| go/no-go | 17 | alcohol, emotion, food, money, social |
| cue induction | 12 | alcohol, cocaine, food, tobacco |
| reappraisal | 26 | emotion, food |
| n-back | 7 | emotion, letters |

The addiction frames are the most productive: `cue exposure` and `cue reactivity` are
essentially a *frame that takes any substance*, which is why 8 and 7 stimuli respectively pass
through them.

**Consequence for this method.** Both dimensions are scientific variables, and which one a
query filters on depends on the question — "all cue reactivity regardless of substance" and
"all alcohol tasks regardless of paradigm" are both legitimate and neither is answerable from
a single cluster label. The clustering here **mixes both dimensions into one distance**, so it
can answer neither cleanly: that is why `smoking cue-reactivity task` has cannabis and heroin
members ([task-categories.md](task-categories.md)), and why cue reactivity
fragments four ways. Separating the two into their own slots — with the cluster carrying the
paradigm and the stimulus read off the member — is the change that would fix both symptoms at
once, and it is not made here.

### 2.3 How to separate them

Three steps, and the first costs nothing.

**Step 1 — split the `setting` channel. No lexicon required.** `Task.setting` is
`stimuli + design_type + response_modality` concatenated into one string and encoded as one
channel. But `stimuli` is *what was shown* and the other two are *how the task ran*: the
schema already separates them and the code joined them back together. Splitting `stimuli` out
and keeping only `design_type + response_modality` in the paradigm distance is a two-line
change that removes the stimulus from a channel that was never meant to carry it.

**Step 2 — strip stimulus terms before encoding `name` and the condition names.** So
`alcohol Go/NoGo task` and `food-specific go/no-go task` present to the encoder as the same
string, and their difference lives in the stimulus column instead of in the distance. This is
the step that needs a lexicon, and it is the one lexicon in this pipeline that is genuinely
small and closed: ~50 terms — substances, emotions, faces, words, money, food — reaching 42%
of names, and the substance half is groundable in ONVOC.

**Step 3 — read the stimulus off as its own column**, from the name, then the conditions,
then the `stimuli` field, first hit wins. It is a label, not a cluster: the set is small
enough to enumerate, so clustering it would add error for nothing.

Measured, cut held at 0.60 throughout. Fragmentation of a paradigm across clusters, lower
being better:

| paradigm | tasks | A shipped | B split `setting` | C + strip |
|---|---|---|---|---|
| cue reactivity | 57 | 12 clusters | 8 | **7** |
| cue exposure | 33 | 13 | 11 | **10** |
| Stroop | 30 | 5 | 4 | **3** |
| go/no-go | 27 | 11 | 10 | **8** |
| n-back | 18 | 5 | 4 | **3** |

and the largest cluster for each paradigm grows accordingly — Stroop 16 → 18 → **25** of 30,
n-back 10 → 15 → **16** of 18, go/no-go 9 → 10 → **17** of 27. Tasks in no category fall from
204 to 157. A and C agree at ARI 0.649, so this moves about a third of the partition.

The output is then a pair per task, which is what the queries actually want:

```
  64  cue-induced craving task   alcohol (34), tobacco (16), cocaine (5), opioid (4), food (2)
  53  cue-reactivity task        tobacco (24), food (11), alcohol (8), cannabis (6), cocaine (1)
  31  delay task                 money (26), alcohol (3), social (1)
  29  Stroop task                emotion (11), words (7), cocaine (3), alcohol (2)
  26  go/no-go task              emotion (7), letters (3), alcohol (3), food (3)
  26  n-back task                emotion (9), letters (7), (unspecified) (7)
```

One paradigm, five substances, in one row. That is the thing neither a single cluster label
nor a flat vocabulary can express.

**Two things it does not fix, stated because the table above looks better than the result is.**
Cue reactivity is still spread over five clusters (`cue-induced craving`, `cue-reactivity
task`, `cue reactivity task`, `cue exposure`, `cue-exposure task`) — once the stimulus is out,
the remaining variance is prose wording, and that is the `family` tier's job rather than this
one's. And stripping degrades the cluster *labels*, because the label is the most frequent
member name and the members are now stripped: `emotion regulation task` becomes `regulation
task`, and one cluster is labelled `task`. Strip for the distance, label from the unstripped
names — the current experiment does not, and should.

### 2.4 Seed from the Cognitive Atlas, cluster only the residue

A vocabulary match gives a category a name somebody else already agreed on, which is the one
thing clustering cannot do. So: match what the Atlas covers, cluster only what is left, then
ask of each residual cluster whether it is a known paradigm under another name.

**Filtering the Atlas first, because it has the same two-dimension problem inside it.** The
Atlas lists `letter n-back task`, `face n-back task` and `spatial n-back task` alongside
`n-back task` — stimulus variants of one paradigm, exactly what [§2.2](#22-how-often-a-name-really-is-stimulus--paradigm)
describes. Three filters take 856 labels to **724**:

| | | |
|---|---|---|
| collapse onto a parent | 6 | `letter n-back task` → `n-back task`; also face/spatial n-back, `color-word stroop task`, `Visual short term memory task` |
| drop, instrument not paradigm | ~100 | anything ending `scale`, `inventory`, `questionnaire`, `battery`, `checklist` |
| drop, too generic to match on | ~26 | whole-label generics (`maze`, `vigilance`, `drawing`) and labels under 7 characters |

The generic filter has to test the **whole** label, not its first word: an earlier version
anchored on a prefix and dropped `counting Stroop task` and `Judgment of Line Orientation
Task`, which are real paradigms whose first word happens to be generic.

**Matching.** Exact fold, then exact fold of the stimulus-stripped name, then bidirectional
token-subsequence containment, then a squashed-substring fallback. Both of the last two were
added after measurement:

- **Bidirectional.** A seed inside the name is the obvious case (`alcohol go/no-go task`
  contains `go/no-go task`). The reverse is as common and was missed: a paper writing
  `emotion regulation` is naming `Emotion Regulation Task` with the generic head dropped, and
  **48 studies did exactly that**.
- **Squashed.** `go/no-go task` tokenises to `('go','no','go','task')` and `alcohol Go/NoGo
  task` to `('alcohol','go','nogo','task')`, so token containment misses a seed that is
  plainly there. Hyphenation is not a distinction.

Together these took seeding from 330 tasks / 49 categories to **392 tasks / 66 categories**.

**Result on the corpus: 392 of 1,672 tasks (23%) seed; 1,280 go to clustering**, yielding 136
residual clusters of 2+ members and 144 singletons.

**Judging the residue.** The useful question is "would the clusterer have merged this with a
seeded category if nothing had been seeded", and that is answerable by clustering everything
jointly and reading off where each group lands — no new threshold. Two earlier formulations
were wrong and are recorded because the failure is instructive: the *mean* distance to a
seeded category buries the one member that matters, and the *max* reports 1.00 because
must-link writes zeros into `D` for tasks sharing a stripped name.

The overlap **count** is the confidence, and it separates cleanly:

| overlap with the seed | residual clusters | trustworthy? |
|---|---|---|
| ≥10 seeded tasks | 10 | yes — `emotion regulation paradigm`, `cognitive reappraisal`, `active regulation` and `Voluntary regulation of negative affect` all land on `Emotion Regulation Task` (91 tasks overlapping) |
| 1–9 | 30 | no — `visual food cues` = `passive viewing` on **1** task, `Faces` = `Emotion Regulation Task` on 1 |
| none | 96 | the paradigms the Atlas lacks |

**What the residue says the Atlas is missing**, by studies: `resting state` (172), `cue
reactivity` (75 + 60 + 13 in three clusters), `cue exposure` (37 + 25), `food cue task` (18),
`Taste Cue Paradigm` (14), `personalized guided-imagery task` (11). The same list
[task-terms.md](task-terms.md) produced by a different route, which is some comfort.

**The judge is usable at overlap ≥10 and should abstain below it.** It is a ranking, not a
decision.

### 2.5 Two more filter rules, from the `drug Stroop` case

`drug Stroop fMRI task` (21 studies) did not reach the seeded `Stroop task`, and an earlier
draft of this file blamed the judge. That was wrong: **it never reached the judge, because it
never seeded.** `Stroop task` tokenises to `('stroop','task')` and the name to
`('drug','stroop','fmri','task')` — `fmri` sits between `stroop` and `task`, so the
contiguous run is broken. Diagnose before prescribing.

The fix is to drop **method words** — `fmri`, `task`, `paradigm`, `test`, `block`,
`event-related` — from both sides before the containment test, leaving a *core*:
`('stroop',)` inside `('drug','stroop')`. That took seeding from 394 mentions to 547, +39%.

It also needs three guards, each added after it broke something measurable:

- **A one-token core must be distinctive.** Cores are often one token, and a bare common word
  matching alone is how `art emotion test` became the Angling Risk Task. Require ≥6
  characters and not one of a small common-word list (`memory`, `recall`, `control`, `faces`).
- **Rank by longest MATCH, then shortest LABEL.** Ranking on raw seed length handed `Go-NoGo
  task` to `Go-NoGo fMRI paradigm` and `N-back working memory task` to `working memory fMRI
  task paradigm`. You want the most specific thing that matched under the plainest name for it.
- **The squashed test needs a word boundary.** Squashing to catch `go/no-go` ≈ `GoNoGo` also
  let `Motion processing` match `emotion processing task` — e-MOTIONPROCESSING-task — taking
  12 tasks with it. Requiring the seed's first core token to start some token of the name
  restores the boundary that squashing removed; the length guard can then drop to 6, which
  `gonogo` needs.

And two more filters on the Atlas itself, for the same reason:

- **Collapse labels identical once method words are dropped.** `Go-NoGo fMRI paradigm` and
  `go/no-go task` both reduce to `gonogo`; keep the shorter. 7 pairs.
- **Drop compound labels.** `cue-based expectancy paradigm combining emotion regulation task`
  is the Atlas describing one study's design, and as a seed it swallowed 27 tasks belonging to
  `Emotion Regulation Task`. Anything with `combining` / `combined with`, or a core over six
  tokens.

Result: 856 → **693** seeds, **536 of 1,696 mentions seeded (31%)** into 101 categories, with
`drug Stroop fMRI task` → `Stroop task`, `alcohol Go/NoGo task` → `go/no-go task`, `N-back
working memory task` → `n-back task` and `emotion regulation paradigm` → `Emotion Regulation
Task` all landing correctly.

### 2.6 Normalisation is not synonymy, and the two must not share a mechanism

The name is a hard constraint: two papers that wrote the same name wrote the same task, and
the clustering does not get a vote. That is only safe while "the same name" means something
purely orthographic. Two things were being done under one name and they are now two stages.

**Stage 1 — orthographic. A constraint.** Case, whitespace, separators, method words
(`task`, `paradigm`, `fMRI`), British/American spelling, the `n` in `n-back`, and the paper's
own acronyms expanded from the corpus abbreviation store. Every one is reversible and asserts
nothing about the research.

Keying on the **core** rather than the folded string is what this turned on. Keeping method
words made `cue reactivity` and `cue reactivity task` two different keys, and one paradigm
came out as three categories. On the core they agree, and cue reactivity goes from three
categories to one of **89 studies** on orthography alone.

**Stage 2 — semantic. A curated claim, marked in the output.** `Taste Cue Paradigm` reduces
to the single token `cue`, and no string rule joins that to `cue reactivity` without also
joining every `cue task` in the corpus. That it belongs there is a fact about addiction
research: the cue family is one design, and `reactivity`, `exposure`, `induction`, `elicited`
and `viewing` are the same study described by different groups.

So `PARADIGM_FAMILY` is its own table, `--no-family` turns it off, and every category it
touches carries `rests_on_family_rule` in the output. Exactly **one** category does:
cue reactivity, which goes from 89 studies to **166**. A reader who rejects the claim can see
precisely what it bought and drop it.

| | categories | cue reactivity | in no category |
|---|---|---|---|
| orthography only | 193 | 89 studies | 122 |
| + the family rule | 189 | 166 studies | 123 |

Two bugs had to be fixed before either worked, and both are the same shape — a constraint
that silently did not apply:

- **Must-link was a star, not a clique.** `d[group[0], j] = 0` for each other member. The
  clustering runs on the unseeded submatrix, so whenever `group[0]` was seeded, every other
  member lost its only zero. 49 tasks keyed `cuereactivitytask` were spread over seven
  categories that way. Every pair now gets the zero.
- **Acronym expansion, applied to task names, destroyed them.**
  `vocabularies.expansions_in` accepts any token containing an uppercase letter, which is
  right for running prose and catastrophic for a Title Case name: every word qualifies.
  `Cue Reactivity Task` came back as *"congruent and five for incongruent Reactivity task to
  assess planning"*. Restricted to ALL-CAPS tokens, 209 mangled names become 61 correct ones
  — `WM` → working memory, `CANTAB`, `EBA` → extrastriate body area, `MJ` → marijuana.

## 3. The pipeline

```
1. weak labels    name ladder -> connected components -> positive pairs within,
                  negative pairs across
2. pair model     logistic regression over the six channel similarities,
                  P(same paradigm | pair)
3. identity       agglomerative, average linkage, on 1 - P, cut at `identity`
4. rescue         attach a singleton to its nearest cluster when P >= `rescue_at`
5. family         agglomerative, cosine, over the `prose` centroid of each identity,
                  cut at `family`
```

### 3.1 What the name ladder is

`name_links()` in `_clustering.py`, plus `components()`. Two rules over task names, then the
transitive closure of both. It is the cheap pass that settles the easy pairs so the model can
be trained on them — and, today, it is also written into the distances as a certainty, which
is [§4.1](#41-stop-the-name-ladder-from-being-a-hard-constraint).

**Rule 1 — folded equality.** Lowercase, drop punctuation, drop spaces, compare. `Go/No-Go
Task` and `go no go task` both squash to `gonogotask`, so they are one task.

**Rule 2 — token-subsequence containment.** Fold to a token tuple; if one is a *contiguous
run* inside the other and both have at least two tokens, link them. `n-back task` →
`('n','back','task')` sits inside `emotional faces n-back task` →
`('emotional','faces','n','back','task')`, so those are one task.

Contiguous *tokens*, not substrings, and that is deliberate: `saccade task` is a substring of
`reward cue antisaccade task`, and joining an antisaccade study to a prosaccade one would
then train the pair model on the error.

It is strict in both directions, which is worth seeing:

| | |
|---|---|
| `Go-NoGo task` ~ `alcohol go/no-go task` | **not linked** — `nogo` is one token, `no go` is two |
| `n-back task` ~ `letter n-back with olfactory emotion induction` | **not linked** — `task` does not follow `back` |

**Then union-find.** `components()` takes the transitive closure, and that is where the
trouble is. A ~ B and B ~ C makes A ~ C even when A and C share no word at all. Real chains
from this corpus:

```
'Decision-Making Task'
  ~ 'emotion regulation and risky decision-making task'   (rule 2)
    ~ 'emotion regulation'                                (rule 2)
=> 'Decision-Making Task' and 'emotion regulation' are one task

'amygdala neurofeedback'
  ~ 'Amygdala neurofeedback regulation task'              (rule 2)
    ~ 'regulation task'                                   (rule 2)
=> 'amygdala neurofeedback' and 'regulation task' are one task
```

A **compound task name welds two paradigms together**: a paper that ran emotion regulation
*and* a decision-making task names it once, and that single name is a bridge. 22 such chains
exist inside the largest component alone.

> Naming collision, since the word appears twice in this codebase with different shapes. The
> *matching* ladder in `vocabularies/onvoc.py` is an ordered fallback — exact, synonym,
> variant, acronym, contains, stem, overlap — where the first hit wins and the layer that
> produced it is the confidence. The *name* ladder is not that: it is two rules both applied,
> plus transitive closure, with no ordering and no layer recorded.

**Distant supervision, then a grouped split.** Positives come from inside a name-ladder
component, negatives from across. The split for evaluation is *by component*, so no task
appears in both train and test. The honest number is the one with the `name` channel held out,
because the pairs the model exists to judge are exactly those the name could not settle:
**AUC 0.916 overall, 0.868 without the name channel**, measured under grouped 5-fold on 1,672
tasks by an evaluation script that has since been removed with the rest of the superseded
clustering code.

**The model decides identity; geometry decides families.** A logistic probability saturates
near 0 for non-matches, so it is a good decision score and a bad distance — using it for the
coarse cut piled every pair at 1.0. Families are therefore plain cosine over identity
centroids.

**Hand-written paradigm regexes were tried and removed.** A marker table (SSRT for
stop-signal, and so on) made clustering *worse*: 243 clusters / ARI 0.600 with it, 185 / 0.619
without. Giving the name its own weighted channel had already solved what it was for.

## 4. Three changes the measurements demand

### 4.1 Drop the ladder from training; keep only its equality rule

An earlier draft of this section said the fix was to stop writing the ladder into the
distances as must-link. **That was measured and it is not enough.** Three variants, same
channels, same cut:

| | clusters | largest | blob members land in |
|---|---|---|---|
| A supervised + must-link (shipped) | 136 | 416 | — |
| B supervised, must-link removed | 133 | **431** | 15 clusters, biggest 379 |
| C unsupervised, no ladder anywhere | 531 | 171 | **99 clusters, biggest 167** |

Removing the constraint alone does nothing — A and B agree at ARI 0.788 and the blob gets
*larger*. The reason is that the ladder's influence is not only in the constraints: the pair
model is **trained on ladder-derived labels**, so it has already learned to predict "same" for
exactly the pairs the ladder chained. Deleting the constraint leaves the lesson.

Remove the ladder from training as well and the blob does not exist. Under C the shipped
run's 416-task cluster scatters into 99, and what comes out is coherent on inspection:

```
171  resting-state protocol, resting state, resting-state fMRI
171  emotion regulation task, cognitive reappraisal task, emotion regulation
 34  cigarette cue exposure, cigarette cues task, smoking cue and control videos
 33  visual food cues, food cues, viewing photographs of high and low-calorie foods
 31  cue-reactivity task, smoking cue-reactivity task, Cue Exposure Task
```

**So the blob is manufactured by the supervision, not present in the data.** A vs C agree at
only ARI 0.332; these are different answers, not tunings of one.

Two further reasons not to trust the learned weights. They were fit against a flawed teacher,
and it shows in what they learned:

```
prose +5.96   name +5.17   prose_lex +3.46   setting +2.84   measures +1.09   conditions -0.25
```

`conditions` — the channel that carries the factorial structure, and the one most likely to
tell two paradigms apart — is given a *negative* weight. The model learned that condition
overlap is mild evidence **against** two tasks being the same, because the ladder's positives
are name-driven and condition sets vary freely within a name.

**Keep rule 1.** Pure unsupervised clustering does lose something real: of the 8,225 pairs
that share an identical folded name, only 74% stay together at cut 0.55 and 80% at 0.60 —
their median channel similarity is 0.528, right at the cut, because two papers can use one
name and describe it differently.

| cut | clusters | largest | singletons | identical-name pairs kept |
|---|---|---|---|---|
| 0.50 | 710 | 107 | 480 | 46% |
| 0.55 | 531 | 171 | 335 | 74% |
| 0.60 | 373 | 221 | 201 | 80% |
| 0.65 | 248 | 263 | 123 | 85% |

So force those 8,225 pairs and let the channels do the rest. **Equality is safe to close
transitively and containment is not** — that is the whole distinction. `A = B` and `B = C`
genuinely implies `A = C`; `A ⊂ B` and `B ⊃ C` implies nothing about A and C, which is how
`Decision-Making Task` reached `emotion regulation`.

### 4.1b What replaces it

```
distance   1 - weighted mean of the six channel similarities
must-link  folded equality only (8,225 pairs; largest component 87, and it cannot chain)
cluster    agglomerative, average linkage, cut ~0.55-0.60
```

No pair model and no distant supervision. Weights start equal, which is a guess — but a
defensible one next to weights fit on 270,000 pairs generated by a rule known to chain. Six
weights is also something the 304 labelled pairs of [§5.1](#51-label-the-304-pairs-the-ladder-cannot-settle)
can actually fit, where they could never fit a model that needs hundreds of thousands.

The cut needs care at the tail: at 0.60 the 201 singletons still include `cue-reactivity
paradigm` and `MJ cue reactivity task`, which plainly belong with the cue-reactivity cluster.
Under-merging at the tail is the failure mode to watch once over-merging at the head is gone,
and the singleton rescue pass already in `_clustering.py` is the place to fix it.

### 4.2 Give `cannot_link` something to carry

`distances()` already takes a `cannot_link` argument and nothing has ever passed one. The
obvious source — two tasks listed by one paper are two tasks — **does not survive contact with
the corpus**, and that is worth knowing before anyone builds it:

161 studies (11%) name two or more distinct tasks, giving 319 within-paper pairs. Inspected,
they are mostly one task written twice — `emotion picture task` ~ `perception of emotional
pictures`, `alcohol cue-induction paradigm` ~ `alcohol-induction paradigm`. Some are genuinely
different (`Go-NoGo response inhibition` ~ `spatial working memory`). So the signal is real
and it points *both ways*; as a hard cannot-link it would be wrong more often than right.

Use it as [§5.1](#51-label-the-304-pairs-the-ladder-cannot-settle) instead.

### 4.3 Re-derive the thresholds, but not yet

`identity=0.5` gives 121 clusters over 1,672 tasks with a 421-task largest. `identity=0.3`
scores better on every available measure. But every available measure is 93% the name ladder
(see [§5](#5-validating-without-a-gold)), so tuning on them selects for agreeing with the
ladder. **Change 4.1 first, then re-derive the thresholds against §5.1.** Doing it in the
other order bakes the chaining into the threshold.

## 5. Validating without a gold

This is the part that has been missing, and it is not "get a gold" — it is that three of these
four need no gold at all.

### 5.1 Label the 304 pairs the ladder cannot settle

Not 1,696 tasks: **304 pairs**. They are the within-paper pairs from §4.2 that the name ladder
calls different — an enriched pool, because every one is a judgement the cheap rule already
failed. Each is a yes/no question about two names from one paper, answerable from that paper's
Methods in under a minute.

That is a few hours of work and it yields the only thing this method currently lacks: a
precision/recall curve for `P(same paradigm)` on hard pairs. `benchmarks/gold/direction/` is
the format — numbered annotators, `tier`, `disputed` — and the proof the project can afford it.

### 5.2 Channel-holdout agreement

Cluster twice: once on `{name, prose_lex}`, once on `{prose, setting, measures, conditions}`.
These are near-disjoint views of the same task. Where two disjoint views agree, the grouping is
evidence; where they disagree, it is one channel's artefact. No labels required, and it is the
one check that would have caught the blob, since the blob is a `name` artefact that the prose
channels do not support.

### 5.3 Stability under resampling

Bootstrap the task set, recluster, measure pairwise co-assignment agreement across resamples.
Reports how much of the partition is a property of the data and how much of the threshold.
Cheap, and it needs nothing.

### 5.4 The mixing count, read by eye

Number of clusters containing two or more distinct Cognitive Atlas paradigms: **23 of 52**
today. The metric is weak — that gold is 93% the ladder — but the *instances* are checkable
without any gold. A cluster holding `go/no-go task`, `monetary incentive delay task` and
`temporal discounting task` is wrong on inspection, and no scoring choice rescues it. Track the
list, not the number.

## 5.5 Would a topic model do this instead?

[turftopic](https://x-tabdeveloping.github.io/turftopic/) was raised, and it is worth being
precise about which half of this problem it solves. Not installed or run here — this is read
off its documentation.

**Where it does not fit.** It is a *topic* model: the API is `Model(k).fit(corpus)` where
`corpus` is a list of one string per document, and there is no documented way to pass
precomputed embeddings or a precomputed distance matrix. A custom `encoder` class is the only
hook, so using it would mean flattening the six channels into one string or one vector — which
is the move this corpus has already measured as harmful (folding `performance_measures` into a
single signature shrank the stop-signal / go-no-go margin from +0.040 to +0.031). Its
`ClusteringTopicModel` is also TSNE-or-UMAP → OPTICS/HDBSCAN, and a *stochastic* reduction
before clustering is the wrong trade when 1,672 items fit comfortably in an exact pairwise
matrix.

**Where it fits well, and these are real.**

- **Naming.** Every category in [task-categories.md](task-categories.md) is labelled by its
  most frequent member name, which is crude — the `cue-induced craving task` category is named
  after one of its 28 members. c-TF-IDF or centroid terms over the members' prose would
  produce a description rather than a borrowed name, and turftopic also wires an LLM up to it.
  This is the single biggest weakness of the current output.
- **Outliers as a first-class label.** HDBSCAN's `-1` is a principled "no category", where
  §4.1b produces "clusters of size 1" and cannot tell a genuinely singular paradigm from an
  under-merged one. That distinction is exactly what the uncategorised list needs and does not
  have.
- **A second view for [§5.2](#52-channel-holdout-agreement).** A topic model over the prose
  alone is a near-disjoint view of the same tasks. Where it agrees with the channel clustering,
  that is evidence neither one alone provides.

So: not as the clusterer, plausibly as the labeller and as an independent check. If a single
model is wanted for the clustering itself, the constraint to shop against is *accepting a
precomputed distance matrix* — scikit-learn's agglomerative and HDBSCAN both do, which is why
the current approach uses them.

## 6. Order to build it in

1. **4.1b** — replace the pipeline's first two stages with equality-only must-link and an
   equal-weight channel mean. This deletes code rather than adding it: no ladder rule 2, no
   pair sampling, no logistic regression.
2. **5.2** — channel-holdout agreement, as a standing check alongside
   `pondie/normalization/task.py`. Needs no labels and catches this class of failure.
3. **5.1** — label the 304 pairs. Everything numeric depends on it, and it is what turns the
   equal weights into fitted ones.
4. **4.3** — re-derive the cut and the rescue threshold against 5.1, and only then.

Steps 1 and 2 are a day. Step 1 makes the pipeline *simpler* than it is now. Step 3 is the
one that needs a person, and it is four hours, not four weeks.
