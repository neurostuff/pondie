# The normalization layer

> Supersedes the coverage numbers in [normalizing-with-onvoc.md](normalizing-with-onvoc.md)
> (measured on 328 records) and the `is_healthy` recommendation in
> [normalization-pipelines.md](normalization-pipelines.md). The per-field *method* table in
> normalization-pipelines.md still stands; this file is the contract those methods run under.
>
> **Superseded in turn for `medical_condition`** by
> [condition-normalization.md](condition-normalization.md), which is later than this file
> on three points. The negation gate described below is anchored at the start of the value
> and tested before the split, so it misses every cue that is not in the first position.
> `alcohol dependence` (106 studies) and `nicotine dependence` (101) are listed below as
> ONVOC coverage gaps, which measures ONVOC's flat surface rather than what it can name:
> both reach `Substance Dependence` through MONDO's hierarchy, and the real gap is the
> substance. And the MONDO route it refers to did not run at all — nothing in the
> repository fetched `mondo.json`.

Measured over the **2,115 records** on `beast-proxy` — `cue` 550, `emotion_regulation` 525,
`dementia` 448, `sud` 244, `depression` 191, `ptsd` 62, plus the benchmark runs. Every number
below comes from that corpus, not from the 328-record sample the earlier file used.

## The contract

A record keeps the paper's own words; the storage schema says so and says mapping them onto a
vocabulary "is a later stage". This is that stage, and it **never edits a record**. A mapping
is an assertion *about* a record — "this paper's `bvFTD` is ONVOC's Dementia" — and writing it
into the field destroys the thing that makes it checkable, which is the paper's wording next
to it.

Every mapping carries:

| | |
|---|---|
| `text` | the field's value, verbatim |
| `head` | the part of it this row is about, after triage split a comorbidity list |
| `concept` | the vocabulary term, or `None` |
| `method` | which layer of the ladder produced it — `exact` … `overlap` |
| `rollup` | whether the term was reached by *dropping* part of what the value said |
| `corpus_term` | the finest grain the literature itself uses |
| `qualifiers` | course and state lifted off before lookup — `first-episode`, `remitted` |
| `sentinel` | `NO_CONDITION` when the value states an absence |

Two artefacts come out, and the second is as important as the first: `mappings`, one row per
routed value, matched or not; and `candidates`, the values the vocabulary could not carry,
grouped and counted **by distinct study**.

## Three defects, and what they cost

### 1. The highest-value route addressed a field that does not exist

`ROUTES` carried `groups.diagnosis` from the table's first version. Neither the extraction
schema nor the storage schema has ever had that slot — it is `medical_condition` — so the
route fired on zero values while the table read as though diagnoses were covered. **3,898
mentions across 1,589 distinct forms, the single densest clinical field in the corpus, were
never normalized at all.**

Nothing failed, because a route matching no field and a field whose values all miss look
identical from outside. `test_every_route_names_a_field_the_schema_has` now walks every route
against the storage schema from `Study`, so the next one is a red test.

### 2. `contains` produced almost all the coverage, unguarded

Per-field coverage before the fixes, and what layer produced it:

| field | mentions | distinct | matched | via `contains` |
|---|---|---|---|---|
| `groups.medical_condition` | 3,898 | 1,589 | 44.6% | 577 of 616 distinct hits |
| `regions.name` | 3,645 | 1,405 | 50.2% | 373 of 451 |
| `tasks.name` | 1,696 | 1,120 | 43.3% | 436 of 497 |
| `measures.source_label` | 2,649 | 1,034 | 21.7% | 140 of 151 |
| `tasks.conditions.name` | 3,415 | 1,649 | 12.8% | 175 of 247 |
| `groups.name` | 4,841 | 2,501 | 9.9% | 260 of 294 |
| `assessments.name` | 4,944 | 2,691 | 5.5% | 71 of 96 |

**85–95% of every field's coverage came from the least trustworthy layer in the ladder.**
`contains` asks only whether some vocabulary label appears as a substring, and the median
`medical_condition` value it fired on runs to six words. Three failure modes, all measured:

**No negation gate.** **22% of `medical_condition` values state an absence** — 452 as
`no` / `none` / `without` and 414 as a bare `healthy` — and they are overwhelmingly the
*control* cohorts, so the error is not scattered. It lands on exactly the groups a
patients-versus-controls query has to tell apart.

```
'No history of major depression or other mental illness'  -> Depressive Disorder
'absence of major depressive disorder'                    -> Depressive Disorder
'Healthy controls  No current medical, psychiatric, or …' -> Substance Use Disorders
'No induced neuropathic pain; sham-operated controls'     -> Pain
```

**Comorbidity lists mapped as one value.** The slot is multivalued in the schema and arrives
as one string anyway: 13% of values carry a separator, and 29% run to five words or more. The
ladder returned whichever comorbidity it reached first and the cohort's other diagnoses were
not in the output at all.

The splitter is deliberately conservative and recovers **4%** of values as more than one head,
not 13%. It splits on ` or `, `/` and `;` and never on `and`, because "attention deficit and
hyperactivity disorder" is one disorder. The rest of the long values are prose the extraction
should not have put in the slot, and they stay one head so the residual shows up as an
extraction defect rather than being quietly shredded into fragments.

**Longest-first picked the wrong condition.** `contains` took the longest label inside the
phrase, which in a comorbidity list is whichever comorbidity has the longer name:

```
'behavioural-variant frontotemporal dementia amyotrophic lateral sclerosis'
                                          -> Amyotrophic Lateral Sclerosis   (now Dementia)
'treatment-resistant depression bipolar disorder'
                                          -> Bipolar Disorder                (now Depressive Disorder)
'generalized social anxiety disorder generalized anxiety disorder …'
                                          -> Depressive Disorder             (now Anxiety)
```

English puts the head of a noun phrase last but puts the **primary** condition first. The
layer now sorts contained surfaces by position, then by length, so the earliest wins and
longest still breaks ties — `Selective Serotonin Reuptake Inhibitor` still beats the shorter
labels nested inside it.

### 3. The string decides a negation, not `is_healthy`

[normalization-pipelines.md](normalization-pipelines.md) recommended `Group.is_healthy` as the
primary gate with the regex as fallback. Measured on the full corpus, that is backwards.

The two agree on **93%** of the 3,624 mentions where both are known. But of the 237 mentions
where `is_healthy` is true and the string carries no negation, most name a real trait the
cohort was *selected for*: `obesity`, `cannabis use`, `postmenopause`, `nicotine dependence`,
`heavy drinking`. A cohort can be healthy in the study's sense and still be chosen for a
trait. Letting the flag overrule the string threw those away — including one cohort described
as `extracranial carcinoma/metastases under evaluation`.

So: **the string decides; `is_healthy` fills the 274 mentions whose string says nothing either
way**, which is the gap it was brought in to close. Only 8 mentions have the regex firing
against an `is_healthy` of false.

A third category shows up in the residue and is neither: `high risk for depression`,
`elevated genetic risk of schizophrenia`, `familial risk for major depressive disorder`,
`presymptomatic familial frontotemporal dementia risk`. 32 mentions. Risk is not diagnosis and
not absence, and nothing in the layer says so yet.

## What ONVOC actually is

Built from the pinned turtle release (`data/vocab/onvoc.ttl`, `owl:versionInfo 1.0.0`) rather
than a BioPortal class dump. The two carry the same 752 concepts with identical labels —
checked, not assumed — so nothing is gained on content; what the release adds is that it is
one versioned file rather than a snapshot of an API with no version on it.

**752 concepts, exactly three levels, and no synonyms.** 9 top concepts → 54 branches → 689
leaves, uniformly. **Zero `skos:altLabel`, zero `skos:definition`, in both the turtle and the
dump.** That single fact shapes everything else: ONVOC gives a label and a place in a tree,
so every surface form beyond the label has to come from the crosswalks it publishes (764
synonyms over MeSH, MONDO, DOID, SNOMED), from the paper's own abbreviations, or from
morphology.

The nine top concepts are the **facets**, and a mapping now carries the one it landed in.
`branch` says which list a term came from; `facet` says what *kind* of thing it is, which is
what a query filters on and what a wrong mapping shows up as:

| facet | concepts | |
|---|---|---|
| Drugs and Medications | 181 | by class: Antidepressants, Anti Psychotics, Opioids, Psychedelics |
| Disorders | 115 | Psychiatric 42, Medical 38, Neurological 31 |
| Brain Regions | 106 | a Desikan-Killiany parcellation |
| Population Characteristics | 100 | Age, Handedness, Weight, Ethnicities, Health |
| Psychological Concepts | 81 | **the process vocabulary** — Working Memory, Inhibitory Control |
| Symptoms | 77 | |
| Study Design | 64 | of which `Tests` is 53 |
| Behaviors | 19 | Substance Use, Substance Abuse, Sleep, Technology Use |
| Sensation and Perception | 9 | |

Two consequences worth stating, because both were got wrong before:

**ONVOC files substance terms under `Behaviors`, not `Disorders`.** `Alcohol Use`, `Smoking`,
`Tobacco Use`, `Substance Dependence`, `Alcohol Abuse`. A `medical_condition` route scoped to
the disorder branches therefore missed the two heaviest conditions in this corpus — `alcohol
dependence` at 148 mentions and `nicotine dependence` at 135. Which branch a term is filed
under is the ontology's judgement; which branches a *field* may draw from is the route table's,
and `medical_condition` asks about both. The disorders group now includes them.

**ONVOC does have a process vocabulary, and it is not a task vocabulary.**
`normalizing-with-onvoc.md` concluded "ONVOC has no task vocabulary", which is true of
*paradigms* and false of what a paradigm measures: `Emotion Regulation`, `Working Memory`,
`Inhibitory Control`, `Cognitive Load`, `Reward Responsiveness`, `Selective Attention` are all
there. That distinction is the whole task scheme below.

## The routing table

Which branches a field may draw from is part of the mapping, not a filter applied afterwards:
`Wechsler Abbreviated Scale of Intelligence` contains the word `Intelligence`, and an unrouted
match returns that concept confidently and wrongly.

| route | vocabulary | branch groups | gated |
|---|---|---|---|
| `design.arms.agent`, `design.arms.name` | ONVOC | drugs | |
| `groups.medical_condition` | ONVOC | disorders (incl. Substance Use/Abuse) | ✓ |
| `groups.name` | ONVOC | disorders + population | |
| `assessments.name` | ONVOC | tests | |
| `regions.name` | ONVOC | regions | |
| `tasks.name`, `tasks.conditions.name` | Cognitive Atlas | — | |
| `measures.source_label` | Cognitive Atlas | — | |

**Gated** means triage runs first: negation to a sentinel (22% of `medical_condition`
values), compound split into one row per head (4%), qualifiers lifted. Only `medical_condition` is gated, and getting that boundary right
took a measurement, because gating `groups.name` looks obviously correct and is not.

The two fields ask different questions. `medical_condition` asks *what condition*, so "none"
is an answer to it. `name` asks *who was this group*, so `healthy controls` is not an absence
— it is a study role, and one ONVOC has no term for. Gating it suppressed the best-supported
proposal in the corpus: **640 mentions of `healthy controls` / `healthy volunteers` /
`healthy participants` across 592 studies**, classified as absences and therefore never
proposed. And it bought nothing: of the 653 `groups.name` mentions the rule caught, 631 match
nothing either way, and all 22 that would have matched ungated matched *correctly*, to the
population branch — `healthy elderly` → Elderly, `healthy weight children` → Weight. Not one
disorder false positive.

The general form: **a gate belongs to a field, not to a vocabulary.** Whether "no X" is an
answer or an error depends on what the slot was asking.

`measures.source_label` is routed to the wrong vocabulary and is left that way for now: 21.7%
coverage, almost all via `contains`, and the residual is `BOLD signal` (77 studies), `BOLD
response`, `functional connectivity` (164), `gray matter volume` (126), `fractional
anisotropy`. Cognitive Atlas has no measure branch and neither does ONVOC — but the storage
schema already has `MeasureType` and `MeasureFamily` enums for exactly these. The route is
wrong, not the matching, and the fix is a closed-enum normalizer rather than a link.

## Two grains, and the threshold between them

ONVOC's grain is fixed and is not always the grain a query needs. It has `Dementia` and
nothing below it, so `behavioural variant frontotemporal dementia` — 148 studies — maps to
`Dementia` and loses the variant that the whole 448-record dementia corpus is *about*.

Every mapping therefore carries both: `concept` is the ONVOC term, `corpus_term` is what the
literature said, and `rollup` says whether the two differ by a dropped qualifier. Neither is
canonical. Which grain a meta-analysis needs is the meta-analysis's decision, and a layer that
picks one has made that decision for it silently.

`candidates()` then emits two kinds of proposal, and the distinction matters to whoever
receives them:

**A gap in coverage** — nothing matched. ONVOC has no term for this at any grain.

```
229 studies   'healthy controls (HCs)'   + 118 'Controls', 71 'HC', 53 'healthy control subjects'
170 studies   'neutral'                                       (conditions; wrong vocabulary)
164 studies   'functional connectivity (FC)'                  (measures; wrong vocabulary)
144 studies   'Structured Clinical Interview for DSM-IV (SCID)'
126 studies   'gray matter volume (GMV)'                      (measures; wrong vocabulary)
120 studies   'Fagerström Test for Nicotine Dependence (FTND)'
106 studies   'alcohol dependence'
102 studies   "bvFTD (Pick's)"
101 studies   'nicotine dependence'
 92 studies   'Beck Depression Inventory (BDI-II)'
 84 studies   'Mini International Neuropsychiatric Interview (MINI)'
 77 studies   'BOLD response' / 76 'BOLD signal'               (measures; wrong vocabulary)
 71 studies   'AD'
 68 studies   'ventral striatum (VS)'
 60 studies   'smokers'
 59 studies   'resting state'                                 (tasks; no vocabulary has it)
 52 studies   'Alcohol Use Disorders Identification Test (AUDIT)'
```

**A gap in grain** — it matched, by generalizing, and the finer term recurs across studies.
This is the kind that only exists because both grains are kept, and it is off unless a
threshold is passed, because one study's `remitted anorexia nervosa` is that study's wording
and forty studies' `bvFTD` is the literature's:

```
148 studies   'behavioral-variant frontotemporal dementia (bvFTD)'   [under Dementia]
104 studies   'major depressive disorder (MDD)'                      [under Depressive Disorder]
 71 studies   'frontotemporal dementia (frontal variant FTD)'        [under Dementia]
 41 studies   'semantic dementia'                                    [under Dementia]
 24 studies   'semantic variant primary progressive aphasia'         [under Aphasia]
 18 studies   'AUD'                                                  [under Alcohol Use]
 15 studies   'cigarette smoking'                                    [under Smoking]
```

Support is counted in **distinct studies, not mentions**: a paper naming its diagnosis once
per group is one piece of evidence that the term exists, not four. `candidates(rows,
minimum=5)` gives the coverage gaps; `candidates(rows, grain=8)` gives the grain gaps.

One class of "grain gap" is not a subtype and should not be proposed as a term: **laterality**.
`left amygdala` (116 studies), `right amygdala` (103), `bilateral amygdala` (52) all roll up
to `Amygdala`, and the residue is a side rather than a finer region. Laterality is its own
facet and belongs in a field, not in the vocabulary.

Related, and unfixed: ONVOC's region branch is a Desikan-Killiany parcellation, so it has
`Rostral anterior cingulate cortex` and `Caudal anterior cingulate cortex` and no plain
`anterior cingulate cortex`. The literature's `anterior cingulate cortex` is genuinely
ambiguous between the two and currently resolves to `Caudal` by containment. That is a
mapping the layer should refuse rather than resolve, and it does not yet.

## The scheme for tasks and conditions

The question this section answers: what should a task and a condition be normalized *to*, such
that a meta-analysis query can be written against them?

### A task name is not one term, it is a composition

The corpus says so plainly. Read the task names for one paradigm:

```
'alcohol Go/NoGo task'          conds: alcoholic beverage no-go, non-alcoholic beverage go,
                                       geometric no-go, geometric go
'food-specific go/no-go task'   conds: Go, Nogo, High-calorie, Low-calorie
'go/no-go task'                 conds: go, no-go
'cocaine-word Stroop task'      conds: cocaine words, neutral words
'Stroop task'                   conds: congruent, incongruent, cannabis-related words, neutral words
'letter n-back task'            conds: 0-back, 2-back
'emotional faces n-back task'   conds: 0-back, 2-back, happy faces, fearful faces, neutral faces
```

Every name is a **paradigm** — the response logic, `go/no-go`, `Stroop`, `n-back` — plus a
**content**: what the stimuli were about, `alcohol`, `food`, `cocaine`, `letters`, `faces`. And
the conditions carry the *same two dimensions*: a paradigm level (`go` vs `no-go`, `0-back` vs
`2-back`, `congruent` vs `incongruent`) crossed with the same content.

This is why flat normalization of either one fails, and the earlier measurement shows exactly
how badly. Over 1,696 task mentions: 1,120 raw forms, 966 token bags, **17% of bags appear in
two or more papers, covering 51% of mentions.** The top 100 bags reach 44%. Conditions are
worse: 3,415 mentions, 1,547 bags, top 100 reach 43%, and the single most common condition in
the corpus is `neutral` (183 mentions) — a word that means nothing without the thing it was
contrasted against.

Splitting on the facet is what recurs, because the paradigm recurs even when the whole name
does not: `cue reactivity` in 106 papers, `resting state` in 172, `emotion regulation` in 149,
`Stroop` in 26, `monetary incentive delay` in 23.

### Three vocabularies, and each covers a different facet

Probed directly against ONVOC and Cognitive Atlas:

| | Cognitive Atlas task | ONVOC |
|---|---|---|
| `n-back task` | n-back task | — |
| `go/no-go task` | go/no-go task | — |
| `stop signal task` | stop signal task | — |
| `flanker task` | Eriksen flanker task | — |
| `monetary incentive delay` | monetary incentive delay task | — |
| `oddball task`, `Wisconsin card sorting test` | both | WCST only |
| **`cue reactivity`** | — | — |
| **`cue exposure`** | — | — |
| **`resting state`** | — | — |
| **`picture viewing`**, **`fear conditioning`** | — | — |
| `emotion regulation` | — | Emotion Regulation |
| `working memory` | — | Working Memory |

Cognitive Atlas covers the **classically named paradigms** — the ones with a proper name.
ONVOC covers the **process** a paradigm targets. Neither covers the **descriptive paradigms
that dominate this corpus**: cue reactivity, cue exposure, resting state, picture viewing,
reward anticipation. Those are the heads of the distribution and they are exactly the list to
propose.

So the scheme is three slots, filled from three sources:

| facet | source | example |
|---|---|---|
| **paradigm** | Cognitive Atlas task; else corpus cluster | `go/no-go task`, `cue reactivity` |
| **content** | ONVOC Drugs + Behaviors; else corpus cluster | `Smoking`, `Cannabis`, faces, food |
| **process** | ONVOC Psychological Concepts / CogAtlas concept | `Inhibitory Control`, `Working Memory` |

### Measured, including where it does not reach

Every task name in the corpus was decomposed by taking the longest token span each vocabulary
can claim at `exact` or `synonym` — no `contains`, because the whole point is to not repeat
defect 2. Terms are counted as recurring when two or more *studies* use them:

| facet | mentions filled | distinct terms | recur in ≥2 studies |
|---|---|---|---|
| the whole name | 1,696 / 1,696 | 1,074 | 156 (**15%**) |
| **paradigm** | 493 (29%) | **73** | 38 (**52%**) |
| **process** | 333 (20%) | 95 | 48 (51%) |
| **content** | 87 (5%) | 3 | 3 |
| residue | 1,412 (83%) | 775 | 141 (18%) |

**Where a vocabulary reaches, the facet is 15× fewer terms than the name and 3.5× more
recurrent.** 73 paradigm terms against 1,074 name forms, and half of them are shared between
studies where six names in seven are not. That is the result the scheme is for, and it is the
tractable list the flat vocabularies never produced.

**The list itself is [task-terms.md](task-terms.md)** — all 73 paradigms, 95 processes and
3 contents with their study support and the names each folded, plus the 647 unnamed residues.
The script that produced it was folded into `pondie/normalization/task.py`; the measurement
stands.

**It reaches under a third of mentions.** 83% of names have residue left over, and 775
distinct residues is not a vocabulary. So the facets are a *naming layer*, not a replacement
for clustering: the corpus is still clustered against itself for whatever the Atlas does not
name, and the unseeded whole-corpus clustering reached 167 identities / ~130 families at
ARI 0.619.

> **That unseeded route no longer exists.** It was a second `normalize` in a second module,
> and because it was the one exposing the field contract it was the one `pondie normalize
> task` ran, while the seeded route was reachable only from a script. `task.py` is the
> seeded route now: Atlas first, clustering for the residual. The ARI figure above was
> measured on the deleted route and is kept because it is what the comparison in this
> section rests on; it is not a measurement of what runs today. What the facets add is
that a cluster gets a term a query can be written against, instead of being named by whichever
of its members is most frequent.

The content row is the sharpest ONVOC finding in this file. The corpus is saturated with
stimulus content — `food` appears in 124 papers' task names, `alcohol` in 68, `smoking` in 61 —
and **ONVOC can name three of them**: Smoking (61 studies), Cannabis (11), Heroin (9). There
is no `Alcohol`, no `Cocaine`, no `Nicotine`, no `Food` anywhere in it. The content facet is
the most query-relevant of the three, because it is the one that joins a task to a group to a
condition, and it is the one with almost no vocabulary behind it.

### Conditions are normalized *within* a paradigm, not across the corpus

"Identify the task, then the conditions fall out" is half right, and the measurement says
which half. For task name bags used by three or more papers, only **20% of condition bags
recur inside their own task bag** (159 of 792). Grouping conditions by task *name* alone does
not make them converge: the naming variance is inside the paradigm too.

What does converge is the facet. `go`/`Go`/`Go trials`/`respond alcohol` are all the go level;
`0-back`/`0B`/`find X` are all the low-load level. Those are **paradigm levels** — a small
closed set *per paradigm*, defined by the paradigm rather than by the corpus — and they are
what a contrast is written against. `2-back > 0-back` is a query; `2-back > find X` is the
same query in one paper's words.

So: identify the paradigm, then read the condition as `(paradigm level, content)`. The paradigm
supplies the level vocabulary, which is why the order is task first.

The content slot is where the payoff is, because it is the *same slot* `medical_condition`
fills. A query for "alcohol studies" would reach `alcohol dependence` in a group, `alcohol cue
reactivity` in a task and `alcohol` in a condition through one concept — one term joining
three entities that no current query can join. That is also the measurement's sharpest
negative: ONVOC can name three of this corpus's stimulus contents, so the join exists in the
design and not yet in the vocabulary. It is the first thing to fix and the cheapest — four or
five substance terms.

### Conditions do help decide whether two tasks are the same, with one caveat

They already do — `task.py` carries a condition-set overlap channel, soft-matched so `win`
stays beside `gain` without a long condition list drowning a short one. The caveat the corpus
shows: **conditions do not always name the paradigm level.** `emotional faces n-back` has
conditions `happy`, `fearful`, `neutral` and no `0-back`/`2-back` anywhere, so a paper can
describe an n-back entirely in content terms. The name and the conditions are complementary
channels, not redundant ones, which is why holding the name channel out is the honest ablation
and why neither alone should decide.

### `condition_kind` is a within-paper judgement, not a corpus-stable term

[task-condition-normalization.md](task-condition-normalization.md) recommends leaning on
`Condition.condition_kind` for the target/baseline split. It is 92% filled on this corpus
(`task_state` 2,112, `control_state` 871, `fixation` 92, `rest` 55, 284 unset) and the split it
gives *within* an analysis is sound.

Across papers it is not a term. Of 206 condition names appearing three or more times with a
kind, **49 (24%) get different kinds in different papers, covering 37% of the mentions**:

| condition | kinds assigned |
|---|---|
| `neutral` | control_state 145, task_state 17 |
| `look neutral` | task_state 6, **control_state 8** |
| `look` | control_state 7, task_state 8 |
| `attend` | control_state 9, task_state 5 |
| `maintain` | task_state 27, control_state 4 |
| `congruent` | task_state 14, control_state 2 |
| `baseline` | control_state 6, fixation 4, rest 2 |

`look neutral` is a coin flip. That is not an extraction defect: whether a condition is a
control *is* relative to what the paper contrasted it with, which is what `ConditionKind`'s own
description says. The conclusion is narrower than the earlier doc's — use it to pick sides
*inside* one analysis, and do not treat it as a normalized value that means the same thing in
two papers.

### The modality fields are empty, and that is the old schema

`Task.stimulus_modality` and `Task.response_modality` are in the storage schema now.
`response_modality` is populated on 1,577 of 1,696 tasks (button_press 840, none 548, speech
42, hand_movement 46). **`stimulus_modality` is populated on 22.** The records were extracted
before the slot existed.

This matters for the grain question and cuts the way the scheme wants. A particular n-back's
stimulus and response modalities are **attributes of the task instance, not part of its
identity** — `letter n-back` and `emotional faces n-back` are the same paradigm with different
content and different stimulus modality, and the whole point of the paradigm slot is that they
answer one query. So the modalities slot in beside the facets as filters, and no re-clustering
is needed when a re-extraction fills them. Nothing in the scheme has to change; the field just
starts carrying values.

## What to propose to ONVOC

Ranked by studies unlocked. Terms first:

- **`Healthy Control`**, or a study-role branch. The top proposal in the corpus and by some
  distance: **229 studies** write `healthy controls`, 118 plain `Controls`, 71 `HC`, 53
  `healthy control subjects`, plus `healthy volunteers` and `healthy participants` — around
  590 studies in all once the phrasings are pooled. Nothing else recovers them:
  `Population Groups` is ethnicities, and `Typical Health` is a health state rather than a
  study role.
- **A clinical rating scale branch**, distinct from `Tests`. SCID 144, FTND 120, BDI 92, MINI
  84, AUDIT 52, HAMD 46, plus PANSS, SANS, SAPS, BPRS, MADRS, YMRS, Edinburgh Handedness, NART.
  PANSS is not a Wechsler scale, and "studies that measured negative symptoms" needs SANS and
  PANSS to be distinct terms.
- **Substance use disorders by substance.** ONVOC has `Tobacco Use Disorder` and `Cannabis Use
  Disorder` and stops. `alcohol dependence` 106 studies, `nicotine dependence` 101, plus
  cocaine, heroin, opioid, methamphetamine.
- **The substances themselves.** Separately from the disorders, and this is the gap that costs
  most across fields: there is no `Alcohol`, `Cocaine`, `Nicotine` or `Food` anywhere in the
  drug or behaviour branches, so the content facet of a task — the thing that joins `alcohol
  dependence` in a group to `alcohol cue reactivity` in a task to `alcohol` in a condition —
  can be named for 3 of the corpus's stimulus contents and not the rest.
- **The FTD spectrum under `Dementia`.** bvFTD 148 studies, FTD 71, semantic dementia 41, svPPA
  24, PPA, FTLD.

Branches ONVOC does not have and arguably should not:

- **Descriptive task paradigms** — cue reactivity (106 studies), cue exposure (41), resting
  state (172), picture viewing, fear conditioning, reward anticipation. Cognitive Atlas is the
  right home and does not have them either; this is the list to send there.
- **Measures and modalities** — `functional connectivity` 164 studies, `gray matter volume`
  126, `BOLD signal` 76. The storage schema's `MeasureType` / `MeasureFamily` enums already
  cover these; the route is wrong rather than the vocabulary.

## Where it is wrong

Hand-judged on a random sample of 22 distinct (value → concept) claims per route, drawn from
the full corpus. This is an error *profile*, not a precision estimate — 22 is too few for the
second, and the point of the sample is that the errors cluster into namable classes rather
than scattering.

| route | wrong in sample | what they are |
|---|---|---|
| `groups.medical_condition` | 3–4 / 22 | risk phrasings; a comorbidity list whose primary ONVOC lacks |
| `regions.name` | 3 / 22 | **all** the ACC parcellation problem |
| `assessments.name` | 3 / 22 | a generic phrase matching a named battery |
| `tasks.conditions.name` | 3–4 / 22 | a Cognitive Atlas *concept* standing in for a condition |
| `groups.name` | 1–2 / 22 | a comorbidity or exposure promoted over the cohort's own trait |

Four classes, and three are already named elsewhere in this file. The new ones:

**Earliest-first is right only when the primary is in the vocabulary.** `methamphetamine
dependence marijuana dependence panic disorder` → `Panic Disorder`, because ONVOC has no term
for either of the first two and the earliest *matching* label is the third. Position-in-value
is known at match time and is not recorded; a match at word 6 of a comorbidity list is a
weaker claim than one at word 0 and should say so.

**Risk is read as diagnosis.** `Familial risk for bipolar disorder` → Bipolar Disorder,
`hypomanic personality traits, higher risk to develop…` → Bipolar Disorder. 32 mentions
measured. A cohort selected for *risk* is not a patient cohort, and pooling them is the kind
of error a meta-analysis cannot see downstream.

**`tasks.name` conflates paradigm with process.** The route matches against the whole
Cognitive Atlas — tasks, concepts and disorders together — so `food perception task` →
`perception` and `worry modulation task` → `internalizing` are returned in the same shape as
`Go-No-Go task` → `go/no-go task`. That is why the route reports 43% coverage while only 29%
of mentions get a *paradigm*: the difference is concept matches that look like task matches.
`pondie/normalization/task.py` separates them, and it is the production route now -- when
this was written the separating code was reachable only from a script.

## What is not done

- **Generality is untested.** The ladder, the routing and the two-grain artifact are
  vocabulary-agnostic and would run against any `Vocabulary`. The lexicons are not: 33
  psychiatric course qualifiers, 9 study-role nouns, 29 neuroanatomical modifiers, 42 named
  ONVOC branches. Every one was written against this corpus, and `cohort_role()` is the
  standing warning — a hand-written regex that looked fine at 13% failure on schizophrenia
  and turned out to fail on 57% of MID and 60% of depression group names. Nothing here has
  been run on a literature outside neuroimaging.
- `measures.source_label` still routes to Cognitive Atlas. It wants the schema enum.
- `regions.name` resolves `anterior cingulate cortex` to the caudal parcel by containment,
  where it should refuse. Laterality should come off into its own slot rather than showing up
  as a grain gap.
- Risk and familial-risk phrasings (`high risk for depression`, 32 mentions) are read as
  diagnoses. They are neither diagnosis nor absence and need a third sentinel.
- The paradigm/content/process facets are specified here and measured, and `task.py` still
  emits a single clustered identity. Wiring the facets into it needs a GPU run on beast to
  re-measure the clustering against them.
