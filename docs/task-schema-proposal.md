# Two slots `Task` is missing, and what they would fix

Implemented and tested; results at the bottom. The evidence for the change is [task-clustering-method.md](task-clustering-method.md)
§2.2–2.3 and the categories in [task-categories.md](task-categories.md); the test split is
`data/task-facets/schema-test-split.json` and was drawn before any of this was written.

## The problem, stated as a measurement

A task name is a **paradigm** crossed with a **stimulus**, and `Task` has a slot for neither.
Over the 1,696 task mentions in the corpus:

- 49% of names carry a paradigm term, 42% carry a stimulus term, **14% carry both explicitly**
  (26% counting a bare presentation frame like `visual food cues` as naming the paradigm).
- The cross is productive: **13 of 25 paradigms appear with two or more stimuli, and 11 of 13
  stimuli appear with two or more paradigms**, over 60 combinations actually used. `cue
  exposure` crosses eight substances; `Stroop` crosses seven.

Both are query axes, and they are independent ones. "All cue reactivity regardless of
substance" and "all alcohol tasks regardless of paradigm" are both legitimate questions, and
neither is answerable today.

## Where the information currently goes, and why that is wrong

| slot | holds | problem |
|---|---|---|
| `name` | the source's own words | both dimensions, fused, in the paper's phrasing |
| `stimuli` | free text: "photographs of alcoholic beverages and matched neutral images" | the content is in there as prose, not as a filter |
| `stimulus_modality` | `visual`, `auditory` | the *channel*, not the *content*. Populated on 22 of 1,696 |
| `design_type` | `block`, `event-related` | apparatus, and correct |

So the stimulus content exists only inside free text, and the paradigm exists only inside the
name. Downstream this is not a cosmetic loss:

- `Task.setting`, the clusterer's channel, is `stimuli + design_type + response_modality`
  concatenated — the *content* sat inside the channel meant to measure how the task ran.
  Splitting it measurably reduced paradigm fragmentation (cue reactivity 12 clusters → 8).
- Because the two dimensions share one distance, the clustered category
  `smoking cue-reactivity task` holds members whose cues are cannabis and heroin. The
  paradigm is right and the label's content claim is wrong, and no query can tell.
- Recovering the split post-hoc needs a ~50-term stimulus lexicon and a curated
  same-paradigm table, both of which are guesses about what the paper meant. The paper knows.

## The change

Two slots on `Task` in `neuroimaging-study-storage/task.yaml`; the extraction schema is
generated from it. `paradigm` is single-valued, `stimulus_content` multivalued -- a design
contrasts two or more contents and the contrast is what the analysis tests.

The descriptions give the **rule** and then examples spanning domains, deliberately not this
corpus's vocabulary. Written against a corpus that is addiction and emotion regulation by
construction, a description illustrated with alcohol, food and cue reactivity would read as
correct here and mislead on a memory, language or motor study. So `paradigm`'s examples run
n-back, Sternberg, delayed match-to-sample, go/no-go, stop-signal, Stroop, flanker, oddball,
mental rotation, verbal fluency, semantic decision, finger tapping, monetary incentive delay,
fear conditioning, cue reactivity, resting state; `stimulus_content`'s run faces, houses,
scenes, tools, body parts, words, pseudowords, sentences, digits, letters, shapes, tones,
music, odours, food, alcohol, cigarettes, money, pain, autobiographical memories, social
feedback. Both say explicitly that the list is not a vocabulary to choose from.

Both also say when to leave the slot **empty**, which is the part that makes them safe: a
paradigm the study invented, and stimuli that are not about anything -- resting state, finger
tapping, a tone with no content. The develop run confirms the abstention works.

The full wording is in the schema file rather than duplicated here, so there is one copy.

`paradigm` deliberately does **not** take an enum. The corpus's three most common paradigms —
`resting state` (174 studies), `cue reactivity` (155), `cue exposure` — are in neither the
Cognitive Atlas nor ONVOC, so an enum would be wrong on the head of the distribution. Free
text normalised afterwards against the Atlas seed list
([cognitive-atlas-seeds.md](cognitive-atlas-seeds.md)) is the shape that fits what exists.

## What would count as the change helping

Fill both slots on the eight **develop** articles, then compare against the current pipeline's
answer for those tasks:

1. **Agreement on paradigm.** Does the model's `paradigm` match the category the pipeline
   assigns? Where they differ, which is right on reading the Methods?
2. **Recovery where the name is silent.** 58% of names carry no stimulus term and 51% no
   paradigm term. The interesting cases are the ones the string cannot reach —
   `attentional bias paradigm` (tobacco), `fast event-related fMRI task` (cocaine). If the
   model fills these from the Methods, the slot earns its place; if it only restates the
   name, it does not.
3. **Cost to `name`.** The slots must not tempt the extractor to normalise `name` itself.
   `name` stays the source's own words — that is what makes every mapping checkable.

Then, and only once the wording is settled, the same on the eight **holdout** articles.

## Tested on the develop set

Implemented in `study_schema/neuroimaging-study-storage/task.yaml` and regenerated. Run on
`beast` with `@psyc-aid338-ope-333f18/gpt-5.6-luna`, stages `tables prose split demands
satisfy fill build`, no evidence pass. **Eight develop papers, nine tasks, zero failures.**
Two control runs of the *unmodified* schema on the same papers give the noise floor, because
without one no difference can be attributed to the change -- the lesson
`normalization-pipelines.md` records from the `population_characteristics` rounds.

**Both slots fill: 8 of 9 tasks each.** The single abstention is correct and is the
behaviour the description asks for: `letter repetition task during PET uptake` is a bespoke
procedure instantiating no established paradigm, and `paradigm` was left empty.

### 1. Agreement on paradigm -- the slot as first written was wrong

**What the two columns are, because they are not two methods at the same task.**
*pipeline* is the category `pondie/normalization/task.py` assigns from the OLD record's
`tasks[].name` plus prose embeddings -- no model, no paper. *slot* is the new `paradigm`
field, filled by a model reading the full text. One has the article, the other has a string.

The first draft of this section said the slot "wins every time" against the pipeline. It does
not, and the interesting failure is on the two cases I had called its best work.

`PoiCaeLDyEjZ` reads: *"two fMRI behavioral tasks (an emotional reactivity task (**Hariri et
al., 2002a**) and a gambling task (**Delgado et al., 2000**))"*. The slot returned
`Hariri face-matching emotion task` and `Delgado card-guessing gambling task`. Checked
against the text, every ingredient is somewhere in the article -- `face-matching` names a
block ("five shape-matching blocks interleaved with four face-matching blocks"),
`card-guessing game` describes the procedure, the authors are in the citations -- and
**neither label appears anywhere in the paper**. The paper calls them "an emotional
reactivity task" and "a gambling task".

So the model **composed** a paradigm name out of the paper's descriptive vocabulary and a
citation's author. That is a worse failure than being wrong, because:

- it cannot be checked against a span, which is the contract every other value in the record
  is held to;
- two papers running one paradigm will compose different strings, so the slot is **no more
  joinable than the name it was meant to improve** -- and it now reads authoritative.

The slot's own description invited it: "Use the name the field uses for the procedure ... not
this study's name for its own version" is an instruction to depart from the source. It has
been rewritten to require the source's own words and to leave the slot empty otherwise.

Scored on grounding rather than on plausibility:

| paper | slot | in the paper? | verdict |
|---|---|---|---|
| o8WEzV9Uy5GB | dot probe task | "attentional bias **dot probe task**" | grounded |
| 8Nkn4bHtgHmz | oddball target detection | "An **oddball target detection task** was employed" | grounded |
| SdvwpgEnZm4u | go/no-go task | task is named that | grounded |
| 4UoCgF3UJSXq | emotional picture encoding task | "**encoding** of ... IAPS pictures", "**affect-laden pictures**" | partly -- composed, but from adjacent words |
| MBPUwjmmCaL7 | gender discrimination of emotional faces | "identify the **gender** of each face"; task named "faces emotion recognition test" | describes the procedure, is not the name |
| PoiCaeLDyEjZ | Hariri face-matching emotion task | **no** | composed |
| PoiCaeLDyEjZ | Delgado card-guessing gambling task | **no** | composed |

Three of seven grounded, two composed outright, two in between.

### 1b. What the citations are actually good for

The right use of `(Hariri et al., 2002a)` is not to let the model translate it into a name
from memory. It is a **reference**, and a reference identifier is a stable join key across
papers in exactly the way a generated string is not: two papers citing Hariri 2002 for their
task are running the same paradigm, and that is checkable against the reference list.

That makes it a downstream text-processing stage, not a slot: extract the task with its
evidence span, pull the citations inside or adjacent to that span, resolve them against the
bibliography. Measured on the develop papers, **5 of 9 tasks have a citation within 300
characters of a task mention** -- so it would reach about half. The other half name the
paradigm in their own words (`dot probe task`, `oddball target detection task`,
`Go-NoGo response inhibition task`), which is where the vocabulary matching in
[cognitive-atlas-seeds.md](cognitive-atlas-seeds.md) already works. The two halves are
complementary rather than competing.

Note this run used `--no-evidence`, so the spans that would carry those citations were not
produced. Testing the citation route needs a run with the evidence pass on.

### 2. Recovery where the name is silent -- yes, and contrast-aware

`stimulus_content` does not restate the name. The pipeline could only reach a single token
from the string; the slot comes back with what the design actually contrasted:

```
gambling task            pipeline: emotion        -> numbers; monetary outcomes
attentional bias task    pipeline: emotion        -> angry faces; happy faces; neutral faces
target detection task    pipeline: emotion        -> sad scenes and faces; neutral images
emotional faces task     pipeline: emotion        -> negative emotional faces; scrambled pictures
Go-NoGo task             pipeline: (unspecified)  -> (empty -- correctly)
```

The gambling case is the clearest: the string pipeline had `emotion` because the paper is an
emotion study, and the task's stimuli are numbers and money.

### 3. Cost to `name` -- not measurable at this sample size

| comparison | names identical | differ |
|---|---|---|
| control vs control-2 (same schema, two runs) | 7 | **2** |
| with-slots vs control | 5 | **4** |

Four against a noise floor of two, on nine tasks, is not a detectable effect -- and
`o8WEzV9Uy5GB` flips in *both* directions between the two controls, so at least one of the
four is the same noise. The honest statement is that no disturbance was detected, not that
none exists; nine tasks cannot rule out a small one.

### Cost

982,712 + 1,395,106 + 870,920 + 914,483 input tokens across the four runs (treatment, the
sealed check set, and two controls), 438,571 output.

## Status and what to do next

`stimulus_content` **earns its place** and is unchanged: grounded, contrast-aware, and it
recovered content no string method reached (`gambling task` -> `numbers; monetary outcomes`
where the pipeline had `emotion`).

`paradigm` **has been rewritten** after the develop run showed it composing names. It now
requires the source's own words and an empty slot otherwise. **That change is untested** --
the run above measures the first wording, not this one.

The holdout is run and sealed: `data/runs/schema-chk` on beast holds the 7 holdout and 6
generalisation papers under the FIRST wording. Looking at them now would answer a question
about a slot that no longer exists as written, so they stay shut.

Next, in order:

1. Re-run develop under the rewritten `paradigm`, with the **evidence pass on** -- this run
   used `--no-evidence` and the spans are what the citation route needs.
2. Check that composition has stopped: no slot value should be absent from its paper's text.
3. Only then open the holdout and the generalisation set, and re-run them if the wording
   moved again.
4. Separately, prototype the citation stage: task evidence span -> citations within it ->
   reference list -> identifier. It is not a schema change and can be built against the
   records that already exist.
