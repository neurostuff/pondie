# Why `pondie.normalization` is shaped the way it is

Every decision in the package, with the measurement that forced it. The code states the
rules; this file states the reasons. A docstring here answers *what* a function is and
what it promises; when you want to know *why* it is that way, the answer is in this file
under the module's own heading.

Companions: [normalization-pipelines.md](normalization-pipelines.md) is the measured claim
that a field's shape decides its method; [normalization-layer.md](normalization-layer.md)
is the contract for the layer as a whole; [condition-normalization.md](condition-normalization.md)
and [task-clustering-method.md](task-clustering-method.md) carry the two hardest fields in
full. Where any of those disagree with this file, they are the more specific document and
win.

## The package

One module per field. A field's shape decides its method, and four shapes recur.

| shape | the field has | mechanism | fields |
|---|---|---|---|
| closed target | a small fixed answer set, free-text input | `_lexicon` | `coordinate_space`, `multiple_comparison_method`, `correction_scope`, `medication_status`, `sex_distribution`, `handedness_distribution`, `modality`, `prespecification` |
| link | an external vocabulary | `pondie.vocabularies`, `_embedding` | `medical_condition` |
| seed + cluster | a target covering part of it; the corpus covers the rest | `atlas`, `_embedding` | `task` |
| partition | one field holding two kinds of value | rules, in the module | `population_characteristics` |

Two conventions every field shares.

**UNKNOWN is not OTHER.** `OTHER` asserts an answer outside the known set; `UNKNOWN`
asserts we cannot tell. They license different downstream actions — a transform must
refuse `OTHER` and may fall back on a default for `UNKNOWN` — so collapsing them loses the
distinction that matters.

**Nothing is bucketed silently.** An input no rule matched is `UNKNOWN` with
`reason="unmatched"` and is reported, so a new surface form forces a rule instead of
disappearing into `OTHER`.

### What is and is not a field module

`fields()` derives the list rather than keeping one, because a hand-kept list drifts: five
of the eight closed-target modules bound `normalize` and not `report`, and `pondie
normalize coordinate_space` raised on a field its own help text offered as an example. A
field module is one that exposes `normalize`; `tests/test_normalization_contract.py`
enforces that it also exposes `report`.

Deliberately outside that list:

- `corpus` maps a whole corpus rather than one field, and has its own CLI.
- `contrasts` answers one question across many records — which contrast is treatment
  against control — and is read by `query.engine` and `corpus`.
- `atlas` is a vocabulary normaliser, not a field: it is the target half of `task`.
- `is_healthy` **fills** a slot rather than normalizing one, from fields
  it does not touch, so it has no `normalize` to expose, only `apply`.
  `population_characteristics` does both, which is why `apply` sits beside `normalize`
  there and in no other field module.

Modules with a leading underscore are shared machinery, not interface. Term lists both
packages fetch and share live in `pondie.vocabularies`, not here.

### Two closed targets whose answer set is the schema's own

`modality` and `prespecification` map onto the *schema's* permissible values rather than a
downstream set like MNI or RIGHT. Both slots were required closed enums until one
unrecognised word was found to discard the whole entity — `create` answered "Acquisition
would be missing modality" and everything else the proposal carried went with it. They are
open ranges now and these two put the wording back on the vocabulary.

For `modality` that is load-bearing rather than tidy: only a vocabulary value carries an
`instantiates`, so an unmapped one leaves `Acquisition.acquisition_type` underivable and
the record resolving to the base class.

## `_lexicon` — the closed-target mechanism

Rules instead of an encoder, for **auditability rather than accuracy**. Sixteen surface
forms over two coordinate spaces, or 293 over four correction methods, is a case where a
rule can be read and argued with and a cosine cannot — and where a wrong answer is acted
on, not merely displayed.

`Rule.decisive` exists for negation. "not medicated" contains "medicated", so the two
compete and the negation has to win rather than register as an ambiguity. Several distinct
non-decisive matches is an ambiguity, not a choice, so it answers `ambiguous_to`.

`scan` and `field_report` are free functions rather than only methods on `ClosedField`
because `medication_status` is a closed target that is *not* rule-based — it reads a
dependency parse — and it had hand-copied both. One definition of "read a field's strings
and summarise the decisions" now serves both.

## `coordinate_space`

`Analysis.coordinate_space` → MNI, TAL, OTHER or UNKNOWN. The record keeps the source's own
words for the same reason `Measure.source_label` does; this maps them onto the four values a
query and a coordinate transform need.

More than a spelling exercise, because this field decides whether coordinates are moved: a
wrong answer displaces foci by 5–10 mm. So `OTHER` (a third space, refuse to transform) and
`UNKNOWN` (no information, a caller may default) must not be collapsed.

**Resolution precedence.** The schema's own: the analysis's field beats a table's, and both
beat the spaces stage 1 read off the coordinates. That last fallback is not decoration — it
answers **11% of analyses**, where the model left the field blank. `Table.coordinate_space`
sits in the middle and is empty in every table measured, so the middle step never fires on
this corpus; it is kept for the schema's sake rather than its yield.

Point spaces are normalised before being compared, because stage 1 writes "MNI" for one
sentence and "MNI152" for the next, and a set of raw tokens reads two spellings of one space
as a conflict.

**Why the patterns are shaped that way.** No trailing boundary on the space names: "MNI152",
"ICBM152" and "fsaverage6" are each one token, and `\bmni\b` misses every one — `\bfsaverage\b`
did, on this corpus. Each name is spelled out as well as abbreviated because a paper writes
either. ICBM and Colin27 reach MNI rather than OTHER on substantive grounds: the MNI152
template *is* the ICBM-152 average and Colin27 is the MNI single-subject brain, so coordinates
read off them are already MNI and need no transform. `NMI` and `Talaraich` are transpositions
measured in the corpus, not hypothetical.

The `OTHER` rule's second half names templates a study built for itself — DARTEL, SUIT,
FMRIB58, an SPM release's own, anything "*-specific". Those are third spaces and a transform
must refuse them. It deliberately excludes the bare word "template": "template image space"
names no template, and not knowing which space a paper used is UNKNOWN, not a third space.

## `modality`

`Acquisition.modality` → the Modality value the wording names. Load-bearing rather than
cosmetic: only a vocabulary value carries an `instantiates`, so `Acquisition.acquisition_type`
cannot be derived from anything else, and without the designator the record resolves to the
base class where every modality-specific parameter is undeclared.
`rules.check_acquisition_subclass` reports each one that lands there.

No `OTHER`: the vocabulary already has `other`, and a second value meaning the same thing
would be two spellings of one answer.

**Rule order.** The three qualified MRI values are tested before the bare one, and `MRI`
carries a lookbehind for each qualifier. Without it "functional MRI" matches both `fMRI` and
`MRI`, which `classify` reads as an ambiguity and answers UNKNOWN — the commonest wording in
the corpus would have been the one value this cannot resolve. The acronyms need no lookbehind:
"fMRI" is one token, so `\bmri\b` does not reach the MRI inside it. The `other` rule matches
only the bare word, because reading "other imaging" as the vocabulary's `other` asserts the
paper placed itself outside the named modalities.

Seeded from the vocabulary's own synonyms rather than measured drift: at the time of writing
no committed record holds an off-vocabulary modality, because the field was closed and
`values.cast` refused them before they could be counted. `report()` is how the real surface
forms arrive.

## `prespecification`

`Analysis.prespecification` → whether the contrast was planned before the data were seen.
The distinction a reader needs: an exploratory contrast searched a space the paper does not
report, and pooling it with a planned one treats the two as equal evidence.

No `OTHER`. The field asks a yes-or-no question about *when* the contrast was decided, so a
third answer would not be a third kind of prespecification — it would be a statement that the
wording does not say, which is `UNKNOWN`.

The negation rule is decisive and first: "not pre-registered" contains "pre-registered". Same
case as `medication_status`. Seeded from the vocabulary's synonyms, same reason as `modality`.

## `multiple_comparison_method`

293 surface forms over 921 values, for four answers. "Corrected results only" is a standard
meta-analysis inclusion criterion, and it cannot be applied against `FWE`,
`family-wise error (FWE)`, `Family-Wise Error` and `FWE correction` as four values.

`UNCORRECTED` is not `UNKNOWN`: a paper stating it did not correct has told us something, and
a paper silent on the matter has not. Only the first is safely excludable.

**Rule order.** `permutation` is tested before `FWE` because a permutation-derived family-wise
threshold is usually written as both, and the resampling is the specific claim. Cluster-level
thresholding is `OTHER`, not a fifth value: it names a unit, saying where the correction was
applied and not which error rate it controls.

## `correction_scope`

The distinction a meta-analysis needs is whole-brain against a restricted volume: a
small-volume-corrected result survived a much lower bar than a whole-brain one, and pooling
them treats the two as equal evidence. 217 surface forms over 760 values.

`RESTRICTED` is tested first: "whole brain and a priori ROIs" restricts somewhere, and the
restricted half is the one that changes how the result should be weighed. `cluster level` is
not an answer to this question — it names the unit a threshold applied to, not the volume
searched — so it is `OTHER`; `searchlight` is the same kind of answer.

Every separator class admits an underscore as well as a space and a hyphen, because the value
this field holds most often is the schema's own `whole_brain`. Over 1,817 records it is 212 of
433 values, and a class of space-or-hyphen matched **none** of them: the field answered 51% of
what it saw, and every miss was the permissible value spelled exactly as the enum spells it.

## `sex_distribution` and `handedness_distribution`

Case and plural folding, not vocabulary work: 18 surface forms over 859 values for two sex
answers, 14 over 265 for three handedness answers. Here rather than inline so a query
grouping by sex reads one value and not eight.

Kept as two modules because the answer sets differ, and a shared "demographics" module would
hide that. `OTHER` holds a reported category outside the binary — a value to preserve rather
than a failure to classify.

The handedness negations are load-bearing: "non-left-handed" is an inclusion criterion meaning
right or ambidextrous, and reading it as LEFT inverts the group it describes.

## `medication_status`

493 surface forms over 711 values, the messiest field measured, and the standard moderator in
schizophrenia and depression meta-analyses.

**The discriminating feature is negation, not vocabulary.** Every affirmative cue appears
inside its own negation — "not medicated" contains "medicated", "free of antipsychotics"
contains "antipsychotics" — so a keyword rule inverts the cohort it describes unless it models
scope, and a proximity rule cannot tell `no` the negator from `no` the determiner. Scope comes
from a dependency parse (`_negation`), and the domain part shrinks to a lexicon of concept
words. A phrasing nobody anticipated is then handled by syntax rather than by another regex.
Growing `CONCEPTS` is how this field is extended.

**An unnegated mention settles it.** A cohort described as taking something is taking it,
whatever else the sentence goes on to deny: "taking antidepressants; no medication changes" is
a medicated cohort.

`NAIVE` is kept apart from `FREE` deliberately: never-medicated and withdrawn-before-scanning
are different populations, and a moderator analysis that merges them cannot see a
treatment-history effect.

Morphological negation is stripped to its scope-visible form first, because a parse cannot see
it — "unmedicated" is one token with no syntactic negation to attach to. "not drug-naive" is
checked with an adjacency pattern rather than parse scope, because scope reaches across
clauses: in "never-medicated; antipsychotic naive" the negation belongs to the first clause and
not to the marker.

There is no `available()` guard before the parse. Without one this returned UNKNOWN, which is
also what a paper that never mentions medication returns — so an uninstalled model read as a
corpus that stopped reporting. `mentions` raises instead, naming the package.

## `_negation` — scope from syntax

Written because the alternative does not generalise. A proximity regex — a negation cue
within N characters of a concept word — has to be retuned for every new phrasing, and it
cannot tell `no` the negator from `no` the determiner: in "taking antidepressant
medication; no medication changes" the cohort *is* medicated, and a proximity rule reads it
as the opposite. A dependency parse distinguishes them because they are different relations.

The domain part shrinks to a lexicon of concept words, small and stable. The linguistic part
is a general parser, so a phrasing nobody anticipated is handled by syntax rather than by
another rule.

Scope follows the standard clinical-NLP treatment (Chapman et al., NegEx, 2001): a negation
governs its syntactic subtree, so a mention is negated when the negation attaches to it or to
any of its ancestors.

**Why ancestors are searched over the left subtree only.** English negation precedes what it
scopes over, and the restriction is what stops a later clause reaching back: in "taking
medication; no changes" the `no` is to the right of the mention and does not negate it. An
ancestor that *is* the negation is checked before its subtree, because a preposition governs
its object and has nothing to its left — so `without` in "adults without neurologic
disorders" was invisible to the left-subtree scan.

**Why a missing parser raises.** Without scope, "not medicated" and "medicated" contain the
same words, so the field would read UNKNOWN — and papers that never mention medication also
read UNKNOWN, making a missing model look like missing data.

`cue_forward_scope` is NegEx's forward scope, doing two jobs: the residue a parse leaves when
it drops a cue's object but not the cue, and the whole negation layer where no parser is
installed. Scope runs to the end of the part because the caller has already split on the
separators that terminate one.

`scope` parses the whole value **before** the caller splits it into heads, so a cue scopes
over a coordination. What survives is cut out of the original string by character offset: a
tokenizer does not round-trip, and rejoining turned `treatment-resistant` into
`treatment - resistant`.

## `_records` — reading a record

Value access goes through `formats.values.value_of`, which takes the wrapper and the slot's
declared shape from the LinkML schema. Hand-rolling that unwrap conflates three different
claims — absent, `not_reported`, and reported-empty — and each conflation is a silent wrong
answer. See [pipeline-architecture.md](pipeline-architecture.md), "The contract at each seam".

`_descend` has to tell a wrapped value from a nested entity: both are mappings and only one
has a `value`. `value_of` reads a mapping without one as `not_reported`, which is right for a
wrapper and wrong for an entity — a Task is an object to descend into, not a slot the paper
declined to fill.

## `_embedding` — which encoder, and why two

Chosen by input length, not by domain. Measured on this corpus the two models invert
completely:

| input | SapBERT | MiniLM |
|---|---|---|
| short entity strings (17–30 chars) | R@1 66.3% / 62.9% | 50.6% |
| task descriptions (~400 words) | 24.5% (last) | 58.5% |

Neither is "the biomedical model". `for_phrases` and `for_prose` name the choice so a caller
states the input's shape instead of guessing a model.

The device is detected rather than pinned: the machines this runs on differ and a hardcoded
`cpu` left a GPU host idle. It is deliberately **not** part of the cache key, so a run on one
machine may reuse another's cache.

## `is_healthy` — derived, not asked

The slot was `model_extracted` until it was measured. Asked directly, a model answers with the
source's wording: across 1,817 records **168 groups came back True beside a real diagnosis** —
"healthy male smokers" with nicotine dependence, "obese subjects" with obesity — and 132 of
those were `value_source: reported`, because the addiction and obesity literature says
"healthy" to mean free of comorbidity. Rewriting the slot description to say otherwise moved 2
of 5 test cases and left `is_healthy=True` beside `medical_condition=[obesity]` untouched.

So the question is removed instead of rephrased. The flag then cannot contradict the field it
summarises, which is the invariant `schema-tutorial.md` already declared and nothing enforced.

**Three states, and the third matters: unset is not False.** A group whose `medical_condition`
was never read is an unread cohort, not a sick one. That third state is reachable from the
value as well as from `extraction_status`: an extractor writing "unknown" into the slot has
said the same thing the status says.

`apply` writes into the same `ExtractedValue` shape as the rest of the record, marked
`derived`, and replaces any previous value — the point is that the two cannot disagree.

## `group_role` — built, measured, removed

A derived `Group.role` (`case` / `comparison`) was added and then taken out again. What it
was for is real and is now handled where it belongs; what it could not do is why it went.

**The problem it addressed.** Every cohort criterion in a coordinate meta-analysis asks
which side of the comparison a cohort is on, and each query rebuilt it from the words —
wrongly, in the same way: **a comparison group is named after the condition it does not
have.** "nonsmoking control subjects" matches a nicotine pattern, "comparison group,
non-use of marijuana" matches a cannabis one, and with both sides read as the cohort the
contrast has no other side. It cost 13 of 15 gold papers on one published map and 7 of 17
on another.

**What it was worth.** Over the 11 cohort keys of the benchmark's published maps, 212 gold
papers: the cohort pattern alone finds 120 at 0.445 mean coordinate F1; adding `is_healthy`
finds **160 at 0.546**; adding a negated-naming rule finds **164 at 0.549**. So the fix was
almost entirely `is_healthy` — a derived field that already existed and that no query was
reading — plus four papers from the negation rule.

**Why the field came out.** Reading the derived role *instead of* those two readers scores
0.547, and every point of the difference is a study whose cohorts cross two questions.
22445480 has `Control Smoker` beside `MA-dependent Smoker`: that cohort is the comparison
for the methamphetamine map and the case for the nicotine one. **A role is relative to the
question and a per-group enum is not**, so the field has to pick one, picks comparison, and
loses the nicotine contrast. 19645730 loses its contrast the other way, to `Non-alcoholic
control status`, where the denial is real and the word form (`alcoholic` against `alcohol`)
hides it from a field computed once.

A field that reproduces two readers on 14 of 14 keys, scores slightly worse where they
differ, and cannot express the thing that makes the hard cases hard, is a slot to maintain
and a second place for the answer to live. The rule stays where it can see the question:
`names_cohort` in `scripts/query_contrasts.py`, beside `is_healthy`.

**What it leaves behind, for whoever tries again.** The negation reading has to be
relational — `phrases.triage` reads the syntactic negations ("no history of alcohol misuse")
and must not read the morphological ones, because `non` is not a negator in "non-fluent
variant PPA" or "non-Hodgkin lymphoma". What makes `cannabis non-consuming` an absence is
that another cohort in the same study asserts cannabis. And it has to read the cohort's
name as well as its `medical_condition`: `non-PTSD subjects` denies an acronym that the
other cohort spells out as `post-traumatic stress disorder`.

## `task` — seed, then cluster

Three steps, and the order is the design: **seed** each task name against the normalised
Cognitive Atlas, **cluster** everything unmatched on a paradigm distance over six channels,
**label** the stimulus from `Condition.stimulus_content` as its own column.

**Seeding comes first because the Atlas is a target and clustering is not.** There used to be
a second module doing this job that skipped the seeds and clustered the whole corpus against
itself with a fitted pair model. Both were reachable — the seeded one only through a script,
so `pondie normalize task` ran the other — and they disagreed. On the 100-paper defect set the
unseeded route merged `novelty oddball task`, `Go/No-go tasks` and `sustained attention task`
into one identity called `stop signal task`; the Atlas names those as three separate paradigms
and keeps them apart.

The unseeded route also had no way not to. Its pair model trained by distant supervision on
name components of three or more members, and over 90 tasks **only two such components exist** —
so 20 of 90 tasks appeared in any training pair, and the model learned a single axis:
resting-state or not. Full working in [task-clustering-method.md](task-clustering-method.md).

So the seeds are the vocabulary and the clustering is the residual. Only folded name
**equality** is a hard constraint: equality closes transitively, containment does not. The
constraint is applied as a group (`same_name` returns the members and `paradigm_distances`
zeroes every pair within them) rather than as a star of pairs anchored on the first member — a
star does not survive losing its hub, and the hub is dropped routinely because clustering sees
only the unseeded half. 49 tasks sharing one folded name came out in seven categories that way.

**The six channels are kept separate rather than concatenated.** A sentence embedding is a mean
over its passage, so folding a weak field into one signature averages away the token that
discriminates. `prose` and `prose_lex` are dense and sparse views of the same text; conditions
are a set, compared by soft overlap. `_similarities` returns the columns rather than combining
them so the caller decides; `paradigm_distances` means them, and a *fitted* weighting is what
the deleted route did instead.

`Condition.stimulus_content` is deliberately **not** a channel: the stimulus must not separate
two tasks running one paradigm — alcohol and food cue reactivity differ there and must still
cluster together. `stimulus_of` reads it as its own output column instead. For the same reason
`Task.apparatus` is the design and the response modality only; `Task.stimuli` used to be folded
in beside them and the channel excluded it, so the slot was read off every record and never
used. It is not loaded any more.

**Abbreviations are article-scoped.** `paper_stores` reads each paper's own text and hands it to
`Abbreviations.for_paper`, which mines that text and adds only that paper's rows. The returned
store *is* one paper's, so reaching past it is not possible rather than merely not done. There
was a module-level cache holding the whole corpus file with `normalise` looking up
`(short, paper)` in it; lookups were scoped so nothing leaked, but a global abbreviation list is
the shape this repository has ruled out — an expansion is a fact about the paper that wrote it —
and it never read the paper, so a short form defined only in a paper's own Methods resolved to
nothing unless someone had mined the corpus file first. The corpus file is read once for the
whole loop, passed rather than cached in a global.

Measured gain on the 100-paper defect set: `SVF test` reaches the Atlas's `verbal fluency task`,
taking seeded tasks from 16 to 17 and Atlas-named categories from 15 to 16.

`rescue` is off by default (`--rescue 0`). Average linkage votes down a task adjacent to one
member of a large cluster, which is what it exists to undo; it reports what it moved through the
result rather than printing, because `categorise` is a library call and `report` and `main` are
the only things here that write to a terminal.

`cut` and `encoder` are in the result because a category list is not interpretable without the
cut that made it.

## `atlas` — the target half

Normalises the Cognitive Atlas task list into seed categories: 856 labels to 639 seeds. Reads
the `alias` field, which 259 of the 857 entries carry and nothing was reading —
`temporal discounting task` aliases `delay discounting task`, `balloon analogue risk task`
aliases `BART`.

Hand-curated where no rule works, and the curated lists are short and named in the module. The
four error classes the rules exist to avoid — negation, junk parents, wrong parent, eponyms —
are worked through in [cognitive-atlas-seeds.md](cognitive-atlas-seeds.md).

Re-parenting is **earliest in the child, then longest**. Shortest-first sent
`Motor Selective Stop Signal Task` to a motor label; longest-only left `Stop signal task with
dot motion discrimination` on `dot motion task`, because both candidate cores are two tokens.
The paradigm is named first and the qualifier follows it, so position breaks the tie — the same
rule the ONVOC `contains` layer needed.

`report()`'s eight probes are regression cases, each one a failure the rules were changed to fix.

## `population_characteristics` — the partition shape

The field holds what a study chose its cohort for: habitual exposure, training, occupation,
lifestyle, atypical body habitus. Its value is that a query can filter on it, and that value
survives only if every entry is discriminative.

**Asking for that did not work.** Three rewrites of the slot description, each tested by
re-extracting the same ten papers, left non-selective entries at 24%, then 14%, then 7% —
differences of two and three entries out of thirty, on a sample too small to tell the versions
apart. `is_healthy` had already shown why: a description cannot outvote the source's own
wording. So the question stays as it is and the answer is partitioned afterwards, which is a
rule that can be read, tested against the whole corpus, and changed without a re-extraction.

**What moves.** A value goes to `other_characteristics` when every plausible cohort could carry
it — "normal weight", "right-handed", "normal or corrected-to-normal vision", "no psychiatric
history", "MRI compatible", "native English speakers". *Moved, not dropped*: the value is not
wrong, and a reader auditing a cohort wants to see it. It simply cannot share a field with a
trait a filter would select on. `other_characteristics` is `deterministic` in the storage
schema, so the generator never puts it to a model and nothing is asked twice.

**Two asymmetries, both of which a blunter rule gets backwards.**

*Handedness.* "right-handed" is normative; "left-handed" and "mixed-handed" are selective,
because a study recruiting left-handers recruited for that.

*Negation.* A negated **condition** is normative — "no neurological or psychiatric disorder" is
carried by every control cohort in the corpus. A negated **exposure** is selective: "no history
of smoking" is the control arm of a smoking study, and "cannabis use less than 50 times" is how
a cue-reactivity paper defines its comparison group. `EXPOSURE` is therefore decisive against
every rule here, which is what makes the negation rule safe to state broadly.

**Matching is full-string against a reduced core, never a substring search.** "Otherwise healthy
adult smokers" reduces to "healthy smokers", which no rule matches in full, so it stays — where
`search(r"healthy")` would have moved it and lost the cohort's defining trait.

## `contrasts` — one question across many records

Which contrast is treatment against control. Normalization, not query: a trial names its arms
`active iTBS`, `REAL`, `MPH`, `paroxetine 20mg`, and this resolves them to the roles a synthesis
can pool on — "a record's own wording → shared values". It sat under `query/` where nothing in
`query/` used it, with one import from `normalization.corpus` pointing back up.

Three things have to line up, and only one of them is hard.

| | |
|---|---|
| the role | already normalised. `ArmKind` is a schema enum and it splits cleanly: pharmacological / stimulation / behavioural_intervention / active_comparator on the intervention side, placebo / sham / usual_care / no_intervention on the comparator side |
| the agent | free text, mapped onto ONVOC so `escitalopram` in one paper and `Escitalopram` in another are one row |
| the link | a `Cell.level` is a string, and which arm it names is the open question. Matched on **words, never a similarity score**, for the reason `derive_direction` gives: `men` is a substring of `women` |

An analysis qualifies only when one cell resolves to an intervention arm and another to a
comparator arm. An analysis contrasting two groups, or two timepoints, is not a treatment
contrast however much it mentions a drug.

## Open findings

Things this audit surfaced and deliberately did **not** change, because each is a behaviour
decision rather than a redundancy.

### The derived-slot fills are never called

Three modules fill a slot rather than normalizing one, and each exposes `apply`:

| module | slot | caller |
|---|---|---|
| `is_healthy` | `Group.is_healthy` | none |
| `population_characteristics` | `Group.other_characteristics` | none |

`study_schema/extraction-deviations.yaml` says of both `is_healthy` and `role` that the slot
is `deterministic` in storage and that
`pondie.normalization.<module>.apply` "fills it from … after extraction". Nothing in the
package or the pipeline invokes any of the three. The extraction schema re-adds the slots, so
a model answers them and the derivation that is supposed to overwrite the answer never runs —
which is the exact failure `is_healthy` was made deterministic to prevent.

What it wants is one seam — a `normalization.apply_derived(record)` that runs all three and
returns their tallies — plus a caller after `build`. That is a new pipeline step, not a
cleanup, so it is a decision to take rather than a change to slip in.

### `is_healthy.UNREAD` invents extraction statuses

`UNREAD = {"not_reported", "not_applicable", "unknown", "None", "none"}` is compared against
`extraction_status`. The authority is `formats.values`:
`ExtractionStatus = Literal["extracted", "not_reported"]` and `STATUSES = ("extracted",
"not_reported")`. Of the five, only `not_reported` can legitimately occur; `not_applicable` is
an *evidence* status and the other three are not statuses at all.

Narrowing it to the canonical set would remove a second, looser definition of what a status
is — but any record in the wild carrying a junk status would change answer, so it needs the
corpus checked first rather than a blind edit.

### `contrasts` has three unreferenced patterns

`_CONTROL`, `_PATIENT` and `_RISK` are defined and read by nothing. Left alone because
`contrasts` is under active development and they may be half of an intended change.
