# Normalizing records so one query can cross papers

A record keeps the paper's own words. That is deliberate --
`neuroimaging-study-storage.yaml` says so, and adds that mapping those words onto ONVOC or
the Cognitive Atlas "is a later stage that reads the free text and its evidence
sentences". This is that stage, and it is a separate layer on purpose: a mapping is an
assertion *about* a record, and writing it into the field would destroy the thing that
makes it checkable, which is the paper's wording sitting next to it.

The question being served: **for every intervention trial in the corpus, which contrast is
treatment against control, and which way did it go?**

## The question decomposes into three, and only one is hard

| | where it comes from | state |
|---|---|---|
| the **role** of an arm | `ArmKind`, a schema enum | already normalized |
| the **agent** | free text, mapped to ONVOC | the layer below |
| the **link** from a cell's level to an arm | word-set containment | the open problem |

`ArmKind` splits cleanly without any vocabulary work: `pharmacological`, `stimulation`,
`behavioural_intervention` and `active_comparator` are the intervention side;
`placebo`, `sham`, `usual_care` and `no_intervention` the comparator side.
`active_comparator` sits on the intervention side deliberately -- a head-to-head trial has
no inert arm at all.

Measured over 37 records, `arm_kind` was populated on every arm of all 16 records that had
arms. Nothing needed to be inferred.

## Two vocabularies, and why both

**ONVOC** (752 classes, [BioPortal](https://bioportal.bioontology.org/ontologies/ONVOC))
supplies the nouns a trial's arms, groups and assessments are made of: drugs by class
(Antidepressants, Anti Psychotics, Psychedelics, Opioids), Psychiatric / Neurological /
Medical Disorders, Cortical and Subcortical Regions, Tests, Population Groups.

**Cognitive Atlas** (918 concepts, 857 tasks, 221 disorders) supplies what a paradigm
*is*, which ONVOC does not attempt.

Both are cached in `data/vocab/`, with ONVOC's own crosswalks to MeSH, MONDO, DOID and
SNOMED from [open-neuro-catalog](https://github.com/ckindermann/open-neuro-catalog), and
the CUI bridge from [mesh-2-onvoc](https://github.com/ckindermann/mesh-2-onvoc) (3,898
rows of `cui -> mesh_id -> onvoc_id`).

## Abbreviations are expanded first

ONVOC spells everything out; papers do not. `ADOS`, `dlPFC`, `MADRS` reach nothing by
string matching, and the bridge is almost always in the paper itself, in the one shape
that makes it recoverable -- `the Autism Diagnostic Observation Schedule (ADOS)`.

Finding those is Schwartz & Hearst's algorithm, and **scispacy's `AbbreviationDetector`
is what runs it**. A hand-written version is kept only as a fallback for hosts without
scispacy, and it is a fallback for a demonstrated reason: on four test definitions it
missed `MADRS`, whose letters are plainly there in `Montgomery Åsberg Depression Rating
Scale`. On a real paper scispacy found 18 definitions to its 13. It needs no trained
model -- a blank English pipeline with a sentencizer is enough -- so the dependency is
spacy, not a download.

Every expansion goes into one referenceable file, `data/vocab/abbreviations.json`:
**361 entries, 332 mined from the corpus and 29 curated** for abbreviations papers use
without defining. One file rather than a per-paper step, because an abbreviation resolved
one way in one record and another way in the next is a bug no single record can show.

Three things the file has to get right:

- **Manufacturer strings are not abbreviations.** A detector looking for `long form (SF)`
  also finds `(Philips Medical Systems, Best, The Netherlands)`, whose letters happen to
  fit. Rejected on shape -- several commas, or a quoted fragment.
- **Spelling variants are not disagreements.** `echo planar imaging`, `echo-planar
  imaging` and `echoplanar imaging` are one expansion written three ways; comparing on
  stems stops them being reported and leaves the ones that matter visible.
- **A paper's own definition beats the corpus.** `FA` is fractional anisotropy in a
  diffusion paper and flip angle in an acquisition section; `AD` is axial diffusivity or
  Alzheimer's disease. A corpus-wide expansion picks whichever was commoner and is then
  wrong for every paper that meant the other, so `for_paper()` layers the paper's own
  definitions on top.

Expansion is worth **+3 points** of coverage on its own (20% to 23%), all of it through
the `variant` layer, which goes from 1 match to 12.

## Matching, layered from certain to plausible

Each mapping records which layer produced it, because an exact label match and a token
overlap are different claims and one confidence number would flatten them.

| layer | what it does |
|---|---|
| `exact` / `synonym` | the label or one of its surface forms, folded |
| `variant` | the same, on a stripped or **expanded** form -- the acronym out of the brackets, the laterality removed, the abbreviation spelled out |
| `acronym` | initials of a multi-word label, **only when the record spells the expansion out** |
| `contains` | a label appearing whole inside the phrase, on word boundaries |
| `stem` | `depression` to `Depressive Disorder`, which share no substring |
| `overlap` | content words equal, order ignored |

Crosswalk terms are folded in as synonyms, which is the cheapest widening available: no
model, no network, and the pairings are the ontology author's own.

## What measurement said, including where it said no

447 field values over 37 records.

| routing | matched |
|---|---|
| vocabulary only | 31% |
| **vocabulary + branch scoping** | **20%** |

Coverage fell and that was the right trade. Unscoped, `Wechsler Abbreviated Scale of
Intelligence (WASI-IV)` matched ONVOC's *Intelligence* -- a psychological concept returned
for a test, confidently and wrongly. ONVOC's own README makes the point: *Study Focus:
Schizophrenia* and *Exclusion Criteria: Schizophrenia* are different claims about the same
term, so which branch a field may draw from is part of the mapping rather than a filter
applied to it afterwards.

Two false positives were found by reading the output and are now tested against:

- **`ASD` to `Alzheimer's Disease`.** Folding the apostrophe leaves a stray `s`, turning a
  two-word name into a three-letter acronym for a different disorder.
- **`MDD` to `Mood Dysregulation Disorder`.** ONVOC contains exactly one label with those
  initials and it is not the one any paper means. An acronym unambiguous *inside* a
  vocabulary can still be the wrong referent outside it, so an acronym match is now
  accepted only when the record itself spells the expansion out.

Where it genuinely cannot reach:

| field | matched | why |
|---|---|---|
| `regions.name` | 44% | laterality strips cleanly; ONVOC has 94 regions |
| `tasks.name` | 38% | Cognitive Atlas is strong here |
| `design.arms.agent` | 33% | drug names match exactly; **drug classes do not** -- ONVOC has no `SSRI` |
| `groups.name` | 20% | descriptive phrases reach the disorder by containment |
| `measures.source_label` | 11% | measures are not what either vocabulary is about |
| `assessments.name` | 3% | **ONVOC's Tests branch has 53 entries and lacks ADOS, MADRS, HAMD, BDI** |

The assessment number is an ONVOC coverage gap, not a matcher failure, and it is the
clearest candidate for a contribution back to the vocabulary.

## The query

`pipeline/query.py` yields one row per analysis that puts an intervention arm against a
comparator arm. An analysis contrasting two groups, or two timepoints, is not a treatment
contrast however much it mentions a drug -- and a level naming two arms identifies neither,
so it is dropped rather than guessed.

The direction reported is always the **intervention cell's**, because each paper names its
contrast whichever way round it likes and only one of the two readings survives pooling.

Over the 16 records with arms it finds 9 treatment contrasts:

    KryfAKT9dcby  'REAL' vs 'PHNT' (sham): increased
    eaEGQiVtDp9e  'Tuina' vs 'Sham' (sham): increased
    eaEGQiVtDp9e  'Tuina' vs 'CCD' (no_intervention): increased
    rxaz3qhEmJhx  'THC' vs 'Placebo' (placebo): increased

## What could not be mapped becomes a proposal

The useful output of a normalization layer is not only what it mapped. Unmatched values
are grouped on their parenthetical-stripped form, counted across papers, and emitted as
proposals for terms the vocabularies lack -- with the abbreviation expansion attached,
since an unmapped acronym with a known expansion is a far better proposal than the
acronym alone. A term used once is a paper's idiosyncrasy; a term used in ten is a term.

At support >= 2 over 37 records, 15 proposals. The pattern in them is the ONVOC gap the
coverage table already implied:

| support | group | proposal |
|---|---|---|
| 5 | disorders+population | `healthy controls` |
| 3 | regions | `left dorsolateral prefrontal cortex` |
| 2 | tests | `Positive and Negative Syndrome Scale (PANSS)` |
| 2 | tests | `Mini-International Neuropsychiatric Interview` |
| 2 | disorders+population | `healthy volunteers` |

`healthy controls` is the striking one: the commonest comparison group in the literature,
and ONVOC's Population Groups branch has no term for it.

`normalize_records.py` writes all three artefacts -- mappings, proposals and treatment
contrasts -- to one JSON.

## What has not been tried yet, and why it is next

Everything above is deterministic: no model, no network at query time. That was the point
-- it establishes what string handling alone is worth (20% at high precision) so anything
added has a number to beat.

The three candidates, in the order their expected yield suggests:

1. **The UMLS bridge.** Now the obvious next step rather than a speculative one:
   abbreviation expansion turns `ADOS` into `Autism Diagnostic Observation Schedule`,
   which is exactly the kind of string a UMLS linker resolves and ONVOC's 752 labels do
   not contain. `scispacy`'s entity linker maps free text to CUIs over ~3M
   concepts, and `mesh-2-onvoc` maps CUIs to ONVOC. That is a far wider net than ONVOC's
   own 752 labels and it reuses the ontology author's own mapping. It is the only option
   here that could move `assessments.name` off 3%.
2. **Embeddings for the level-to-arm link**, which is the part word containment is
   weakest at and the part the whole query rests on.
3. **An LLM, last.** It would map `SSRI` to a drug class and `left dlPFC parcel` to a
   region that ONVOC does not contain -- but it would also produce a plausible mapping for
   everything, and the two false positives above are the argument for making a normalizer
   say no.
