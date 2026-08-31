# Field-by-field extraction audit — todo

Scope: every field in the **extraction** schema (182 attributes), not just the ones intuition
suggested. For each: what was tried, and the maximum precision any surface method can reach.

## Method ladder, applied per field

1. **Verbatim location** — does the value appear in the paper text at all? This bounds every
   surface method: string match, regex, spaCy and a transformer NER all read the surface, so a
   value that is not there cannot be recovered by any of them.
2. **Normalised match** — case/punctuation folded, numbers coerced, token-subset containment.
3. **Ambiguity count** — how many candidate matches per instance. One candidate is
   deterministically resolvable; k candidates need a scope rule to choose between them.
4. **Section scoping** — which section the correct match sits in, using the `paper_sections`
   offsets already in `extraction_metadata`. Tests whether "look only in Methods" disambiguates.
5. **Targeted regex** — for fields where 1-4 say it is worth writing one.
6. **spaCy / scispaCy NER** — for entity-name fields, where the task is span typing.
7. **Hugging Face** — researched per field family; only run where 1-6 leave a gap a
   recogniser could close.

## Status

- [x] enumerate the 182 extraction-schema attributes and join instance counts
- [x] within-paper invariance screen (26 invariant / 24 single-instance / 103 varies)
- [x] targeted regex on the paper-scoped candidates (8 shipped, measured)
- [x] research DocRE / SciREX for the scoping problem (SOTA F1 16.9 — unusable)
- [x] research biomedical NER on Hugging Face (OpenMed anatomy/disease families)
- [x] distant-supervision test for salience (features invert; labels are recall-noise)
- [x] verbatim-recoverability ceiling for every field
- [x] ambiguity count per field
- [x] section distribution of the correct match per field
- [x] spaCy NER on entity-name fields
- [x] final per-field table with max precision and every method tried

- [x] spaCy `en_core_web_sm` on the entity-name fields (CPU)
- [x] GLiNER `gliner_medium-v2.1` on the entity-name fields (CPU)
- [x] GLiNER2 `fastino/gliner2-base-v1` schema extraction on paper-scoped fields (CPU)
- [x] NuExtract-2.0-2B on paper-scoped fields (beast, RTX 3070)
- [x] survey Ai2 / AllenAI models (SciBERT, Longformer, OLMo 3, Asta)
- [x] NuExtract3 systematic sweep: 10 entity classes, 40 fields, 6 papers (beast)
- [x] NuExtract3 nested-template scoping test (TR/TE per acquisition)
- [x] NuExtract3 table-splitting test against 15 stage-1 parses
- [x] NuExtract3 few-shot leave-one-paper-out ablation
- [x] field-name hint ablation (answer-vocabulary leakage)

- [x] second NuExtract3 sweep: analyses, effect.cells, analyses.details, model terms, design, tables
- [x] score NuExtract3 `Cell.direction` against the reviewer direction gold

## Coverage

The method ladder (surface / uniq / medC / section) is filled for **all 158** filled fields.
NuExtract3 now covers **78** fields across 16 templates, including `effect.cells` scored on the
reviewer gold. GLiNER covers 5 entity-name fields, GLiNER2 covers 6.

Not covered by a model, and deliberately: the long tail of `analyses.details` subtypes
(`LatentDecompositionDetails`, `ConjunctionDetails`, `SimilarityDetails`, `MediationDetails`)
and the rarer `Acquisition` subclasses (PET tracer fields, EEG/fNIRS rig fields). Between them
they hold fewer than 30 instances across the whole corpus — too few to measure a precision
against, which is itself the finding: those fields need more reviewed papers before any
method can be evaluated on them.

## Not done, and why

- **scispaCy `en_core_sci_sm`** — wheel install did not finish. General spaCy plus GLiNER
  established the shape of the result, and the repo's prior evaluation covers scispaCy's other
  components.
- **NuExtract3-4B / W4A16** — the 2B version answered the question at 2-3s/paper, so the
  larger one is a cost-quality tuning question rather than a new finding.
- **Fine-tuning SciBERT or Longformer** — the right base models, no zero-shot mode. Blocked on
  labels: 4 answered, 281 available once the entity-inventory review is done.
- **A trained classifier for entity salience** — same blocker.
