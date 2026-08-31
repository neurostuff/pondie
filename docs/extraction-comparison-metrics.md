# Scoring an extraction against a gold record

`compare_extractions.py` answers one question -- *how much of this paper did the extractor
get right?* -- by decomposing it into four that can be measured separately. This file is
why those four and why those measures; the script says how.

    python compare_extractions.py data/records/xevP8UDRAVh9.extraction.json \
        --gold data/gold/xevP8UDRAVh9.extraction.json -v

## The dependency that shapes everything

Nothing can be compared until the two records' entities have been put in correspondence.
Field accuracy is accuracy *on matched pairs*; relationship F1 is over triples *in gold
identifiers*. So the entity map is computed first, and every later number is conditional
on it. Two consequences the report never hides:

- **Unmatched entities are printed next to the field scores.** An extractor that emits two
  of ten analyses and fills both perfectly has a field accuracy of 1.0 and a recall of 0.2.
  Reading the first without the second is the standard way to be wrong about an extractor.
- **A reference touching an entity that never matched keeps a `?local_id` marker** and so
  can never coincide with a gold edge. That is the correct verdict: it names something the
  gold record does not contain.

## 1. Objects

Extractors do not agree on `local_id` -- gold's `dev_verio` is the candidate's
`dev_magnetom_verio` -- so entities are matched by content, not by name.

**Optimal bipartite assignment, not greedy.** Entities of one class are scored pairwise and
assigned by the Hungarian / Jonker-Volgenant shortest-augmenting-path algorithm. Greedy
best-first matching is not merely suboptimal here: one strong pair can consume the only
partner a second pair had, and the mis-map then corrupts the field and relationship scores
of everything hanging off it. This is the same construction as CEAF in coreference
resolution (Luo 2005), which likewise scores a clustering by the best alignment to a key
rather than by a link-by-link comparison.

**Four kinds of evidence, to a fixed point.** An entity's identity is where it sits in the
record as much as what it says. A `Measure` is *the quantity these three analyses measured*;
a `ModelTerm` is *the term those cells are contrasts on, declared by that model*. So the
score combines:

| evidence | weight | what it reads |
|---|---:|---|
| attributes | 0.45 | fields, weighted by how well each separates instances of its type |
| outgoing | 0.20 | references it makes, translated through the map so far |
| incoming | 0.25 | who points *at* it, and through which slot |
| containment | 0.10 | whether the entity declaring it is itself matched |

Each component is dropped, not counted as disagreement, when an entity has none of it.

**Incoming references matter most for the entities that matter most.** An entity that exists
to be referenced makes few references itself, and a *continuous* `ModelTerm` makes none at
all: it declares no levels, so it has no outgoing edges, and a matcher reading only outgoing
edges is left matching it on its name. That is not a hypothetical. An extractor that wrote
`"cerebral perfusion"` where gold says `"perfusion condition"` scored 0.50 on name, fell
below threshold, and every cell hanging off that term vanished from the direction metrics as
*unaligned* rather than being reported as wrong. Reading the four analyses whose cells name
it recovers the match (incoming agreement 0.86) and leaves the genuine defect — the candidate
made the term continuous where gold has a two-level factor — visible as the low attribute
score it is.

References held by inline objects (`Cell.term`, `AnalysisGroup.group`, `FactorLevel.arms`)
count as their owning entity's, with list indices stripped; they are most of what connects
the analysis side of the graph.

**Attributes are weighted by what they discriminate.** A field every `Acquisition` in a paper
agrees on cannot say which `Acquisition` you are looking at, however prominent it is. Each
field's weight is scaled by the fraction of same-type instance pairs its values actually
separate, measured on the gold record rather than assumed, so thirty agreeing boilerplate
slots cannot outvote the one that differs. A type with a single instance has no pairs to
measure, and sits at the midpoint.

The passes feed each other -- attributes first, then attributes plus the relational three
read through the map so far -- until the map stops moving (collective entity resolution, as
in Bhattacharya & Getoor 2007). Four passes is a cap, not a target; these records converge in
two.

**Matched within the declared family, not the concrete class.** An `Acquisition` slot may
hold an `MRI` or a `PET`. Matching per concrete class would make a candidate that typed a
scan wrong into two unmatched objects -- one missing, one spurious -- when it is one entity
with one wrong field. So the family is the matching unit and the type designator is scored
as a field.

**Below-threshold pairs are dropped.** The assignment problem will happily marry the last
gold entity to the last candidate one whatever they are. A pair scoring under 0.45 is left
unmatched, becoming a false negative and a false positive, which is what it is.

Reported per class: precision, recall, F1, the gold and candidate counts, the mean score of
the matches made -- a low mean over an F1 of 1.0 means the matching was forced -- and the
identifiers actually missed and invented.

## 2. Relationships

With the map in hand every cross-reference is a triple `(source, slot, target)`. Candidate
triples are rewritten into gold identifiers and the two sets compared: micro precision /
recall / F1, a breakdown per slot, and both false-positive and false-negative lists in
full. List positions are normalised (`effect.cells[1].term` reports under
`effect.cells[].term`) so the per-slot rates are not fragmented by index.

Both endpoints are treated alike. An edge *out of* a hallucinated entity is as much a false
positive as an edge *into* one, and symmetrically a missed gold entity takes its outgoing
edges with it into the false negatives. The alternative -- suppressing edges whose source
never matched -- makes precision and recall describe different graphs, and hides the fact
that a spurious entity is usually spurious *because* of what the extractor wired it to.

## 2b. Structure: is each object wired where its counterpart is?

Relationship F1 is one number over the whole graph. It says a structure is wrong without
saying where, and it averages away the failure worth naming: an object whose *attributes* are
right and whose *place* is not.

So each matched pair also gets its own neighbourhood precision, recall and F1 over both
directions — the references it makes and the references made to it — with the specific links
missing and extra listed. Entities scoring high on attributes and low on neighbourhood are
called out as **misplaced**: a plausible object wired into the wrong slot, which is a
different defect from a wrong value and needs a different fix.

Matched entities with no neighbourhood at all — a `Timepoint` no factor level reaches, the
`Study` root — are excluded rather than scored zero. There is no structure there to get wrong.

Worked example, from a real run: `measure_cbf` matched on attributes at 0.75 but scored 0.0
on neighbourhood, with three `analysis -measure-> here` links missing. The candidate created
the right measure and then failed to point three of the four analyses at it. Nothing in the
field metrics would have said so.

## 3. Fields

For each matched pair, every field is compared at its own path -- `age_mean`,
`effect.statistic.family`, `levels[0].level`.

**Presence and value are separate defects.** `not_reported` is a positive claim that the
paper is silent, and it is wrong in two different directions: a value the paper had and the
extractor dropped, and a value the extractor supplied from nowhere. Those get their own
precision / recall / F1 (`presence`), and value accuracy is computed only where both sides
extracted something. Folding them together would let an extractor that reports nothing look
identical to one that reports everything correctly.

**Numbers are coerced, then compared with a relative tolerance.** `2`, `2.0` and `"2"` are
one value. Tolerance is relative (1% by default) because these fields span twelve orders of
magnitude -- an alpha level of 0.05 and a permutation count of 5000 cannot share an absolute
tolerance. Beyond the pass/fail, matched numeric pairs get MAE, RMSE, MAPE, signed bias and
a within-tolerance rate, so a systematic misreading (unit confusion, a consistent
off-by-one in a count) is visible as bias rather than hidden in an accuracy.

**Strings get a graded score, not a verdict.** Surface similarity is the best of a character
edit ratio, an order-free token Dice coefficient, and a discounted containment ratio.
Containment is what carries an abbreviation against its expansion -- `SCID-II` inside
`Structured Clinical Interview for DSM-IV Axis II Disorders (SCID-II)` -- which neither of
the others will find; it is discounted so a one-word substring cannot claim a perfect match.
With `--semantic`, embedding cosine is taken as well and the larger of the two is used, so
semantics can only rescue a pair that surface overlap missed. Cosine is rescaled from its
practical floor (~0.3 for unrelated text in a modern embedding space) so the two scores are
on a comparable footing. A pair scores as *matched* at 0.85; the graded mean is reported
alongside because free-text fields like `description` are paraphrases even when correct.

**Enums are exact,** with half credit on the graded score for a near miss, because several
of these vocabularies are open (`variation_level`, `assessment_type`) and a paper's own
wording is a legal value.

**Multivalued fields are matched, not zipped.** A list of inclusion criteria has no
meaningful order; the elements are aligned and scored as a set (F1 at the match threshold,
or the graded mean, whichever is higher).

Inline objects that are not entities -- `Cell`, `FactorLevel`, `AnalysisGroup`,
`SexDistribution` -- have no identifier to join on, so they are aligned within their parent
by the same content similarity and then compared at the gold-side index.

## 4. Direction -- the weighted metric

Direction is the fact a synthesis cannot recover from anywhere else in the record. Every
other field can be wrong and leave the finding legible; a flipped sign inverts it. It
carries 0.45 of the composite.

**A cell is only credited when its term is grounded.** `Cell.direction` is meaningless
without `Cell.term`: `positive` on a gray-matter slope and `positive` on a treatment
condition are different claims. A cell pair is scored only when its `term` reference
resolves, through the §1 entity map, to the same `ModelTerm` on both sides. A right sign on
a wrong term is counted as ungrounded, not as correct.

**Cells are aligned on term and level, never on direction.** Matching cells by their
direction would manufacture the agreement being measured, so `direction` is excluded from
the cell alignment score (`ALIGN_EXCLUDE`).

**F1, not accuracy, is the headline.** Accuracy over grounded cells is gameable: an
extractor that emits one easy cell per analysis and drops the rest scores 1.0. Precision is
over every candidate cell and recall over every gold cell, so a dropped cell is a miss and
an invented one a false positive. The observed failure on
`data/records/xevP8UDRAVh9.extraction.json` is exactly this shape -- one cell per analysis
where gold has two -- and accuracy alone would have understated it.

The rest of the panel names the *kind* of error, because they are not interchangeable:

| metric | what it catches |
| --- | --- |
| `sign_flip_rate` | gold `positive`, candidate `negative`. The catastrophic case: the finding is reported backwards. |
| `sign_loss_rate` | gold signed, candidate `undirected` / `held` / `not_reported`. Conservative -- uninformative, not wrong. |
| `sign_invention_rate` | gold unsigned, candidate signed. A direction asserted that no test produced. |
| `kappa` | agreement above chance. These vocabularies are skewed; an extractor answering `positive` always gets 0.8 accuracy on a record that is 80% positive, and kappa 0. |
| `macro_f1`, per-class P/R/F1 | keeps the rare labels (`held`, `undirected`) visible, which a micro average buries. |
| confusion matrix | which substitutions the extractor actually makes. |

**A contrast is only right as a whole,** so each analysis also gets one label:

- `exact` -- every gold cell grounded and matched, no extra candidate cells, all directions agree.
- `reversed` -- every sign flipped, unsigned cells unchanged. Split out from `wrong_direction`
  on purpose: it is one diagnosable mistake, the comparison read backwards, and it behaves
  differently downstream from a contrast that is merely partly wrong.
- `wrong_direction` -- structure right, some direction wrong.
- `structure_mismatch` -- the cells do not correspond, so no direction claim can be checked.
- `analysis_missed` -- no candidate analysis matched at all.

### The prose-leakage bracket

An Analysis's `name` and `definition` say which way its contrast went -- "Positive
correlation" -- so an entity alignment that reads them has partly seen the answer before
scoring it. The script therefore reports the direction metrics twice: once under the primary
alignment, and once under an alignment blind to `Analysis.name`, `definition` and
`interpretations`, matching instead on what the analysis was *of* (measure, groups, model,
inference settings, cell terms and levels).

This is a **bracket, not a correction**, and it can move either way:

- Blind number *lower* than primary: the identity of the analyses was being carried by their
  prose, and the primary number is optimistic.
- Blind number *lower because two analyses became indistinguishable*: some pairs -- an
  `A > B` and its `B > A` -- differ *only* in cell direction, so removing the prose leaves
  nothing to separate them and the blind alignment swaps them. That is a property of the
  record, not evidence against the extractor.

When the two agree, the headline is robust to the question. When they diverge, read the
per-analysis detail rather than either number.

### Uncertainty

A single record supplies few cells, and a direction accuracy of 0.86 over fourteen of them
is not distinguishable from 0.7. The headline carries a 95% percentile bootstrap interval
over cell pairs. It is a *within-record* interval: it covers the sampling of cells, not of
papers. Across several records the report gives both a micro pool (every cell in the corpus,
so a paper with forty analyses dominates -- how much of the evidence was recovered) and a
macro mean over records (how the extractor does on a typical paper). They answer different
questions and are printed side by side.

## The composite

    0.45 direction F1  +  0.20 entity F1  +  0.20 relationship F1  +  0.15 field accuracy

Renormalised over whatever components exist -- a record with no analyses has no direction
term. It is a ranking convenience for comparing extractor versions, and it is always printed
with its parts, never alone. Any single number over a structure this heterogeneous hides
more than it says; the parts are the finding.

## Known limits

- **Everything downstream of §1 is conditional on the entity map.** Read the recall column.
- **Evidence spans are not compared.** Whether a value is supported by the right characters
  of the paper is a different question, and `review/validate_record.py` and the review layer
  already ask it.
- **Ordered lists are compared as sets.** A permuted `acquisition_voxel_size_mm` is a real
  error this will not flag; the field's own description states the ordering rule, and
  encoding a per-field order policy here would duplicate the schema.
- **One gold record is one annotator.** Nothing here estimates the ceiling. Agreement
  between two independent gold extractions of the same paper, scored with this script, is
  what would give the numbers a scale.
