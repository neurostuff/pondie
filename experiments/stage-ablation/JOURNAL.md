# Stage ablation journal

Which of pondie's stages earn their complexity, measured by what a meta-analysis query can do
with the records. Forward stepwise, like building a regression model: start with the whole
extraction schema in one prompt, then separate stages one at a time and keep what buys
structure, fidelity or queryability.

## Setup (2026-10-02)

- **Meta-analysis**: VBM of PTSD, Pankey 2022 (`36100907`). 22 included studies in
  neurometabench, structural only, one pooled contrast "non-PTSD > PTSD". Chosen because it is
  small, has gold NiMADS coordinates, and its criteria are specific enough to test precision
  (adult, English, grey matter, voxel-wise, whole brain, PTSD measured, no longitudinal or
  treatment effects, no within-group effects, no null effects, no overlapping samples,
  coordinates reported).
- **Model**: `@psyc-aid338-ope-333f18/gpt-6-luna` ("luna-6"), `service_tier=flex`,
  `reasoning_effort=low` (pondie's default), on every arm.
- **Where**: code synced to `beast-proxy:/data/james/pondie-ablation/code`, harness in
  `.../exp` (= this folder), runs in `.../runs/<run>`. Nothing outside
  `/data/james/pondie-ablation` is written; the catalog is opened `mode=ro`.
- **Target** (from the user): ≥90% recall and ≥90% precision on PTSD, then the next
  meta-analysis. Gold coordinates may be used directly. I use them for scoring and for the
  "reported coordinates" criterion. I do **not** inject them into model inputs: a parse entry
  that exists only for included papers would leak the label into exactly the precision test.

### Arms

| arm | what runs | isolates |
|---|---|---|
| `mono` raw | one call, whole extraction schema, no parse; merged payload, no repairs | the baseline |
| `mono` built | same payload through pondie `build` (repairs, validation, derived slots) | `build` |
| `mono_parse` | `tables`, `prose`, `split`, then one call with the stage-1 listing | the parse listing |
| `ds` | `tables`, `prose`, `split`, `demands`, `satisfy`, `build` | splitting the call |
| `ds_fill` | + `fill` | `fill` |
| `ds_fill_ev` | + `evidence` | `evidence` |
| `full` | + `repair` | `repair` |

### Scoring

`queries.py` is the meta-analysis's criteria as predicates at the **analysis** level. A paper
is selected strictly when every study criterion is True and some analysis has every
analysis criterion True; permissively when nothing is False (None, "the record cannot say",
counts as a pass). The gap between the two is the part of the record that cannot answer.
Foci: the coordinates of the selected analyses, via `source_table_analysis` into the run's
own parse, against the gold foci (1 mm).

## Data findings, before any model

**D1. The existing corpus hides tables from 11 of 19 gold papers.** In
`/data/james/pondie-vs-fulltext/corpus`, 11 gold papers' texts contain no table and have an
empty stage-1 parse. `build_corpus.py` now renders each paper from either the ns-pond catalog
(text + every table's HTML as markdown + the catalog's coordinate parse) or the old corpus,
whichever carries more parsed points, then more tables. It also covers the 3 gold papers the
old corpus did not have.

**D2. Input ceiling** (`ceiling.py`): of 159 gold foci, 120 are in the stage-1 parse and
134 appear as number triples anywhere in the text. Six papers' gold foci are absent from the
text entirely (figure-only, or a different rendering), so no arm can recover them from
inputs. This is why the "reported coordinates" criterion is answered from gold coordinates.

**D3. Benchmark error, adjudicated.** Gold `29740753` is "Facial Expression Enhances Emotion
Perception Compared to Vocal Prosody", an fMRI study with no PTSD cohort. The benchmark's
`matched_studies_merged.csv` matched "Zhang et al., 2018" by author and year alone (score 1.0).
The study the coordinates come from is **30555358**, "Altered Gray Matter Volume and Its
Correlation With PTSD Severity in Chinese Earthquake Survivors" (Zhang X, 2018), whose Table 2
prints the gold focus (48, 24, −29) as PTSD < TEC. `gold.REMAP` substitutes it.

## Experiments

### E1. `mono` vs `ds` on the gold papers

**Hypotheses.**
- H1: a single prompt over the whole schema drops analyses, as pondie's docs report for an
  older model (19% of papers returned none).
- H2: demands→satisfy produces records the query can answer more often than one prompt.

**Results** (22 gold papers; `mono-v1`, `ds-v1`, `mono_parse-v1`; query v2 = `queries.py` with
the date window and the adult rule below).

| arm | strict recall | veto recall | permissive | gold foci recovered | calls | input tok (cached) | output tok |
|---|---|---|---|---|---|---|---|
| `mono` raw (no repairs) | 13/22 | – | 18/22 | n/a | 23 | 1.26M (1.04M) | 178k |
| `mono` built | 14/22 | 18/22 | 19/22 | n/a | 23 | same call | same |
| `mono_parse` | 12/22 | 15/22 | 17/22 | **124/159** | 22 | 1.22M (0.93M) | 135k |
| `ds` (demands→satisfy) | 6/22 | 11/22 | 19/22 | 49/159 | 60 | 2.61M (1.81M) | 323k |

*veto* = the defining criteria (structural, grey-matter voxel-wise, PTSD effect) must be True on
one analysis; every other criterion excludes only on an explicit False.

**What the numbers say.**
- **H2 is rejected on these papers.** Splitting into demands→satisfy gives the *least*
  answerable records at 2.6× the calls and 2.1× the input tokens. Two concrete causes:
  1. `satisfy` builds only what `demands` declared, and `demands` never declares an
     `Acquisition`. So 8/22 records have no acquisition, or none linked to an analysis, and
     "structural MRI" is unanswerable. The single prompt fills it on 20/22.
  2. `demands` gave up on 17825801 (all 15 listing entries `omitted: undetermined`) and on
     22453299 (its only prose entry declined as `localizer`), and 3 retries with the failure
     named did not recover either. The single prompt extracted both.
- **H1 is weakly supported.** The single prompt returned no analyses on 2/22 papers
  (26347628 stopped cleanly after `model_estimations`; 32490056 returned every list empty),
  not 19%. Neither was truncated (`finish_reason=stop`). A cheap retry-on-empty is the
  obvious fix.
- **`build`'s repairs barely matter to the query** (13→14 strict, 18→19 permissive).
- **The parse listing is what links an analysis to coordinates**, and one call can use it:
  `mono_parse` recovers 124 of 159 gold foci against 49 for `ds`, the same cost as `mono`.
  Study-level recall is a little lower than `mono` (12 vs 14 strict), so the listing slightly
  distracts from study facts, or this is noise at n=22.
- **`build` repairs are not why `ds` scores lower.** It is what the demands pass declares.

### The query, and what the gold papers taught it

- `adult` was unanswerable on 8/22 gold papers because ages sit in demographics tables that
  never reached the text (e.g. 22453299: "summarized in Table 1", no table in the render).
  The criterion, "reporting results among adult humans", exists to exclude children. It now
  answers False on a minor's age or a child/adolescent cohort, and True on stated adult ages
  or adult populations (veterans, parents, "aged 18–50"). First version read "child abuse" as
  minors (20673548, an adult complex-PTSD study); fixed.
- A PTSD *effect* must admit a severity regression: gold 23021615's only coordinate table is
  a CAPS regression across 28 combat veterans (13 PTSD, 15 not), labelled "TrauE > PTSD".
- The meta-analysis's search window ("from 2002 to 2020") is a criterion. 2 negatives
  (32938511, 2022; 33169525, 2021) fall outside it. PubMed's pubdate.

### Adjudication of the gold set

| pmid | verdict | why |
|---|---|---|
| 29740753 → 30555358 | benchmark error, remapped | see D3 |
| 21118656 (Eckart 2011) | **inconsistent with the criteria** | VBM restricted to a priori ROIs: "we focused on the ROIs that have been included in the cortical parcellation analysis". The criteria exclude "papers reporting a priori regions of interest". The record's `roi` is right. |
| 16838824 (Li 2006) | input limitation | the text is abstract + references only (11k chars); gold's 4 foci are nowhere in it. Whole-brain vs ROI cannot be read. |

### The negatives (33 papers screened by autonima's search, not in the gold set)

Read by title and record; the reason each is out:

- children/adolescents: 11950456, 16199014, 19349151, 22948482, 25212487, 26535944, 28888350, 30343133
- a priori ROI / manual volumetry, not whole-brain VBM: 15734342, 17892884, 18330460, 19914045, 25000505, 20673548, 26424424
- not grey-matter VBM (cortical thickness, FreeSurfer, DTI, metabolic): 19996042, 27082610, 31662209, 32977211
- no PTSD measured / not PTSD vs non-PTSD: 24058706, 28416565, 26138235 (dissociative subtype), 21592738 (flashbacks, within PTSD)
- treatment or longitudinal: 29761009, 30937515, 33169525
- outside 2002–2020: 32938511, 33169525
- **null result**: 16701903 ("Neither method detected any volume or structural differences")
- **overlapping sample**: 23113800 (Nardo 2013, same train-driver cohort as gold 19942229 Nardo 2010, and it does not cite it), 16371250 (Chen 2006, 12 vs 12 survivors of the same fire as gold 16838824 Li 2006), 30127342 (Gong 2019, "subsets of the data used here have been used in previous studies")
- unclear: 16508348, 29939345, 30937515

Pondie's own `docs/meta-analysis-queries.md` called 16371250 a gold-set omission. The overlap
with 16838824 (same fire, same 12 vs 12) is the likelier reason it was left out.

**Two criteria the schema cannot answer from these records:**
- *null effects*: `Cell.direction` is signed whether or not anything survived the threshold,
  and `Analysis` has no slot for "found nothing". Where the analysis is linked to the parse,
  zero foci can stand in for it.
- *overlapping samples*: `Group.sample_source = previously_reported` exists, but all three
  overlap negatives say `recruited`. Two never cite the earlier paper, so only a cross-record
  check (shared authors + same population) can find them.

### E3. Check-and-retry inside the single call

**Hypothesis H3**: most of what pondie's stage split buys is protection against a structurally
broken reply, and that can be had inside one call by holding the reply to pondie's own
deterministic checks and retrying with the failures named.

`Mono(check=True)`: up to 3 attempts. A reply is rejected for no analyses, for dangling or
duplicate local ids (`fix.check_local_ids`, with the deterministic `tables.json` included),
and, with the parse, for listing entries neither emitted nor declined
(`render.unconsumed_listing`). The fewest-failures answer is kept. Seeded from the E1 replies,
so attempt 1 is the same reply and only failing papers are re-asked.

- **22% of single-call replies fail a structural check** (12/55 `mono`, 9/55 `mono_parse`).
  The typical failure is analyses referencing model estimations and measures that the reply
  then emits as empty lists (19538748: 22 dangling references).
- Re-asking costs +12 and +9 calls on 55 papers (about +20%).

### E4. Criteria the query was missing

- *reported coordinates*: from the inputs (parsed points + prose coordinates), or, for an
  included paper, the gold foci (`--gold-coords`, permitted by the user). 18 of 33 negatives
  carry no coordinates in their inputs.
- *within-group effects*: a severity regression counts only if the analysed sample contains a
  non-PTSD cohort, and a contrast counts only on the levels the cells actually contrast. The
  first version read every level of a three-level group term, so "ASD < controls" in
  30937515 passed as a PTSD contrast.
- *null effects*: `reported foci` from the parse link; an explicit zero excludes.
- *the pooled direction*: `PTSD decrease` picks which analyses' foci are pooled. It is not a
  study criterion. Foci precision rose from 128/167 to 128/144.

### E5. Overlapping samples

**The benchmark is inconsistent about overlap.** It includes two re-reports of the same
people:
- Hunan fire (Nov 2003), 12 PTSD (8F/4M) vs 12 non-PTSD: 16371250 (Chen 2006, Jan; excluded),
  16838824 (Li 2006, Jun; gold), 19538748 (Chen 2009; gold), all reporting non-PTSD vs PTSD.
- Coal-mine flood (July 2007), 10 vs 10: 21498053 (gold) and 23155380 (gold). 23155380 adds
  20 unexposed controls and contributes a different contrast (HCs > PTSD).

Adjudicated rule: a paper is excluded only when **all** its cohorts were already reported in an
earlier paper. That keeps the coal pair (a new cohort), drops Nardo 2013 (a subset of 2010),
and keeps the first fire report: 16371250 in, 16838824 and 19538748 out. `gold.ADJUDICATED`
also removes the two a-priori-ROI papers (21118656, 19794316). Scores are given against both
`benchmark` and `adjudicated` labels.

The declared slot (`Group.sample_source = previously_reported`) cannot be a veto. Gold
17825801 truthfully says all four twin cohorts were reported before, but in a hippocampal-
tracing paper outside this pool.

**Correction: the first overlap implementation broke the design constraint.** It asked luna-6
to judge each pair of selected papers from their records' group fields. The user's hard
limit is that LLM calls only fill records and selection is a deterministic query. Replaced by
`overlap.py`: two papers sharing ≥2 PubMed authors, the earlier one selected, and every cohort
of the later paper fitting inside a same-status cohort of the earlier one (size ≤, each
reported sex count ≤). The later paper is then excluded. It makes **exactly the same
exclusions** the LLM judge did: 19538748 ~ 16371250, 23113800 ~ 19942229. It keeps 23155380,
whose 20 controls do not fit inside 21498053's 10, and Bossini 2017, whose PTSD cohort has more
men than Bossini 2012's. The cached judge verdicts were deleted; no number below uses them.

### Results after E3–E5 (veto mode, `--gold-coords --overlap`, 55 papers)

| arm | labels | recall | false pos | precision | gold foci (pooled) |
|---|---|---|---|---|---|
| `mono_check` | benchmark | 19/22 | 3/32 | 0.86 | n/a |
| `mono_parse_check` | benchmark | 17/22 | 1/32 | 0.94 | 130/159 |
| `mono_check` | adjudicated | 19/19 | 3/35 | 0.86 | n/a |
| `mono_parse_check` | adjudicated | **18/19** | **0/35** | **1.00** | 127/143 |

The 90/90 target is met by `mono_parse_check` on adjudicated labels, but with two caveats:
the query was written while looking at these 55 papers, and this is one draw. A fresh replicate
(`mono_parse_check-rep2`) is running to measure noise, and the next meta-analysis is the
held-out test of the method.

### E6. Replicate, and the overlap rule's first bug

`mono_parse_check-rep2` is a fresh, unseeded run of the best arm (66 calls on 55 papers).

| run | labels | recall | false pos | precision | gold foci |
|---|---|---|---|---|---|
| v1 | adjudicated | 18/19 | 1/35 | 0.95 | 127/143 |
| rep2 | adjudicated | 16/19 | 1/35 | 0.94 | 116/143 |
| v1 | benchmark | 17/22 | 2/33 | 0.89 | 130/159 |
| rep2 | benchmark | 16/22 | 2/33 | 0.89 | 119/159 |
| `mono_check` v1 | adjudicated | 19/19 | 2/35 | 0.90 | n/a |

**Noise is about ±2 papers out of 19.** What moves between draws is borderline-field
instability, not structure:
- 21418787's PTSD contrast present in v1, absent in rep2.
- 22453299: no analyses after 3 attempts in rep2.
- 19794316: rep2 found a whole-brain analysis v1 did not (see below).
- 32490056 misses its PTSD vs non-PTSD analysis in **both** draws: the record has one PTSD
  cohort and a social-support regression, though the title says "PTSD diagnosis". A
  consistent extraction miss.

**19794316 re-adjudicated to unscored.** Its pooled gold foci are hippocampus and ACC peaks
from AAL masks (ROI), but the paper also reports, in prose and without coordinates, whole-brain
"nonhypothesized" reductions. A paper with an ROI result and a whole-brain result is not
clearly against the criteria, so it is out of both recall and precision under
`adjudicated` (`gold.unscored`).

**The overlap rule had a hole.** rep2 excluded gold 21498053 (coal-mine flood) as a re-report of
16838824 (Hunan fire). 16838824's record gives no sex counts, so "10 men" fit inside "12 of
unknown sex". Now a later cohort may report *fewer* sexes than the earlier one (16838824 omits
what 16371250 reported) but not *more*.

**The persistent false positive is 30127342 (Gong 2019).** It states "subsets of the data used
here have been used in previous studies [13–16]", and the record says `previously_reported`
on every cohort, but the earlier papers are outside the pool. So no deterministic rule over
these records can exclude it without also excluding gold 17825801, which says the same thing
about a non-VBM paper. Answering it would need the cited reference resolved to a paper and
that paper's method, i.e. `sample_source_reference` as an identifier rather than a citation
string.

## Porting into pondie (branch `stage-ablation`; study_schema branch `analysis-outcome`)

**Constraint, from the user:** LLM calls only fill records; selection is a deterministic query.
Everything in `queries.py`, `pondie.query.overlap` and the PubMed fields obeys it.

1. **`StageName.single` / `stages.Single`.** The whole record in one call with the stage-1
   listing, held to the demands pass's listing checks (`unconsumed_listing`,
   `unsupported_omissions`) and a new `render.dangling_references` (built on
   `fix.check_local_ids`, with the `tables` stage's ids as `existing`). It runs the `shape`,
   `demands` and `satisfy` payload repairs. `stages.SINGLE_PASS` is selected when a run's
   `--stages` names `single`. **The default pipeline is unchanged** until the held-out
   meta-analysis agrees. Tests: `tests/test_single_pass.py`.
2. **`Analysis.outcome`** (`significant_effect | no_significant_effect`) in the storage
   schema, regenerated into extraction. It answers "null effects excluded" without the parse
   link. The query reads it first and falls back to the parse-link foci count.
3. **`pondie.query.overlap`**: the deterministic overlap rule, with the cohort status passed in
   by the query (default: derived `is_healthy`). **`pubmed.authorship()`** supplies authors and
   pubdate; the record has no slot for them, and `pubmed.fill` writes every key it fetches
   onto the record, so they are kept off it. Tests: `tests/test_query_overlap.py`.
4. `queries.py` is now a registry of per-meta-analysis `Spec`s. Cohort status got stricter
   in one place: a comparison cohort must read as a control (or be derived healthy), so
   "PTSD vs OCD" is no longer a PTSD contrast. It also reads the description when name and
   condition decide nothing (rep2's "non-symptomatic (NS)" … "did not develop PTSD").
   PTSD scores after the refactor: v1 unchanged; rep2 adjudicated 17/19, 0.94.

Note on pondie's `single` vs the harness's `mono_parse_check`: the stage also enforces
`unsupported_omissions`, so more replies are re-asked (many papers take 2–3 calls) and some
still fail after 3. It is a stricter, costlier variant, measured below.

## E7. Held-out: Dementia (bvFTD, Kamalian 2022, `35664889`)

The query (`queries.DEMENTIA_*`) was committed (4851a9e) from the published criteria before
any dementia record was extracted. Pool: 25 gold sampled from the 66 gold papers autonima
screened, plus 30 negatives (15 that autonima's full-text screener *included* but the
benchmark did not, 15 random), seed 0. Its gold studyset merges papers into lab blocks
(e.g. ~28 Kumfor/Irish/Hornberger papers as one study), so scoring is study-level only, and
an included paper's membership answers "reported coordinates".

**Held-out result** (`dem_pondie_single-v1`, pondie's `single` stage, query as committed in
4851a9e, benchmark labels, `--gold-coords --overlap`):

| mode | recall | false pos | precision |
|---|---|---|---|
| strict | 13/25 | 1/30 | 0.93 |
| veto | **16/25** | 1/30 | **0.94** |
| permissive | 17/25 | 2/30 | 0.89 |

108 calls, 6.15M input (4.84M cached), 813k output. Precision transfers from PTSD; recall does
not. Gold losses: "bvFTD vs control" False on 6, the modality predicate False on 3, one
record with no analyses. Diagnosis follows; every change from here is fit to this data and
reported as such.
