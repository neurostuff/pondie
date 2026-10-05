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

**Diagnosis of the held-out misses** (everything below is fit to the dementia data):

- *Query bug*: VBM's quantity is often typed `structural_morphometry_other` with the label
  "grey matter intensity", and the dementia modality predicate read only the type, which
  dropped 26401935's real bvFTD-vs-controls atrophy analysis. It now reads the label, as the
  PTSD predicate already did. Veto recall 16 → 17/25.
- *Query bug*: the window is "- 5/2020" and I coarsened it to the year. The only false
  positive, 32353756, was published July 2020. Precision 0.94 → 1.00. The window is now
  month-precise (PubMed pubdate via `pubmed.authorship`).
- *Benchmark structure*: 7 of the 9 gold misses sit in merged gold "studies" (the 28-paper
  Kumfor/Irish/Hornberger block, 634 foci; the Cousins/McMillan/Massimo block). The meta
  pooled each lab's data as one study.
- *Extraction misses, not benchmark problems*: the Sydney papers' main tables are
  **covariance** analyses (grey matter covarying with a score across patients + controls),
  and their bvFTD-vs-controls atrophy contrast is reported in the text with its coordinates
  only in the supplement (25797589 "t-tests (Supplementary Material)", 30718430 "SI
  Appendix, Fig. S3 and Table S4", 31873787 "Supplementary Table 2", 31461580 "Table S1",
  22952324 "supplement table E-1"). The single pass does not emit those as analyses. One
  paper (28304289, FDG-PET) declined to emit any analysis because "the detailed imaging
  results" were in supplementary tables, and held that through three retries naming the
  fault. Only 25009480 mentions no group contrast at all.

`dem_pondie_single-v2` adds to the single-pass note that a comparison whose coordinates are
only in a figure or supplement is still a tested effect. Veto recall stayed at 17/25,
precision 0.94: it recovered 22952324 and 28304289 but lost 23576128 and 23805313 to empty
records. Draw-to-draw variation is as large as the effect.

### E8. Why single-pass records come back empty

Records with no analyses, per run: `mono` 5/56, `mono_check` 0/55, `mono_parse` 5/55,
`mono_parse_check` 0/55 and 1/55, `ds` 2/23, **pondie `single` 6/55, dementia v1 6/55,
v2 4/55**. Two causes, both now fixed in pondie:

1. **The retry loop kept the last attempt, not the best.** pondie's `_ModelPass.run`
   overwrote `payload` on every attempt. The single pass holds replies to more post-
   conditions than the harness did (`unsupported_omissions` too), so a reply with sound
   analyses and one unvetted omission reason was re-asked, and an empty retry replaced it.
   That is why the harness, which kept the fewest-failures reply, had 0–1 empties against
   pondie's 6. Now the attempt with the lowest `(empty?, number of faults)` is kept. This
   applies to `demands` too. Regression test in `tests/test_single_pass.py`, verified to fail
   on the old code.
2. **The reply ended after the entity lists.** On 23576128 (29 listing entries) the model
   wrote `study` and every entity list, then closed the object with no `analyses` key,
   three times, finishing normally at ~8k output tokens each. That is the failure pondie's
   docs give for splitting into demands → satisfy ("a single call puts the analyses behind
   thirty-odd entity classes and drops them"). H4: asking for `analyses` as the **first
   key** gets the demands-first ordering inside one call. `dem_pondie_single-v3` tests H4
   alone (synced before fix 1); v4 will have both.

`dem_pondie_single-v3` (analyses-first only): veto 17/25, precision 1.00, **7/55 records still
empty**. **H4 not supported**: asking for analyses first did not stop replies ending without
them. Fix 1 (keep the best attempt) is the other candidate; v4 tests both together.

### E9. `fill` on top of `single`

`pondie_single_fill-v1`: a fresh `single` draw (the harness wrote no pipeline stamps, so
seeding could not reuse `pondie_single-v1`'s payloads; fixed in `run_arm.stamp`) plus `fill`.

| | adjudicated recall | precision | pooled foci | calls |
|---|---|---|---|---|
| `single` v1 | 18/19 | 0.90 | 117/143 | 117 |
| `single` (new draw) + `fill` | 14/19 | 1.00 | 116/143 | 188 (113 single + 75 fill) |

**Inconclusive on recall, clear on cost.** The single draw underneath had 7/55 empty records
(before the best-attempt fix), which is most of the recall drop. `fill` added 64% to the calls.
What `fill` is for, completing slots like `sex_distribution`, did matter to the overlap rule:
this run's Nardo 2010 record had sex counts, and the earlier one did not.

**Overlap rule: 2 shared authors was too few.** In this run the rule excluded gold 21498053
(coal-mine flood) against 16371250 (Hunan fire). They share two authors (Li L, Zhang J), and
the coal record gave no sex counts, so its 10 "fit" inside 12. Every true re-report in the PTSD
pool shares 3–5 authors. `min_shared_authors` is now 3 (fit to this data). Rescoring every run:
all true exclusions kept, the false one gone (single+fill 13 → 14/19).

### E10. v4: both fixes, on both meta-analyses

`pondie_single-v4` / `dem_pondie_single-v4`: the `single` stage with best-attempt selection and
analyses-first, fresh draws.

- **Empty records: PTSD 0/54, dementia 1/55** (from 6–7 per run). The best-attempt fix is
  what worked; analyses-first alone (v3) did not.
- Dementia veto **21/25 (84%), precision 0.95** (from 17/25).
- PTSD adjudicated 15/19 at first. Two query fixes (fit to this data) brought it to **17/19
  (89%), precision 0.89**:
  - 17825801 encodes its twin design as pair diagnosis, so the "PTSD" level holds the PTSD
    twins *and* their unexposed co-twins. A level holding any case cohort is now the case
    side.
  - 22453299's VBM contrast points at the paper's only declared acquisition, the fMRI one.
    The T1 was never emitted. A grey-matter or structural-morphometry measure now answers
    "structural MRI".
  - 21418787 this draw encoded "combined clinical groups (PTSD + depression) < controls"
    with a level label matching no declared level. Left as extraction variation.
- PTSD's two false positives are the known pair: Gong 2019 (overlap outside the pool) and
  Nardo 2013, whose 2010 record this draw names the trauma-exposed cohort just "NS", with no
  condition and no description, so its status is unknown and nothing can be matched to it.
  Not loosened further: that is how the coal/fire false match happened.

Previous runs rescored under the final query: PTSD adjudicated `mono_parse_check` v1 18/19 0.95,
rep2 17/19 0.94, `single` v1 18/19 0.90, v4 17/19 0.89; dementia v1–v3 17/25 at 0.94–1.00,
v4 21/25 0.95.

### E11. A pondie caching bug: every resume re-paid every model call

Seeding a run from another (to add one stage) re-ran `tables`, which made `single` stale, and
it re-asked the model. The harness was writing stamps correctly (`run_arm.stamp`); the stale
input was pondie's own.

`Tables` declared a dependency on the stage-1 parse file's hash. `prose` and `split` run after
it and rewrite that file. So on any resume `tables` is stale, its digest changes, and every
model pass downstream, `single` or `demands` + `satisfy`, becomes stale with it. In pondie's
own driver this means **resuming a run re-pays every model call.**

Fixed in 545896c: `Tables.depends_on` hashes the manifest and the parse's **table list**
(`source_tables()`, which excludes prose entries and survives a sign split), which is all it
reads. Regression test `test_tables_stays_fresh_after_prose_and_split_rewrite_the_parse`,
verified to fail on the old code. `restamp.py` re-stamped v4's unchanged outputs under the new
code, and the seeded `fill` and `evidence` runs now skip `single` (0 calls).

### E12. `fill` and `evidence`, stepwise, on the same `single` draw (PTSD)

Seeded from `pondie_single-v4`, so `single` was reused (0 calls) and each run differs from v4
by exactly one stage.

| added stage | adjudicated recall | precision | benchmark recall | precision | calls | input tok |
|---|---|---|---|---|---|---|
| — | 17/19 | 0.89 | 17/22 | 0.85 | 115 | 6.42M |
| `fill` | **18/19** | **0.95** | 18/22 | 0.90 | +83 | +1.39M |
| `evidence` | 17/19 | 0.89 | 17/22 | 0.85 | +55 | +0.67M |

- **`fill` earns its place for querying.** It settled every open slot it was asked about
  (e.g. "121 of 121"), and two of those slots decided the query. Sex counts on the Nardo
  cohorts let the overlap rule exclude Nardo 2013, and cohort links on 21418787's group term
  recovered its PTSD contrast. +72% calls but +22% input tokens (the paper sits in the
  cached system half).
- **`evidence` changes nothing the query reads.** It attaches quotes ("87 warranted, 107
  unsupported, 13 not_reported") and leaves values alone. It is for review and provenance,
  not selection, and should be judged on that.
- `single` + `fill` meets the target on PTSD: **18/19 (95%) recall, 0.95 precision**
  (adjudicated). The remaining miss is 32490056 (its PTSD-vs-non-PTSD analysis is never
  extracted), and the remaining false positive is Gong 2019 (overlap with papers outside the
  pool).

### E13. `repair`, `fill` on dementia, and two query corrections

**`repair` on top of `single` + `fill` (PTSD, seeded):** +110 calls, 1.26M input, 717 slots
written, 0 reported as introduced. Query unchanged: 18/19, 0.95 adjudicated, same foci.
At first it read *worse* (an extra false positive). `repair` gave Nardo 2013's "all subjects"
row a medical condition, and that 32-person row then fit inside no 2010 cohort, so the overlap
rule let the paper back in. Fixed in the rule, not in `repair`: a row named as a whole
sample ("all", "whole", "total", …) whose size is the sum of the other cohorts is the union of
the cohorts, not one of them. The name is required because the coal-mine paper's 20 new
controls also equal 10 + 10.

**`fill` on dementia (seeded from `dem_pondie_single-v4`):** veto 21/25, precision 0.95, and
**strict rises from 16/25 to 21/25**. `fill` completes the slots the strict query needs (group
sizes for "a group of six" were unanswerable before).

**Overlap is a per-meta-analysis criterion, not a scorer default.** Applied to dementia it
excluded gold 25009480 (Kumfor 2014) against Kumfor 2013. The dementia criteria never state an
overlap exclusion: the meta merged each lab's papers into one study instead (both are inside
the Kumfor block). `Spec.excludes_overlap` is True for PTSD only.

**The four remaining dementia misses all lack the contrast in the text we have.** 25797589,
30718430 and 31461580 describe a bvFTD-vs-controls atrophy contrast whose coordinates are in an
unfetched supplement. 26682697 reports only covariance analyses in its text, so its inclusion
rests on the lab's merged data. This is a corpus limitation (ns-pond does not fetch
supplements), not a query or schema one.

## Where it stands

**Best pondie workflow measured:** `tables → prose → split → single → fill → build`, then a
deterministic query (`queries.py` predicates, PubMed fields, `pondie.query.overlap` where the
criteria exclude overlap).

| meta-analysis | labels | recall | precision |
|---|---|---|---|
| VBM of PTSD (developed on) | adjudicated | **18/19 (95%)** | **0.95** |
| VBM of PTSD | benchmark | 18/22 (82%) | 0.90 |
| Dementia (query written blind, then fixed) | benchmark | 21/25 (84%) | 0.95 |

Stage verdicts (stepwise, same `single` draw where marked):
- `single` (one call, checks, best attempt) over `demands → satisfy`: more queryable at half
  the calls.
- `fill`: **keep**. +1 PTSD paper, +1 correct overlap exclusion; dementia strict 16 → 21.
  +72% calls, +22% input tokens.
- `evidence`: changes nothing a query reads. It is for review and provenance.
- `repair`: no change to the query, +110 calls. Not worth it for selection.
- `build` repairs: small, deterministic, keep.

Remaining errors: PTSD 32490056 (its PTSD-vs-non-PTSD analysis is never extracted, in every
draw) and Gong 2019 (overlap with papers outside the pool). Dementia: 4 papers whose contrast is
not in the fetched text.

## E14. Held-out #2: VBM of substance use (Hill-Bowen 2022, `36115222`)

The frozen workflow (`single` + `fill`) and a query committed from the criteria before any
record existed (bf24302). Pool: 25 gold from the 65 autonima screened, plus 30 negatives (15
autonima-included non-gold, 15 random), seed 0. Input ceiling: 148 of 195 gold foci in the parse.

**Held-out result, 52 of 55 records** (3 still running when first scored), benchmark labels,
`--gold-coords`:

| mode | recall | false pos | precision |
|---|---|---|---|
| strict | **22/25 (88%)** | 2/27 | **0.92** |
| veto | 22/25 | 3/27 | 0.88 |

167 calls, 6.37M input (4.14M cached), 1.07M output.

"no pharmacological manipulation" answers on all 25 gold (25/0/0). pondie's
`docs/meta-analysis-queries.md` found the old records could not answer it on 49 of 76 gold,
because `allocation: non_randomized` could not tell "assigned a drug non-randomly" from
"split by diagnosis". `StudyDesign.assignment_structure = observational_cohorts` and the
current extraction answer it.

**All 55 records.** Held out (before the fix below): strict 22/25, 0.92. After the one query
fix, fit to this data:

| labels | strict recall | false pos | precision |
|---|---|---|---|
| benchmark | 22/25 (88%) | 1/30 | 0.96 |
| adjudicated | **22/23 (96%)** | 1/31 | **0.96** |

Diagnosis:
- *Query fix*: 26000879 is adolescents with substance use **and conduct disorder**, and the
  criteria exclude participants with other mental disorders. The pattern lacked "conduct".
- *Benchmark, against its own criterion*: 20875635 (Kühn 2010) measures cortical thickness,
  and the criterion is "assessing GM volume differences". Adjudicated out.
- *Ambiguous, unscored*: 29065207's VBM is cerebellum-only (SUIT); a whole-brain DARTEL
  version is said to be in the supplement, and the pooled foci are cerebellar.
- *Record error*: 18165464 restricted its analysis with "an explicit mask created from
  Automatic Anatomic Labeling that limited the analysis to gray matter regions", a
  whole-brain grey-matter mask, and the record says `roi`. This is the scope defect in
  `docs/meta-analysis-queries.md` ("the record reads the names as a restriction").
- *Remaining false positive*: 29058369 (Bach 2019, alcohol dependence vs controls plus a
  genotype effect). No stated criterion excludes it that I can find.

## Change made: `single` is pondie's default (183dbdf)

`Settings.stages` now excludes `demands` and `satisfy`, so the default sequence is
`SINGLE_PASS`: tables, prose, split, single, fill, evidence, build, repair. The split still runs
when both are named in `--stages`. The CLI defers to that default instead of passing every
stage. README's stage table and `tests/test_models.py::test_the_pipeline_is_one_ordering`
updated.

## Summary across the three meta-analyses (`single` + `fill`, deterministic query)

| meta-analysis | how the query was made | labels | recall | precision |
|---|---|---|---|---|
| VBM of PTSD | developed on it | adjudicated | 18/19 (95%) | 0.95 |
| | | benchmark | 18/22 (82%) | 0.90 |
| Dementia (bvFTD) | written blind, then 2 fixes | benchmark | 21/25 (84%) | 0.95 |
| VBM of substance use | written blind, then 1 fix | adjudicated | 22/23 (96%) | 0.96 |
| | | benchmark | 22/25 (88%) | 0.96 |

Held-out numbers before any fix: dementia veto 16/25 at 0.94; substance use strict 22/25 at
0.92.

What still limits recall: contrasts reported only in unfetched supplements (dementia, 4
papers), an analysis never extracted in any draw (PTSD 32490056), and the `spatial_scope: roi`
misreading of a grey-matter mask (substance use 18165464). What still limits precision:
overlap with papers outside the pool (Gong 2019), and a few papers no stated criterion
excludes.

# Part 2: prompt experiments

Same method as the stages: one change per arm, against a fresh baseline draw, on a development
meta-analysis (**dementia**: most headroom, and its main failure looks like a prompt problem),
then winners checked on **PTSD** and **substance use**. Variants are in-process patches of
pondie's renderer (`prompts.py`, `run_arm.py --variant`); the stage, its post-conditions, the
build and the query are unchanged. `compare.py` gives one row per run, including **probes**:
dementia papers that failed in at least one earlier draw (22952324, 25009480, 25797589,
26682697, 28304289, 30718430, 31461580, 31873787, 23576128, 23805313). A variant aimed at a
failure should move its probes whatever the rest of the draw does.

Single-pass system prompt, ~50k tokens: schema 30.2k, conventions 9.8k, worked models 8.5k,
rules 1.0k, the single note 0.4k.

| id | change | hypothesis |
|---|---|---|
| B | current prompt, fresh draw | baseline; with v4, a noise estimate |
| H1 `no_worked` | drop the worked models | −8.5k tokens per call, no loss |
| H2 `no_conventions` | drop the conventions | −9.8k tokens; may hurt cells and direction |
| H3 `inventory` | a `results_inventory` list first: every reported comparison, wherever it is (text, table, figure, supplement), each mapped to an analysis or a reason | recovers the comparisons the pass skips (dementia supplement-only contrasts, PTSD 32490056) |
| H4 `cohort_condition` | every cohort states its condition, including absence | fewer unknown cohort statuses (Nardo "NS"; the overlap rule) |
| H5 `scope` | a grey-matter or brain mask is whole-brain; ROI is a priori named regions | fewer ROI misreads (18165464) |
| H6 effort `medium` | not a prompt change; the cheapest knob | everything |

Existing dementia rows for reference (`compare.py`, benchmark labels):

| run | veto R | P | strict R | empty | key T | probes | calls | in Mtok |
|---|---|---|---|---|---|---|---|---|
| v1 | 17/25 | 1.00 | 14/25 | 6 | 18 | 3/10 | 108 | 6.15 |
| v4 (best-attempt) | 21/25 | 0.95 | 16/25 | 1 | 21 | 6/10 | 106 | 6.12 |
| v4 + fill | 21/25 | 0.95 | 21/25 | 1 | 21 | 6/10 | +90 | +1.59 |

Round 1 (dementia, `single` + build): B, H3, then H1, H6.

## P1. Round 1 on full dementia: one draw per variant is not enough

| run | veto R | P | strict R | empty | key T | probes | calls | in Mtok |
|---|---|---|---|---|---|---|---|---|
| v4 (current prompt) | 21/25 | 0.95 | 16/25 | 1 | 21 | 6/10 | 106 | 6.12 |
| **B** (same code, fresh draw) | **17/25** | 1.00 | 16/25 | 2 | 17 | 3/10 | 100 | 5.74 |
| H3 `inventory` | 18/25 | 1.00 | 14/25 | 4 | 18 | 4/10 | 111 | 6.44 |
| H1 `no_worked` | 20/25 | 1.00 | 12/25 | 1 | 20 | 6/10 | 105 | **5.21** |

**B and v4 are identical code and differ by four gold papers.** Across the six full dementia
draws, seven gold papers are selected in some draws and not others (22952324, 23576128,
23805313, 25009480, 26682697, 28724588, 31887311), and two are never selected (30718430,
31461580). A single draw over 55 papers cannot detect a prompt effect smaller than that.
(Strict recall moves with "a group of six" answerability, which `fill` settles anyway.)

**Changed design:** a *flip panel* of the 16 dementia papers that varied or always failed (15
gold, 1 negative, `panel_dementia.pmids`), with 4 replicate draws per variant, scored by
`panel.py` as a selection rate over paper-draws (60 gold paper-draws per variant). About 30
calls a draw instead of about 100. `lanes.py` runs the draws two at a time. (It was first
named `queue.py`, which shadowed the stdlib `queue` that urllib3 imports, and every job died at
import.)

H6 (effort `medium`) is running as a full draw: flex calls take 5–28 minutes each at medium,
against about 4 at low, with 1–6k reasoning tokens per paper.

## P2. Why panel papers fail, and H7 (complete the missing references)

Of 55 gold misses on the dementia panel across 7 full draws, 40 are "bvFTD vs control =
False", 6 are empty records, 6 are "whole brain = False", 2 are modality, 1 is no single
analysis meeting all criteria. "bvFTD vs control = False" is two different failures:

1. **The group contrast is never emitted**: 22952324, 25009480 and 31461580 hold only
   covariance analyses in a failing draw. The supplement-only contrast, which H3 targets.
2. **The contrast is emitted and its cohorts are not.** 31887311 has a correct "bvFTD versus
   Controls grey matter intensity" contrast whose levels point at `grp_bvftd`, `grp_ad` and
   `grp_controls`, and the reply wrote `groups: []`, **in all three attempts**, with all 23
   dangling references named in each retry. Re-asking for the whole record re-writes
   everything and drops the entity lists again.

How often (2) survives the retries, per run: 2, 8, 8, 3, 7, 5 and 8 of 55 single-pass records
end with unresolved references; 0–6 end empty. It is the larger of the two.

**H7**: after the attempts, one call in `satisfy` mode for only the missing local_ids (kind
from the id prefix, the extracted analyses shown for context) resolves them where re-asking
for the whole record does not. In pondie as `Single.complete`, behind
`Settings.complete_references` (off by default); merged by local_id, never replacing an entity,
a model estimation gaining only the terms it lacked; kept only if no worse by `_severity`.
Tests in `tests/test_single_pass.py`.

Tested **paired**: `complete_existing.py` copies a run and applies only the completion to the
replies that need it, so the comparison is the same draw with and without it.

| draw | veto R | P | papers completed | references resolved | extra calls |
|---|---|---|---|---|---|
| `dem_p_B` | 17/25 | 1.00 | | | |
| `dem_p_B+complete` | **18/25** | 1.00 | 3 | 11 of 12 | 3 |

31887311 recovered. Five more paired draws (dementia v4, no_worked, inventory; PTSD v4;
substance use) are running.

**H7 across six paired draws** (completion only where references still dangled):

| draw | veto recall without → with | precision | extra calls |
|---|---|---|---|
| dementia B | 17 → **18**/25 | 1.00 → 1.00 | 3 |
| dementia v4 | 21 → 21/25 (strict 16 → 17) | 0.95 | 4 |
| dementia no_worked | 20 → 20/25 | 1.00 | 3 |
| dementia inventory | 18 → 18/25 | 1.00 | 4 |
| PTSD v4 (adjudicated) | 17 → 17/19 | 0.94 | 4 |
| substance use (adjudicated) | 22 → 22/23 | 0.92 | 4 |

The completion repairs records reliably: most asks resolve every missing id (21/21, 17/17,
10/10, …); two resolved nothing and were discarded; two papers' missing ids had no mintable
prefix, so nothing was asked. It changed one selection in six draws and lost none, because
most dangling references are not the cohorts a criterion reads. At about 3% more calls,
**adopted**: `Settings.complete_references` now defaults to True. The harness keeps
`--variant no_complete` for comparisons. Not synced to beast until the panel queue finishes,
so the later baseline replicates stay on the code they started with.

## P3. Interim panel, and what the inventory revealed

After 2 of 4 replicates (30 gold paper-draws each): B 0.63, `inventory` 0.53, `no_worked` 0.50
(SE about ±0.09, nothing separable yet). The three supplement-only papers (30718430, 31461580,
31873787) are 0/2 under every variant.

**The inventory shows the obstacle is a policy, not a miss.** Where it was produced, the model
*listed* the supplement contrast and declined it:

- 30718430: `supplement | analysis: null | "Gray matter intensity reductions in patients
  relative to controls …" | why: The cited supplementary figure and table are not available
  here`
- 31461580: `supplement | analysis: null | "Pairwise differences in cortical grey matter
  density between AD, bvFTD, and controls" | why: Table S1 and Figure S1 results are
  referenced, but their rows … are not`

So the model believes it may not emit a result whose numbers it cannot see. The schema already
says what to do: such slots take `not_reported` with `unreported_reason: outside_text`. Also, in
replicate 2 the inventory was not produced at all for these papers (0 entries).

**H3b `inventory_strict`**: inventory plus "required, never empty", and a result stated in the
text with its numbers in a supplement or figure ALWAYS gets an Analysis, its unseen slots
`outside_text`. Four panel replicates queued.

## P4. Panel results: no prompt change beats the baseline

Four replicate draws per variant on the 16-paper dementia panel (`panel.py`, veto selection,
benchmark labels):

| variant | gold paper-draws selected | rate | negative selected | calls/paper | Mtok/paper |
|---|---|---|---|---|---|
| **B** (current prompt) | 37/60 | **0.62** | 2/4 | 2.00 | 0.119 |
| H3 `inventory` | 33/60 | 0.55 | 3/4 | 2.28 | 0.136 |
| H1 `no_worked` | 28/56 | 0.50 | 1/3 | 2.17 | 0.110 |
| H3b `inventory_strict` (2 draws so far) | 16/30 | 0.53 | 0/2 | 2.28 | 0.136 |

Standard error about ±0.06 per row. **No variant beats B.** `no_worked` saves 8% of input tokens
and is, if anything, worse, so the worked models earn their 8.5k tokens. The inventory adds
14% tokens and 14% calls for nothing measurable. Under `inventory_strict`, 25009480 gained its
"bvFTD patients versus controls" analysis, but the record had no groups or measures (the
missing-reference failure; completion was off in panel draws), so it still did not select.

## P5. H8: two draws per paper, selected if either record qualifies

Per-paper selection on the unstable papers behaves like a coin with p ≈ 0.6, so two
independent draws should raise it to about 0.85. Computed from existing runs, no new calls:

| | single draws | union of two draws, every pair |
|---|---|---|
| dementia (`+complete` runs) | 21, 18, 20, 18 of 25 (mean 77%), P 0.95–1.00 | 22, 22, 23, 22, 20, 23 of 25 (mean **88%**), P 0.96–1.00 |
| PTSD adjudicated | 18, 17, 18, 17 of 19 | 18/19 in all six pairs, P 0.90–0.95 |

The largest gain of anything tried on the prompt side, at 2× extraction cost. Precision barely
moves because false positives are stable: the same papers in every draw. Selection stays a
deterministic query; it reads two records per paper and selects if either qualifies.

## P6. H6: reasoning effort `medium` (the first change that looks better than baseline)

One full dementia draw at `effort=medium`, otherwise identical to B:

| run | veto R | P | strict R | empty | analyses/rec | calls | in Mtok | out Mtok |
|---|---|---|---|---|---|---|---|---|
| B (low) | 17/25 | 1.00 | 16/25 | 2 | 5.0 | 100 | 5.74 | 0.77 |
| v4 (low) | 21/25 | 0.95 | 16/25 | 1 | 5.2 | 106 | 6.12 | 0.83 |
| **medium** | **22/25** | 0.96 | 16/25 | **0** | **6.4** | 99 | 5.73 | **1.27** |

On the 15 panel gold papers inside these draws: medium 12/15, v4 11/15, B 7/15 (the B panel
rate over four replicates is 0.62). Input tokens and calls are unchanged; output +65% (reasoning
1–6k tokens per paper). Flex wall time is far longer: 5–28 min per call against about 4, so a
55-paper draw took about 2.5 hours at 12 workers. Three panel replicates at medium are queued
to check that this is not one lucky draw.

`inventory_strict` final (4 draws): 34/60 (0.57), no better than B.

**Medium on the panel, three replicates:**

| | panel gold selected | rate | empty | calls/paper | input Mtok/paper |
|---|---|---|---|---|---|
| B (low), 4 draws | 37/60 | 0.62 | 2 | 2.00 | 0.119 |
| **medium**, 3 draws | **36/45** | **0.80** | **0** | **1.73** | **0.104** |

With the full draw's 12/15, medium is 48/60 (0.80) against 0.62, about 2 SE. It also needs
fewer retries, so fewer calls and less input per paper. Output (reasoning) rises, and flex
latency rises a lot. Papers that went from coin-flip to reliable: 25009480 (2/4 → 3/3),
26682697 (2/4 → 3/3), 31873787 (2/4 → 3/3), 22952324 (1/4 → 2/3), 25797589 (1/4 → 2/3). Still
0: 30718430 and 31461580, the supplement-only contrasts.

Confirmation on the other two meta-analyses: full medium draws on PTSD and substance use with the
current defaults (`single`, completion on, `fill`), plus a second low-effort substance-use draw.

### P6 continued: medium on PTSD and substance use, and the query bugs it exposed

First scores did not transfer (PTSD 17/19 0.94; substance use 20/23 **0.83**). The losses were
**query bugs that richer records exposed**. Medium extracts more per record (6.8–7.1 analyses
against 4.5–5.1, and longer condition lists), and three predicates failed on that:

1. **Negation.** Medium lists control cohorts' exclusions as conditions: "No current or past
   psychiatric disorders", "no major depression", "no history of … psychotic disorders". The
   "no other disorder" predicate matched the disorder words. Two substance-use gold papers were
   lost that way. Now `queries.asserted()` drops entries that state an absence ("no", "without",
   "never had", "free of", …) before the disorder and comorbidity predicates read them.
2. **A second disorder named beside the substance use.** 26000879's "severe substance and conduct
   problems" was skipped because the entry also matched a substance; a named other disorder now
   excludes regardless.
3. **"Sixteen" contains "teen".** PTSD 12853571 failed `adult` on "Sixteen Tokyo subway
   sarin-attack victims". `\bteen`.

After the fixes (no earlier low-effort score moved):

| | low effort (`single` + `fill`) | medium |
|---|---|---|
| dementia | 21/25, 0.95 (18–21 over draws) | **22/25, 0.96** |
| PTSD adjudicated | 18/19, 0.95 | 18/19, 0.95 |
| substance use adjudicated | 22/23 0.92; 21/23 0.91 | 22/23, 0.88 |

**Verdict on medium:** it helps where the miss is a *skipped* analysis (dementia panel 0.62 →
0.80), is neutral on PTSD, and costs one false positive on substance use, where more analyses per
record give a wrong one more chances to qualify. Output tokens rise about 60%, and flex latency is
several times longer. A reasonable choice when recall matters and latency does not; not adopted as
the default here.

**Lesson for the query layer:** a query has to be robust to a *better* record, not just a
sparser one. The negation bug was invisible until extraction got richer. pondie already has
negation tooling (`normalization._negation`, `is_healthy`'s triage); the per-meta predicates
should go through it rather than through ad hoc regexes.

## P7. Reasoning effort `high`

Three draws of the dementia panel at `effort=high`, completion off to match the B and medium
panels (43 of 45 gold paper-draws scored; two calls still running when recorded).

| effort | gold paper-draws selected | rate | panel negative selected | empty | calls/paper | output tok/paper | reasoning tok/paper | wall s/paper |
|---|---|---|---|---|---|---|---|---|
| low (B) | 37/60 | 0.62 | 2/4 | 2 | 2.00 | 17k | 0.8k | 182 |
| medium | 36/45 | 0.80 | 2/3 | 0 | 1.73 | 28k | 6.6k | 737 |
| high | 37/43 | 0.86 | **3/3** | 0 | 1.72 | **48k** | **12.9k** | 691 |

- **High edges medium (0.86 against 0.80), within one standard error.** Most of the gain came
  at low → medium. It is the first setting to select 30718430 in any draw (1/3); 31461580 is
  still never selected.
- **It costs about 1.7× medium's output and 2.8× low's.** Input and calls barely change, and wall
  time is no worse than medium (both dominated by flex queueing).
- **The panel's one negative (28474365) was selected in every high draw**, against half the
  time at low and medium. That is the precision risk medium showed on substance use, more
  analyses giving a wrong one more chances, and the panel has only one negative, so it cannot
  measure it. Full PTSD and substance-use draws at high would.

## The workflow as pondie runs it (CLI)

`pondie extract` now runs `tables → prose → split → single → fill → evidence → build →
repair` on flex, at the per-stage efforts (`single` and `repair` medium, `fill` and `evidence`
low), with one INFO line per paper: stages run and cached, calls, tokens, elapsed, ETA. On a
terminal a tqdm bar per paper is drawn below those lines; redirected, the lines are the
progress. Smoke run, 3 PTSD papers, 3 workers: 18 calls, 413k input / 93k output tokens,
11m41s; every stage ran, no failures.

Two cache bugs found by resuming it:
- `tables` depended on the whole parse file, which `prose` and `split` rewrite afterwards, so
  a resume re-ran every model stage (fixed earlier, 545896c). Resume now: 0 calls.
- `prose` and `split` both write the parse and shared one stamp file, so each made the other
  stale on every resume (cheap, but it made "N run" lines lie). Stamps are now named per
  step, and an unqualified stamp still counts for the step that wrote it, so existing runs
  keep their caches. Resume: 8/8 cached per paper.

## Record formatting on the 55-paper default run

`check_records.py` runs pondie's validator over every record (quote spans checked against
the paper text). The `ptsd_default` records: **0/55 valid, 573 errors, median 10 per
record**. Most errors were systematic, and code caused them rather than the model:
- Code-filled slots (`is_healthy`, the normalized demographics) were written `extracted` +
  `not_applicable`, a shape the schema forbids: 407 errors.
- `other_characteristics` was written but never declared on the extraction schema.
- The prompt asked for slots that code overwrites (`language`, `is_healthy`, …), and the
  model's wrapper survived where code didn't write one.
- Model shape slips: `tables` filed under `effect`, cells written as wrappers, `"direction":
  "not_reported"` (bare or wrapped), debris keys (`"}rayele"`, `":{"`).

These fixes run at `build`, so the same payloads could be rebuilt with no model calls:
**26/55 valid, 115 errors, median 1**. The remaining errors are content: ROI analyses or
corrections that never name their regions (85), and cell levels that don't match the term's
declared levels (16). Those are for the prompt or repair. Commits 092c852 (pondie) and
7fbf270 (study_schema).

## The adjudicator settles the validator's content findings

The 107 errors left after the formatting fixes were content, and repair's adjudicator only
knew one kind (a whole-brain scope beside named regions). It now also takes an ROI scope
naming no region (85), a cell level none of its term's declared levels spells (2), and an
effect kind its cells contradict (6), still in one call per record, at `repair`'s medium
effort. A level repeating the analysis's only group (`'PTSD group'` on a CAPS correlation
within PTSD) is dropped by code instead.

`adjudicate_only.py` over the 55 rebuilt PTSD records, no proposer sweep: 25 calls,
**107 → 36 errors, 27 → 40 records valid**. 57 ROI cases named their regions, 11 were
rescoped to whole brain, 11 left unresolved, 4 refused by the warrant guard, 2 quotes
rejected.

The first attempt validated better (42) and was worse: the prompt's "or the paper's own
words" came back verbatim as `definition_method`, and with no way to say "not named" the
model invented "Parcellation region 1 (prefrontal cortex)". With `not_reported` allowed and
"answer unresolved if the paper does not name them one by one", the regions are anatomical
names and the methods are the paper's ("manually traced", "FSL FIRST"). 13 quotes were also
rejected for tidying `(Figure  1 )` into `(Figure 1)`; the resolver now allows any spacing
beside brackets and punctuation.

Remaining: 14 ROI analyses (unresolved or refused), 6 levels on crossed `group × score`
terms typed continuous (a model-structure question), 6 kinds, and singletons.

## Whose slope a moderation's sign describes (schema change)

The 6 "declares no levels" errors were `level: 'PTSD'` on product columns. Two were unsigned
F-tests, where a level says nothing; `build` now drops it. Four were signed moderations ("age
negatively predicted GMV in PTSD youth"), where the level is the only thing saying which group's
slope the sign is: a product coefficient's sign depends on which level the difference is taken
from. The schema now lets a signed product cell name a level of its categorical component
(study_schema 'Let a signed product cell name whose slope its sign describes'), and §5.4's worked
model shows `level: PTSD group`, the referent paper's own reading. Adjudicated PTSD records:
36 → 32 errors, 40 → 42 valid, before a rebuild clears the two unsigned cells.
Narrowed the same day: only a product with a continuous component may carry the level. Two
factors cross in their own cells, so `build` now rewrites a signed product of two two-level
factors as crossed cells when the other factor's levels carry `order` (later positive). Both
30343133 group x time analyses came out as `PTSD ±, TD ∓, follow-up +, baseline -`, kind
`interaction`. Rebuilt PTSD records: no level errors left, 28 → 30 valid before adjudication.

## Models declared once and borrowed

30343133's three "kind" errors were not kind errors. The extractor wrote one seed's model in
full and left three sibling seed models with no terms, their analyses' cells naming the first
model's terms. The derivation read those unresolvable cells as plain signed cells
(`simple_effect`), the validator called that a contradiction with the stated `interaction`,
and the adjudicator was asked a question with no right answer ("kept interaction", 3/3).
Now: a cell outside its model chain derives no kind (the out-of-chain reference is reported
on its own), and `build` copies into an empty model the terms its analyses borrow from one
other model, scoped `<model>.<term>`. Only the terms its analyses cell, the uncelled
covariates, and their product components: 17923164 put the IES total, intrusion and
avoidance scores under one model, and copying all of it would have claimed each subscale
regression was adjusted for the others. Every empty model in four runs was a borrower (3, 9,
1, 1 per 55); all but the substance-use one (two donors) are filled. Out-of-chain + kind
errors: PTSD 9 → 6, PTSD medium 26 → 10, dementia 29 → 23. Rebuilt PTSD: 31/55 valid.
The substance-use "two donors" case was one: 21338692's correlation model borrowed `age of
first use` (declared by two models) and `years of use` (declared by one of them, which also
declares the other). The donor is now any model declaring every borrowed term, and several
are fine when they would give the same design (16701903's one `group` factor, declared twice).
All 14 empty models in the four runs are filled; substance use 19 → 17.

## Misnested structure, repaired from the schema

Correction to the last entry: the dementia crash was not "a reference written as an object" —
there are none in ~12,460 reference values over four runs. It was a FactorLevel written inside
its own `level` slot. A schema-driven census of the raw payloads (220 papers) found 61 pieces
of structure one level from where the schema puts them, in 20 papers. Three shape repairs now
decide placement from which class declares a key, replacing `rehome_misplaced` (analysis/effect
only) and `rehome_stray_tables` (tables only): `lift_misnested` (a parent's slot in a child; an
object inside its own slot), `rehome_keyed_entities` (an entity written as a key named after
its id), `unwrap_entities` (wrapper keys on an entity). After the build chain every such
case is gone; 21078704 keeps its 12 analyses instead of 1. Left reported: an analysis's slots
at study level (which analysis?), MRI slots on a PET acquisition (a type question), and
`omitted` under `study` in payloads written before it was hoisted at reply time.

## Seven more deterministic repairs, checked for information loss

Of 448 validation errors over 220 rebuilt records, ~10% were decidable from the record.
`compare_builds.py` builds the same payloads under two code trees and lists, per record,
validation errors, entity lists that shrank, and every leaf value outside evidence that
disappeared. Committed code → new: 448 → 408 errors, 113 → 119 valid, no list shrank, 8
values gone, all intended (`"arms"` debris ×2, `"inferred"` → generated, a moot
`unreported_reason`, and four duplicate copies whose identical values stay in place).

Dropping strays that contradict the value in place (the user's alternative) changes one
paper, 21078704: 6 fewer errors for 18 values lost, including an interpretation sentence.
Three of its "conflicts" were the same value with different evidence, which the first
version compared as different; compared by value they are duplicates. The three real
conflicts (`prespecification: exploratory` vs a sentence; `coordinate_space: MNI` vs a
description; a result sentence in `definition`) are kept both ways, reported.

A study-level copy of an analysis (19996042, medium run) is a second analysis with no id or
name; no rule recovers it, and the "only child lacking them" rule written for it fired on
nothing in 220 papers, so it was removed.

## S. Structured Outputs (strict JSON schema) instead of JSON mode

Every call so far used Chat Completions JSON mode (`response_format: json_object`): valid
JSON, no shape guarantee. The adjudicator's prompt never said "JSON", so the gateway refused
JSON mode on every adjudicator call and it silently ran unconstrained. The gateway accepts
strict `json_schema` for gpt-6-luna.

`prompt/reply_schema.py` generates one schema per reply shape from the extraction schema:
`single` (82–87 defs, ~700 properties, ~500 enum values, ~85k chars -- inside strict limits),
`fill` and `evidence` per call. Closed vocabularies become enums, open ones strings, a
type-designated class the choice of its subclasses with the designator fixed, a wrapper the
choice {extracted, typed value} | {not_reported, reason}. Strict mode requires every key, so
an optional slot is answered `null` and stripped on return. Raw `single` replies are now kept
(`payloads/<id>/raw/single.json`, before normalizing or repair); before this, the faults the
repairs fix were invisible after the fact.

Bugs found on the way: `Acquisition` was concrete, so the schema offered only
`acquisition_type: "Acquisition"` and no MRI slots (fixed in the generator, and the class
made abstract in study_schema -- its description already said so). And strict decoding
answers the forced reason key with `undetermined` for every absent slot (52 on one paper),
which `fill` then re-asks.

### One paper (19538748), configurations

| arm | evidence | analyses | extracted / not reported | evidence resolved | single out tokens |
|---|---|---|---|---|---|
| original (JSON mode) | quotes | 1 | 73 / 7 | 77%* | – |
| S0 structured | quotes | 1 | 104 / 52 `undetermined` | 93%* | – |
| S1 + silence | quotes | 1 | 94 / 48 `silent_default` | 91% | 9.9k |
| S2 + silence | indexed | 1 | 106 / 5 | 93% | 6.5k |
| S3 + silence | inverted | 1 | 127 / 66 | 73% | 10.5k (3x slower) |

\* after `fill` and `evidence`; S1–S3 are `single` + `build` only.

- An earlier draw of the original (`ptsd_default`) found 3 analyses here (VBM + two fMRI);
  this draw found 1 in every arm. The "structured loses recall" reading of S0 was variance.
- `silent_default` replaced every `undetermined`.
- Indexed (the paper shown as `[S12]` sentences; fields cite numbers): all 106 extracted
  fields cite, 155 citations over 36 distinct sentences, and 12 of 12 hand-checked citations
  state the value. A third fewer output tokens; quotes exact by construction.
- Inverted (top-level verbatim sentences, each naming the fields it supports): all 31
  sentences verbatim, 6 of 108 paths unresolved, but the model leaves many fields out of
  `support`, so less evidence resolves; and it took 3x as long.
- Raw JSON-mode replies hold the faults strict decoding cannot: 37 bare `"negative"`
  directions and 33 bare levels in five replies.

Next: indexed on the 20 papers against the original and S0, then the meta-analysis.

### Under strict decoding, the schema's key order is part of the prompt

S0 on 20 PTSD papers, read after 8 had finished: 3 came back with every entity list empty
(17892884: 0 analyses vs the original's 13; 26535944: 0 vs 8; 22453299: 0 vs 1) and others
lost most of theirs (21498053: 8 → 1). The records were "valid" -- empty records are -- so the
progress line looked healthy. Cause: strict decoding writes keys in schema order, and the
schema put `analyses` after every entity list, while `SINGLE_NOTE` says "write `analyses`
FIRST", because a reply that spends itself on entities first ends without them. Made to start
on `groups`, the model wrote `[]` and moved on. With `analyses` first, the same two papers:

| paper | original | S0 (groups first) | S1 quotes+silence | S2 indexed+silence |
|---|---|---|---|---|
| 26535944 | 8 analyses | 0 | 10 | 11 |
| 17892884 | 13 | 0 | 9 | – |

S1/S2 cover every result the original found on 26535944 and split the three-group comparisons
more finely. On 17892884 S1 missed four secondary analyses (adjusted hippocampus; the
group × gender ANCOVA). S2 cited sentences for all 366 values, none out of range.

The S0 and first indexed arms were stopped once this was found (lesson saved: read the first
outputs of any run over ~10 minutes).

### Baseline: the original pipeline on the 20 papers

Recall 9/10 and precision 0.90 (1 false positive of 9) against the adjudicated labels;
10/20 records valid (52 errors); 104 calls (single 35 -- about 15 retries --, fill 34,
evidence 35), 689k output tokens (single 468k).

### The evidence stage discarded `single`'s quotes

`Evidence` asked the model for a quote for every extracted field and overwrote what the field
carried, so every quote `single` wrote (a large share of its output) was thrown away and paid
for again. It now keeps a field's own evidence when every quote resolves to the text `build`
reads, and asks only for the rest. In the indexed format `fill` also cites sentence numbers.

### 20 papers: structured arms against the original

S1 = strict schema, analyses first, explicit silence, quotes. S2 = the same with indexed
evidence. Both ran with a bug, since fixed: the schema wrote omissions as `entry` while the
listing check reads `key`, so every recorded omission read as an entry ignored and `single`
was re-asked. S2 was stopped at 19/20 (17825801, a gold paper, stuck in those retries).

| arm | recall | precision | foci recall | foci precision | calls / paper | out / paper |
|---|---|---|---|---|---|---|
| original | 9/10 | 0.90 | 30/33 | 57/89 (64%) | 5.1 | 33k |
| S1 quotes | 10/10 | 0.91 | 30/33 | 57/71 (80%) | 3.4* | 38k |
| S2 indexed | 9/9 | 0.90 | 22/26 | 25/31 (81%) | 2.2* | 28k |

\* inflated by the omission bug. Calls and tokens over the 16 papers all three finished.

- S2 first scored 8/9: gold 22453299 failed "PTSD effect". The query's case pattern read the
  control group "veterans negative for PTSD" as a PTSD cohort (its condition was empty, so
  the name decided). A query bug, not an extraction one: `NEGATED` now includes "negative
  for PTSD". Every arm rescored.
- On the 16 shared papers: analyses 84 / 92 / 100 and final values 4,177 / 4,352 / 4,576
  (original / S1 / S2).
- Hand review of S2: of ~40 citations read, 1 wrong, a few weak or padded with a second
  sentence; automated "mismatches" were numbers written as words or derived counts. Links:
  inference settings 100% vs 65%, ROI regions 94% vs 42%, no dangling references; fewer
  table links, mostly correct omissions (a localizer ROI, duplicate prose peaks). Faults:
  one duplicated ROI analysis, one lumped four-measure analysis.
- Explicit silence closes the slots `fill` would re-ask: S1/S2 papers often took one call in
  total, but on some papers the original's `fill` adds 50–140 values the single pass does
  not. S2r (indexed, no silence) tests whether that second look is worth its calls.

Removed after measurement (recoverable from git before the "Prune the S2r workflow"
commit): the `inverted` evidence format (lower evidence coverage, 3x slower) and
`explicit_silence` (29% of its `silent_default` claims contradicted by a value the original
pipeline found, and it closes the slots `fill` re-asks). S1 and S2 above used silence.

### S2r: indexed evidence, no silence, `fill` re-asks (20 papers)

| arm | recall | precision | foci precision | analyses | values | validation errors | calls |
|---|---|---|---|---|---|---|---|
| original | 9/10 | 0.90 | 64% | 121 | 5,666 | 52 | 104 |
| S2r | 10/10 | 0.91 | 76% | 137 | 6,765 | 30 | 65 |

No S2r reply needed a shape repair. `evidence` kept 97.8% of its quotes as `single` cited them
and added the other 2.2%.

### Judging a reply after the repairs it will get anyway

Retries of `single` came from cross-reference failures, some of which the deterministic
repairs fix at no cost (a term declared once per model, a reference with one possible
target). The postcondition now judges a copy after this pass's repairs and the merge
repairs (`_ModelPass._judged`); the payload itself is still repaired once, where it always
was. Replayed on the 20 S2r replies, 2 of the 4 that were retried pass on the first attempt.

The first 55-paper launch on this code retried 9 of its first 9 papers on
`coordinate_sets[i].analysis -> unknown local_id`. The cause was `derived_ids`, which
renames an analysis to an id from its parse key (`ana_ptsd_control_gm_density` ->
`a_prose_1`). It followed only `mirror_of`, on the premise that nothing else references
an analysis by id, but `CoordinateSet.analysis` does. The 20-paper replies carried no
coordinate sets, so the replay missed it. The judged copy exposed it, but the stored
record dangled too. It now renames through every reference slot whose range is
Analysis. The run was stopped (kept as `ptsd55_s2r_aborted_derivedids`) and relaunched.

### Regression check: 23021615 on the current workflow

S2r + `fill` + `evidence` + `repair`, against the original and the 20-paper S2r record:
one `single` attempt, 5 calls, 126k in / 23k out.

- The four S2r analyses are unchanged in kind, outcome, scope and cells.
- Two new analyses, both true: the hippocampal ROI regression ("no significant
  correlations with any of the above variables and hippocampal volume") and the
  hippocampal group comparison ("no significant difference in hippocampal volumes between
  the PTSD and non-PTSD groups"). Values 226 -> 244; evidence spans 405 -> 446.
- `prespecification` is now `planned` on all six, where S2r left most empty.
- One validation error: the null group comparison is `kind: contrast` with both cells
  `undirected`, which derives `omnibus`. The adjudicator kept `contrast`, but nothing
  rewrites the cells, so the contradiction stands. A contrast's cell sign is its weight,
  not the result's direction; the model left it undirected because the effect was null.

## Post-mortems

### `derived_ids` left coordinate sets on the old analysis id

**What happened.** The repair renames each analysis to an id from its parse key and
repointed only `mirror_of`. `CoordinateSet.analysis` (study_schema `eaf4f94`, 2026-10-02)
is a second reference to an analysis. Every coordinate set then dangled. The first
55-paper launch retried `single` on 9 of its first 9 papers for it, and the stored records
dangled too.

**How it came to be.** The premise "`mirror_of` is the only pointer at an analysis" was
true when written (2026-08-31). It sat in a docstring, a code comment and a test comment,
and the code hard-coded it. Adding a reference slot to the schema changed no code, so no
check failed. The same schema change made three more claims stale: "`source_table_analysis`
is the only exact route to an analysis's coordinates" (two docstrings and the `single`
prompt).

**Why it was not obvious.**
- The error named the model's own id (`ana_morphometric_ptsd_control`), so it looked like a
  model fault. The retry note even sent it to the model as one.
- Nothing said a repair had renamed that analysis.
- No earlier reply had carried coordinate sets, so no run and no test had both together.

**What changed so it cannot recur.**
- `walk.repoint` takes the reference slots from the schema. `derived_ids`,
  `fill_empty_models` and `scope_duplicate_terms` all rename through it.
  `scope_duplicate_terms` used to rewrite any matching string in any slot.
- `apply_all` counts dangling references around every repair. It names any repair that
  adds one, in `RepairLog.introduced` and a warning. Here that would have said
  `derived_ids` on the first paper.
- Rebuilding 75 records (`ptsd_default`, `ptsd20_s2r`) with both changes gave identical
  errors and values. No repair introduces a dangling reference on them.
- The stale "only route" docstrings were corrected. The prompt sentence is unchanged
  while the 55-paper run uses it.

**Left open.** Nothing in pondie reads `coordinate_sets` (the model emits them because the
strict schema includes the list). In the first 17 sets of the 55-paper run, 14 agree with
their analysis's `source_table_analysis`. All 3 that differ are from 28549317, keyed by
the table's `local_id` (`tbltable2#1`) where the analyses carry the parse key (`823#1`).
One of them links an `anchor` set to an analysis, which the schema allows only for
results. No rule checks that the two joins agree.

### `single` retried on table references no retry could clear

**What happened.** 26952803 spent two full `single` attempts on `Analysis.tables:
["table2"]` before a third left the slot out. That cost about 6 minutes on flex and two
full-prompt calls.

**Upstream cause: the tables were lost at ingestion.** The `old-corpus` text keeps each
table's caption and footnotes but no rows. Its `tables.jsonl` is empty and the parse is
`autonima/empty`. PubMed has no PMC copy (no PMCID). Two of the 55 PTSD papers are like
this:

| paper | tables lost | in the meta-analysis |
|---|---|---|
| 26952803 | 3, two of them coordinate tables | yes |
| 22948482 | 6, including Table 5's VBM findings | no |

Their coordinates cannot be recovered from this corpus.

**Why retries could not succeed.** With an empty parse the prompt has no table listing,
so no `[no table local_id — OMIT tables]` line. The model sees "Table 2" in the text and
cites it. The retry note says "every local_id you reference must be an entity you emit",
but `single` cannot emit a Table; only the `tables` stage declares them.

**How it came to be.** The same reasoning was written down once, for *declarations*
("Tables are copied from the pubget manifest, never extracted, so a declaration naming one
asks this pass for something it is forbidden to emit. Demanding it spends the whole retry
budget on a fault no retry can clear"). It was never applied to *references*. The fact
that Tables come from code lives in a comment and a `startswith("tbl")` test, not in one
place both checks consult. The `tables` stage even noted "no Table records, so no
`Analysis.tables` target exists", but no code reads stage notes.

**What changed.** `settle_table_references` (merge repairs, so `_judged` applies it too)
works on any reference whose range is Table. It repoints to the only declared table
carrying the reference's number, keeping supplementary tables (`S2`) apart from main ones
(`2`), and drops the reference otherwise. It waits for the `tables` stage: until
`table_map` exists, a table may yet be declared. On 75 rebuilt records, dangling table
references went from 6 to 0: all six were 17892884's `table_2`, repointed to `tbltable2`.
Nothing else changed.

**Noted.** The `Validator`'s error count does not include dangling references
(`check_local_ids` reports them separately, in the build report's `dangling`). So
17892884's 12 errors did not move when its 6 dangling references were fixed.

### The adjudicator "kept contrast" and left the cells contradicting it

**What happened.** Three null group comparisons in the 55-paper run were stated `contrast` over
two `undirected` cells, which derive `omnibus`: 23021615, 22952599 and 28549317. Each says
the groups did not differ. The adjudicator answered `contrast`, the report said "kept
contrast", and the record still contradicted itself.

**Which half is wrong.** The cells. `undirected` marks a test with no per-level sign (an F or
χ² over the factor). A t or z comparison that found nothing still had a sign, and when the
paper does not print it the cell is `not_reported` (extraction-readme.md §2), which derives
`contrast`.

**How it came to be.** A contradiction between two halves of the record was put to the model
as a question about one slot, `effect.kind`. Answering the stated kind wrote nothing. No
option could ever reach the cells.

**Why it was not obvious.** "kept contrast" reads as a resolution, and nothing recomputed the
contradictions afterwards.

**What changed.**
- A `Case` now says what else each answer writes. When every cell is `undirected` and
  withholding their signs would derive the stated kind, the question explains the two
  readings, and answering the stated kind marks those signs `not_reported`.
- After adjudication the contradictions are recomputed. An answered case still standing is
  reported as "still contradicted after the answer", not counted as settled.
- Re-adjudicating the three records (3 calls): all answered `contrast`, the cells became
  `not_reported`, and all three validate with nothing introduced.

**A bug I introduced along the way.** The first version wrote the withheld sign as a
hand-built `{"extraction_status": "not_reported"}`. The schema requires
`evidence.status: not_applicable` on such a field, so each rewritten cell failed
validation (2 new errors per record). `values.wrap(None, ...)` is the constructor that
encodes that invariant. My unit test checked only the derived kind, and
`adjudicate_only.py` skipped the `Validator.diff` that `stage.run` performs. Now the
wrapper goes through `values.wrap`, the test validates the record it produces, and
`adjudicate_only.py` prints what an adjudication introduces.

**Upstream.** The model writes `undirected` for a null result, reading "no direction in the
result" as "no direction in the test". It does so for slopes too: 23021615's four null
regressions each carry one `undirected` slope cell. That derives a regression, so no
contradiction flags it and only the prompt can correct it. The enum's own description
never reaches the prompt (value descriptions are not rendered); extraction-readme.md §2
does. §2 now says a null t or z result is `not_reported`, for group cells and for slopes,
and `undirected` is only for an F or χ².

**Tested** (`runs/nullsent`: `single` + `build`, one draw each, on the three papers). Every
unsigned cell came back `not_reported`, 13 in all (group comparisons and slopes), and none
`undirected`. Before: 3 group comparisons and 4 slopes were `undirected`. None of the three
records derives `omnibus` any more. 23021615 found 14 analyses, against 8 in the 55-paper
draw. The records' ROI-without-regions errors (11) are not new: the 55-paper `single`
payload had the same gap (4 on 23021615), and there `fill`/`repair` named the regions. The
sentence is on study_schema PR #15 (`965f7b9`).

## 55 PTSD papers on S2r + repair (`runs/ptsd55_s2r`, code `ce94af0`)

Scored as the headline numbers were (`--labels adjudicated --overlap --gold-coords`; the
queued script left out `--overlap --gold-coords`, which turned two overlap exclusions into
false positives):

| run | veto recall | precision | strict recall | foci recall | foci precision |
|---|---|---|---|---|---|
| S2r | 18/19 | 0.95 | 17/19 | 128/143 | 155/173 (0.90) |
| original (`ptsd_default`) | 18/19 | 0.95 | 17/19 | 128/143 | 159/182 (0.87) |

Same selection. The miss is 32490056 (its PTSD vs non-PTSD analysis was not extracted in
this draw either); the false positive is 30127342 (overlap with papers outside the pool).

**Final records, validated as `build` validates (`final_errors.py`):** 43/55 valid, 30
errors. The run log's "records valid 36/55" counts `build`, before `repair`. By class: 8
cell levels on level-less terms (7 are 21592738's `continuous`), 6 null contrasts
`undirected` (fixed upstream since), 6 ROI analyses with no regions (all total-volume
comparisons whose cited `whole_brain` answer the warrant guard refused), 4 terms the model
does not reach (17825801), and a few one-offs.

### Fixes from those errors

- `drop_redundant_cell_levels`: a level that restates the term's type (`continuous`) is
  dropped. A level saying more than the term (19914045's `left amygdala volume` on a
  `medial temporal volumetric measure`) is kept.
- `refuses_losing_the_warrant`: an adjudicated answer may swap one of the case's own options
  for another when it brings a verified quote. A free-text value outside the options
  (12853571's compound scope) is still protected.
- `complete_partial_models` (new): a model gets the celled terms it lacks from the model
  declaring them. Only those terms and their components are copied, not the donor's
  covariates (17825801: `mod_age` lacked `trm_exposure`).
- `scope_duplicate_terms` moved to the merge, after the models are filled. Inside `single`,
  analyses on still-empty models named the bare id, the rename was reverted, and the
  duplicate reached the record.

**A regression I introduced, caught by the rebuild diff.** `complete_partial_models` first ran
before `cell_terms`. 19794316's whole-brain model declares its own `group` term, and its
cells named the ROI model's `trm_group`. `cell_terms` repoints such a cell to the
same-named term in scope; run first, the new repair copied `trm_group` in beside the
model's own term, in 5 records. How it came to be: I placed the repair by what it depends
on (`empty_models`) and not by what it must not pre-empt. Why it was visible: rebuilding
130 stored records and diffing every leaf value showed renamed term ids on records whose
errors did not change, which made no sense for a fix aimed at one paper. Now `cell_terms`
runs first, a test holds the 19794316 shape, and only 17825801 and 21592738 change (errors
170 -> 159 over the 130 records).

**A second slip of mine.** I committed with a failing test: `pytest | tail` hid the exit
status from `&&`. I now use `set -o pipefail` before chaining.

### `fill` ran unconstrained on a quarter of the papers

"gateway rejected response_format; retrying without JSON mode" appeared 14 times in the
55-paper run. Strict decoding allows 1,000 enum values per schema. A 250-row `fill` batch
repeated the five unreported reasons, and each closed vocabulary, in every row. 14 of 55
first batches exceeded the limit (up to 1,278 values), matching the 14 rejections exactly.
Each distinct enum is now defined once under `$defs`. Checked live: 16701903's 250-row
schema is accepted and answered in full.

### Query: negations read per clause

`asserted()` dropped a whole `medical_condition` entry for one negated clause, taking the
disorder its other clauses assert with it. Across the PTSD, dementia and substance-use
records there were 46 mid-string negations, e.g. "17 survivors were diagnosed with
recent-onset PTSD; ... survivors without PTSD", and "A subgroup of the 40
amphetamine-dependent patients; ... without a diag...". It now reads per clause. Rescored
dementia and substance use (the two queries that use it): no selection changes.

### Tables lost at ingestion, in every pool

The `old-corpus` route lost table bodies, keeping captions and footnotes only, in every
pool, not just PTSD. Papers with table headings in the text, no manifest and an empty parse:

| pool | papers | gold among them |
|---|---|---|
| PTSD | 6 | 26952803 |
| dementia | 6 | 28724588 (one heading, likely a text mention) |
| substance use | 2 | none |

A negative whose coordinate table was lost fails `reports coordinates` for the wrong
reason. Under `--gold-coords` a gold paper passes that criterion on its gold foci whatever
its inputs say. That asymmetry flatters precision; see the review below, item 1.

### Review of this journal for untested assumptions (subagent, 19 items)

The most consequential, and what was done with each:

1. `--gold-coords` makes `reports coordinates` depend on the label: a gold paper passes on
   its gold foci, a negative only on its inputs. Every precision figure rests on it.
   *To do:* rescore inputs-only for both classes and list what flips.
2. The shipped defaults (single and repair at medium, evidence and repair in the pipeline)
   were never scored as a whole. *Now running:* full S2r + repair draws on PTSD
   (`ptsd55_s2r2`) and dementia (`dem55_s2r`); substance use next.
3. The headline tables predate many repairs. *Done for PTSD:* rebuilt and rescored.
4. The stage verdicts come from single draws inside the ±2-paper noise. *Open.*
10. `asserted()` too broad. *Done (above).*
14. S2r rested on 20 PTSD papers. *Done for 55 PTSD; dementia running.*
17. Two joins link an analysis to its coordinates, unchecked. *Open.*
18. "Valid" ignored dangling references. *Done:* `final_errors.py` reports reference
    problems and a "clean" count.
19. Lost tables were checked in PTSD only. *Done (above).*

Others (open): the panel's standard errors treat repeated papers as independent; the query
fixes were each fitted to one paper; the overlap thresholds were fitted to about 3 pairs;
the adjudicated labels are one rater's; `Analysis.outcome`'s accuracy is unmeasured; the
recall effect of `_judged` accepting first attempts is unmeasured.

**Item 1 measured on PTSD.** Without `--gold-coords` (coordinates from inputs for both classes)
veto recall goes from 18/19 to 17/19, and precision from 0.95 to 0.94. The paper that flips
is 26952803, the gold paper whose tables were lost at ingestion. No negative changes. So on
PTSD the flag compensates for one ingestion loss and admits no false positive. The negatives
excluded by `reports coordinates` alone are 16701903 (a null result) and 19914045 (a priori
ROI), each with a reason of its own in the negatives list above. Both scorings are now
reported for each meta-analysis.

## The PTSD rerun (`ptsd55_s2r2`, code `d11af84`: null-result sentence, adjudicator fixes)

| run | veto recall | precision | foci recall | foci precision |
|---|---|---|---|---|
| `ptsd55_s2r` | 18/19 | 0.95 | 128/143 | 155/173 |
| `ptsd55_s2r2` | 18/19 | 0.95 | 117/143 | 139/156 |

Inputs-only scoring, without `--gold-coords`, gives 17/19 at 0.94 for both, the loss again
being 26952803. The foci drop is one paper: **21418787 went from 11/12 gold foci to 0/12.**
Its analysis `a_803_1` "Combined PTSD and major depression groups < Controls (brain volume
reduction)" carried cells PTSD +, major depression +, control −. That is the reverse of its
own name and of the parse's entry, so it left the PTSD-decrease pool. The first run had it
right.

**Name against cells.** A new adjudicator case fires when an analysis's name states `A < B`
or `A > B`, every signed cell's level names one side only, and all of them carry the
opposite sign. P-value thresholds ("P < 0.01") are excluded. Across ~380 stored records it
fires on exactly two analyses, both in this rerun:
- 21418787: the signs were wrong.
- 30343133: the name was wrong. The parse titles that row group "PTSD > TD", the cells
  agree, and the model renamed it "PTSD < TD".

The answer is 'name' (reverse the signs) or 'cells' (reverse the name's comparison).

**The first version of the question got 30343133 wrong.** Asked only "which does the paper
report", the model quoted the *other* contrast's result ("sustained decreases in GMV ... in
youths with PTSD") and reversed the cells. That turned the null PTSD > TD contrast, an empty
row in Table 3, into a copy of TD > PTSD. Telling it to quote what states the direction of
*this* comparison, and that a null contrast keeps what was tested, did not change the
answer. Saying where the analysis is ("It is row group 3 of Table 3") did. Both are now
right: 21418787's signs are reversed, and 30343133's name becomes "PTSD > TD".

Lesson, for the adjudicator generally: a verified quote proves the sentence exists, not that
it is about the case. A case should locate its subject as precisely as the record allows.

Whether the null-result sentence caused 21418787's flip cannot be told from one case. In
three earlier runs no such reversal occurred, so the dementia draw (same sentence) is
watched for it.

**Review item 13 (`Analysis.outcome` against linked foci).** Over the linked analyses of the
two 55-paper PTSD runs: 107 `significant_effect` with foci > 0 in each run, and 13 and 10
`no_significant_effect` with foci > 0. No analysis says significant with zero foci. Spot
check on gold 17923164: `prose#2`'s two peaks are printed, but the sentence says the
difference "did not survive a SVC", so `no_significant_effect` is right and the foci count
alone would have been wrong. Reading `outcome` before foci is supported here. The remaining
disagreements are not hand-checked.

### More from the rerun's errors

- **Hyphenated levels.** `fold_label` now treats hyphens and dashes as spaces: 17825801 wrote
  `combat exposed` against the declared `combat-exposed` on 12 cells. A longer name
  (`PTSD twin pairs` against `PTSD`) is still not a spelling, and is still reported.
  Rebuilding 185 records changed only that paper (24 -> 12 errors).
- **Coordinate-set keys.** `rekey_coordinate_sets` maps a set keyed by its Table's id
  (`tbltable2#1`) back to the parse key (`823#1`) through `table_map`, and only to a key the
  parse has. It runs beside `single` ("demands"), so it applies to new draws, not rebuilds:
  `build` runs only the merge repairs.
- **Kept `whole_brain` left its regions.** The adjudicator's clearing of `regions` ran only
  when the scope changed. 33169525's five analyses and 15734342's one were answered
  "kept whole_brain" and still contradicted. This is the second consequence found running
  on the write path only, after "kept contrast" over undirected cells. Both now go through
  one `_consequences` step that runs whichever way the slot went, and the re-check after
  adjudication is what exposed this one.

The dementia draw's failed attempts so far are references to models and terms declared
nowhere (`mod_dti_fbi_regression`, `trm_lobule_vi_volume`). That is a real omission, for the
retry or the completion call.

**Review item 15 (does `analyses` first under strict decoding bring back dangling
references?).** Calls by stage over 55 PTSD papers, from `usage.jsonl`:

| run | single | fill | evidence | repair | total |
|---|---|---|---|---|---|
| original (JSON mode) | 101 | 84 | 89 | 107 | 381 |
| S2r | 66 | 72 | 14 | 119 | 271 |
| S2r rerun | 71 | 78 | 15 | 127 | 291 |

`single` needs fewer attempts under strict decoding (1.2-1.3 per paper, against 1.8), so it
does not. `repair` is now the largest consumer, 44% of calls. Its proposer sweep was judged
"not worth it for selection" on an older pipeline, and has not been re-measured on S2r.

**`repair` and selection, re-measured on S2r.** Scoring each PTSD draw's `unrepaired/` records
against its final ones gives identical selection and foci in both draws (18/19, 0.95; foci
128/143 and 117/143). `repair` buys record validity (36 -> 43 of 55 valid on the first draw)
and nothing in selection, for 44% of the calls. Kept: the adjudicator settles the
validator's contradictions, and the sweep fills real gaps (27082610: regions, an assessment).

## Dementia (bvFTD, 35664889) on S2r + repair (`runs/dem55_s2r`, code `5561211`)

The first held-out check of the S2r workflow; nothing was tuned on this pool today.

| run | veto recall | precision | strict recall |
|---|---|---|---|
| S2r | 23/25 | 0.96 | 22/25 |
| medium-effort baseline (`dem_p_effort_medium`, JSON mode) | 22/25 | 0.96 | 16/25 |

Misses: 30718430 (`bvFTD vs control`=F, the supplement-only contrast that has failed in every
draw) and 31461580 (`no comorbidity`=F). 31873787 passes veto only (`a group of six`=None).
The false positive is 28474365.

**Inputs-only, without `--gold-coords`, dementia drops to 16/25 (precision 0.94).** Eight gold
papers carry no coordinates in their inputs: 0 parsed points and 0 prose coordinates,
though their text describes voxel-wise VBM results. Four have empty parses; four have table
manifests (`catalog/ace`, `pubget`, `elsevier`) whose captured tables are demographic or
empty. Their coordinate tables never reached the corpus. For dementia `--gold-coords` means
"being in the included set counts", so 23/25 is a claim about the extraction given complete
inputs, not about the corpus as fetched. The gap is ingestion.

**Final records:** 44/55 valid, 24 errors. Fixed from them:
- **A p-threshold was read as a comparison.** `direction.polarity` read "FTD-MND compared
  with FTD at P <0.001" as FTD < 0.001, and `fill_directions` signed both of 10526199's
  levels negative (also 28219620's "at p<0.05"). A left side ending in a lone p/q, or a
  right side with no letters, is now no comparison.
- **The pass's `ambiguous` was overwritten.** `fill_directions` filled a `not_reported`
  direction that carried a reason, in place, so the wrapper kept its `unreported_reason`
  beside the new value (6 errors). A reasoned `not_reported` is the pass's answer, and a
  filled direction is built by `values.wrap`. This is the second wrapper bug of the day from
  writing a wrapper by hand instead of through its constructor (the first was mine).
- **A non-text quote crashed the build.** `warrant` raised on a quote that was an object, a
  JSON-mode reply shape. That lost `dem_p_effort_medium` one paper's whole build, which is
  why that run has 54 records. It is now an unresolved quote.

Rebuilding 329 records changes only the two dementia papers (9 errors removed). Left as
reported: 25797589's group-named levels on continuous terms (analyses over a patient group
plus controls, so not a restatement), and the "term name on both stages" findings.

## Substance use (36115222) on S2r + repair (`runs/sud55_s2r`, code `101d37e`)

Three papers failed on flex capacity (429 "Flex does not have sufficient resources", and a
timeout, each after 4 tries over about two hours). They were resumed into the same run;
completed stages are cached.

| run | veto recall | precision | strict recall |
|---|---|---|---|
| S2r | 22/23 | 0.85 | 22/23 |
| medium baseline (`sud_medium-v1`) | 22/23 | 0.88 | 22/23 |

Inputs-only scores are identical. Every substance-use gold paper's inputs carry coordinates.

**The miss, 18165464, is now fixed.** VBM explicitly masked to AAL grey matter was recorded
as `roi` over a region named "gray matter regions", so `whole brain`=False. The schema
already prescribes the fix in its Region rule: "Record the mask as
InferenceSettings.search_volume and leave the analysis whole-brain". But its name pattern
did not include "regions", and nothing applied the prescription.
- The pattern now includes "regions" (study_schema `21369ac`, local, not pushed).
- `rescope_tissue_masks` applies the prescription, reading the pattern from the schema
  rule. It acts only where every region named is a mask and an inference setting holds or
  takes the mask.
- Rebuilding 440 records changes only that paper, and the query then selects it: 23/23
  on a new draw.

**The four false positives are disputed labels, not record errors.** Autonima's full-text
screener included three of them:
- 29058369: alcohol-dependent patients vs controls, VBM.
- 30082140: synthetic-cannabinoid users vs controls.
- 30643026: 14-year-olds with one or two cannabis uses vs THC-naive controls.

The fourth, 26133201 (stimulant dependence vs controls, group × sex, women-only group
effect), the screener excluded for having no main between-group contrast. The baseline
excluded it only because its record said `English`=False, which was wrong (PubMed: "eng").
None is excluded by anything in the criteria text ("GM volume differences between
substance users and controls"; no pharmacological manipulation, lesions, or other
disorders). I have not fit the query to them. The 90% precision target is not met here,
in either pipeline, and the limit is the labels.

## A fresh PTSD draw on current code (`runs/ptsd55_s2r3`, code `54622dc`)

| | veto recall | precision | foci recall | foci precision | valid records | errors |
|---|---|---|---|---|---|---|
| `--gold-coords` | **19/19** | 0.95 | 128/143 | 155/178 | 45/54 | 14 |
| inputs-only | 18/19 | 0.95 | | | | |

One negative (19996042) failed on flex capacity. 32490056, missed in every earlier draw,
was found. The inputs-only miss is 26952803, whose tables were lost at ingestion. Errors
fell from 30 (first S2r run) to 14.

Adjudicator outcomes across the S2r runs show the warrant-guard fix at work. Refusals:

| run | refusals |
|---|---|
| `ptsd55_s2r2` (before the fix) | 42 |
| `ptsd55_s2r3` | 1 |
| `dem55_s2r` | 1 |
| `sud55_s2r` | 3 |

Rejected quotes are rare, 3 in one run, so moving the adjudicator to indexed citations is
not worth doing yet.

## Cue reactivity (Hill-Bowen 2021, 34400176): a fourth meta-analysis, held out

The first task-fMRI meta-analysis. Selection rests on visual cues
(`Task.stimulus_modality`) and a **within-participant** cue > control contrast, read off
cells, their levels' conditions, and those conditions' `stimulus_content` and
`condition_kind`. The three VBM pools instead compare groups.

**Pool and corpus.** `make_pool.py` (new; the earlier pools were drawn inline): 25 gold
sampled from the 140 benchmark papers autonima screened at full text, plus 30 negatives (15
that autonima's screener included and the benchmark did not, 15 others it screened), seed
0. The corpus was built with `build_corpus.py` into a new directory, and only the 54 papers
not already in the base corpus were copied in. 16 of 25 gold papers carry parsed
coordinates; two negatives are abstract-only texts.

**Query** (`43d66cb`), committed from the criteria text before any record existed. The run
was `cue55_s2r`: `tables` + `prose` + `split`, then S2r + repair, on code `43d66cb`.

| | veto recall | precision | inputs-only recall |
|---|---|---|---|
| **held out** | **9/25** | **0.64** | 7/25 |
| after three reading fixes (fit to this data) | 15/25 | 0.71 | 13/25 |

**The records were right and the query read them wrong:**
1. *Control first, in the description.* A cue condition's description compares ("sexual
   pictures ... masked by neutral pictures"), so looking for a control word there first
   called every cue a control. The side now comes from the conditions' `condition_kind`
   (`control_state`), then their stimulus content, label and names; the description is
   used only when those say nothing. Fixation and rest are neither side: the criterion
   says "control *stimuli*".
2. *A held cohort counted as a group crossing.* A contrast taken within one group is not a
   group comparison. ("romantic" was also dropped from the cue words: romantic-partner
   photos made 22860092's "partner + pen" control condition a cue.)
3. *An SVC is not recorded as I assumed.* The query's comment said an SVC "is a
   correction over an ROI on a whole-brain model, so `whole_brain` already admits it". That
   was never checked. Records give it `spatial_scope: roi` on a voxel model: 46 of 179 ROI
   analyses here, against 133 region-mean ones. Now an ROI analysis on a voxel or vertex
   model passes, and a region-mean (`roi`/`parcel` unit) one fails.

**What still limits it:**
- *Recall (9 misses, all `cue > control`=None).* Many cue-reactivity papers report group
  comparisons of the cue > neutral map ("heroin-dependent vs controls: heroin-related >
  neutral cues"). The model cells only the group term and leaves out the first-level
  condition cells the schema asks for. I did not loosen the query to accept a bare group
  comparison as a within-participant contrast. This is an extraction gap.
- *Precision (6 false positives).* Five are papers autonima's own full-text screener
  included, each with visual cues, a within-participant cue > control contrast and
  whole-brain results. The sixth, 29108734, reports only uncorrected results. Against the
  15 random negatives, precision is 1 false positive in 15; against the 15 the benchmark
  left out despite autonima including them, it is 5 in 15. The benchmark's 191 papers are
  a selection from a larger eligible literature.

### Cue reactivity: a group difference *in* the cue contrast

**The benchmark pools such a difference.** 25214465's gold analysis "H>N" (32 foci) is its
heroin-dependent vs controls difference on heroin > neutral cues: the table the record
links, and the paper's own heading "Heroin-dependent individuals versus healthy controls:
heroin-related > neutral cues". The query had returned None for a cue pair crossed with a
group term, reading the criterion's "within-participant contrast" narrowly. It now counts
the pair (16/25 at 0.73, fit to this data).

**Extraction left the contrast in the name.** 29 analyses, 19 in gold papers, compared
groups on a cue > neutral map and celled only the group term. That loses which contrast
was compared, though the schema's `Cell.term` says a group contrast of a first-level
condition is celled on that stage's term.

**Probe, one change.** A paragraph in `SINGLE_NOTE`: such a comparison cells both factors,
the group levels and the condition levels, each on its own term. Run on the 11 affected
papers (`runs/cue_note`, `single` + `build`, one draw, 13 calls), against the same papers
in `cue55_s2r`:
- 3 gold papers become selectable: 20172508; 25214465 (0/6 analyses read as cue > control,
  now 8/10); and 30217552.
- The 2 negatives were already selected and stay so; no new false positive.
- 4 gold papers do not move. Their failures are other shapes: a correlation with sexual
  desire; a cue vs fixation contrast; a cue-type main effect on a four-cell term.

Adopted (`f5f53d5`). A full cue draw on it (`cue55_s2r2`) is the confirmation.

### Optional objects that say nothing

Strict decoding requires every slot of an object a model opens. 25533729 wrote
`mediation: {mediator: "", path: not_reported}` on four analyses that had no mediation.
`drop_vacuous_objects` removes an optional, single-valued nested object whose every value
is blank or unreported, which is what `null` would have said. Rebuilding 385 records changed
only that paper.

### Confirmation draws

**Cue reactivity with the group-difference note (`cue55_s2r2`, code `f5f53d5`):**

| run | veto recall | precision | inputs-only recall | valid records | errors |
|---|---|---|---|---|---|
| `cue55_s2r` (query as fixed) | 16/25 | 0.73 | 13/25 | 44/55 | 31 |
| `cue55_s2r2` (+ note) | **19/25** | 0.70 | 16/25 | 36/55 | 45 |

- **The note held on the full pool:** +3 gold papers.
- **Precision:** the 8 false positives include the 5 autonima-included papers again. The
  new ones are 27459715 and 27654662.
- **Errors rose.** The note made the model cell the neutral condition, but it did not always
  declare the level: 24695721's cue term listed cocaine, sexual and aversive and not
  neutral, though the record declares `cond_neutral`. `complete_condition_levels` now
  declares such a level when the cell's name folds to exactly one unclaimed condition of a
  condition factor. A contrast label ("cocaine vs neutral") names no condition and is
  still reported. Rebuilding 330 records: 401 -> 394 errors, no value lost.
- **Remaining recall misses** (`cue > control`=None) are other shapes:
  - a correlation with sexual desire (22514316);
  - cue vs fixation (30165099; the gold analysis is "F>baseline");
  - a craving regressor with the cue contrast implicit (23359677);
  - a cue-type main effect on a four-cell term (22860092);
  - an n-back analysis (30991248, a surprising gold paper).

**Substance use on current code (`sud55_s2r2`, code `7c88557`):** veto recall 21/23,
**precision 0.91** (2 false positives: 29058369, 30082140). This time 26133201 and 30643026
were not selected. Across the two SUD draws, the disputed negatives move in and out, which
is draw variance at the borderline.

The tissue-mask repair did not fire on 18165464 here: this draw named the mask region
"explicit mask created from Automatic Anatomic Labeling limiting the analysis to gray
matter regions", which the bare-tissue pattern does not match, by design. Matching
descriptions that mention a mask would risk real structures ("grey matter of the
hippocampus"), so it is left. 28887180 (`users vs controls`=F) was found in the previous
draw.

## Standing, all four meta-analyses (veto, `--gold-coords`, current code or the latest draw)

| meta-analysis | held out? | recall | precision | inputs-only recall |
|---|---|---|---|---|
| PTSD | developed on it | 19/19 | 0.95 | 18/19 |
| dementia | held out for S2r | 23/25 | 0.96 | 16/25 (8 gold papers' coordinate tables missing from the corpus) |
| substance use | developed earlier | 21-22/23 | 0.85-0.91 | same |
| cue reactivity | held out (9/25, 0.64), then fit | 19/25 | 0.70 | 16/25 |

## Decision making (Poudel 2020, 32078973): a fifth meta-analysis, held out

Chosen from autonima's projects for criteria a record can answer: English fMRI (PET
excluded), whole brain (ROI excluded), coordinates, by 3/2019, any population. The
criterion that needs judgement is the task: risky, ambiguous or perceptual decision making,
read off the analysis's task name, description and conditions (`DECISION`). Executive
function (22282036) was passed over: its benchmark row carries the criteria of the
nicotine-administration meta-analysis, a data error in neurometabench.

**Query** committed before any record (`5fde88e`).

**Pool:** `make_pool.py`, seed 0. Ten of the draw (5 gold, 5 negatives) had no text in the
catalog or the old corpus. They were replaced from the same strata (gold; autonima-included;
other screened) with seed 1, taking the first candidates that have text. So this pool is 25
gold and 30 negatives with text, not a pure seed-0 draw.

Run: `dm55_s2r` (code `476deaa`, every fix so far).

**Cue reactivity, one more query change.** 27459715's cues are beer-flavour sprays
(gustatory), and the second draw recorded the task as gustatory + visual, so it passed
`visual cues`. A visual channel beside a gustatory, olfactory or tactile one now answers
None, following the criterion: "other sensory cues ... were not considered". Scores are
unchanged, because `visual cues` is not a required criterion and None passes veto. Making
it required would also fail the 2 gold papers whose modality is unrecorded.

**Review item 9 (query rules fitted to one paper), measured.** `ablate_rules.py` reverts one
rule at a time in a copy of `queries.py` and rescores 9 stored runs (4 PTSD, 2 dementia, 3
substance use):

| rule | reverting it |
|---|---|
| "negative for PTSD" is a comparison cohort | moves no paper |
| a grey-matter label answers "structural MRI" | moves no paper |
| a level holding any case cohort is the case side | loses gold 21418787 in `ptsd55_s2r` ("combined PTSD and major depression" level); no negative |
| dementia modality read from the measure's label | loses 2-3 gold per dementia draw (25009480, 25797589, 31873787, 31887311); no negative |

No rule admits a negative. The first two guarded against draws that did not recur; the
last two earn their place on gold papers.

**Review item 11 (overlap threshold), measured on the 4 PTSD runs.** `min_shared_authors` = 2 and
3 make the same three exclusions in every run. At 4, 19538748 ~ 16371250 is lost (precision
0.95 -> 0.90); at 5, 23113800 ~ 19942229 too (0.86). The threshold sits at the top of a
plateau, not on an edge: lowering it changes nothing, and raising it misses real re-reports.
Substance use and the other pools have no overlap criterion.

## A code review of tonight's repairs (subagent, read-only)

The reviewer read `git diff 97dad25..HEAD -- pondie/` for correctness and found eight faults.
All are fixed, each with a test:

1. **`scope_duplicate_terms`, my regression, high severity.** After a revert
   (`body.clear(); body.update(snapshot)`), the models list read before the loop still
   pointed at the pre-revert dicts. The next collision renamed terms no longer in the
   record while repointing the live cells, leaving them dangling, and the new check
   (`dangling[name]`) passed, because the bare declarations were still in the record.
   - *How it came to be:* I replaced a crude whole-record string check with a narrower one
     and did not notice that the crude check also covered this.
   - *Fix:* the models are re-read for each name, and any increase in dangling references
     reverts.
2. **`_reverse_name`** flipped every `<` and `>` in a name, the p-threshold in "PTSD > HC
   (p < 0.001)" too. Now only the comparison the case matched.
3. **`complete_partial_models`** raised on a donor term without a `local_id`, which would
   lose the record's build.
4. **`drop_vacuous_objects`** would drop `Study.design` with reasoned `not_reported`
   fields, before `fill` could ask about it. Now only an object whose required reference is
   blank, and a reasoned `not_reported` is an answer.
5. **Ordering:** `table_effects` and `coordinate_space` ran before `table_references`, so a
   repointed table got neither.
6. **`_table_number`** misread "Supplementary Table 2", "Table 2 (n = 30)" and the `tbl1_2`
   collision suffix. On rebuild this un-did a real mislink: 30739462's "Supplementary
   Table 1" had been linked to main Table 1.
7. **`rescope_tissue_masks`** wrote the mask into inference settings shared with a genuine
   ROI analysis. It now does so only where no unmasked analysis shares them, and only when
   an existing `search_volume` already names the mask.
8. **`fold_label`** folded a leading minus away (`-1` equal to `1`). Now only a hyphen
   between word characters is folded.

Rebuilding 385 records changes only 30739462 (fault 6). The review also confirmed as
correct: `walk.repoint`, `complete_condition_levels`, the polarity threshold guard,
`_named_against_cells`, the guard's `choices`, the kind re-check, `_shared_enums` and
`rekey_coordinate_sets`.

Lesson: a second reader on a night's repairs found a record-corrupting regression that the
rebuild diff could not, because no stored record had that shape. The rebuild diff tests
what the data contains; review tests what the code allows.
