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
