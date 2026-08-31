# What is actually inside `unknown`

`docs/evidence-union-design.md` scores two evidence locators against human spans and puts
half of every column in `unknown` — the pick matched nothing a reviewer marked. That
bucket means two incompatible things: a pick that supports the value but was not the
sentence the reviewer happened to highlight, or a genuine miss. Until it is split, the
42% union figure is uninterpretable, and so is every decision resting on it.

So: 42 cases, sampled across the five disagreement buckets and spread over fields so no
one repeated field dominates, read by hand against the papers. For each one this file has
the value, the retriever's top-1, the LLM's quote, what the reviewer marked, what the
reviewer deleted, and my judgement. Corroborate freely — the whole point is that the
automated verdict cannot be trusted here.

## The count

| judgement | n | share |
|---|---|---|
| a system's pick does support the value | 21 | 50% |
| partially supports | 2 | 5% |
| scored wrong against a deletion of the correct sentence | 4 | 10% |
| the gold is on the wrong entity | 1 | 2% |
| genuine miss | 14 | 33% |

**Two thirds of what the metric calls `unknown` or `wrong` is a scoring failure, not a
retrieval failure.** Only 14 of 42 are real misses.

## Three things this changes

**1. `unknown` is dominated by correct-but-unhighlighted.** In the 18 sampled cases where
*both* systems scored `unknown`, 13 had at least one pick I would accept as a citation.
Reviewers mark one supporting sentence and move on; papers state facts more than once. So
the union's 42% is a floor, and the true figure is a good deal higher. It also means
improvements measured this way are compressed — a change that finds a second valid
sentence scores as no change at all.

**2. `wrong` does not mean wrong.** Every one of the five sampled cases where the
retriever scored `wrong` turned out to be a pick I judge correct — including three where
the picked sentence is the recorded value nearly verbatim (#38 "Two-sample t-test was used
to compare the MDTs of ASD and TD children" for exactly that definition; #40, #41). They
score `wrong` because the reviewer *deleted* that passage, and a deletion is not a ruling
that the sentence fails to support the value. Reviewers delete to re-scope a label, to
trim an over-broad span, or to replace a fragment with a fuller sentence.

The data says the same thing: of 68 slots carrying a deletion, only 15 also gained a
replacement span, and 7 deleted spans *overlap a span the same reviewer added back*. A
bare deletion is the common case and it is uninterpretable as a negative.

The consequence is that `precision on adjudicated` — 74.2% for the LLM, 76.6% for the
retriever — is measuring something it should not, and both are understated. The
abstention-threshold sweep in the union design's open question 3 is built on that same
`correct / (correct + wrong)` ratio and needs re-basing on `correct` alone, or on the 15
deletions that came with a replacement.

**3. 22 slots were queried with a local_id.** `Acquisition`, `Measure`, `Preprocessing`
and `InferenceSettings` have no `name` field, so the eval fell back to the local_id and
asked the retriever to find `acquisition_fmri` or `measure_diffusion_metric_5` in the
paper. Those strings do not occur in any paper. It is a bug in the harness, not in either
locator, and it costs both of them:

| slots | retriever correct | LLM correct |
|---|---|---|
| queried with a local_id (22) | 13.6% | 4.5% |
| everything else (204) | 33.8% | 33.3% |

Excluding them, retriever top-1 is 33.8% rather than 31.9%. Those classes need a
descriptor built from their own fields — modality plus sequence for an acquisition — not
their identifier.

## What the 14 real misses are made of

They are not one problem, and none of them is a ranking problem:

- **Derived values that are never printed** (#1 voxel size 1.875 = 24 cm ÷ 128; #24
  approached_count 29 = 27 + 2). No surface form exists to match.
- **Notation flattened by text extraction** (#19 `11C` is written `11C-PiB` with a
  superscript the extractor dropped to `C-PiB`).
- **Controlled terms whose evidence shares no vocabulary with them** (#20 `preregistered`
  is warranted by "ClinicalTrials.gov ID: NCT01191333"; #26 `prior_literature` by "lying in
  the proximity of previously published coordinates"). Both are reachable by an alias and
  neither is reachable by anything else.
- **The homonym trap** (#22 matched "randomize" in "the randomize function in FSL" —
  permutation testing, not allocation, and it argues *against* the recorded value).
- **Sibling confusion** (#21 picked the caption for the positive-correlation analysis when
  the value was the negative one; #9 cited the sham arm for the active arm's field).
- **Malformed query** (#17, #25 — the local_id bug above).

Two of these — aliases for registry identifiers and for `prior_literature`-style terms,
and superscript-aware value variants — are the same kind of fix as the alias and unit work
already in `review/evidence_retrieval.py`, and would land in the same table.

## What the harness fixes were worth

All three were applied and the whole set re-scored. They were not cosmetic:

| | before | after |
|---|---|---|
| LLM evidence pass correct | 30.5% | **39.9%** |
| retriever top-1 correct | 31.9% | **42.2%** |
| retriever top-12 correct | 52.7% | **69.9%** |
| union | 42.0% | **55.5%** |
| retriever `wrong` | 9.7% | 0.6% |

Nothing about either locator changed. The whole movement is the measurement no longer
being broken: nameless classes now carry a descriptor built from their own fields
(`Acquisition acquisition_fmri` became `fMRI`), bare deletions are `retracted` rather than
counted against a pick, and `correct` is the headline instead of a precision over a
denominator this gold cannot supply. Only 8 usable negatives survive across the whole set,
which is the honest size of the "confirmed wrong" evidence and the reason no precision is
quoted anywhere now.

The abstention curve sharpened with them: a margin cut at 2.62 now keeps 40% of slots at
80% confirmed-correct, and a literal hit is worth 80.9% against 27.8%.

Raw sample: `data/eval/unknown-sample.json`.

## The cases

### 1. `acquisitions.acquisition_voxel_size_mm` — genuine miss

- **bucket** both unknown · **paper** 84rGLhCbUJTh
- **value** `1.875`
- **retriever top-1** (section: methods)
> Voxels of T>2.674 (p<0.001, uncorrected, cluster size>10 voxels;
- **LLM evidence pass**
> [9] - we smoothed all the transformed parameter maps with a 6-mm FWHM Gaussian kernel, ready for further statistical testing. In addition, because the study was focused on white matter abnormality of OCD patients, a whit
- **reviewer marked**
> For each volume, 24 continuous slices, each 5-mm-thick, were collected on a 24 cm×24 cm field of view, with the following imaging parameters: repetition time (TR) = 9000 ms, echo time (TE) = 105.5 ms, and matrix = 128*128.

- **judgement** (MISS) Voxel size 1.875 is never printed; it is 24 cm FOV over a 128x128 matrix. The reviewer's sentence carries the two numbers you divide. Neither system found it and neither could have matched on the value.

### 2. `acquisitions.name` — the gold is on the wrong entity

- **bucket** both unknown · **paper** 7HPLh5nJzmP5
- **value** `acquisition_fmri`
- **retriever top-1** (section: intro)
> Functional magnetic resonance imaging (fMRI) is a well-established method for measuring FC by correlating fluctuations in cortical blood oxygenation (oxy-Hb), which is considered a proxy for neural activity due to neurovascular coupling.
- **LLM evidence pass**
> For FC analysis, a BOLD T2*-weighted image sequence was acquired
- **reviewer marked**
> An anatomical T1-weighted image was acquired using a 3D multi-shot spin echo sequence (TR/TE 8.2/3.8 ms, flip angle = 8°, field of view 256x256 mm with voxel sixe 1x1x1mm )

- **judgement** (GOLD WRONG) The LLM cited 'For FC analysis, a BOLD T2*-weighted image sequence was acquired' -- correct for the fMRI acquisition. The reviewer marked the *anatomical* T1 sentence, which belongs to a different acquisition. The LLM is right and the gold is on the wrong entity.

### 3. `analyses.details.performance_metrics.reference_value` — genuine miss

- **bucket** both unknown · **paper** ngDTY5BgJUuX
- **value** `50.0`
- **retriever top-1** (section: methods)
> (3) 0.64, SD ±2.09;
- **LLM evidence pass**
> alysis. The names of the brain regions are reported with reference to the AAL3 ( Rolls et al., 2020 ). ## Results ### Questionnaire subjective ratings The behavioral ratings showed
- **reviewer marked**
> A one-sample t test was used to determine whether the observed decoding accuracy was significantly higher than chance (50%) with intersubject difference treated as a random factor (df = 24)

- **judgement** (MISS) The value 50.0 is the chance level; the supporting sentence says 'significantly higher than chance (50%)'. The retriever picked a stray decimal from a results list. Genuine miss.

### 4. `analyses.inference_settings.voxelwise_threshold_type` — a system's pick does support the value

- **bucket** both unknown · **paper** ngDTY5BgJUuX
- **value** `p`
- **retriever top-1** (section: methods)
> We used an uncorrected threshold of p < 0.001 at the voxel-level and a threshold of p < 0.05 FWE corrected at the cluster-level for each type of classification analysis.
- **LLM evidence pass**
> us and asynchronous conditions. Parameter estimates (β values) of each trial of voxels within ROIs were used as inputs to the classifie
- **reviewer marked**
> Activation was reported with a threshold of p < 0.05 corrected for FWE at the voxel-level with an extent threshold of 10 voxels

- **judgement** (SUPPORTS) The field is the *type* of voxelwise threshold, and the pick states an uncorrected p threshold at the voxel level. It supports 'p' as well as the reviewer's sentence does.

### 5. `analyses.inference_settings.voxelwise_threshold_value` — genuine miss

- **bucket** both unknown · **paper** ngDTY5BgJUuX
- **value** `0.05`
- **retriever top-1** (section: methods)
> We used an uncorrected threshold of p < 0.001 at the voxel-level and a threshold of p < 0.05 FWE corrected at the cluster-level for each type of classification analysis.
- **LLM evidence pass**
> us and asynchronous conditions. Parameter estimates (β values) of each trial of voxels within ROIs were used as inputs to the classifie
- **reviewer marked**
> Activation was reported with a threshold of p < 0.05 corrected for FWE at the voxel-level with an extent threshold of 10 voxels

- **judgement** (MISS) Same pick as #4, and here it is wrong: the recorded value is 0.05 at the voxel level, and this sentence puts 0.05 at the *cluster* level and 0.001 at the voxel level. Right paragraph, wrong number -- and instructive that one sentence is correct evidence for one field and wrong for its neighbour.

### 6. `analysis.definition` — a system's pick does support the value

- **bucket** both unknown · **paper** xevP8UDRAVh9
- **value** `Positive correlation between gray matter and placebo-associated perfusion.`
- **retriever top-1** (section: results)
> Correlation between gray matter with placebo-associated perfusion .
- **LLM evidence pass**
> Biological Parametric Mapping analysis integrating GM and perfusion data after the placebo treatment revealed a significant positive correlation between perfusion and GM volume in frontal areas on both hemispheres including the precentral gyrus, the inferior, middle and superior frontal gyrus, the frontal pole, and the right paracingulate cortex, including parts of the ACC (Table 1 and Figure 2 , upper panel).
- **reviewer marked**
> In BPM, each perfusion condition (heroin and placebo) was correlated separately with the VBM data

- **judgement** (SUPPORTS) Both. The retriever picked the table caption naming the analysis; the LLM picked a Results sentence that states the correlation is positive, which is better still. The reviewer marked the Methods sentence. Three defensible spans, one counted.

### 7. `analysis.interpretations` — a system's pick does support the value

- **bucket** both unknown · **paper** xevP8UDRAVh9
- **value** `There was no significant negative correlation between modalities.`
- **retriever top-1** (section: results)
> No significant cluster was found in the temporal lobe and there was no significant negative correlation between modalities (Table 1 ).
- **LLM evidence pass**
> there was no significant negative correlation between modalities (Table 1 ).
- **reviewer marked**
> This negative correlation is difficult to explain in terms of our hypothesis that hypoperfusion is the driving force leading to GM reduction. The cuneus serves as visual processing and inhibitory control centers and electroencephalographic studies have revealed abnormalities in heroin-dependent individuals ( 31 , 32 ) and decreased regional homogeneity ( 33 ) in this region

- **judgement** (SUPPORTS) Both picks contain the value verbatim: 'there was no significant negative correlation between modalities'. The reviewer marked a Discussion sentence interpreting a correlation. Both systems are better than the gold here.

### 8. `arm.arms[0].agent` — a system's pick does support the value

- **bucket** both unknown · **paper** xevP8UDRAVh9
- **value** `heroin`
- **retriever top-1** (section: abstract)
> Using a double-blind, within-subject design, heroin or placebo (saline) was administered to 14 heroin-dependent patients from a stable heroin-assisted treatment program, in order to observe acute short-term effects.
- **LLM evidence pass**
> heroin
- **reviewer marked**
> In a cross-over, double-blind design, placebo (5 ml saline), and half of the daily heroin dose (mean = 176 ± 89.1 mg) were administered intravenously over a period of 30 s by a study nurse 20 min before the scanning session started. Heroin was dissolved in 5 ml of sterile water and aspirated into a syringe, according to the procedure described by Stohler et al.

- **judgement** (SUPPORTS) 'heroin or placebo (saline) was administered to 14 heroin-dependent patients' supports agent=heroin plainly. The LLM cited the bare word 'heroin', which is six characters and warrants nothing.

### 9. `arm.arms[0].arm_kind` — genuine miss

- **bucket** both unknown · **paper** 7HPLh5nJzmP5
- **value** `stimulation`
- **retriever top-1** (section: methods)
> In the sham group, a mild TENS was delivered synchronously to the magnetic pulses and in proportion to iTBS stimulation intensity, with a maximum current of 4 mA, to mimic the sensation of the active treatment.
- **LLM evidence pass**
> active treatment group
- **reviewer marked**
> Each session comprised 40 trains of stimulation, with each train consisting of two seconds stimulation, and eight seconds off.

- **judgement** (MISS) The pick describes the *sham* arm's TENS; the arm being evidenced is the active one. Wrong entity.

### 10. `arm.arms[0].description` — a system's pick does support the value

- **bucket** both unknown · **paper** 7HPLh5nJzmP5
- **value** `Active intermittent theta-burst stimulation over the dorsomedial prefrontal cortex for ten treatment`
- **retriever top-1** (section: methods)
> ## Methods 42 patients with ongoing depression were recruited and randomized to receive active or sham intermittent theta-burst stimulation (iTBS) over the dorsomedial prefrontal cortex (dmPFC) twice daily for ten days at target intensity.
- **LLM evidence pass**
> The treatment was given for ten days of target intensity, defined as 90% of the patient’s individual resting foot motor threshold, with two sessions per day separated by a fifteen-minute intersession interval ( Tse et al., 2018 ).
- **reviewer marked**
> Each session comprised 40 trains of stimulation, with each train consisting of two seconds stimulation, and eight seconds off. The stimulation comprised ten bursts at 5 Hz, and three biphasic pulses at 50 Hz per burst, thus delivering 1200 pulses per session. For all patients, transcutaneous electrical nerve stimulation (TENS) electrodes were applied directly under the TMS coil

- **judgement** (SUPPORTS) 'randomized to receive active or sham intermittent theta-burst stimulation (iTBS) over the dorsomedial prefrontal cortex (dmPFC) twice daily for ten days' covers nearly the whole recorded description. The reviewer marked the pulse-train detail instead.

### 11. `assessments.name` — a system's pick does support the value

- **bucket** both unknown · **paper** 7HPLh5nJzmP5
- **value** `Clinical Assessment Interview for Negative Symptoms`
- **retriever top-1** (section: methods)
> Clinical assessments of depression symptoms, such as the Clinical Assessment Interview for Negative Symptoms (CAINS) ( Kring et al., 2013 ) or the Montgomery Åsberg Depression Rating Scale (MADRS) ( Svanborg & Åsberg, 1994 ), and functional magnetic resonance imaging (fMRI) were conducted one work day before treatment start (baseline), and once again four weeks later (follow-up).
- **LLM evidence pass**
> Clinical Assessment Interview for Negative Symptoms
- **reviewer marked**
> Mini International Neuropsychiatric Interview (M.I.N.I.) version 6.0.0 ( Sheehan et al., 1998

- **judgement** (SUPPORTS) The pick contains 'Clinical Assessment Interview for Negative Symptoms' character-for-character. The reviewer's span is the M.I.N.I. -- a different instrument. Gold is on the wrong entity.

### 12. `definition` — a system's pick does support the value

- **bucket** both unknown · **paper** DTpwdoGbjqsq
- **value** `Results from ANCOVA reflecting regions where VBM-GM differences were not explained by 11C-PiB SUVR l`
- **retriever top-1** (section: results)
> Results from ANCOVA reflecting regions where VBM-GM differences were not explained by 11C-PiB SUVR levels in AD BA, Broadman areas;
- **LLM evidence pass**
> tween neocortical PiB and hippocampal volume when pooling 93 normal controls (32 PiB positive) 33 aMCI (20 PiB
- **reviewer marked**
> BPM ANCOVA analysis of GM maps superimposed on the T1-weighted template in the axial plane for the contrast AD < CT and controlling for 11C-PiB SUVR levels

- **judgement** (SUPPORTS) The pick is the table caption the definition was written from, word for word. The reviewer marked a figure caption for a different contrast (AD < CT).

### 13. `design.assignment_structure` — a system's pick does support the value

- **bucket** both unknown · **paper** 84rGLhCbUJTh
- **value** `within_subject`
- **retriever top-1** (section: methods)
> Pre and post-treatment and brain regions with abnormal DTI-derived parameters were set as within-subject factors.
- **LLM evidence pass**
> 8*128. The acquisition time was 4 min 12 sec for each participant. ### Image processing We adopted voxel-based analysis (VBA) in our study, which had been frequently us
- **reviewer marked**
> After receiving clinical feature measurements and DTI scans at baseline, 15 of the 27 OCD patients agreed to accept SSRI treatment only (6 of 15 OCD patients were drug-naive).

- **judgement** (SUPPORTS) 'set as within-subject factors' is about as direct as evidence for within_subject gets. The reviewer marked the SSRI enrolment sentence.

### 14. `groups.name` — a system's pick does support the value

- **bucket** both unknown · **paper** 6oTrCJA43Jcd
- **value** `typical development children`
- **retriever top-1** (section: abstract)
> As compared to typical development children, weak relevance condition (the strength of a large number of connectivities in the state was less than means minus standard deviation of all connection strength) was maintained for a longer time between brain areas of ASD children, and ratios of weak connectivity in brain states varied dramatically in the ASD.
- **LLM evidence pass**
> TD children
- **reviewer marked**
> The site includes 79 (7.1–39.1 years) ASD and 105 TD (6.5–31.8 years) children.

- **judgement** (SUPPORTS) The pick contains 'typical development children' verbatim; the reviewer's sentence says 'TD'. The pick is the better citation for the name as recorded.

### 15. `inferencesettingss.name` — a system's pick does support the value

- **bucket** both unknown · **paper** 84rGLhCbUJTh
- **value** `inference_none_the_roi_level_tests_are_reported_uncorrecte`
- **retriever top-1** (section: results)
> p<0.001 uncorrected and cluster size>10 voxels.
- **LLM evidence pass**
> d without a history of medication, mean FA, RD, AD and MD values were extracted in regions
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> The tests were two-tailed and the threshold of statistical significance was set at p<0.05.
- **judgement** (SUPPORTS) Weakly: 'p<0.001 uncorrected' does support an inference setting recorded as uncorrected. The value here is a local_id, not a name -- see the note on malformed queries below.

### 16. `interpretations` — a system's pick does support the value

- **bucket** both unknown · **paper** TgcHKMRfrVog
- **value** `BA25–paracentral lobule rsFC decreased following treatment.`
- **retriever top-1** (section: results)
> In whole-brain analyses, BA25 ~ paracentral lobule and rACC ~ paracentral lobule connectivities decreased post-treatment.
- **LLM evidence pass**
> ion in this region [ 13 ] th
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> BA25-PCL rsFC also decreased
- **judgement** (SUPPORTS) 'BA25 ~ paracentral lobule and rACC ~ paracentral lobule connectivities decreased post-treatment' states the interpretation in full. It was scored `unknown` because this slot has only a deletion and no positive span, so it could not have scored `correct` whatever the pick was.

### 17. `measures.name` — genuine miss

- **bucket** both unknown · **paper** 84rGLhCbUJTh
- **value** `measure_diffusion_metric_5`
- **retriever top-1** (section: intro)
> Diffusion tensor imaging (DTI) is a non-invasive method that maps the diffusivity of water molecules in tissue [5] .
- **LLM evidence pass**
> lin oligodendrocyte glycoprotein (MOG) [42] and oligodendrocyte lineage transcription factor 2 (OLG2) [43] . Atmaca et al. provided further evidence t
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> The effect of pre and post-treatment in brain regions with abnormal RD values was found to be statistically significant (F = 7.079, df = 1,14, p = 0.019)
- **judgement** (MISS) The pick is the Introduction's definition of DTI. The query was a local_id (`measure_diffusion_metric_5`), so there was nothing to match on.

### 18. `model_estimations.model_family` — a system's pick does support the value

- **bucket** both unknown · **paper** 84rGLhCbUJTh
- **value** `glm`
- **retriever top-1** (section: methods)
> A General Linear Model repeated-measure analysis of covariance (ANOVA) was used in the analysis of each DTI-derived parameter before and after treatment.
- **LLM evidence pass**
> ults ### DTI-derived parameters at baseline The results indicate significantly decreased FA in patients compared with health
- **reviewer marked**
> We performed a whole-brain voxel-based analysis using the normalized and smoothed FA, AD, RD and MD maps for a 2-sample group t-test

- **judgement** (SUPPORTS) 'A General Linear Model repeated-measure analysis of covariance' contains the model family verbatim. The reviewer marked a t-test sentence, which is the weaker citation.

### 19. `acquisitions.tracer_radionuclide` — genuine miss

- **bucket** LLM correct, retriever unknown · **paper** DTpwdoGbjqsq
- **value** `11C`
- **retriever top-1** (section: methods)
> For example, the signal relative to C-PK11195 PET uptake is lower than the signal of C-PiB PET uptake, resulting in part from the lower density of activated microglia with regard to Aβ plaques, and also due to the lower signal to-noise ratio of C-PK11195 PET tracer.
- **LLM evidence pass**
> the intravenous bolus injection of approximately 555 MBq of C-PiB or 370 MBq of C-PK11195
- **reviewer marked**
> the intravenous bolus injection of approximately 555 MBq of C-PiB or 370 MBq of C-PK11195

- **judgement** (MISS) The tracer is written '11C-PiB' with a superscript the text extraction flattened to 'C-PiB', so the recorded value '11C' has no surface form to match. The LLM read it correctly.

### 20. `analyses.prespecification` — genuine miss

- **bucket** LLM correct, retriever unknown · **paper** aVGe9BmFTMDR
- **value** `preregistered`
- **retriever top-1** (section: intro)
> ## Introduction Repetitive pulse transcranial magnetic stimulation (rTMS) of the dorsolateral prefrontal cortex (DLPFC) is an important therapy option for patients with treatment resistant major depression (TRMD) [ 1 , 2 ], although success varies [ 3 − 5 ].
- **LLM evidence pass**
> ClinicalTrials.gov ID: NCT01191333
- **reviewer marked**
> ClinicalTrials.gov ID: NCT01191333

- **judgement** (MISS) The evidence for `preregistered` is 'ClinicalTrials.gov ID: NCT01191333', which shares no vocabulary at all with the field name or the value. A registry alias would reach it; nothing else will.

### 21. `definition` — genuine miss

- **bucket** LLM correct, retriever unknown · **paper** xevP8UDRAVh9
- **value** `Negative correlation between gray matter and placebo-associated perfusion; no significant clusters r`
- **retriever top-1** (section: results)
> Correlation between gray matter with placebo-associated perfusion .
- **LLM evidence pass**
> No significant cluster was found in the temporal lobe and there was no significant negative correlation between modalities (Table 1 ).
- **reviewer marked**
> No significant cluster was found in the temporal lobe and there was no significant negative correlation between modalities (Table 1 ).

- **judgement** (MISS) The pick is the table caption for the *positive* correlation analysis and the value is the negative one. A sibling-analysis confusion, and exactly the case the entity bonus is supposed to break.

### 22. `design.allocation` — genuine miss

- **bucket** LLM correct, retriever unknown · **paper** TgcHKMRfrVog
- **value** `non_randomized`
- **retriever top-1** (section: methods)
> For all GLM analyses, we also explored a whole-brain voxel-wise correction for multiple comparisons using the randomize function in FMRIB Software Library (FSL) with nonparametric threshold free cluster correction and 1,000 permutations [ 37 ] at p < 0.05, since this correction is more stringent and less sensitive to weaker, spatially distributed signals.
- **LLM evidence pass**
> Fifty-three participants (20 HVs and 33 patients with MDD) enrolled and completed baseline rsfMRI.
- **reviewer marked**
> Fifty-three participants (20 HVs and 33 patients with MDD) enrolled and completed baseline rsfMRI.

- **judgement** (MISS) Matched 'randomize' in 'the randomize function in FSL' -- permutation testing, not group allocation. The homonym trap, again, and here it argues against the recorded value.

### 23. `groups.acquired_count` — genuine miss

- **bucket** LLM correct, retriever unknown · **paper** aVGe9BmFTMDR
- **value** `12`
- **retriever top-1** (section: methods)
> As shown, all Rlocations target DLPFC (green overlay, Fig.
- **LLM evidence pass**
> active-responder = (−42,34,38; N = 12)
- **reviewer marked**
> active-responder = (−42,34,38; N = 12)

- **judgement** (MISS) The count is inside a coordinate annotation, 'active-responder = (-42,34,38; N = 12)'. The LLM found it; the retriever picked a figure caption.

### 24. `groups.approached_count` — genuine miss

- **bucket** LLM correct, retriever unknown · **paper** 84rGLhCbUJTh
- **value** `29`
- **retriever top-1** (section: methods)
> ### MRI acquisition setting We scanned OCD patients for DTI data before and after 12-week SSRI treatments.
- **LLM evidence pass**
> Two OCD patients who fulfilled inclusion criteria refused to participate in the MRI scanning. They were not counted within the OCD sample total of 27.
- **reviewer marked**
> Two OCD patients who fulfilled inclusion criteria refused to participate in the MRI scanning. They were not counted within the OCD sample total of 27.

- **judgement** (MISS) approached_count = 29 is 27 enrolled plus the 2 who refused. Derived, never printed, so no surface form exists.

### 25. `preprocessings.name` — genuine miss

- **bucket** LLM correct, retriever unknown · **paper** rxaz3qhEmJhx
- **value** `preprocessing_fmri`
- **retriever top-1** (section: methods)
> fMRI Task Inside the scanner, participants performed the emotional (fear) processing task, which has been described in detail elsewhere [ 10 , 40 ].
- **LLM evidence pass**
> a 7.2 mm full-width-at-half-maximum Gaussian filter
- **reviewer marked**
> Firstly, correction for head motion was completed by realigning the images, to a template created by computing a 3D volume from the average intensity at each voxel throughout the whole period scanning [ 47 ]. Realignment of the 3D image volume at each time-point to the template was computing using a combination of rotations (around the x , y and z axes) and translations (in x , y and z ) that maximised the correlation between the template 3D volume and the image intensities of the volume in question. In order to smooth the data a 7.2 mm full-width-at-half-maximum Gaussian filter was applied to

- **judgement** (MISS) Query was the local_id `preprocessing_fmri`. Nothing to match.

### 26. `regions.definition_method` — genuine miss

- **bucket** LLM correct, retriever unknown · **paper** SULKxviGFurw
- **value** `prior_literature`
- **retriever top-1** (section: methods)
> ### Effective connectivity analysis Dynamic Causal Modeling (DCM) is an effective connectivity analysis method for making inferences about causal relationships between brain regions.
- **LLM evidence pass**
> Voxels meeting p < 0.05 (FDR correction) threshold requirement and lying in the proximity of previously published coordinates of dorsomedial prefrontal cortex (dmPFC) [-6,2,58], ventromedial prefrontal cortex (vmPFC) [− 32,6,54], dorsolateral prefrontal cortex (dlPFC) [-48,24,28], and anterior cingulate cortex (ACC) [0,32,2] were taken to be the ROIs used in this study.
- **reviewer marked**
> Voxels meeting p < 0.05 (FDR correction) threshold requirement and lying in the proximity of previously published coordinates of dorsomedial prefrontal cortex (dmPFC) [-6,2,58], ventromedial prefrontal cortex (vmPFC) [− 32,6,54], dorsolateral prefrontal cortex (dlPFC) [-48,24,28], and anterior cingulate cortex (ACC) [0,32,2] were taken to be the ROIs used in this study.

- **judgement** (MISS) 'lying in the proximity of previously published coordinates' is the evidence for `prior_literature`, and no part of that phrase overlaps the value. An alias would reach it.

### 27. `acquisitions.name` — a system's pick does support the value

- **bucket** retriever correct, LLM unknown · **paper** aVGe9BmFTMDR
- **value** `acquisition_resting_fmri`
- **retriever top-1** (section: methods)
> #### MRI acquisition Structural MRI and resting state fMRI were collected ( eMethods: MRI Acquisition ) while patients wore a treatment cap with a fiducial marker ( eMethods: Fiducial Marker Protocol ) identifying the rTMS treatment location, thus enabling identification of the underlying brain location stimulated.
- **LLM evidence pass**
> resting state fMRI
- **reviewer marked**
> resting state fMRI

- **judgement** (SUPPORTS) Both. The LLM cited the exact phrase 'resting state fMRI' and was scored `unknown` only because it resolved to a different occurrence of the same words than the reviewer's. A scoring artifact, not a miss.

### 28. `analyses.inference_settings.clusterwise_threshold_value` — a system's pick does support the value

- **bucket** retriever correct, LLM unknown · **paper** ngDTY5BgJUuX
- **value** `0.05`
- **retriever top-1** (section: methods)
> We used an uncorrected threshold of p < 0.001 at the voxel-level and a threshold of p < 0.05 FWE corrected at the cluster-level for each type of classification analysis.
- **LLM evidence pass**
> SD ±2.10], indicating that the participant’s feeling of ownership for the hand on the screen was less stron
- **reviewer marked**
> a threshold of p < 0.05 FWE corrected at the cluster-level for each type of classification analysis

- **judgement** (SUPPORTS) The retriever picked the sentence carrying 'p < 0.05 FWE corrected at the cluster-level' and was scored correct. The LLM cited an unrelated behavioural result.

### 29. `analyses.inference_settings.correction_scope` — a system's pick does support the value

- **bucket** retriever correct, LLM unknown · **paper** ngDTY5BgJUuX
- **value** `whole brain, grey matter only`
- **retriever top-1** (section: methods)
> The searchlight moved over the gray matter of the whole brain.
- **LLM evidence pass**
> the hand on the screen was their own in the synchronous condi
- **reviewer marked**
> The searchlight moved over the gray matter of the whole brain

- **judgement** (SUPPORTS) 'The searchlight moved over the gray matter of the whole brain' is the reviewer's own span. Retriever correct, LLM unrelated.

### 30. `analyses.inference_settings.inference_level` — a system's pick does support the value

- **bucket** retriever correct, LLM unknown · **paper** ngDTY5BgJUuX
- **value** `cluster`
- **retriever top-1** (section: methods)
> We used an uncorrected threshold of p < 0.001 at the voxel-level and a threshold of p < 0.05 FWE corrected at the cluster-level for each type of classification analysis.
- **LLM evidence pass**
> ±1.18 and visuomotor asynchronous condition: (5) −1.64, SD ±1.81; (6) −1.24, SD ±2.10], indicating that the participant’s feeling of ownership for the hand on the screen was less stron
- **reviewer marked**
> We used an uncorrected threshold of p < 0.001 at the voxel-level and a threshold of p < 0.05 FWE corrected at the cluster-level for each type of classification analysis

- **judgement** (SUPPORTS) Same sentence as the reviewer's. Retriever correct, LLM cited behavioural ratings.

### 31. `definition` — a system's pick does support the value

- **bucket** retriever correct, LLM unknown · **paper** eaEGQiVtDp9e
- **value** `ALFF differences showing greater values in the Sham group than in the Tuina group 28 days after surg`
- **retriever top-1** (section: results)
> Brain regions showing ALFF differences between the Tuina group and the Sham group 28 days after surgery.
- **LLM evidence pass**
> he primary peak in MNI space; ALFF, the amplitude of low frequency fluctuations; M
- **reviewer marked**
> ALFF differences between the Tuina group and the Sham group 28 days after surgery.
- **reviewer deleted**
> ALFF differences between the Tuina group and the Sham group 28 days after surgery.
- **judgement** (SUPPORTS) The retriever picked the table caption; the reviewer both added and deleted essentially that caption, so the slot carries the same passage as positive and negative at once.

### 32. `groups.name` — a system's pick does support the value

- **bucket** retriever correct, LLM unknown · **paper** JzsUUQbDr2bm
- **value** `medicated chronic schizophrenia (CSZ) patients`
- **retriever top-1** (section: methods)
> ## Methods In this study, we recruited 44 first-episode drug-naive SZ (FESZ) patients, 44 medicated chronic SZ (CSZ) patients, and 56 normal controls (NCs) and acquired their structural and resting-state functional magnetic resonance imaging (MRI).
- **LLM evidence pass**
> 44 CSZ patients (the CSZ group)
- **reviewer marked**
> 44 medicated chronic SZ (CSZ) patients

- **judgement** (SUPPORTS) Both. The retriever's Methods sentence names the group in full; the LLM's '44 CSZ patients (the CSZ group)' also supports it and was scored `unknown` only for landing on a different phrasing.

### 33. `acquisitions.name` — partially supports

- **bucket** LLM wrong, retriever unknown · **paper** xevP8UDRAVh9
- **value** `acq_mprage`
- **retriever top-1** (section: methods)
> Voxel-Based Morphometry Structural data (MPRAGE) were preprocessed with VBM implemented in SPM8.
- **LLM evidence pass**
> using a 3D T1-weighted sequence (MPRAGE) for high resolution anatomical data
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> using a 3D T1-weighted sequence (MPRAGE) for high resolution anatomical data
- **judgement** (PARTIAL) The pick mentions MPRAGE but in the preprocessing step rather than the acquisition. The span the reviewer *deleted* -- 'using a 3D T1-weighted sequence (MPRAGE) for high resolution anatomical data' -- is the correct evidence, so the deletion is not a judgement that the LLM was wrong.

### 34. `arm.arms[0].name` — genuine miss

- **bucket** LLM wrong, retriever unknown · **paper** xevP8UDRAVh9
- **value** `heroin`
- **retriever top-1** (section: intro)
> Within the brain, the heroin metabolites 6-monoacetylmorphine (6-MAM) and morphine bind as agonists to μ and κ receptors ( 11 ).
- **LLM evidence pass**
> heroin
- **reviewer marked**
> perfusion condition (heroin and placebo)
> perfusion condition (heroin and placebo)
- **reviewer deleted**
> heroin
- **judgement** (MISS) The pick is an Introduction sentence on heroin pharmacology. Here the deletion is meaningful: the LLM cited the bare word 'heroin', which cannot warrant an arm name.

### 35. `definition` — a system's pick does support the value

- **bucket** LLM wrong, retriever unknown · **paper** TgcHKMRfrVog
- **value** `Baseline correlation between aSCC resting-state functional connectivity and BDI in patients with MDD`
- **retriever top-1** (section: results)
> Anterior subcallosal cingulate (aSCC) and rostral anterior cingulate (rACC) resting-state functional connectivity associated with depression symptom severity (Beck Depression Inventory [BDI]) in major depressive disorder group at baseline ( N = 30).
- **LLM evidence pass**
> Baseline rsFC of the aSCC seed with a cluster in left lateral prefrontal cortex (lPFC) correlated negatively with baseline BDI
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> Baseline rsFC of the aSCC seed with a cluster in left lateral prefrontal cortex (lPFC) correlated negatively with baseline BDI
- **judgement** (SUPPORTS) The pick is the figure caption naming aSCC/rACC rsFC against BDI at baseline -- the definition as recorded. The LLM's deleted span also supports it.

### 36. `group.medications` — a system's pick does support the value

- **bucket** LLM wrong, retriever unknown · **paper** xevP8UDRAVh9
- **value** `heroin`
- **retriever top-1** (section: abstract)
> Using a double-blind, within-subject design, heroin or placebo (saline) was administered to 14 heroin-dependent patients from a stable heroin-assisted treatment program, in order to observe acute short-term effects.
- **LLM evidence pass**
> heroin
- **reviewer marked**
> Patients received their regular morning dose of heroin, corresponding to half of their daily individual dose.
- **reviewer deleted**
> heroin
- **judgement** (SUPPORTS) 'heroin or placebo (saline) was administered' supports medications=heroin. The LLM's bare 'heroin' does not.

### 37. `groups.name` — a system's pick does support the value

- **bucket** LLM wrong, retriever unknown · **paper** 84rGLhCbUJTh
- **value** `OCD patients`
- **retriever top-1** (section: methods)
> The OCD patients were recruited from an outpatient clinic at Shanghai Mental Health Center, Shanghai, China.
- **LLM evidence pass**
> Twenty-seven patients with OCD
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> Twenty-seven patients with OCD
- **judgement** (SUPPORTS) 'The OCD patients were recruited from an outpatient clinic' names the group. The LLM's deleted 'Twenty-seven patients with OCD' would also have done.

### 38. `definition` — scored wrong against a deletion of the correct sentence

- **bucket** LLM unknown, retriever wrong · **paper** 6oTrCJA43Jcd
- **value** `Two-sample t-test comparing the mean dwell time of each k-means connectivity state between ASD and T`
- **retriever top-1** (section: methods)
> Statistical analysis Two-sample t -test was used to compare the MDTs of ASD and TD children.
- **LLM evidence pass**
> able 2 shows the percentage of three connectivity strength levels at k
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> Two-sample t -test was used to compare the MDTs of ASD and TD children.
- **judgement** (SCORING ARTIFACT) The pick is 'Two-sample t-test was used to compare the MDTs of ASD and TD children' -- which *is* the definition. It scored `wrong` because the reviewer deleted that exact passage. Deleting a span is not a ruling that the sentence fails to support the value.

### 39. `groups.name` — scored wrong against a deletion of the correct sentence

- **bucket** LLM unknown, retriever wrong · **paper** 84rGLhCbUJTh
- **value** `OCD patients receiving SSRI treatment`
- **retriever top-1** (section: methods)
> ### Pharmacotherapy After receiving clinical feature measurements and DTI scans at baseline, 15 of the 27 OCD patients agreed to accept SSRI treatment only (6 of 15 OCD patients were drug-naive).
- **LLM evidence pass**
> 8*128. The acquisition time was 4 min 12 sec for each participant. ### Image processing We adopted voxel-based analysis (VBA) in
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> After receiving clinical feature measurements and DTI scans at baseline, 15 of the 27 OCD patients agreed to accept SSRI treatment only
- **judgement** (SCORING ARTIFACT) '15 of the 27 OCD patients agreed to accept SSRI treatment' is the group as recorded. Scored `wrong` against a deletion of the same passage.

### 40. `inferencesettingss.name` — scored wrong against a deletion of the correct sentence

- **bucket** LLM unknown, retriever wrong · **paper** 84rGLhCbUJTh
- **value** `inference_lsd_post_hoc`
- **retriever top-1** (section: methods)
> ANOVAs with least significant difference (LSD) post hoc tests were used to compare regional DTI-derived parameter changes before and after treatment in OCD patients.
- **LLM evidence pass**
> sor imaging (DTI) parameters. A. Fractional Anisotropy (FA) decrease; B. Mean Diffusion (MD) increase; C. Axial diffusion (AD) increase; D. Radial diffusion (RD) in
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> ANOVAs with least significant difference (LSD) post hoc tests were used to compare regional DTI-derived parameter changes before and after treatment in OCD patients
- **judgement** (SCORING ARTIFACT) 'ANOVAs with least significant difference (LSD) post hoc tests were used' is exactly the inference setting. Scored `wrong` against a deletion of the same sentence.

### 41. `interpretations` — scored wrong against a deletion of the correct sentence

- **bucket** LLM unknown, retriever wrong · **paper** QQCjAAT6SwwQ
- **value** `There was no significant main effect of group on aINS-based functional connectivity.`
- **retriever top-1** (section: results)
> Notably, there was no significant main effect of group on aINS-based functional connectivity.
- **LLM evidence pass**
> tex ( Zhou et al., 2020 ), and nucleus accumbens ( Shao et al., 2020 ), which may be correlat
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> Notably, there was no significant main effect of group on aINS-based functional connectivity.
- **judgement** (SCORING ARTIFACT) The pick is the interpretation verbatim: 'there was no significant main effect of group on aINS-based functional connectivity'. Scored `wrong` against a deletion of itself.

### 42. `definition` — partially supports

- **bucket** LLM unknown, retriever wrong · **paper** 84rGLhCbUJTh
- **value** `Areas of abnormal radial diffusivity (RD) in 27 patients with obsessive-compulsive disorder and 23 h`
- **retriever top-1** (section: results)
> Areas of abnormal DTI-derived parameters in 27 patients with obsessive-compulsive disorder and 23 healthy controls before medication.
- **LLM evidence pass**
> ant (F = 6.869, df = 3,43, p = 0.001). As to the regional differences, decreased RD was observed in the left striatum (baseline = 0.6
- **reviewer marked**
> _(no positive span -- this slot carries only a deletion)_
- **reviewer deleted**
> Areas of abnormal DTI-derived parameters in 27 patients with obsessive-compulsive disorder and 23 healthy controls before medication.
- **judgement** (PARTIAL) The pick is the table caption for abnormal DTI-derived parameters generally; the value names radial diffusivity specifically. Near, not exact -- and again scored against a deletion of that caption.
