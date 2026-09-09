"Seed-based connectivity of the left amygdala, computed per participant with white-matter, CSF and
motion regressors, then compared between patients and controls in a group model with age, sex and
scanner as covariates of no interest."

Two design matrices, so two records — and one model, so a link:

{{block}}

{{block}}
→ **`contrast`**, adjusted for `term-trs`, `term-motion` **and** `term-vim-timecourse`.

Two of those three covariates are columns of a record this analysis does not name. That is the
point: motion regressed out at the first level adjusts the group betas, and without the link the
record asserted the opposite by omission.

**Do not cell the seed.** The link makes an unsigned `{term: term-vim-timecourse}` cell constructible,
and it is the trap: a cell says the contrast *tested* that column, and a tested continuous
within-subject term derives `parametric_modulation` by step 2 of §3 — so the diagnosis contrast
stops reading as a contrast. What the map is *of* is `Measure` and `ConnectivityDetails`; what the
contrast *compared* is the cells. The seed belongs in the adjustment set, which is exactly what the
connectivity beta is conditional on.

**Name the seed once, as a `Region`.** The place and the column carrying its signal are two
things: `term-vim-timecourse` is the column, and the region is a `Region` that `term-vim-timecourse` points at through
`ModelTerm.region` and every analysis built on that map names in `seed_regions`. As bare
strings one seed becomes three spellings — `left VIM`, `VIM seed time series`,
`Left VIM connectivity` — that nothing joins. Its provenance is
`Region.definition_method`, so a seed taken from this study's own earlier contrast is
`same_study_analysis`, and that contrast names it in `defines_regions`.

The rule is unchanged by stages: cell what the comparison compared, and nothing else. Where a
first-level column genuinely *is* what was compared — a group contrast of a task condition fitted
per subject — cell it, and the derivation reads it as it reads any other factor.

**A crossing spanning the stages** is a product column on the stage that fitted it, naming the
lower stage's column directly: a group-level `term-dx-x-vim` with
`interaction_with: [term-diagnosis, term-vim-timecourse]` is "the seed's connectivity related to diagnosis", and derives
`interaction` by §5.4's rule. The lower column is never copied upward.

**Two seeds are two chains.** The left-VIM and right-VIM group models have identical term lists
and different inputs, and the input is part of the specification, so they are two records rather
than one shared by four analyses.

**When not to split.** `inputs_from` records a stage the source describes. A one-sample activation
map has a group stage too — an intercept over the first-level contrast images — and papers say
nothing about it, so it takes no record of its own and the first-level record stands alone. §5.1 is
that case, and most of §5 with it: a two-stage fit reported as one map is one record. Where such a
stage *is* described, its `terms` is legitimately empty; do not invent an intercept term.

**In the paper.**

> a VIM seed-based functional connectivity (FC) analysis of resting-state functional magnetic resonance imaging (RS-fMRI) data was performed to characterize the VIM FC network in ET patients.
>
> Fisher's z-transformation was applied to improve the normality of these correlation coefficients, and individual VIM-related RS-FC maps were constructed.
>
> We combined the group-level significant brain regions into a mask, within which we further identified the group differences using the random-effects two-sample t-test.
>
> Compared with HCs, ET patients displayed VIM-related FC changes, primarily within the VIM-motor cortex (MC)-cerebellum (CBLM) circuit, which included decreased FC in the CBLM and increased FC in the MC.

**Referent** `6WJs2gBAhcQL` (pmid 26467643) — Fang et al. 2016, a VIM seed connectivity map
computed per participant and then compared between essential tremor patients and controls. The
seed is named as a region and cell-ed in nothing; what the contrast compared is diagnosis.

---
