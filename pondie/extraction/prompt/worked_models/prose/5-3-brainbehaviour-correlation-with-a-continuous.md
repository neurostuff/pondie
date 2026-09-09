{{block}}
→ **`cross_subject_regression`**, adjusted for every other term in the model. The cell has no
`level` because a slope has none, and its `direction` is the sign of the fitted coefficient.

`term-sex` is categorical and declares its levels even though no contrast names it. A covariate is
a column like any other: what makes it a covariate is the absence of a cell, not a different kind
of term.

`variation_level: between_subject` is what makes this a cross-subject regression. Change it to
`within_subject` — a value regressor varying trial to trial — and the same cell pattern derives
**`parametric_modulation`**. That one field is the whole difference, and it is on the model because
it is a property of the measurement rather than of the contrast.

The covariates appear in no cell, which is how the record says the correlation was adjusted for
them. There is no covariate list.

**In the paper.**

> First, a whole-brain correlation analysis was conducted to uncover the brain areas related to perceived stress.
>
> A whole-brain correlation analysis showed that higher levels of perceived stress were associated with greater fALFF in the left superior frontal gyrus (SFG)
>
> the framewise displacement (FD; Van Dijk, Sabuncu, & Buckner, 2012) was calculated as a measure of head motion and was treated as a covariate in the subsequent data analyses
>
> Controlling for age, sex, and head motion

**Referent** `3KGhvY7MhanA` (pmid 31397949) — perceived stress against resting-state fALFF,
controlling for age, sex and head motion. Transcribed, not corrected.

`3qC7anyYszL4` (pmid 22076840) is the same shape in structural data — grey-matter volume on a trait
impulsiveness score — and differs in one instructive way: there the extractor typed `gender` as
`continuous`. Two levels and no order is a categorical term, and a covariate being uninteresting
does not make it exempt.
