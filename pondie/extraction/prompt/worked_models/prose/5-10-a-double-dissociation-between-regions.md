{{block}}
→ **`interaction`**. The point of signing the region axis is that it makes the group direction
*scoped*: "controls > patients" is asserted at the posterior seed, not of the analysis as a whole.
Without it the record reads as a plain group contrast and asserts a difference true in one region
and false in the other.

**In the paper.**

> a region at the inferior frontal sulcus involved in cognitive action control was partitioned into an anterior and posterior subdivision based on their whole-brain co-activation profiles
>
> To analyze seed-specific FC differences between patients and healthy controls, that is, FC group differences that are significantly higher for one seed compared to the other, we tested for the “seed × subject group” interaction effects in conjunction with the positively correlated network of the respective seed in the respective subject group.
>
> When testing for specific connectivity differences, i.e., the “seed × subject group” interaction (in the direction of a PD-related posterior right dlPFC connectivity decrease) for the medical OFF condition

**Referent** `6Ts55HvrSTEJ` (pmid 28611616) — connectivity of an anterior and a posterior right
dlPFC seed in Parkinson's disease. The extractor signed **both** axes unprompted, and the region
levels join their declarations; the correction above is on the group levels, which it cell-ed `HC`
and `PD` against declarations reading `healthy controls` and `Parkinson's disease patients` — §5.5's
24%, not a failure of this shape.

A factor comparing places is scarce in coordinate tables — across 39,192 pubget table captions, a
region axis crossed with a condition appears essentially only where the regions are seeds, as here,
because everywhere else such a factor is reported as an ROI analysis and ROI analyses do not
produce the tables stage 1 reads.

`3agtZxaWUcQV` (pmid 16154453) — Simons et al. 2005, medial against lateral anterior PFC over task
and position memory — is the same shape and shows the failure. The extractor built its region factor
correctly, filled `FactorLevel.regions`, and then **cell-ed only the context axis**, producing
exactly the plain condition contrast this example warns against. That record also settles something
this example used to assert: it carried the comment "no entity slots: a region is not a study
entity", and that is wrong.
