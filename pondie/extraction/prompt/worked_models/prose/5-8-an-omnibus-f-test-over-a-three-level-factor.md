{{block}}
→ **`omnibus`**. All three levels take part and none is signed. Giving them cells rather than
omitting them is what keeps the factor out of the adjustment set: it was tested, not controlled for.

`undirected` and not the other two unsigned values, and §4's two questions say why. The levels were
not held: the F compares them against each other rather than taking a contrast within any of them,
so `held` on all three would claim the factor was held at three levels at once — the shape
`check_unsigned_cells` flags as a miscoded F. Nothing was withheld: an F over three levels
returns one statistic for the set and has no per-level direction to print, so this is not a withheld
sign either. Reporting the follow-up contrasts would supply signs, but those are separate Analyses
(§2) and would not sign *these* cells.

That distinction is what makes this an `omnibus` at all. Had the paper run a two-level comparison
and merely omitted which way it went, the cells' `direction` would be `not_reported` and step 6
would derive a `contrast`.

**In the paper.**

> Accuracy and reaction time were compared using 3 × 3 analysis of variances (ANOVAs) with group (monolingual, bilingual, IA) and task (0-back, 1-back, 2-back) as factors.
>
> There was a significant main effect of condition in the left superior frontal gyrus, inferior parietal lobule and posterior cingulate, and in the right anterior insula, anterior cingulate and middle frontal gyrus

**Referent** `7EEyXsyEDf2Q` (pmid 26624517), §5.7's paper and §5.7's model — the "main effect of
condition" coordinate table off the 3 × 3 ANOVA. The same three load levels §5.7 contrasted at
its extremes, here all cell-ed and none signed.
