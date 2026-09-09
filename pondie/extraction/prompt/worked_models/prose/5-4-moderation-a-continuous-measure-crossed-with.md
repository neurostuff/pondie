"Regions showing a symptom-severity by diagnosis interaction."

{{block}}
→ **`interaction`**, adjusted for `term-ctq`, `term-ptsd` and `term-age`.

This is the one crossing the cells alone cannot express, and the reason `interaction_with` exists. A
continuous term has no levels, so it cannot be crossed; without the product column this would read as
a plain regression on maltreatment severity. The product column also holds the only thing that can
carry the moderation's *direction* — "maltreatment was negatively associated with rACC activation in
the PTSD group but not in controls" is a fact about the crossing, not about either term's own slope.

**Do not add a product column for a crossing of two categorical factors.** There the crossed levels
already say it (5.5), the column decides nothing, and records that add one are flagged for review.

**In the paper.**

> To test for regions showing different effects of child maltreatment in the PTSD versus control groups, group-level models included an interaction term for CTQ total score × PTSD, main effects of CTQ score and PTSD status, and an age covariate.

**Referent** `5P7tnuyp5NTP` (pmid 27062552) — Stevens et al. 2016, "Interaction of CTQ and PTSD
diagnosis". The paper also reports each group's slope separately, and the extractor cell-ed those
the way §5.5's last row does: `{term: term-ctq, direction: positive}` alongside
`{term: term-ptsd, level: PTSD, direction: held}`, the held level marking the group the
slope was taken within.
