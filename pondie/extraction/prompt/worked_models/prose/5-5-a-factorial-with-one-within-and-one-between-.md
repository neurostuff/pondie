Subject group (between) × task (within), one model, five results.

{{block}}

| result | cells | derives | adjusted for |
|---|---|---|---|
| main effect of group | both group levels `undirected` | `omnibus` | task, disease activity, DAS28 |
| main effect of task | both task levels `undirected` | `omnibus` | group, disease activity, DAS28 |
| group × task, F-test | all four levels `undirected` | `omnibus` | disease activity, DAS28 |
| rotation > comparison, within RA | task crossed + `held` on `RA patients` | `contrast` | disease activity, DAS28 |
| RA > HC, within rotation | group crossed + `held` on `rotation` | `contrast` | disease activity, DAS28 |

All five are this paper's, off the one model, and the record holds two more: the same simple
effect of task within the control group, and a comparison of active against remission patients.

Three things to read off that table. **Averaging over a factor is the absence of its cells** — the
main effect of group simply has no task cells, which is also how it comes to be adjusted for them.
Rows three and four are §4's pair in one model: the F-test cells every level `undirected` because
the test yields no per-level sign, and the simple effect's `RA patients` cell is `held` because the
comparison was taken within that level, which puts it on both sides.
**The two factors differ only in `variation_level`**; the cells have the same shape for a
between-subject and a within-subject factor, because what differs is a property of the design,
recorded once on the model.

The last two rows are the same comparison read along its two axes, and they are what makes the
held cell load-bearing: without it, "rotation vs comparison within RA" and "rotation vs comparison
within HC" are the same two cells and the record cannot tell them apart.

**In the paper.**

> A 2 × 2 factorial design analysis of variance (ANOVA) was designed for fMRI analyses, with the group (RA and HC group) and task (rotation and comparison) as factors.
>
> The differences of activation were analyzed for the main effect and simple effect of the group, task and the interaction effects of group by task.
>
> Compared to the control group, RA patients showed enhanced activation in the left precuneus, left superior frontal gyrus and right cingulate gyrus during the rotation task, with left hemisphere dominance.

**Referent** `Qa5HqrHq97Pm` (pmid 37559139) — a mental-rotation task in rheumatoid arthritis. This
one is **audited by hand**, and the audit is
[corrections/Qa5HqrHq97Pm.corrections.json](../corrections/Qa5HqrHq97Pm.corrections.json): 34
operations that take the record from five validator errors to none.

Four of the five defects are worth knowing because they are not this paper's:

- the three omnibus cells carried `extraction_status: not_reported` **in place of a level**, so the
  factor being tested named none of its own levels;
- `term-group` and the two DAS28 scores sat on `me-first-level`, a per-participant GLM, which cannot
  carry a between-subject column at all;
- task and group were each declared twice, once per stage, which §5.12 rejects;
- the two simple effects of task came out with **identical cells**, because the held group level was
  missing — the defect the fourth and fifth rows above exist to prevent.

The fifth is the ordinary one: `healthy control subjects` cell-ed against a declaration reading
`healthy controls`. Naming the model was wrong too — both simple effects named the first-level
stage, and an `Analysis` names the top stage and reaches downward through `inputs_from`, never up.

`6qSfdQCVbYhH` (pmid 11050021) is the same design in autism and shows the level-string rewrite on
its own: `explicit processing of emotional facial expressions` declared, `explicit` cell-ed, four of
its fifteen errors from that one abbreviation.
