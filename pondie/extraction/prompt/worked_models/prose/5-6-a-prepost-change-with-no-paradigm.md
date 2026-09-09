{{block}}
→ **`contrast`**, adjusted for `term-vbm-group`. A longitudinal structural analysis has no
`Condition` and no `Task`; it links to data through `Analysis.acquisitions`. Nothing about the
encoding differs from 5.1 — the levels name `timepoints` instead of `conditions`, and that is what
makes it a change over time.

A crossover comparison of arms is the same with `arms:` on the levels. When the arms are separate
cohorts rather than a within-person crossing, the levels name `groups` and the allocation is on
`Group.arm`; `StudyDesign.assignment_structure` says which a study is.

**The same paper's other contrast runs the other way.** Ilg et al. also report a practice-related
*decrease* — "Decrease of mirror-reading-related activation after practice compared with before
practice". That result is the fMRI model's rather than this one's, so its cells sit on the fMRI
time factor; the point is that the two occasions carry the opposite signs, and nothing else about
the encoding changes:

{{block}}

The sign follows the measure, not the clock. A later level is not the plus side because it came
second; it is the plus side where the measure is **higher** and the minus side where the measure is
**lower**. The verb decides it — increased, decreased, attenuated, diminished — so read the verb
before copying either block: the two are identical apart from which level carries which sign.

**When the paper prints no sign at all**, neither block applies. A pre–post *t* test that the paper
reports only as "changed significantly after treatment", with no direction anywhere in the text,
tables or figures, has a sign that was withheld rather than absent: both cells take
`direction: not_reported`, and the wording goes in `Analysis.definition`. Signing them from the fact
that something changed asserts a direction the paper never gave. This is the row of §3's table that
no worked model above shows, and it is not rare.

**In the paper.**

> The comparison of GM before and after practice shows a significant increase in GM in a subset of the regions (gray) activated in the right occipital cortex during mirror reading (green).
>
> The longitudinal voxel-based morphometry analysis yielded an increase of gray matter in the right dorsolateral occipital cortex that corresponded to the peak of mirror-reading-specific activation.

**Referent** `39HoutR6iLMj` (pmid 18417700) — Ilg et al. 2008, a longitudinal VBM comparison of
grey matter after two weeks of mirror-reading practice against before it. The paper also runs an
fMRI task, and this analysis is not of it: the structural contrast names two scans and no
condition.
