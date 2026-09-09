{{block}}
→ **`cross_subject_regression`**, adjusted for whatever else is in the model but *not* for
`term-gmd`. `Mediation` is present only for a mediation analysis, and both its fields are
required, so a path always names its mediator.

Which path was tested decides the mediator's status. A `direct` path is by definition the effect
holding the mediator constant, so there it *is* adjusted for. An `indirect` path is undefined without
it and a `total` path is estimated without conditioning on it, so in neither is it a covariate.

**In the paper.**

> of GM density (GMD) and used this information to reassess the age-related relationships between age (the predictor variable) and ICA activity in the bilateral PFC network (the outcome variable)
>
> mediation analysis can be conceptualised as a series of three separate regression equations testing different components of the mediation hypothesis in each voxel within: 1) the age-related decline in GMD (the a effect), 2) the relationship between GMD and ICA loading on the bilateral PFC network, controlling for age (the b effect)

**Referent** `5PXzmhEsxc2e` (pmid 25172389) — age acting on a prefrontal network's loading through
grey matter density. Transcribed, not corrected.

Finding it took eleven papers, and the search that worked is worth recording. Papers *about*
mediation, found by title and abstract, produced no `Effect.mediation` at all — four of them,
including one with the word in its title. Mediation reported in prose and figures never reaches
stage 1, and an `Effect` the pipeline never created cannot carry a `mediation` block. Searching
**table captions** instead — "MNI coordinates for significant mediation clusters", "Path a-, b- and
a×b-related brain activations" — found papers that table their mediation, and three of the first
four produced the block. If a construct is missing across a corpus, check whether the papers put it
in a table before concluding the extractor cannot see it.

`b9EfXh32hvPV` (pmid 36221050) is the same shape reported the other way: an indirect effect of body
mass index on negative symptoms via insular grey matter, with total and direct effects
non-significant. That last part is why `path` is a field and not an inference — the same three
variables carry three different answers, and only the tested one is the record's.
