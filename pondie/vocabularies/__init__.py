"""Fetched term lists, and the machinery for reaching the right entry in one.

    onvoc         ONVOC and the Cognitive Atlas: tasks, conditions, agents
    mondo         MONDO: diseases, with the `is_a` edges a rollup needs
    abbreviations every paper's own definitions, mined once into a store
    labels        do two labels name one thing -- which tokens carry identity, and how
                  two forms of a clinical noun are made to collide
    folding       case, punctuation, accents and plurals -- the orthography all four share

Neither extraction nor normalization owns these. Both use them: the extraction corpus builds
the abbreviation store, and the normalization field modules link values against ONVOC and
MONDO. They lived under `normalization/` with leading underscores, which made them look like
that package's private machinery -- and made `extraction` import `normalization` to reach
them, an edge between two packages the top-level docstring presents as sequential stages.

The two vocabulary classes are deliberately NOT one class. `mondo.Hierarchy` is parallel
arrays plus `is_a` edges, built so a rare subtype can be rolled up to the nearest ancestor
the corpus actually uses; `onvoc.TermIndex` is a flat concept list with four lookup indexes
and no hierarchy at all. Different structures for different questions. They shared the name
`Vocabulary` while sharing a package, which is most of why they read as duplication.

`labels` sits between `folding` and the two vocabularies, and exists because both `onvoc`
and `abbreviations` need it. They used to reach into each other for it -- `abbreviations`
deferring an import of `onvoc.stems` to compare two expansions while `onvoc` deferred one
back for the paper's own abbreviations, a mutual cycle suppressed at three call sites. The
direction is one way now: `onvoc` -> `abbreviations` -> `labels` -> `folding`.

This package imports `paths` and nothing else. Everything else may import it.
"""
