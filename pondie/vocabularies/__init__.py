"""Fetched term lists, and the machinery for reaching the right entry in one.

    onvoc         ONVOC and the Cognitive Atlas: tasks, conditions, agents
    mondo         MONDO: diseases, with the `is_a` edges a rollup needs
    abbreviations every paper's own definitions, mined once into a store
    folding       case, punctuation, accents and plurals -- the orthography all three share

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

This package imports `paths` and nothing else. Everything else may import it.
"""
