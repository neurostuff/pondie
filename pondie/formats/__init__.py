"""What a record and its source text are made of. A second implementation of any of it is a bug.

Four formats, and every one has a history of a second version drifting from it:

    values       the `ExtractedValue` wrapper: a value, where it came from, what warrants it
    text_index   the normalization every `start_char` in a record is relative to
    table_parse  the CSV -> markdown render whose output that text has inlined
    parse_keys   `<table_id>#<ordinal>`, the address of one row group in a table parse

These are agreements, not utilities. `text_index`'s own docstring states the rule for the
whole package -- "Everything that produces or consumes those offsets must agree on the
normalization performed here" -- and the same holds for the other three: a wrapper read two
ways reports a `not_reported` field as a scalar, a parse key numbered two ways attaches an
analysis to another contrast's coordinates.

The history is why this is a directory rather than four loose modules. Across four review
passes: **nine** hand-rolled `ExtractedValue` unwrappers, disagreeing at the edges about
whether a non-wrapper is `None` or itself; two text folds disagreeing on accented characters;
`table_parse` noting that `ns-validate` keeps "its own superset of this module".

It sits at the bottom of the package and imports nothing but `paths`. `schema`, `extraction`,
`normalization`, `query` and `benchmark` all read records through it -- which is also why
`parse_keys` is here rather than in `extraction.corpus.tables`, where it made `query` import
the extraction package and closed a cycle between two of the three pipelines.
"""
