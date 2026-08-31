"""The extraction passes: prompt construction, assembly, repair, evidence, validation.

These are the working implementation, moved here from the schema repository because they are
extraction logic and a schema repository should hold a schema. Carried across unedited apart
from their imports, so the move is a move and not a rewrite -- the benchmark in `benchmarks/`
is what would catch a rewrite going wrong, and it is green either side of it.

An ordinary package: each module imports its siblings relatively, and the schema modules it
reads -- `schema_utils`, `text_index`, `table_parse` -- come from the installed `study-schema`
distribution. Nothing here depends on import order or on which directory a caller ran from.
"""
