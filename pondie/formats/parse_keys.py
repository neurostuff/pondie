"""The address space of a coordinate-table parse: `<table_id>#<ordinal>`.

A format rather than a helper. `Analysis.source_table_analysis` holds one of these and it is
the only exact route from an analysis to the coordinate rows it was read off, so three
separate places have to agree on how they are numbered: the prompt prints them to the model,
the builder resolves what comes back, and the query engine joins on them to find the foci.

It lived in `extraction.corpus.tables`, which made `pondie.query` import the extraction
package to read a record -- the one edge that closed a cycle between two of the three
pipelines the package advertises.
"""

from __future__ import annotations


def parse_keys(analyses: list[dict]) -> list[str]:
    """A stable address per parsed entry, positionally aligned with `analyses`.

    `Analysis.source_table_analysis` holds one of these, and it is the only exact route
    from an analysis to the coordinate rows it was read off. Both sides of that contract
    must number identically: `render.stage1_block` prints the key to the model and
    `builder.resolve_source_table_analysis` resolves what comes back.

    Numbered over EVERY entry, including the withheld half of a sign-split. The prompt
    hides withheld entries -- the paper has no prose for them -- and numbering only what
    is shown makes hiding one renumber its siblings, so the model is told `t1#2` and the
    builder resolves `t1#2` to a different row group. A wrong key that exists is worse
    than a missing one: it passes the join and attaches the analysis to another
    contrast's coordinates.
    """

    ordinals: dict[str, int] = {}
    keys: list[str] = []
    for entry in analyses:
        table_id = str((entry or {}).get("table_id") or "")
        ordinals[table_id] = ordinals.get(table_id, 0) + 1
        keys.append(f"{table_id}#{ordinals[table_id]}")
    return keys
