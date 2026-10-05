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

import json
from pathlib import Path
from typing import Any


#: The `table_id` a prose-derived parse entry carries, making its keys `prose#1`, `prose#2`,
#: distinct from any real table's. Here and not with the stage that writes it, because
#: `benchmark` and `query` both compare against it and neither imports `extraction` -- the
#: same argument this module exists for. It was a constant in the prompt renderer and a bare
#: literal in the two places that write it.
PROSE_TABLE_ID = "prose"


def parse_keys(analyses: list[dict]) -> list[str]:
    """A stable address per parsed entry, positionally aligned with `analyses`.

    `Analysis.source_table_analysis` holds one of these, as does a `CoordinateSet.local_id`:
    the exact route between an analysis and the coordinate rows it was read off. Both sides
    of that contract must number identically: `render.stage1_block` prints the key to the
    model and `fix.resolve_source_table_analysis` resolves what comes back.

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


def split(key: str) -> tuple[str, str]:
    """(table_id, ordinal) of a key: the ordinal follows the last `#`."""
    table_id, _, ordinal = key.rpartition("#")
    return table_id, ordinal


def load(stage1: Path | None) -> list[dict[str, Any]]:
    """A stage-1 parse's entries, or none when there is no parse file."""
    if not (stage1 and stage1.is_file()):
        return []
    return json.loads(stage1.read_text(encoding="utf-8")).get("analyses") or []


def load_table_map(table_map: Path | None) -> dict[str, str]:
    """The tables stage's map, manifest `table_id` -> `Table.local_id`; empty without one."""
    if not (table_map and table_map.is_file()):
        return {}
    return json.loads(table_map.read_text(encoding="utf-8"))
