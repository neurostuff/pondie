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

import re
from pathlib import Path
from typing import Any


#: The `table_id` the upstream parse gives an entry read from the text rather than a table.
#: Here and not with the stage that writes it, because `benchmark` and `query` both compare
#: against it and neither imports `extraction` -- the same argument this module exists for.
PROSE_TABLE_ID = "prose"

#: What such an entry's key is spelled with: `text#1`, `text#2`, as study_schema spells a
#: text analysis's key (`ParsedAnalysis.key`), distinct from any real table's.
TEXT_KEY_PREFIX = "text"

#: The spelling records written before `TEXT_KEY_PREFIX` hold, `prose#1`. Read, never
#: written: `canonical` turns it into the current one.
OLD_TEXT_KEY_PREFIX = "prose"


def parse_keys(analyses: list[dict]) -> list[str]:
    """Each entry's key, positionally aligned with `analyses`.

    An entry read from ingestion's CoordinateParse carries the parse's own `key`, and that
    is its address. Only a stage-1 document, whose entries have none, is numbered by
    `positional_keys`. A document mixing the two is refused: numbering the keyless entries
    would mint positional keys beside the parse's, which is what reading the parse ends.
    """
    keyed = [bool((entry or {}).get("key")) for entry in analyses]
    if all(keyed) and analyses:
        return [str(entry["key"]) for entry in analyses]
    if any(keyed):
        raise ValueError("a parse document mixes keyed and unkeyed entries")
    return positional_keys(analyses)


def positional_keys(analyses: list[dict]) -> list[str]:
    """A stage-1 address per parsed entry, positionally aligned with `analyses`.

    `Analysis.source_table_analysis` holds one of these, as does a `CoordinateSet.local_id`:
    the exact route between an analysis and the coordinate rows it was read off. Both sides
    of that contract must number identically: `render.listing_block` prints the key to the
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
        prefix = TEXT_KEY_PREFIX if table_id == PROSE_TABLE_ID else table_id
        keys.append(f"{prefix}#{ordinals[table_id]}")
    return keys


def canonical(key: object) -> object:
    """`key` in the current spelling: `prose#3` -> `text#3`. Anything else is returned as is.

    For reading a record against keys `parse_keys` mints, so a record written before the
    respelling still joins to its rows.
    """
    old = f"{OLD_TEXT_KEY_PREFIX}#"
    if isinstance(key, str) and key.startswith(old):
        return f"{TEXT_KEY_PREFIX}#{key[len(old):]}"
    return key


#: A CoordinateParse key, `analysis#<16 hex>`, as `study_schema.keys.analysis_key` mints it.
CONTENT_KEY = re.compile(r"analysis#[0-9a-f]{16}")


def is_content_key(key: object) -> bool:
    """Whether `key` is spelled as a CoordinateParse key, `analysis#<16 hex>`."""
    return isinstance(key, str) and CONTENT_KEY.fullmatch(key) is not None


def is_positional(key: object) -> bool:
    """Whether `key` is a stage-1 `<table_id>#<ordinal>` rather than a CoordinateParse key.

    A content key's digest is all digits often enough to matter (0.05% of keys), so the
    digit test alone would call it positional.
    """
    if not isinstance(key, str) or "#" not in key or is_content_key(key):
        return False
    return split(key)[1].isdigit()


def split(key: str) -> tuple[str, str]:
    """(table_id, ordinal) of a key: the ordinal follows the last `#`."""
    table_id, _, ordinal = key.rpartition("#")
    return table_id, ordinal


def load(parse: Path | None) -> list[dict[str, Any]]:
    """A parse's entries, or none when there is no parse file.

    A CoordinateParse is read through study_schema's model; a legacy stage-1 document (read
    only to migrate old keys) as it is.
    """
    if not (parse and parse.is_file()):
        return []
    from pondie.formats.coordinate_parse import read_document

    return read_document(parse).get("analyses") or []
