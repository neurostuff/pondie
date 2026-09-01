"""Reading records: iteration, and pulling a field out by a dotted path.

The value access goes through `values.value_of`, which takes the wrapper and the
slot's declared shape from the LinkML schema. Hand-rolling that unwrap conflates three
different claims -- absent, `not_reported`, and reported-empty -- and each conflation is a
silent wrong answer. See docs/pipeline-architecture.md, "The contract at each seam".
"""

from __future__ import annotations

import glob as globlib
import json
from pathlib import Path
from typing import Iterator

#: `schema_utils` is the reader that knows a slot's declared shape. It ships in the
#: `study-schema` distribution, which is this repository's submodule installed rather than
#: appended to the path, so the import resolves the same from anywhere.
from pondie import paths
from pondie.formats.values import NOT_REPORTED, value_of

__all__ = ["NOT_REPORTED", "iter_records", "strings_at", "value_of"]

DEFAULT = (str(paths.RUNS / "*" / "records" / "*.extraction.json"),)


def iter_records(patterns: tuple[str, ...] = DEFAULT) -> Iterator[tuple[str, dict]]:
    """(study id, record body) for every readable record the patterns reach."""
    for path in sorted({p for pattern in patterns for p in globlib.glob(pattern)}):
        if path.endswith(".raw.json"):
            continue
        try:
            body = json.loads(Path(path).read_text())
        except Exception:  # noqa: BLE001 -- a truncated record is not a reason to stop
            continue
        body = body.get("study") or body
        if isinstance(body, dict):
            yield Path(path).name.split(".")[0], body


def strings_at(body: dict, path: str) -> list[str]:
    """Every string a dotted path reaches, descending through lists as it goes.

    `groups.medication_status` and `groups.sex_distribution.category` are both one path; the
    walk does not care whether a step is a list, a wrapped value or a nested object.
    """
    nodes: list[object] = [body]
    for step in path.split("."):
        nxt: list[object] = []
        for node in nodes:
            nxt.extend(_descend(node, step))
        nodes = nxt
    return [s for s in (value_of(n) for n in nodes) if isinstance(s, str) and s.strip()]


def _listed(value: object) -> list:
    if value is NOT_REPORTED or value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _descend(node: object, step: str) -> list:
    """One step of the walk, telling a wrapped value from a nested entity.

    Both are mappings and only one has a `value`. `value_of` reads a mapping without one as
    `not_reported`, which is right for a wrapper and wrong for an entity -- a Task is an
    object to descend into, not a slot the paper declined to fill.
    """
    if not isinstance(node, dict):
        return []
    child = node.get(step)
    if isinstance(child, dict) and "value" not in child and "extraction_status" not in child:
        return [child]
    return _listed(value_of(child, True))
