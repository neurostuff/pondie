"""Writing what a normalizer made of a value into the value's `normalized` sidecar.

A record is filled in three passes: the value, the evidence for it, and its normalization.
This is the third pass's one writer. An entry sits beside the source's wording, never over
it, and names the code that wrote it in `mapping_tool`; a rerun of one normalizer replaces
its own entries and leaves every other normalizer's alone.

Why a sidecar rather than a sibling slot or an overwrite: docs/normalization-rationale.md,
"The normalized sidecar".
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path
from collections.abc import Callable, Iterator

from pondie.formats import values
from pondie.normalization import UNKNOWN
from pondie.normalization._lexicon import Decision

MAPPED, UNMAPPED, AMBIGUOUS, ABSENT = "mapped", "unmapped", "ambiguous", "absent"


@functools.lru_cache(maxsize=1)
def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- an installed copy has no checkout to ask
        return "unknown"


def tool(module: str) -> str:
    """`mapping_tool` for a pondie normalizer: its module, then the commit it ran at."""
    return f"pondie.normalization.{module} {_commit()}"


def owner(mapping_tool: str) -> str:
    """The normalizer a `mapping_tool` names, without the version: what a rerun replaces."""
    return mapping_tool.split(" ", 1)[0]


def entry(vocabulary: str, status: str, mapping_tool: str, **fields: object) -> dict:
    """One `Normalization`, with the fields that say nothing left out."""
    out: dict = {"vocabulary": vocabulary, "status": status}
    out.update({k: v for k, v in fields.items() if v not in (None, "", (), [])})
    out["mapping_tool"] = mapping_tool
    return out


def from_decision(
    decision: Decision, vocabulary: str, mapping_tool: str, item: int | None = None
) -> dict:
    """A closed field's `Decision` as an entry: mapped, ambiguous, or unmapped."""
    if decision.value in (UNKNOWN, None):
        status = AMBIGUOUS if decision.reason.startswith("matches ") else UNMAPPED
        return entry(vocabulary, status, mapping_tool, item=item, method=decision.reason)
    return entry(vocabulary, MAPPED, mapping_tool, value=decision.value, item=item,
                 method=decision.reason)


def replace(wrapper: dict, mapping_tool: str, entries: list[dict]) -> None:
    """Swap one normalizer's entries on a wrapper for `entries`; keep everyone else's.

    `normalized` is removed when nothing is left, so an unset sidecar still means no
    normalizer ran.
    """
    mine = owner(mapping_tool)
    kept = [e for e in wrapper.get("normalized") or []
            if owner(str(e.get("mapping_tool", ""))) != mine]
    if kept or entries:
        wrapper["normalized"] = kept + entries
    else:
        wrapper.pop("normalized", None)


def wrappers(body: dict, path: str) -> Iterator[dict]:
    """Every wrapper a dotted path ends on, descending through entities and lists of them."""
    *steps, last = path.split(".")
    nodes: list[object] = [body]
    for step in steps:
        nxt: list[object] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            child = node.get(step)
            if values.is_field(child):
                child = values.value_of(child, True)
            nxt.extend(child if isinstance(child, list) else [child])
        nodes = nxt
    for node in nodes:
        if isinstance(node, dict) and values.is_field(node.get(last)):
            yield node[last]


def annotate(
    body: dict,
    path: str,
    decide: Callable[[object], Decision],
    vocabulary: str,
    mapping_tool: str,
) -> dict[str, int]:
    """Read every reported value at `path` and write its entries. Tally of what was written.

    A list-valued slot gets one entry per item, each naming its `item`. A `not_reported`
    wrapper gets none: there is nothing to read, and a normalizer that did not read is unset.
    """
    tally = {"set": 0, "unmapped": 0}
    for wrapper in wrappers(body, path):
        value = values.value_of(wrapper)
        if value is values.NOT_REPORTED or value is None:
            replace(wrapper, mapping_tool, [])
            continue
        items = list(enumerate(value)) if isinstance(value, list) else [(None, value)]
        entries = [from_decision(decide(text), vocabulary, mapping_tool, item)
                   for item, text in items]
        replace(wrapper, mapping_tool, entries)
        tally["set"] += len(entries)
        tally["unmapped"] += sum(e["status"] != MAPPED for e in entries)
    return tally
