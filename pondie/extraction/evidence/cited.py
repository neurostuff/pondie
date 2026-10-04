"""Evidence written once per sentence, turned into the evidence each field carries.

A quote in every field repeats the same sentence wherever one sentence supports several
values -- a group's n, its sex counts and its diagnosis are often one sentence. Two reply
formats write each sentence once instead:

  indexed   the paper is shown as numbered sentences (`[S12] ...`), and a field's
            `evidence` lists the numbers. Nothing is quoted, so nothing can be misquoted.
  inverted  a top-level `support` list: each entry one verbatim sentence and the paths of
            the fields it supports (`grp_ptsd.age_mean`, `ana_1.effect.cells[0].direction`).

`expand` turns either into `{"status": "present", "sets": [{"quotes": [...]}]}` on each
field -- the shape every other reply has -- so nothing after the model pass changes. It also
turns `silent_default`, the structured reply's spelling of plain silence, into no reason.
"""

from __future__ import annotations

import re
from typing import Any

from pondie.extraction.prompt.fill import PLAIN
from pondie.extraction.prompt.preprocess import sentence_spans
from pondie.formats import values


def numbered(text: str) -> str:
    """`text` with `[S<n>]` before each sentence, lines kept, so a field can cite a number."""
    marks = {start: n for n, (start, _end) in enumerate(sentence_spans(text), 1)}
    out, last = [], 0
    for start in sorted(marks):
        out.append(text[last:start])
        out.append(f"[S{marks[start]}] ")
        last = start
    out.append(text[last:])
    return "".join(out)


def expand(payload: dict[str, Any], evidence_format: str, text: str) -> list[str]:
    """Rewrite `payload`'s cited evidence as per-field quotes, in place. Returns notes."""
    notes: list[str] = []
    if evidence_format == "indexed":
        notes += _expand_indexed(payload, text)
    elif evidence_format == "inverted":
        notes += _expand_inverted(payload)
    silent = _drop_silence(payload)
    if silent:
        notes.append(f"{silent} slot(s) marked silent_default: recorded as plain not_reported")
    return notes


def _fields(node: Any):
    if isinstance(node, dict):
        if values.is_field(node):
            yield node
            return
        for value in node.values():
            yield from _fields(value)
    elif isinstance(node, list):
        for value in node:
            yield from _fields(value)


def _cite(field: dict[str, Any], quotes: list[str]) -> None:
    if quotes:
        held = field.get("evidence") if isinstance(field.get("evidence"), dict) else {}
        sets = list(held.get("sets") or []) + [{"quotes": quotes}]
        field["evidence"] = {"status": "present", "sets": sets}


def _expand_indexed(payload: dict[str, Any], text: str) -> list[str]:
    spans = sentence_spans(text)
    unknown = 0
    for field in _fields(payload):
        cited = field.pop("evidence", None)
        if not isinstance(cited, list):
            continue
        quotes = []
        for number in cited:
            if isinstance(number, int) and 1 <= number <= len(spans):
                start, end = spans[number - 1]
                quotes.append(text[start:end])
            else:
                unknown += 1
        _cite(field, quotes)
    return [f"{unknown} cited sentence number(s) name no sentence"] if unknown else []


_STEP = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _resolve(path: str, by_id: dict[str, Any], payload: dict[str, Any]) -> Any:
    """`grp_ptsd.age_mean`, `ana_1.effect.cells[0].direction`, `study.design.allocation`."""
    steps = [m.group(1) if m.group(1) is not None else int(m.group(2))
             for m in _STEP.finditer(path.strip())]
    if not steps:
        return None
    node = payload.get("study") if steps[0] == "study" else by_id.get(steps[0])
    for step in steps[1:]:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return None
    return node if values.is_field(node) else None


def _expand_inverted(payload: dict[str, Any]) -> list[str]:
    support = payload.pop("support", None) or []
    by_id: dict[str, Any] = {}

    def index(node: Any) -> None:
        if isinstance(node, dict) and not values.is_field(node):
            if isinstance(node.get("local_id"), str):
                by_id.setdefault(node["local_id"], node)
            for value in node.values():
                index(value)
        elif isinstance(node, list):
            for value in node:
                index(value)

    index(payload)
    unresolved = 0
    for entry in support:
        if not isinstance(entry, dict) or not isinstance(entry.get("sentence"), str):
            continue
        for path in entry.get("fields") or []:
            field = _resolve(str(path), by_id, payload)
            if field is None:
                unresolved += 1
            else:
                _cite(field, [entry["sentence"]])
    notes = [f"support: {len(support)} sentence(s)"]
    if unresolved:
        notes.append(f"{unresolved} supported path(s) name no field")
    return notes


def _drop_silence(payload: dict[str, Any]) -> int:
    count = 0
    for field in _fields(payload):
        if field.get("unreported_reason") == PLAIN:
            del field["unreported_reason"]
            count += 1
    return count
