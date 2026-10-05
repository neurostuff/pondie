"""Evidence cited by sentence number, turned into the evidence each field carries.

The paper is shown as numbered sentences (`[S12] ...`), and a value's `evidence` lists the
numbers of the sentences that state it. One sentence often supports several values -- a
group's n, its sex counts and its diagnosis -- and a number costs a few tokens where a quote
repeats the sentence. Nothing is quoted, so nothing can be misquoted.

`expand` turns the numbers into `{"status": "present", "sets": [{"quotes": [...]}]}`, the
shape every other reply has, so nothing after the model pass changes.
"""

from __future__ import annotations

import re
from typing import Any

from pondie.extraction.prompt.preprocess import sentence_spans
from pondie.formats import values


def numbered(text: str) -> str:
    """`text` with `[S<n>]` before each sentence, lines kept."""
    marks = {start: n for n, (start, _end) in enumerate(sentence_spans(text), 1)}
    out, last = [], 0
    for start in sorted(marks):
        out += [text[last:start], f"[S{marks[start]}] "]
        last = start
    out.append(text[last:])
    return "".join(out)


def _sentences(numbers: Any, text: str, spans: list[tuple[int, int]]) -> tuple[list[str], int]:
    """The sentences `numbers` name, and how many name none."""
    found, unknown = [], 0
    for n in numbers if isinstance(numbers, list) else []:
        if isinstance(n, int) and 1 <= n <= len(spans):
            found.append(text[spans[n - 1][0] : spans[n - 1][1]])
        else:
            unknown += 1
    return found, unknown


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


def expand(payload: dict[str, Any], text: str) -> list[str]:
    """Replace each field's cited numbers with its sentences, in place. Returns notes."""
    spans = sentence_spans(text)
    unknown = 0
    for field in _fields(payload):
        if not isinstance(field.get("evidence"), list):
            continue
        quotes, missed = _sentences(field.pop("evidence"), text, spans)
        unknown += missed
        if quotes:
            field["evidence"] = {"status": "present", "sets": [{"quotes": quotes}]}
    return [f"{unknown} cited sentence number(s) name no sentence"] if unknown else []


def quote_answers(answers: dict[str, Any], text: str) -> None:
    """Replace each `fill` answer's cited numbers with its sentences, in place."""
    spans = sentence_spans(text)
    for answer in answers.values():
        if isinstance(answer, dict) and isinstance(answer.get("evidence"), list):
            answer["evidence"] = _sentences(answer["evidence"], text, spans)[0]


#: A sentence that states a result, and one about the brain. Both, to pass over the
#: demographic and behavioural tests a Results section also reports.
_RESULT = re.compile(
    r"\b(significant(ly)?|greater|reduced|smaller|larger|lower|higher|increase[ds]?|"
    r"decrease[ds]?|correlat\w+|[tTFZz]\s*[=(]\s*-?\d|p\s*[<=]\s*0?\.\d|peak)\b"
)
_BRAIN = re.compile(
    r"\b(activation|activity|volume|gr[ae]y matter|white matter|density|thickness|"
    r"connectivity|bold|signal|cluster|voxel|fractional anisotropy|cortex|cortical|gyrus|"
    r"hippocamp\w*|amygdala|insula\w*|cingulate|thalam\w*|striatum|talairach|mni)\b",
    re.I,
)


def unanalysed_results(payload: dict[str, Any], text: str, cap: int = 40) -> list[int]:
    """Numbers of the Results sentences about the brain that no analysis cites.

    A result the extraction read and did not encode. `payload`'s citations must already be
    expanded; they are matched back to their sentences.
    """
    from pondie.extraction.evidence.retrieval import sectionize

    spans = sentence_spans(text)
    number = {text[a:b]: n for n, (a, b) in enumerate(spans, 1)}
    analysed = {
        number[q]
        for field in _fields(payload.get("analyses") or [])
        for s in (field.get("evidence") or {}).get("sets") or []
        for q in s.get("quotes") or []
        if q in number
    }
    results = [(a, b) for a, b, label in sectionize(text) if label == "results"]
    out = []
    for n, (a, b) in enumerate(spans, 1):
        if n in analysed or (results and not any(x <= a < y for x, y in results)):
            continue
        if _RESULT.search(text[a:b]) and _BRAIN.search(text[a:b]):
            out.append(n)
    return out[:cap]
