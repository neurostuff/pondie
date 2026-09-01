"""Stage 4: ask for a supporting quote for every value the extraction passes emitted.

Evidence is extracted separately because carrying it inline makes the extraction
worse, not merely more expensive. Measured on the pipeline_eval benchmark: evidence
was 57% of output tokens, and stripping it took analysis recall from 94% to 98%,
unparseable records from 6 to 0, and cost from $0.0110 to $0.0084 per paper. It was
crowding out the values it was meant to support.

The extraction passes emit values without an `evidence` key. This pass adds one to every
field. The model returns quotes, never offsets—it cannot count
characters -- and `builder.py` locates them in the normalized text, which is
what lets the integrity gate assert `text == source[start_char:end_char]`.

Fields are addressed by the dotted path `build_record` already uses in its reports,
so a quote that fails to resolve names the same field in both tools.

    pondie extract --pmids papers.pmids --run <run> --model <model> --stages evidence
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from pondie.extraction.evidence import retrieval
from pondie.extraction.models import EvidenceCounts

#: Paths per call. Large enough that a paper is a handful of calls, small enough

SYSTEM = """You locate supporting quotes in a scientific paper.

You are given a paper and a list of facts already extracted from it, each with an id
and the value that was recorded. For each id, return the single shortest span of the
paper that supports that value.

Rules:
1. Emit ONE JSON object mapping id -> quote. No prose, no markdown fence.
2. A quote MUST be copied character-for-character from the paper text given to you.
   It is located by exact match and a paraphrase is discarded, taking the evidence
   for that field with it.
3. Prefer one sentence. Never return a whole paragraph when a clause will do.
4. If the paper does not state the fact anywhere, OMIT that id entirely. Do not
   guess, do not return an approximate sentence, and do not invent one. An omitted
   id is recorded honestly as unsupported; a fabricated quote is a false citation.
5. Some values are classifications the paper never words that way (a controlled
   term such as "between_subject"). Quote the sentence the classification was read
   from, not a sentence containing the term."""


def iter_fields(node: Any, path: str = ""):
    """Every ExtractedValue in a payload, with the dotted path build_record reports."""

    if isinstance(node, dict):
        if "extraction_status" in node:
            yield path, node
            return
        for key, value in node.items():
            yield from iter_fields(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_fields(value, f"{path}[{index}]")


def owners(node: Any, path: str = "", owner: str = "") -> dict[str, str]:
    """path -> the name of the entity the field hangs off.

    The retriever scores a unit higher when it names the entity, and an entity's name is
    not recoverable from a dotted path. Cheap to collect on the way past.
    """

    found: dict[str, str] = {}
    if isinstance(node, dict):
        if "extraction_status" in node:
            return {path: owner}
        mine = owner
        for key in ("name", "title", "source_label", "modality"):
            value = (
                (node.get(key) or {}).get("value") if isinstance(node.get(key), dict) else None
            )
            if isinstance(value, str) and 3 < len(value) < 80:
                mine = value
                break
        for key, value in node.items():
            found |= owners(value, f"{path}.{key}" if path else str(key), mine)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found |= owners(value, f"{path}[{index}]", owner)
    return found


def rendered_value(field: dict) -> str:
    value = field.get("value")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value in (None, "", []) else str(value)


def union_span(
    reranker, units, path: str, field: dict, owner: str, quote: str | None
) -> str | None:
    """A second supporting passage for this field, or None.

    Only fires when the retriever clears its own gate, and never when it lands on the
    passage the model already quoted -- a duplicate set is not a second warrant. Worth
    +6.3 points of located evidence over the quote pass alone, measured over 173 fields
    with human evidence; see docs/evidence-union-design.md.
    """

    value = rendered_value(field)
    if not value:
        return None
    unit = retrieval.locate(
        reranker, units, re.sub(r"\[\d+\]", "", path), value, owner
    )
    if unit is None:
        return None
    # `unit.text` and not `unit.rendered`: build_record resolves a quote by exact match,
    # and a table row's rendered sentence appears nowhere in the paper.
    if quote and (quote in unit.text or unit.text in quote):
        return None
    return unit.text


def apply_evidence(
    payload: dict[str, Any],
    quotes: dict[str, str],
    reranker: Any = None,
    units: Sequence[str] = (),
) -> EvidenceCounts:
    """Put an evidence block on every field of a payload, in place.

    Every field, not only the ones a quote came back for: `evidence` is REQUIRED on
    `ExtractedValue`, and `build_record` leaves a field without one untouched -- so a
    missing block fails validation at the end of the run rather than here, where the
    reason is still visible.

    Three outcomes, and they are different claims. A field the paper did not report gets
    `not_applicable` -- there is no sentence to quote. A field with a quote, or a span the
    retriever found, gets `present`. A field that is asserted but that neither locator
    could place gets `not_found`, which is a defect a reviewer should see rather than a
    silence.
    """

    counts = dict.fromkeys(EvidenceCounts.model_fields, 0)
    owner_of = owners(payload)
    for path, field in iter_fields(payload):
        if field.get("extraction_status") != "extracted":
            field.pop("value", None)
            field["evidence"] = {"status": "not_applicable"}
            counts["not_reported"] += 1
            continue

        quote = quotes.get(path)
        sets = [{"quotes": [quote]}] if quote else []
        second = (
            union_span(reranker, units, path, field, owner_of.get(path, ""), quote)
            if reranker
            else None
        )
        if second:
            sets.append({"quotes": [second]})
            counts["unioned"] += 1
            if not quote:
                counts["recovered"] += 1

        if sets:
            field["evidence"] = {"status": "present", "sets": sets}
            counts["filled"] += 1
        else:
            field["evidence"] = {"status": "not_found"}
            counts["unsupported"] += 1
    return EvidenceCounts(**counts)


def describe(path: str, field: dict) -> str:
    """One line for the model: where the field sits and what was recorded."""

    value = field.get("value")
    if isinstance(value, list):
        rendered = "; ".join(str(item) for item in value)[:300]
    else:
        rendered = str(value)[:300]
    return f"{path} = {rendered}"
