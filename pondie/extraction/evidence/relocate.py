"""Find a better sentence for a citation, and swap only when it is better.

Ported from the pass this package's grounding came from, where it was the half that made
the numbers good. `grounding.review_spans` can say a citation looks wrong; it cannot say
what the right one is, and acting on doubt alone destroyed 46% of all spans across a
six-paper sample. This asks the proposer for the sentences that *do* support a value, scores
the answer against the incumbent, and keeps it only on a strict improvement.

What can only rise is the *score* of any field this touches. The span count can fall, and on
16759342 did -- 203 to 199 -- because a replacement substitutes the whole evidence block:
four `coordinate_space` fields cited a clean prose sentence *and* a table row rendered as
"| Talairach coordinates | t | | | |", and swapping both for the sentence alone is four
fewer spans and four better citations. Counting spans is not the measure.

The score is used comparatively and never as a verdict, which is the only sound use measured
for it: an absolute threshold rescued 7 of 72 known-good citations, while `new > old` needs
no threshold at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from pondie.extraction.evidence.grounding import Claim, Checker, expand, groundable
from pondie.extraction.record import spans as span_tools
from pondie.extraction.record.edit import Refusal
from pondie.formats import values

#: `field_id` is a short tag rather than the dotted path. Offered the path, the model
#: answered with the leaf -- "modality" for "acquisitions[0].modality" -- and across forty
#: contested fields the leaves repeat, so the answer could not be mapped back.
TEMPLATE: dict[str, Any] = {
    "fields": [{
        "field_id": "string",
        "supporting_sentences": ["verbatim-string"],
    }]
}

#: Asked about every contested field at once the reply ran past `max_new_tokens` and came
#: back as truncated JSON, which parses to nothing and reports as "no proposals" -- a silent
#: failure that reads as the model declining to answer.
BATCH = 12

#: Beyond this many, the block itself crowds out the paper.
MOST = 40


@dataclass
class Contested:
    """One citation worth a second look, and what it currently rests on."""

    tag: str
    path: str
    field: str
    value: Any
    premise: str


def contested(record: Mapping[str, Any], weak: Sequence[tuple[str, float]]) -> list[Contested]:
    """The doubted citations, plus every field the record admits it never found one for.

    `not_found` fields were previously reached only when a fallback score failed, which left
    96 fields across five papers that nothing ever asked about. A field saying no sentence
    was located is the clearest possible case for going to look.
    """
    from pondie.formats.values import iter_fields

    doubted = {path for path, _score in weak}
    rows: list[Contested] = []
    for path, node in iter_fields(record):
        slot = path.rsplit(".", 1)[-1]
        if not groundable(slot, node):
            continue
        quoted = [sp.get("text", "")
                  for s in ((node.get("evidence") or {}).get("sets") or [])
                  for sp in (s.get("spans") or [])]
        premise = " ".join(q for q in quoted if q)
        missing = (node.get("evidence") or {}).get("status") == "not_found" and not premise
        if path not in doubted and not missing:
            continue
        rows.append(Contested(tag=f"f{len(rows) + 1}", path=path,
                              field=slot.replace("_", " "),
                              value=values.read(node), premise=premise[:1500]))
        if len(rows) >= MOST:
            break
    return rows


def block(rows: Sequence[Contested]) -> str:
    """The contested facts written out for the model, with what each currently cites.

    The path, the value and the sentence that failed are all handed over. Naming only the
    class -- which is all an entity sweep needs -- says nothing about the field in question.
    """
    lines = ["## Extracted values whose cited sentence may not support them", ""]
    for row in rows:
        lines.append(f"- [{row.tag}] {row.field} of {row.path.split('[')[0]} "
                     f"= {str(row.value)[:80]}")
        lines.append(f"    cited: {(row.premise or '(nothing cited)')[:170]}")
    lines += [
        "",
        "For each `field_id` above, quote the sentence or sentences from the paper that DO "
        "support the value -- copied character for character, and more than one where it "
        "takes more than one. If the paper does not state it, return no sentences.",
        "",
    ]
    return "\n".join(lines)


def _evidence(document: str, sentences: Sequence[Any]) -> dict[str, Any] | None:
    """Quotes located in the document, as an evidence block. None if none of them resolve.

    Resolved against the document rather than the premise: every offset in the record is
    measured against the normalized text, so a span written against a section slice
    addresses a different string.
    """
    located = []
    for sentence in sentences or []:
        quote = str(sentence or "").strip()
        if len(quote) < 8:
            continue
        try:
            located.append(span_tools.resolve(document, quote).as_record())
        except span_tools.SpanResolutionError:
            continue
    if not located:
        return None
    return {"status": "present", "sets": [{"spans": located}]}


def _retrieved(reranker: Any, units: Sequence[Any], record: Mapping[str, Any],
               rows: Sequence[Contested], document: str) -> dict[str, dict]:
    """The local locator's candidate for each contested field, by tag.

    The same question the proposer is asked -- which sentence supports this value -- put to
    a cross-encoder over the paper's own sentences instead of to a generative model. It used
    to run in `evidence` and write a second span whenever it cleared its own gate, with no
    refusal recorded anywhere. Here its answer is a candidate like any other and has to beat
    the incumbent to be kept, which is the one rule that cannot lower a record's support.
    """
    if reranker is None or not units:
        return {}
    from pondie.extraction.evidence import retrieval

    found: dict[str, dict] = {}
    for row in rows:
        unit = retrieval.locate(reranker, list(units), row.path.split("[")[0],
                                str(row.value), label_for(record, row.path))
        if unit is None:
            continue
        located = _evidence(document, [unit.text])
        if located is not None:
            found[row.tag] = located
    return found


def label_for(record: Mapping[str, Any], path: str) -> str:
    """The name of the entity a leaf sits on, which the locator scores as a bonus."""
    from pondie.extraction.record.edit import label_of

    head = path.split(".", 1)[0]
    name, _, index = head.partition("[")
    entities = record.get(name)
    if not isinstance(entities, list) or not index:
        return ""
    position = int(index.rstrip("]"))
    if position >= len(entities):
        return ""
    return label_of(entities[position]) or ""


def _fragment(candidate: str, incumbent: str) -> bool:
    """Is `candidate` a shorter piece of `incumbent`, whitespace aside?

    Not equality: the proposer's copy of a sentence differs from the record's in spacing,
    and a quote cut mid-word is a prefix rather than a duplicate.
    """
    one, other = " ".join(candidate.split()), " ".join(incumbent.split())
    return len(one) < len(other) and one in other


def relocate(record: MutableMapping[str, Any], document: str, premise: str,
             weak: Sequence[tuple[str, float]], proposer: Any, checker: Checker,
             refused: list, abbreviations: Any = None, paper: str = "",
             reranker: Any = None, units: Sequence[Any] = ()) -> list[str]:
    """Re-cite what `review_spans` doubted. Returns the paths that improved.

    A replacement has to beat what it replaces. A field with no span has nothing to be
    better than, so its floor is zero -- scoring it against the whole paper, as an earlier
    fallback did, set a bar the cited sentence itself never had to clear.
    """
    rows = contested(record, weak)
    if not rows:
        return []
    # Two sources of candidate sentences, one acceptance rule. The retriever answers for
    # every contested field it can place; the proposer answers for the ones it recognises.
    retrieved = _retrieved(reranker, units, record, rows, document)

    from pondie.extraction.recall import Starved
    from pondie.formats.values import iter_fields

    replies: list[Mapping[str, Any]] = []
    for start in range(0, len(rows), BATCH) if proposer is not None else ():
        chunk = rows[start:start + BATCH]
        template = {"fields": [{**TEMPLATE["fields"][0],
                                "field_id": [row.tag for row in chunk]}]}
        try:
            answer = proposer.ask(
                template,
                "Find the sentences in this paper that support each extracted value "
                "listed below.\n\n" + block(chunk),
                premise, what="evidence")
        except Starved as starved:
            refused.append(Refusal("evidence", str(starved)))
            continue
        replies.extend(f for f in (answer.get("fields") or []) if isinstance(f, Mapping))

    nodes = {path: node for path, node in iter_fields(record)}
    by_tag = {row.tag: row for row in rows}
    candidates = []
    for reply in replies:
        row = by_tag.get(str(reply.get("field_id") or "").strip())
        node = nodes.get(row.path) if row else None
        if node is None:
            continue
        fresh = _evidence(document, reply.get("supporting_sentences") or [])
        if fresh is None:
            continue
        candidates.append((row, node, fresh))
    for tag, located in retrieved.items():
        row = by_tag[tag]
        node = nodes.get(row.path)
        if node is not None:
            candidates.append((row, node, located))
    if not candidates:
        return []

    def claim(row: Contested, text: str) -> Claim:
        from pondie.extraction.evidence.grounding import claim_for
        return Claim(claim=claim_for(record, row.path, row.value),
                     premise=expand(text, abbreviations, paper))

    after = checker.score([claim(row, " ".join(sp["text"] for sp in fresh["sets"][0]["spans"]))
                           for row, _node, fresh in candidates])
    before = checker.score([claim(row, row.premise) if row.premise else claim(row, "")
                            for row, _node, _fresh in candidates])

    improved: list[str] = []
    best: dict[str, float] = {}
    for (row, node, fresh), new, old in zip(candidates, after, before):
        # A candidate that is contained in what it would replace is a fragment of it, and
        # carries strictly less. The proposer returns a quote cut mid-word often enough that
        # this fired twice in fifteen changes on three papers: `hrf_model` lost "and
        # temporally smoothed the data." to "and temporally smo", and `software` went from
        # two complete spans to one ending at "2.1 x 2.1 x 7 mm,". Both scored well, because
        # a prefix of a sentence says most of what the sentence says.
        candidate = " ".join(sp["text"] for sp in fresh["sets"][0]["spans"])
        if row.premise and _fragment(candidate, row.premise):
            refused.append(Refusal(
                "evidence", f"the replacement for {row.path} is part of the sentence it "
                            f"would replace, and shorter"))
            continue
        floor = 0.0 if not row.premise else float(old)
        # Two candidates may answer for one field. The better one wins, and only if it also
        # beats what is already there.
        if float(new) <= max(floor, best.get(row.path, 0.0)):
            refused.append(Refusal(
                "evidence", f"no better sentence found for {row.path} "
                            f"({new:.2f} against {floor:.2f})"))
            continue
        node["evidence"] = fresh
        best[row.path] = float(new)
        if row.path not in improved:
            improved.append(row.path)
    return improved
