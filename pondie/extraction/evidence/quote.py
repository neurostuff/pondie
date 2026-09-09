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


#: Sentence enders, for lifting the clause a value sits in out of the document.
_BREAK = re.compile(r"(?<=[.!?])\s|\n")

#: A value is worth searching for only if it has some substance. A one-character value
#: is a token that occurs everywhere; a boolean is a classification the paper never
#: spells, and rule 5 of SYSTEM is about exactly that case.
_TRIVIAL = frozenset({"true", "false", "none", "null", "0", "1", "yes", "no"})


def _anchored(value: str) -> re.Pattern[str] | None:
    """`value` as a pattern that will not match inside a longer token.

    `spans._tolerant_pattern` joins tokens with `\\s+` and anchors nothing, which is
    right for a model's quote -- a long span whose edges are already unambiguous -- and
    wrong here. Searching for a bare `3` with it hits the `3` in `13`, `0.35` and
    `Figure 3b`, so a value could look unique while matching something else entirely.
    The boundary goes on only where the edge character can carry one: a value starting
    `(` has no word boundary to its left.
    """
    tokens = [re.escape(t) for t in str(value).split()]
    if not tokens:
        return None
    body = r"\s+".join(tokens)
    left = r"\b" if re.match(r"\w", str(value)[0]) else ""
    right = r"\b" if re.search(r"\w$", str(value)) else ""
    try:
        return re.compile(left + body + right, re.IGNORECASE)
    except re.error:
        return None


def _clause(text: str, start: int, end: int, cap: int = 400) -> str:
    """The sentence `text[start:end]` sits in, clipped to `cap` characters."""
    lo = 0
    for m in _BREAK.finditer(text, max(0, start - cap), start):
        lo = m.end()
    hi = len(text)
    m = _BREAK.search(text, end, min(len(text), end + cap))
    if m:
        hi = m.start()
    return text[max(lo, start - cap):min(hi, end + cap)].strip()


def literal_quotes(payload: dict[str, Any], text: str) -> dict[str, str]:
    """Quotes for the fields whose own value occurs exactly once in the paper.

    The Evidence pass asks a model to copy a supporting sentence for every extracted
    field. For a value that appears once and only once in the document, that sentence is
    determined -- there is nothing to judge, and a call spent on it buys a slower copy of
    what `str.find` returns. Measured over the 89 records of `depression-full`, a sixth of
    all model-placed spans are on values of exactly this kind.

    Uniqueness is the safety condition, and it replaces the twenty-character floor
    `record/edit.py:_wrap` uses. That floor exists because a *bare short value* cannot be
    searched for safely; it is the wrong test, because it rejects `3 T` and `SPM12`, which
    are short and perfectly locatable, while admitting any long string that happens to
    repeat. What has to be unambiguous is the span this returns, not the value that found
    it -- so the value is matched on word boundaries, must occur once, and the clause
    lifted around it must itself be long enough for `build_record` to resolve.

    Silent on anything it cannot settle. A value that is absent, repeated, listed, or
    trivial is simply left out, and the caller sends it to the model as before.
    """
    out: dict[str, str] = {}
    for path, field in iter_fields(payload):
        if field.get("extraction_status") != "extracted":
            continue
        value = field.get("value")
        # A list is one wrapper over many facts and no single clause carries them all.
        if isinstance(value, (list, dict, bool)) or value is None:
            continue
        rendered = str(value).strip()
        if len(rendered) < 2 or rendered.casefold() in _TRIVIAL:
            continue
        pattern = _anchored(rendered)
        if pattern is None:
            continue
        found = list(pattern.finditer(text))
        if len(found) != 1:
            continue
        quote = _clause(text, found[0].start(), found[0].end())
        # Short clauses go to the model. `build_record` locates a quote by matching it in
        # the document, so a fragment that is itself ambiguous trades one guess for another.
        if len(quote) < 20 or text.count(quote) != 1:
            continue
        out[path] = quote
    return out


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




def apply_evidence(
    payload: dict[str, Any],
    quotes: dict[str, str],
    literal: frozenset[str] | set[str] = frozenset(),
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
        # Labelled, not just ordered. The two sets were already two different locators, but
        # only by position, so nothing downstream could say which warranted a value or count
        # how often each was right. `literal` names the paths `literal_quotes` settled
        # before the model was asked: calling those `model_quote` would claim a reading
        # that never happened, on the one field whose whole purpose is telling the
        # locators apart.
        source = "literal_match" if path in literal else "model_quote"
        sets = [{"source": source, "quotes": [quote]}] if quote else []
        second = None
        if second:
            sets.append({"source": "retriever", "quotes": [second]})
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
