"""Resolving a verbatim quote into character offsets on the document.

Where a quote the extraction model copied out of a paper becomes a `start_char`/`end_char`
pair addressing the frozen text. `resolve` tries exact, then whitespace-tolerant, then
case-insensitive, and raises rather than guessing, so what is checked is both what it finds
and what it refuses. `fold` is 1:1 length-preserving for the same reason `normalize` is.
"""

from __future__ import annotations

import pytest

from pondie.extraction.record import spans as span_tools


def test_fold_preserves_length() -> None:
    tricky = "don’t “quote” me — 4–5 units here"
    assert len(span_tools.fold(tricky)) == len(tricky)


def test_resolve_exact_quote() -> None:
    document = "The mean age was 32.4 years in total."
    found = span_tools.resolve(document, "mean age was 32.4 years")
    assert found.exact
    assert document[found.start_char : found.end_char] == found.text


def test_resolve_tolerates_whitespace_and_curly_punctuation() -> None:
    document = "Participants’ responses were\nrecorded reliably."
    found = span_tools.resolve(document, "Participants' responses were recorded")
    assert not found.exact
    # text always comes from the document, never from the model's quote
    assert found.text == document[found.start_char : found.end_char]
    assert "’" in found.text and "\n" in found.text


def test_resolve_prefers_occurrence_near_hint() -> None:
    document = "the effect" + " filler" * 20 + " the effect"
    second = document.rindex("the effect")
    found = span_tools.resolve(document, "the effect", near=second)
    assert found.start_char == second


def test_resolve_raises_for_absent_quote() -> None:
    with pytest.raises(span_tools.SpanResolutionError):
        span_tools.resolve("some document text", "a phrase that is not present")


def test_resolve_raises_for_empty_quote() -> None:
    with pytest.raises(span_tools.SpanResolutionError):
        span_tools.resolve("some document text", "   ")


def test_verify_rejects_shifted_offsets() -> None:
    document = "The mean age was 32.4 years."
    good = span_tools.resolve(document, "32.4 years").as_record()
    span_tools.verify(document, good)

    shifted = {**good, "start_char": good["start_char"] + 1}
    with pytest.raises(span_tools.SpanResolutionError):
        span_tools.verify(document, shifted)


def test_verify_rejects_out_of_range_offsets() -> None:
    with pytest.raises(span_tools.SpanResolutionError):
        span_tools.verify("short", {"text": "short", "start_char": 0, "end_char": 999})
