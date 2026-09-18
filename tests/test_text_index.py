"""The normalization every `start_char` in every record is relative to.

`text_index` is one of four formats in `pondie.formats`, and its docstring states the rule
for all four: everything producing or consuming those offsets must agree on the
normalization done here. These check the two properties the rest of the package assumes --
normalizing is idempotent, and preserves length on text already normalized -- plus the
section index built on them.
"""

from __future__ import annotations

from pondie.formats import text_index


def test_normalize_is_idempotent_and_folds_crlf() -> None:
    raw = "a\r\nb\rc\nd"
    once = text_index.normalize(raw)
    assert once == "a\nb\nc\nd"
    assert text_index.normalize(once) == once


def test_normalize_preserves_length_for_lf_only_text() -> None:
    raw = "already\nnormalized\ntext"
    assert len(text_index.normalize(raw)) == len(raw)


def test_sections_nest_and_stay_in_bounds() -> None:
    document = "## Methods\nbody\n### Participants\nmore\n## Results\ntail\n"
    sections = text_index.build_sections(document)

    assert [(s.title, s.level, s.parent_section) for s in sections] == [
        ("Methods", 1, None),
        ("Participants", 2, "Methods"),
        ("Results", 1, None),
    ]
    assert all(0 <= s.start_char < s.end_char <= len(document) for s in sections)
    # A section ends where the next same-or-higher-level heading begins.
    assert sections[0].end_char == sections[2].start_char
    assert sections[2].end_char == len(document)


def test_section_path_returns_deepest_breadcrumb() -> None:
    document = "## Methods\nbody\n### Participants\nmore text here\n"
    sections = text_index.build_sections(document)
    offset = document.index("more text here")
    assert text_index.section_path(sections, offset) == "Methods > Participants"


def test_text_hash_changes_with_content() -> None:
    assert text_index.text_hash("a") != text_index.text_hash("b")
    assert text_index.text_hash("a") == text_index.text_hash("a")
