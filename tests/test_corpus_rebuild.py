"""Rebuilding a paper's text with its coordinate tables inlined, and proving it reproduces.

The only tests here that need a synced corpus on disk, which is why they are their own file:
every one skips when the pubget checkout or the synced tables are absent, and together they
were the whole of the suite's skip list while sitting in a file of unit tests. What they
establish is that the rebuild is additive -- the corpus text is a prefix-preserving
subsequence of the rebuilt text -- and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pondie.formats import table_parse as tables
from pondie.formats import text_index

from conftest import REPO, TABLE_PAPERS, TEXTS, _table_fixture, requires_tables

PUBGET = REPO / ".tmp_repos" / "pubget"
if not (PUBGET / "src" / "pubget" / "_text.py").is_file():
    PUBGET = Path.home() / "projects" / "pubget"
TEXT_PAPER = "HU6mqxmtySg3"
ARTICLE_XML = TEXTS / TEXT_PAPER / "source" / "pubget" / "article.xml"

requires_pubget = pytest.mark.skipif(
    not (PUBGET / "src" / "pubget" / "_text.py").is_file() or not ARTICLE_XML.is_file(),
    reason="the pubget checkout or the synced article.xml is not present",
)


@pytest.fixture(scope="module")
def pubget_text():
    from pondie.extraction.corpus import rebuild

    module, _utils, _commit = rebuild.load_pubget(PUBGET)
    return rebuild, module


@requires_pubget
@pytest.mark.parametrize("paper", TABLE_PAPERS)
def test_the_plain_rebuild_reproduces_the_corpus_text(pubget_text, paper: str) -> None:
    """The load-bearing assumption of the whole rebuild, and the one that must scream.

    ns-pond built the corpus with this stylesheet at commit 987fc2d and cross-references
    preserved; this runs one commit later. If the two ever disagree, the offsets in every
    existing record were computed against a text this code cannot reproduce, and nothing
    should be regenerated until that is understood.
    """

    rebuild, module = pubget_text
    root = TEXTS / paper
    article = root / "source" / "pubget" / "article.xml"
    if not article.is_file():
        pytest.skip(f"no article.xml for {paper}")

    rebuilt = rebuild.build(article, root / "source" / "pubget", module, keep_tables=False)
    corpus = (root / "processed" / "pubget" / "text.txt").read_text(encoding="utf-8")
    assert rebuild.check_equivalence(rebuilt, corpus) is None


@requires_pubget
def test_the_checkout_is_refused_when_it_predates_the_table_insertion(pubget_text) -> None:
    """The failure mode is silent: an older checkout regenerates the text that already
    exists, and every offset would be recomputed against it as though it had changed."""

    rebuild, module = pubget_text
    assert hasattr(module, "_insert_tables")
    with pytest.raises(rebuild.BuildError, match="no pubget checkout"):
        rebuild.load_pubget(REPO / "does" / "not" / "exist")


@requires_pubget
def test_the_tables_variant_leaves_no_placeholder(pubget_text) -> None:
    """Placeholder numbering is count(preceding::table-wrap) and has to line up with the
    table_NNN files; a leftover means it did not."""

    rebuild, module = pubget_text
    root = TEXTS / TEXT_PAPER / "source" / "pubget"
    built = rebuild.build(root / "article.xml", root, module, keep_tables=True)
    assert "[pubget-table-" not in built


@requires_pubget
def test_the_tables_variant_carries_the_cell_values_the_corpus_text_lacks(
    pubget_text,
) -> None:
    """The whole point. Before this, a coordinate could not be highlighted because it
    was not in the text at all -- pubget's stylesheet deletes td and th."""

    rebuild, module = pubget_text
    root = TEXTS / TEXT_PAPER / "source" / "pubget"
    corpus = (TEXTS / TEXT_PAPER / "processed" / "pubget" / "text.txt").read_text(encoding="utf-8")
    built = rebuild.build(root / "article.xml", root, module, keep_tables=True)

    peak = "−58"  # a coordinate from Table 3, with the publisher's minus sign
    assert peak not in corpus
    assert peak in built


@requires_pubget
def test_the_tables_variant_only_adds(pubget_text) -> None:
    """Every content line of the plain text survives, in order.

    Cheap proof that the flag adds the grid rather than rewriting the prose the existing
    spans address. Blank and whitespace-only lines are excluded deliberately: inserting a
    table changes how many of them sit around it, and measured on this paper that is the
    only difference -- 70 non-blank lines survive unchanged while the blank count moves
    from 88 to 96.
    """

    rebuild, module = pubget_text
    root = TEXTS / TEXT_PAPER / "source" / "pubget"
    plain = rebuild.build(root / "article.xml", root, module, keep_tables=False)
    tables = rebuild.build(root / "article.xml", root, module, keep_tables=True)

    haystack = iter([line for line in tables.splitlines() if line.strip()])
    for line in plain.splitlines():
        if not line.strip():
            continue
        assert any(candidate == line for candidate in haystack), f"lost: {line[:60]!r}"


@requires_pubget
def test_each_table_follows_its_own_caption(pubget_text) -> None:
    """ "At the position it appears in the article" is the claim; this is it, checked."""

    rebuild, module = pubget_text
    root = TEXTS / TEXT_PAPER / "source" / "pubget"
    built = rebuild.build(root / "article.xml", root, module, keep_tables=True)

    manifest = tables.read_manifest(TEXTS / TEXT_PAPER)
    for record in manifest.values():
        caption = (record["caption"] or "").strip()
        label = (record["table_label"] or "").strip()
        if not caption or not label or caption not in built or label not in built:
            continue
        assert built.index(caption) < built.rindex(label) or built.count(label) > 1


@requires_pubget
def test_the_rebuilt_text_needs_no_further_normalisation(pubget_text) -> None:
    """Offsets are computed against the normalized text, so a build that normalizes to
    something else would put every span one step away from the file on disk."""

    rebuild, module = pubget_text
    root = TEXTS / TEXT_PAPER / "source" / "pubget"
    built = rebuild.build(root / "article.xml", root, module, keep_tables=True)
    assert text_index.normalize(built) == built


@requires_pubget
def test_the_build_is_deterministic(pubget_text) -> None:
    rebuild, module = pubget_text
    root = TEXTS / TEXT_PAPER / "source" / "pubget"
    first = rebuild.build(root / "article.xml", root, module, keep_tables=True)
    second = rebuild.build(root / "article.xml", root, module, keep_tables=True)
    assert first == second


def test_sync_texts_wants_the_article_xml() -> None:
    """One line, and every rebuild depends on it. rsync skips a missing source without
    complaint, so its absence would surface much later as an unexplained build failure."""

    from pondie.extraction.corpus import sync

    assert "source/pubget/article.xml" in sync.WANTED


def test_only_a_markdown_heading_is_a_heading() -> None:
    """The paper is served as markdown, so `#` is the whole of the heading grammar.

    A second spelling lived in the indexer while `build_text` restyled headings as a
    title over a rule, and it needed a heuristic -- the rule must be exactly as long as
    the title -- to keep a table's delimiter row or a rule under a paragraph from
    reading as a section. Neither the spelling nor the heuristic exists now.
    """

    # Asserted on the heading-derived sections, not on the list being empty: everything
    # before the first heading is indexed at level 0, so a text with no heading still has
    # one section. That is the preamble, not a claim that the rule below is a heading.
    fooled = "Some ordinary sentence about the data\n-----\n\nmore prose"
    found = text_index.build_sections(text_index.normalize(fooled))
    assert not [s for s in found if s.level > 0]
    assert [s.level for s in found] == [0]

    real = "## Results\n\nprose\n\n### Whole brain\n\nmore"
    sections = text_index.build_sections(text_index.normalize(real))
    assert [(s.title, s.level) for s in sections] == [("Results", 1), ("Whole brain", 2)]


@requires_tables
def test_a_section_name_does_not_set_a_column_width() -> None:
    """A forty-character contrast name in the first column pushed every coordinate off
    the right of the pane, because the section row was measured with the data rows."""

    table = _table_fixture("HU6mqxmtySg3", "brb3829-tbl-0003")
    markdown = tables.markdown_table(table)
    header = next(line for line in markdown.splitlines() if line.startswith("| kE"))
    assert len(header.split("|")[1]) <= 6, header

    # The section text is still there in full; it simply overruns its cell. Read from
    # the table rather than retyped: this publisher sets the names with U+00A0.
    sections = [row["text"] for row in table["body"] if row["type"] == "section"]
    assert sections, "the premise of this test changed"
    for text in sections:
        assert text in markdown


@requires_tables
def test_the_markdown_table_columns_line_up() -> None:
    """Padding is the whole point: unpadded pipes are not a table anyone can scan."""

    table = _table_fixture("4cRnHYtfSwuK", "t2")
    lines = [
        line
        for line in tables.markdown_table(table).splitlines()
        if line.startswith("|") and not line.startswith("|-")
    ]
    data = [line for line in lines if line.count("|") == lines[0].count("|")]
    assert len({len(line) for line in data}) == 1, "rows are not a common width"
