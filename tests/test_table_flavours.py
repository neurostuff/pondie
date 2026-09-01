"""Reading a paper's tables when the publisher is not pubget.

Anchored on real tables from the ns-pond corpus rather than invented ones: the shapes that
break a parser here -- a header split over two rows, a colspan naming its columns instead of
counting them, a hidden cell inserted for screen readers -- are shapes no one writes by hand.
"""

import json
import re
from pathlib import Path

import pytest

from pondie.extraction.corpus.rebuild import build_appended
from pondie.formats import table_parse as tp

FIXTURES = Path(__file__).parent / "fixtures" / "flavours"
CASES = [("els", "elsevier"), ("ace", "ace")]


def coordinates(study: Path, flavour: str) -> list[dict]:
    lines = (study / "processed" / flavour / "tables.jsonl").read_text().splitlines()
    return [
        c
        for line in lines
        if line.strip()
        for c in (json.loads(line).get("coordinates") or [])
    ]


@pytest.mark.parametrize("name,flavour", CASES)
def test_every_manifest_table_is_readable(name, flavour):
    study = FIXTURES / name
    manifest = tp.read_manifest(study, flavour)
    assert manifest, f"no {flavour} manifest in the fixture"
    for table_id, record in manifest.items():
        data_file = (
            record["data_file"] or f"{table_id}{'.xml' if flavour == 'elsevier' else '.html'}"
        )
        table = tp.read_table(study / "source" / flavour, data_file, flavour=flavour)
        assert table is not None, f"{flavour} {table_id} did not read"
        assert table["width"] > 1 and table["body"], f"{flavour} {table_id} came back empty"


@pytest.mark.parametrize("name,flavour", CASES)
def test_a_coordinate_the_manifest_lists_can_be_spanned(name, flavour):
    """The whole point. A reviewer cannot draw evidence on a row that is not in the text.

    Checked against the manifest's own parsed coordinates, so it fails if the parser drops
    body rows -- which is the failure that would otherwise look like a paper reporting less.
    """

    study = FIXTURES / name
    built = build_appended(study, flavour)
    numbers = {tp.normalize_number(t) for t in re.findall(r"[-−–—+]?\d+(?:\.\d+)?", built)}
    coords = coordinates(study, flavour)
    assert coords, "fixture lists no coordinates"
    missing = [
        c
        for c in coords
        if not all(tp.normalize_number(str(int(c[axis]))) in numbers for axis in "xyz")
    ]
    assert (
        not missing
    ), f"{len(missing)} of {len(coords)} coordinates are not in the built text"


@pytest.mark.parametrize("name,flavour", CASES)
def test_appending_leaves_the_prose_byte_identical(name, flavour):
    """Every offset into the original text has to survive, so the tables go on the end."""

    study = FIXTURES / name
    corpus = (study / "processed" / flavour / "text.txt").read_text()
    assert build_appended(study, flavour).startswith(corpus.rstrip())


def test_a_hidden_screen_reader_cell_is_not_part_of_the_header():
    """Publisher HTML hides a " . " in every header cell. Read literally it gives "X .",
    which no axis pattern matches, and the table's coordinate columns look absent."""

    study = FIXTURES / "ace"
    table_id, record = next(iter(tp.read_manifest(study, "ace").items()))
    table = tp.read_table(
        study / "source" / "ace", record["data_file"] or f"{table_id}.html", flavour="ace"
    )
    flat = [cell for row in table["header_cells"] for cell in row]
    assert not [cell for cell in flat if cell.endswith(" .")], flat
    assert table["axis_cols"], "the x/y/z columns were not found"


def test_a_flavour_with_no_reader_says_so_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="no table reader"):
        tp.read_table(FIXTURES / "ace", "1.html", flavour="springer")


@pytest.mark.parametrize("name,flavour", CASES)
def test_the_flavour_is_chosen_and_built_without_a_pubget_checkout(name, flavour, tmp_path):
    """An elsevier or ace paper never runs pubget's transform, so it must not need it."""

    import shutil

    from pondie.extraction.corpus.rebuild import build_one, choose_flavour

    study = tmp_path / name
    shutil.copytree(FIXTURES / name, study)
    assert choose_flavour(study) == flavour
    info = build_one(study, None, "", allow_drift=False)
    assert info["flavour"] == flavour and info["tables_parsed"] > 0
    assert (study / "processed" / "local" / "text.tables.txt").is_file()


def test_pubget_is_preferred_when_it_can_actually_be_built(tmp_path):
    """Text alone does not make a pubget paper: without the article XML there is nothing
    to rebuild, so it falls through rather than failing."""

    from pondie.extraction.corpus.rebuild import choose_flavour

    study = tmp_path / "s"
    (study / "processed" / "pubget").mkdir(parents=True)
    (study / "processed" / "ace").mkdir(parents=True)
    (study / "processed" / "pubget" / "text.txt").write_text("x")
    (study / "processed" / "ace" / "text.txt").write_text("x")
    assert choose_flavour(study) == "ace"

    (study / "source" / "pubget").mkdir(parents=True)
    (study / "source" / "pubget" / "article.xml").write_text("<a/>")
    assert choose_flavour(study) == "pubget"


def test_the_built_text_is_addressable_as_the_local_flavour(tmp_path):
    """`local` is the only flavour whose file is not `text.txt`, and it is the one the
    extraction passes read. Addressing it as `text.txt` finds nothing, which reads
    downstream as a paper with no text rather than as a path built wrong."""

    import shutil

    from pondie.extraction.corpus.rebuild import build_one
    from pondie.extraction.models import Flavour, Paper

    shutil.copytree(FIXTURES / "ace", tmp_path / "ace")
    build_one(tmp_path / "ace", None, "", allow_drift=False)
    paper = Paper(study_id="ace", root=tmp_path, flavour=Flavour.local)
    assert paper.text.is_file(), paper.text
    assert "Tables (floated" in paper.text.read_text()


def test_an_identical_rebuild_is_allowed(tmp_path):
    """Re-running the build must not need a flag. It changes nothing."""

    import shutil

    from pondie.extraction.corpus.rebuild import build_one

    shutil.copytree(FIXTURES / "ace", tmp_path / "ace")
    first = build_one(tmp_path / "ace", None, "", allow_drift=False)
    again = build_one(tmp_path / "ace", None, "", allow_drift=False)
    assert first["variants"]["tables"]["sha256"] == again["variants"]["tables"]["sha256"]


def test_a_build_that_would_replace_a_different_text_is_refused(tmp_path):
    """The built text is what `source_text_hash` and every span offset address. Replacing
    it with a different one moves them all and invalidates nothing that points at them."""

    import shutil

    from pondie.extraction.corpus.rebuild import BuildError, build_one

    study = tmp_path / "ace"
    shutil.copytree(FIXTURES / "ace", study)
    build_one(study, None, "", allow_drift=False)
    built = study / "processed" / "local" / "text.tables.txt"
    built.write_text(built.read_text() + "\nsomething a record was built against\n")

    with pytest.raises(BuildError, match="already exists and this build differs"):
        build_one(study, None, "", allow_drift=False)

    info = build_one(study, None, "", allow_drift=False, overwrite=True)
    assert info["tables_parsed"] > 0
