"""Reading a paper's tables when the publisher is not pubget.

Anchored on real tables from the ns-pond corpus rather than invented ones: the shapes that
break a parser here -- a header split over two rows, a colspan naming its columns instead of
counting them, a hidden cell inserted for screen readers -- are shapes no one writes by hand.
"""

import json
import re
from pathlib import Path

import pytest
import table_parse as tp

from pondie.extraction.passes.build_text import build_appended

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
