"""Parsing a paper's coordinate table out of pubget's CSV, and rendering it as markdown.

A reviewer cannot draw a span on a coordinate that is not in the document, and pubget's
text extraction deletes every table cell -- so this is what puts the numbers back. Most of
what is checked is column resolution: which three columns are x, y and z when the header
says `MNI` rather than `x y z`, when a statistic column is also called `z`, and when one
column holds the whole triple.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pondie.formats import table_parse as tables

from conftest import TEXTS, _table_fixture, requires_tables


def _data_rows(table) -> int:
    return sum(1 for row in table["body"] if row["type"] == "data")


def _body(*rows: list[str]) -> list[dict]:
    return [{"type": "data", "cells": row} for row in rows]


@requires_tables
# Anchored on the three papers under data/corpus/ rather than on invented
# fixtures, because every defect these guard against was found by running the
# code over real tables and none of them would have occurred to me otherwise.
def test_read_table_joins_on_the_csv_filename_not_the_table_id() -> None:
    """ns-pond sanitizes the id, so an id-equality join finds nothing.

    `tables.jsonl` calls this table `t2`; its own `table_001_info.json` calls it `T2`.
    The ids that flow through this repo -- `stage1/table-map.json` keys, and
    `Analysis.tables` via that map -- are the sanitized ones, so joining on
    `info["table_id"]` (which is what the upstream app did) returns None for both
    coordinate tables of this paper and the reviewer is shown no grid at all.
    """

    root = TEXTS / "4cRnHYtfSwuK"
    manifest = tables.read_manifest(root)
    assert manifest["t2"]["data_file"] == "table_001.csv"

    info = json.loads(
        (root / "source" / "pubget" / "tables" / "table_001_info.json").read_text("utf-8")
    )
    assert info["table_id"] == "T2", "the premise of this test changed"

    table = _table_fixture("4cRnHYtfSwuK", "t2")
    assert table is not None and _data_rows(table) == 14


@requires_tables
def test_header_runs_collapse_into_colspans() -> None:
    table = _table_fixture("4cRnHYtfSwuK", "t2")
    first = [(cell["text"], cell["span"]) for cell in table["header"][0]]
    assert ("MNI coordinate", 3) in first
    assert sum(cell["span"] for cell in table["header"][0]) == table["width"]


@requires_tables
def test_axis_columns_are_not_captured_by_a_z_statistic_column() -> None:
    """`5Rw4BhGBShSR` Table 1 has a "Z" statistic column beside its x/y/z axes.

    Accumulating axis matches across header rows top-down bound z to the statistic at
    index 7 and produced [3, 4, 7], which matches no coordinate: every one of the 77
    rows went unattributed and the table rendered with no colour at all. Resolving one
    header row at a time, bottom-up, and only from a row that names all three, rejects
    the first row -- which has the Z but neither x nor y.
    """

    table = _table_fixture("5Rw4BhGBShSR", "t0005")
    assert table["axis_cols"] == [3, 4, 5]
    # The statistic column is real and still there; it is simply not an axis. Read off
    # the collapsed header, where "Peak Coordinate" occupies one cell of span 3.
    assert [cell["text"] for cell in table["header"][0]] == [
        "Contrast",
        "No of voxels",
        "Region (s)",
        "Peak Coordinate",
        "F/T",
        "Z",
    ]


@requires_tables
def test_a_consecutive_axis_triple_beats_a_leftward_statistic() -> None:
    header = [["Region", "Z", "x", "y", "z", "p"]]
    assert tables._axis_columns(header, 6) == [2, 3, 4]


@requires_tables
def test_section_rows_are_recognised_as_headings_not_data() -> None:
    table = _table_fixture("HU6mqxmtySg3", "brb3829-tbl-0003")
    # Whitespace is folded before comparing: this publisher sets the contrast names with
    # U+00A0 around the ">", and a retyped literal would differ invisibly.
    sections = [" ".join(row["text"].split()) for row in table["body"] if row["type"] == "section"]
    assert sections == [
        "Proverbs > Literal sentences",
        "Transparent proverbs > Literal sentences",
        "Opaque proverbs > Literal sentences",
    ]


# Eight of the corpus's 37 coordinate tables named their axes in a shape `AXIS` could not
# read, and one named them in columns that do not hold them. Where the columns go
# unresolved, row matching falls back to comparing any number in the row, which the review
# layer's own docstring calls the behaviour that over-attributes.
def test_a_pandas_suffixed_colspan_is_three_axis_columns() -> None:
    """`84rGLhCbUJTh` Table 2: one merged header over three columns, which pandas
    de-duplicates into `.1` and `.2`."""

    header = [
        [
            "Diffusion parameter",
            "Region",
            "Peak coordinates (x,y,z)",
            "Peak coordinates (x,y,z).1",
            "Peak coordinates (x,y,z).2",
            "t value",
        ]
    ]
    body = _body(["FA", "L SFG", "-10", "52", "16", "3.79"])
    assert tables._axis_columns(header, 6, body) == [2, 3, 4]


def test_a_colspan_naming_no_axis_letter_still_resolves() -> None:
    """`kzMj26hGWacQ` t0015 heads the run `Peak coordinates` and puts `X Y Z` on the row
    below, where pandas left-aligns them to columns 0-2. The label plus the numbers is
    enough; the misplaced letters are no help."""

    header = [
        [
            "Brain regions",
            "Voxels",
            "Hem.",
            "Voxels in region",
            "Peak coordinates",
            "Peak coordinates",
            "Peak coordinates",
            "Peak t",
        ],
        ["X", "Y", "Z", "", "", "", "", ""],
    ]
    body = _body(["Cluster 1", "2971.0", "", "", "24", "-54", "51.0", "3.891"])
    assert tables._axis_columns(header, 8, body) == [4, 5, 6]


def test_an_axis_triple_no_row_supports_is_rejected() -> None:
    """The same table's misplaced `X Y Z`, on its own. Returning [0,1,2] is worse than
    returning nothing: row matching took the strict path and attributed zero of 34 rows."""

    header = [["Brain regions", "Voxels", "Hem."], ["X", "Y", "Z"]]
    body = _body(["Superior parietal gyrus", "468.0", "B"])
    assert tables._axis_columns(header, 3, body) is None


def test_a_parenthesised_axis_letter_resolves() -> None:
    header = [["Tal(x)", "Tal(y)", "Tal(z)", "Cerebral Region"]]
    body = _body(["-42", "34", "38", "L IFG"])
    assert tables._axis_columns(header, 4, body) == [0, 1, 2]


def test_a_statistic_column_named_z_is_not_an_axis() -> None:
    """The guard `AXIS` was written for, still holding once PAREN_AXIS is in the chain."""

    header = [["Region", "x", "y", "z", "Peak (Z)"]]
    body = _body(["L IFG", "-42", "34", "38", "6.85"])
    assert tables._axis_columns(header, 5, body) == [1, 2, 3]


def test_one_column_holding_the_whole_triple_is_reported_separately() -> None:
    """Reported as `axis_cell`, never as `axis_cols`: that key is three indices at four
    call sites and widening its type there is how `cells[column]` becomes an IndexError."""

    header = [["Region", "Z score", "MNI coordinates (x, y, z)"]]
    body = _body(["L IPL", "4.4", "-52,-42,56"], ["L MCC", "3.71", "-4,-26,36"])
    assert tables._axis_columns(header, 3, body) is None
    assert tables._axis_cell(header, 3, body) == 2


def test_a_triple_column_is_confirmed_by_majority_not_by_one_row() -> None:
    """One triple-looking cell in a column of region names must not carry it."""

    header = [["MNI coordinates (x, y, z)", "Region"]]
    body = _body(["-52,-42,56", "L IPL"], ["not a coordinate", "L MCC"], ["also not one", "R STG"])
    assert tables._axis_cell(header, 2, body) is None


@pytest.mark.parametrize(
    "cell,expected",
    [
        ("-52,-42,56", (-52.0, -42.0, 56.0)),
        ("(-30, -84, 22)", (-30.0, -84.0, 22.0)),
        ("46 -8 -38", (46.0, -8.0, -38.0)),
        ("− 52,− 42,56", (-52.0, -42.0, 56.0)),
    ],
)
def test_a_triple_cell_keeps_every_sign(cell: str, expected: tuple) -> None:
    """The sign is the whole risk. A pattern that skips a leading bracket by consuming any
    non-digit eats the minus with it and relocates the peak to the other hemisphere."""

    found = tables.TRIPLE_CELL.match(tables.normalize_number(cell))
    assert found is not None, cell
    assert tuple(float(value) for value in found.groups()) == expected


def test_normalize_number_closes_the_sign_digit_gap() -> None:
    assert tables.normalize_number("− 54") == "-54"
    assert tables.normalize_number("- 54") == "-54"


# The audit in docs/regex-audit.md, findings 4 and 5. Both are corpus-measured:
# 3,950 coordinate tables resolved no axis at all, and 74 crashed outright.
def test_a_coordinate_column_is_found_when_the_header_says_mni_not_xyz() -> None:
    """`MNI Peak Coordinates` over `(48, 24, 2)` is a triple column and was read as nothing.

    `_axis_cell` keyed on AXIS_TRIPLE, which wants the literal letters `x`, `y`, `z`
    separated by punctuation. Real headers name the space instead, and 3,950 coordinate
    tables of the corpus resolved no axis on that alone.
    """

    header = [["Region", "Cluster Size", "MNI Peak Coordinates"]]
    body = _body(["L IPL", "412", "(48, 24, 2)"], ["R STG", "233", "(-34, -54, -33)"])
    assert tables._axis_columns(header, 3, body) is None
    assert tables._axis_cell(header, 3, body) == 2


def test_a_coordinate_column_with_space_separated_axes_resolves() -> None:
    """`Peak MNI coordinates(x y z)` names all three and still failed AXIS_TRIPLE, which
    requires a comma, semicolon or slash between them."""

    header = [["Region", "Peak MNI coordinates(x y z)", "KE"]]
    body = _body(["L IFG", "-3 47 35", "120"], ["R MFG", "42 38 21", "88"])
    assert tables._axis_cell(header, 3, body) == 1


def test_a_coordish_header_still_needs_the_data_to_agree() -> None:
    """The relaxed header test is only safe because the majority confirmation is untouched:
    a column headed `MNI` whose cells are region names resolves to nothing, as before."""

    header = [["MNI coordinates", "Region"]]
    body = _body(["left insula", "L INS"], ["right putamen", "R PUT"])
    assert tables._axis_cell(header, 2, body) is None


def test_an_axis_triple_header_wins_over_a_coordish_one_in_the_same_row() -> None:
    """Two passes and not one condition. Merged, a COORDISH cell earlier in the row won the
    scan from the AXIS_TRIPLE cell that used to answer, and five tables of the corpus
    changed the column they resolved to for no reason."""

    header = [["MNI coordinates", "Peak (x, y, z)"]]
    body = _body(["-52,-42,56", "-4,-26,36"], ["-30,-84,22", "-8,-30,40"])
    assert tables._axis_cell(header, 2, body) == 1


def test_axis_cell_and_axis_cols_are_never_both_set(tmp_path: Path) -> None:
    """The key documents itself as exclusive -- "and then `axis_cols` is None" -- and eight
    tables of the corpus satisfy both readings once the header test admits COORDISH."""

    directory = tmp_path / "tables"
    directory.mkdir()
    (directory / "t.csv").write_text(
        'Region,MNI coordinates,x,y,z\nL IPL,"-52,-42,56",-52,-42,56\n', encoding="utf-8"
    )
    (directory / "t_info.json").write_text('{"n_header_rows": 1}', encoding="utf-8")
    table = tables.read_table(tmp_path, "t.csv")
    assert table["axis_cols"] == [2, 3, 4]
    assert table["axis_cell"] is None


def test_a_header_cell_holding_a_newline_does_not_crash_the_parse() -> None:
    """`DEDUP` has no DOTALL and `_clean` only strips the ends of a cell, so `.match`
    returned None and the third tier of `_axis_columns` dereferenced it. 74 tables of the
    corpus -- 73 ace, one elsevier -- raised AttributeError out of `read_table`."""

    assert tables.DEDUP.match("Peak\ncoordinates") is None
    assert tables._dedup_base("Peak\ncoordinates") == "Peak\ncoordinates"
    assert tables._dedup_base("Peak coordinates.2") == "Peak coordinates"

    # And having survived it, the colspan tier reads it: three identical bases naming a
    # coordinate over three numeric columns is exactly the shape that tier is for.
    header = [["Region", "Peak\ncoordinates", "Peak\ncoordinates", "Peak\ncoordinates"]]
    body = _body(["L IPL", "-52", "-42", "56"])
    assert tables._axis_columns(header, 4, body) == [1, 2, 3]
