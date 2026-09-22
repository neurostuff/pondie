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
        c for line in lines if line.strip() for c in (json.loads(line).get("coordinates") or [])
    ]


@pytest.mark.parametrize("name,flavour", CASES)
def test_the_manifest_locates_its_own_raw_table(name, flavour):
    """`data_file` must be filled from whichever key the flavour writes it under.

    This test used to supply `table_id + suffix` when `data_file` came back empty, which it
    always did for elsevier -- the reader looked only at `metadata.data_path` and elsevier
    writes `metadata.raw_xml_path`. The workaround hid it: `read_table` was handed "" by
    every real caller and returned None, so a paper with a full table manifest read as a
    paper with no tables at all.
    """

    manifest = tp.read_manifest(FIXTURES / name, flavour)
    assert manifest, f"no {flavour} manifest in the fixture"
    assert all(record["data_file"] for record in manifest.values())


@pytest.mark.parametrize("name,flavour", CASES)
def test_every_manifest_table_is_readable(name, flavour):
    study = FIXTURES / name
    manifest = tp.read_manifest(study, flavour)
    assert manifest, f"no {flavour} manifest in the fixture"
    for table_id, record in manifest.items():
        table = tp.read_table(study / "source" / flavour, record["data_file"], flavour=flavour)
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
    assert not missing, f"{len(missing)} of {len(coords)} coordinates are not in the built text"


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


def test_pdf_flavour_reads_text_txt():
    """PDF is a fetched render, so its text is text.txt like the other fetched ones."""
    from pondie.paths import Flavour

    assert Flavour.pdf.filename == "text.txt"


def test_pdf_ranks_below_xml_renders_and_above_ace():
    """Ordering is load-bearing: Paper.best and best_text take the first hit."""
    from pondie.paths import Flavour

    order = [f.name for f in Flavour]
    assert order.index("pubget") < order.index("pdf")
    assert order.index("elsevier") < order.index("pdf")
    assert order.index("pdf") < order.index("ace")


def test_best_prefers_an_xml_render_over_pdf(tmp_path):
    from pondie.extraction.models import Flavour, Paper

    study = tmp_path / "s1"
    for flavour in ("pdf", "elsevier"):
        target = study / "processed" / flavour
        target.mkdir(parents=True)
        (target / "text.txt").write_text(flavour, encoding="utf-8")

    assert Paper.best("s1", tmp_path).flavour is Flavour.elsevier


def test_best_prefers_pdf_over_ace(tmp_path):
    """ace ships no tables at all, so a PDF render is worth more than it."""
    from pondie.extraction.models import Flavour, Paper

    study = tmp_path / "s2"
    for flavour in ("pdf", "ace"):
        target = study / "processed" / flavour
        target.mkdir(parents=True)
        (target / "text.txt").write_text(flavour, encoding="utf-8")

    assert Paper.best("s2", tmp_path).flavour is Flavour.pdf
def test_stage_one_reads_the_best_flavour_that_has_a_manifest(tmp_path):
    """Stage 1 read pubget and only pubget, and raised on a study that has no pubget."""

    from pondie.extraction.corpus import tables as stage1

    study = tmp_path / "s"
    (study / "processed" / "elsevier").mkdir(parents=True)
    (study / "source" / "elsevier" / "tables").mkdir(parents=True)
    assert stage1.table_flavour(study) is None, "no manifest yet"

    src = FIXTURES / "els"
    (study / "processed" / "elsevier" / "tables.jsonl").write_text(
        (src / "processed" / "elsevier" / "tables.jsonl").read_text(), encoding="utf-8"
    )
    for raw in (src / "source" / "elsevier" / "tables").glob("*.xml"):
        (study / "source" / "elsevier" / "tables" / raw.name).write_text(
            raw.read_text(encoding="utf-8"), encoding="utf-8"
        )
    assert stage1.table_flavour(study) == "elsevier"
    got = stage1.coordinate_tables(study)
    assert got, "no coordinate table came back from an elsevier-only study"
    assert all(t["csv_text"].strip() for t in got), "a table rendered to empty CSV"


def test_a_manifest_naming_a_file_pubget_never_wrote_falls_back_to_the_id(tmp_path):
    """Four of the hundred defect papers say `table_001.csv` and ship `t2.xml`."""

    from pondie.extraction.corpus import tables as stage1

    study = tmp_path / "s"
    (study / "processed" / "pubget").mkdir(parents=True)
    (study / "source" / "pubget" / "tables").mkdir(parents=True)
    raw = next((FIXTURES / "ace" / "source" / "ace").rglob("*.html"), None)
    if raw is None:
        pytest.skip("no ace html fixture to stand in for an unrendered pubget table")
    (study / "source" / "pubget" / "tables" / "t2.html").write_text(
        raw.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (study / "processed" / "pubget" / "tables.jsonl").write_text(
        json.dumps(
            {
                "table_id": "t2",
                "table_number": 2,
                "contains_coordinates": True,
                "caption": "",
                "footer": "",
                "metadata": {"table_label": "Table 2", "data_path": "/nowhere/table_001.csv"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    got = stage1.coordinate_tables(study)
    assert len(got) == 1 and got[0]["csv_text"].strip()
