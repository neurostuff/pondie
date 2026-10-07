"""`scripts/bundle_pubget.py` takes a paper by its result coordinates, tabled or prose.

The upstream parse holds prose coordinates as `table_id: "prose"` entries. Held to the
table tests they named a table no manifest lists, and every paper with one was refused.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bundle_pubget.py"
spec = importlib.util.spec_from_file_location("bundle_pubget", SCRIPT)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def folder(tmp_path: Path, analyses: list[dict], text: str = "Results.\n") -> Path:
    study = tmp_path / "S1"
    (study / "processed/pubget").mkdir(parents=True)
    (study / "stage1").mkdir()
    (study / "identifiers.json").write_text(json.dumps({"pmid": "123"}))
    (study / "processed/pubget/text.txt").write_text(text)
    (study / "processed/pubget/tables.jsonl").write_text(json.dumps({"table_id": "t1"}) + "\n")
    (study / "stage1/analyses.json").write_text(json.dumps({"analyses": analyses}))
    return study


def prose(role: str) -> dict:
    return {"table_id": "prose", "role": role, "name": "A > B",
            "points": [{"coordinates": [-17.0, -3.0, 20.0]}]}


TABLED = {"table_id": "t1", "name": "A > B", "points": [{"coordinates": [10, 20, 30]}]}
TABLE_TEXT = "Table 1\nregion\t10\t20\t30\t4.5\n"


def test_a_prose_only_result_is_taken(tmp_path):
    row = bundle.scan(folder(tmp_path, [prose("result")]))
    assert row["taken"], row
    assert (row["tables"], row["prose_result_points"]) == (0, 1)


def test_prose_seeds_alone_report_no_result(tmp_path):
    row = bundle.scan(folder(tmp_path, [prose("seed"), prose("roi")]))
    assert not row["taken"]
    assert row["reason"].startswith("no result coordinates")


def test_a_prose_entry_is_not_a_table_the_manifest_lacks(tmp_path):
    row = bundle.scan(folder(tmp_path, [TABLED, prose("seed")], TABLE_TEXT))
    assert row["taken"], row
    assert row["tables_not_in_text"] == []


def test_a_prose_result_carries_a_paper_whose_tables_are_not_in_the_text(tmp_path):
    row = bundle.scan(folder(tmp_path, [TABLED, prose("result")]))
    assert row["taken"], row
    assert row["tables_not_in_text"] == ["t1"]


def test_a_table_the_manifest_lacks_still_refuses(tmp_path):
    stray = dict(TABLED, table_id="t9")
    row = bundle.scan(folder(tmp_path, [stray, prose("result")], TABLE_TEXT))
    assert not row["taken"]
    assert "t9" in row["reason"]
