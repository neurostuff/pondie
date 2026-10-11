"""pondie reads a paper's analyses from ingestion's CoordinateParse, by key.

The fixture is paper gx2zzUbbydf9 as ingestion's sync wrote it (parse/) beside the stage-1
document the same run left (stage1/). The stage-1 document has 17 entries, the parse 15:
the text sweep dropped "PC > Sil" and "NC > Sil" as restatements of table rows.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from pondie.extraction.models import Paper, Settings
from pondie.extraction.parse import TableParse
from pondie.extraction.prompt import render
from pondie.extraction.record import fix
from pondie.extraction.stages import SignSplit
from pondie.formats import coordinate_parse, parse_keys

STUDY = "gx2zzUbbydf9"
FIXTURE = Path(__file__).parent / "fixtures" / "coordinate_parse" / STUDY
POSITIONAL = re.compile(r"#\d+$")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE, tmp_path / STUDY)
    return tmp_path


def _raw(corpus: Path) -> dict:
    return json.loads((corpus / STUDY / "parse" / "coordinate_parse.json").read_text())


def _write_raw(corpus: Path, raw: dict) -> None:
    (corpus / STUDY / "parse" / "coordinate_parse.json").write_text(json.dumps(raw))


def test_paper_reads_the_parse_when_ingestion_wrote_one(corpus):
    paper = Paper(study_id=STUDY, root=corpus)
    assert paper.parse == corpus / STUDY / "parse" / "coordinate_parse.json"
    assert paper.parse_source == "coordinate_parse"


def test_paper_falls_back_to_stage1_without_a_parse(corpus):
    shutil.rmtree(corpus / STUDY / "parse")
    paper = Paper(study_id=STUDY, root=corpus)
    assert paper.parse == corpus / STUDY / "stage1" / "analyses.json"
    assert paper.parse_source == "stage1"
    keys = parse_keys.parse_keys(parse_keys.load(paper.parse))
    assert keys[:5] == ["text#1", "text#2", "text#3", "text#4", "tbl0003#1"]


def test_every_key_is_the_parses_own_and_none_is_positional(corpus):
    paper = Paper(study_id=STUDY, root=corpus)
    entries = parse_keys.load(paper.parse)
    keys = parse_keys.parse_keys(entries)
    assert keys == [a["key"] for a in _raw(corpus)["analyses"]]
    assert not [k for k in keys if POSITIONAL.search(k)]


def test_document_carries_tables_roles_and_spaces_from_the_parse(corpus):
    doc = TableParse.load(Paper(study_id=STUDY, root=corpus).parse).document
    by_key = {e["key"]: e for e in doc["analyses"]}
    table = by_key["tbl0003#ac6ef3aacb51"]
    assert (table["name"], table["table_id"], table["table_number"]) == ("PO > NO", "tbl0003", "3")
    assert table["table_caption"].startswith("Whole-brain random-effects tests")
    assert table["role"] == "result"
    assert {p["space"] for p in table["points"]} == {"MNI"}
    assert by_key["text#3d3ed67111b0"]["table_id"] == parse_keys.PROSE_TABLE_ID


def test_keys_survive_reordering_and_a_dropped_sibling(corpus):
    before = dict(
        zip(parse_keys.parse_keys(parse_keys.load(Paper(study_id=STUDY, root=corpus).parse)),
            (e["name"] for e in parse_keys.load(Paper(study_id=STUDY, root=corpus).parse)))
    )
    raw = _raw(corpus)
    raw["analyses"] = list(reversed(raw["analyses"]))
    del raw["analyses"][-3]  # tbl0003's first entry, once reversed
    _write_raw(corpus, raw)
    entries = parse_keys.load(Paper(study_id=STUDY, root=corpus).parse)
    after = dict(zip(parse_keys.parse_keys(entries), (e["name"] for e in entries)))
    assert len(after) == len(before) - 1
    assert all(before[key] == name for key, name in after.items())


def test_listing_prints_parse_keys_and_demands_them(corpus):
    doc = TableParse.load(Paper(study_id=STUDY, root=corpus).parse).document
    block = render.stage1_block(doc, {"tbl0003": "tbl3", "tbl0004": "tbl4", "tbl0005": "tbl5"})
    printed = re.findall(r"\[parse key: ([^\]]+)\]", block)
    assert printed == [a["key"] for a in _raw(corpus)["analyses"]]
    assert render.demandable_keys(doc) == set(printed)


def test_listing_shows_a_proposed_non_result_role(corpus):
    raw = _raw(corpus)
    raw["analyses"][2].update(role="anchor", anchor_kind="seed")
    _write_raw(corpus, raw)
    doc = TableParse.load(Paper(study_id=STUDY, root=corpus).parse).document
    block = render.stage1_block(doc, {"tbl0003": "tbl3"})
    line = next(line for line in block.splitlines() if "tbl0003#f87da7d6063a" in line)
    assert "role anchor (seed), proposed" in line


def test_record_links_resolve_to_parse_keys_end_to_end(corpus):
    paper = Paper(study_id=STUDY, root=corpus)
    body = {
        "analyses": [
            {"local_id": "a1", "name": "PO > NO", "tables": ["tbl0004"]},
            {"local_id": "a2", "name": "PO > Sil",
             "source_table_analysis": {"value": "text#3d3ed67111b0"}},
            {"local_id": "a3", "name": "Made up", "source_table_analysis": "tbl0003#9"},
        ]
    }
    notes = fix.resolve_source_table_analysis(body, paper.parse)
    a1, a2, a3 = body["analyses"]
    assert a1["source_table_analysis"]["value"] == "tbl0004#fabbcc2b3dc4"
    assert a2["source_table_analysis"]["value"] == "text#3d3ed67111b0"
    assert "source_table_analysis" not in a3
    assert any("'tbl0003#9' names no parsed row group" in n for n in notes)


def test_stage1_keys_in_an_existing_record_map_to_the_parse(corpus):
    paper = Paper(study_id=STUDY, root=corpus)
    body = {
        "analyses": [
            # stage 1's third tbl0003 entry is "PO > NO"
            {"local_id": "a1", "name": "x", "source_table_analysis": {"value": "tbl0003#3"}},
            # stage 1's "PC > Sil", which the parse dropped
            {"local_id": "a2", "name": "y", "source_table_analysis": "text#3"},
        ],
        "coordinate_sets": [{"local_id": "tbl0003#3"}],
    }
    notes = fix.map_legacy_keys(body, paper.parse)
    assert body["analyses"][0]["source_table_analysis"]["value"] == "tbl0003#ac6ef3aacb51"
    assert body["coordinate_sets"][0]["local_id"] == "tbl0003#ac6ef3aacb51"
    assert body["analyses"][1]["source_table_analysis"] == "text#3"
    assert any("'text#3' maps to no one analysis" in n for n in notes)


def test_stage1_keys_are_left_and_reported_without_a_stage1_to_map_from(corpus):
    shutil.rmtree(corpus / STUDY / "stage1")
    body = {"analyses": [{"source_table_analysis": "tbl0003#3"}]}
    notes = fix.map_legacy_keys(body, Paper(study_id=STUDY, root=corpus).parse)
    assert body["analyses"][0]["source_table_analysis"] == "tbl0003#3"
    assert notes and "no stage1/analyses.json" in notes[0]


def test_legacy_path_is_untouched_by_key_mapping(corpus):
    shutil.rmtree(corpus / STUDY / "parse")
    body = {"analyses": [{"source_table_analysis": "tbl0003#3"}]}
    assert fix.map_legacy_keys(body, Paper(study_id=STUDY, root=corpus).parse) == []
    assert fix.resolve_source_table_analysis(body, Paper(study_id=STUDY, root=corpus).parse) == []
    assert body["analyses"][0]["source_table_analysis"] == "tbl0003#3"


def test_a_sign_split_in_the_parse_is_withheld_and_mirrored(corpus):
    raw = _raw(corpus)
    original, inverse = raw["analyses"][4], raw["analyses"][5]
    original["split"] = {"half": "original", "rule": "sign_of_directional_statistic"}
    inverse["split"] = {
        "half": "inverse",
        "original_analysis": original["key"],
        "rule": "sign_of_directional_statistic",
    }
    _write_raw(corpus, raw)
    paper = Paper(study_id=STUDY, root=corpus)
    doc = TableParse.load(paper.parse).document
    assert [e["key"] for e in doc["analyses"] if e.get("withhold")] == [inverse["key"]]
    assert inverse["key"] not in render.demandable_keys(doc)
    body = {
        "analyses": [
            {"local_id": "a_po_no", "name": {"value": "PO > NO"},
             "source_table_analysis": {"value": original["key"]}}
        ]
    }
    fix.mirror_withheld(body, paper.parse)
    mirrored = [a for a in body["analyses"] if a.get("mirror_of") == "a_po_no"]
    assert mirrored
    assert mirrored[0]["source_table_analysis"]["value"] == inverse["key"]


def test_sign_split_stage_writes_nothing_and_says_which_parse(corpus, tmp_path):
    settings = Settings(payloads=tmp_path / "p", records=tmp_path / "r", model="m")
    paper = Paper(study_id=STUDY, root=corpus)
    before = paper.parse.read_bytes()
    outcome = SignSplit().run(paper, settings, caller=None)
    assert paper.parse.read_bytes() == before
    assert outcome.produced == ()
    assert "coordinate_parse.json" in outcome.notes[0]
    with pytest.raises(PermissionError):
        TableParse.load(paper.parse).save()

    shutil.rmtree(corpus / STUDY / "parse")
    outcome = SignSplit().run(paper, settings, caller=None)
    assert "stage1/analyses.json" in outcome.notes[0]


def test_a_document_mixing_keyed_and_unkeyed_entries_is_refused():
    with pytest.raises(ValueError):
        parse_keys.parse_keys([{"key": "tbl1#ab", "table_id": "tbl1"}, {"table_id": "tbl1"}])


def test_legacy_key_map_leaves_an_ambiguous_name_unmapped():
    parse = [
        {"key": "t#a", "table_id": "t", "name": "A > B", "points": [{"coordinates": [1, 2, 3]}]},
        {"key": "t#b", "table_id": "t", "name": "A > B", "points": [{"coordinates": [1, 2, 3]}]},
    ]
    stage1 = [{"table_id": "t", "name": "A > B", "points": [{"coordinates": [1, 2, 3]}]}]
    assert coordinate_parse.legacy_key_map(stage1, parse) == {}
    parse[1]["points"] = [{"coordinates": [4, 5, 6]}]
    assert coordinate_parse.legacy_key_map(stage1, parse) == {"t#1": "t#a"}
