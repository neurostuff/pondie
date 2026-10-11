"""pondie reads a paper's analyses from ingestion's CoordinateParse, by key.

The fixture is paper gx2zzUbbydf9: parse/ as ingestion's sync writes it, beside the stage-1
document an earlier sync left (stage1/). Both hold 17 analyses. "PO > Sil" and "PC > Sil"
report the same peak in one sentence, as do "NO > Sil" and "NC > Sil", and each is its own
analysis in both documents.
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


def _field(value):
    return {"value": value, "extraction_status": "extracted"}


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
    assert not [k for k in keys if parse_keys.is_positional(k)]
    # A digest made only of digits is still a parse key: a digit test alone calls it
    # positional.
    assert not parse_keys.is_positional("tbl0004#294634710878")


def test_document_carries_tables_roles_and_spaces_from_the_parse(corpus):
    doc = TableParse.load(Paper(study_id=STUDY, root=corpus).parse).document
    by_key = {e["key"]: e for e in doc["analyses"]}
    table = by_key["tbl0003#b67d45b7c84f"]
    assert (table["name"], table["table_id"], table["table_number"]) == ("PO > NO", "tbl0003", "3")
    assert table["table_caption"].startswith("Whole-brain random-effects tests")
    assert table["role"] == "result"
    assert {p["space"] for p in table["points"]} == {"MNI"}
    assert by_key["text#dcf3d5417a44"]["table_id"] == parse_keys.PROSE_TABLE_ID


def test_keys_survive_reordering_and_a_dropped_sibling(corpus):
    before = dict(
        zip(
            parse_keys.parse_keys(parse_keys.load(Paper(study_id=STUDY, root=corpus).parse)),
            (e["name"] for e in parse_keys.load(Paper(study_id=STUDY, root=corpus).parse)),
        )
    )
    raw = _raw(corpus)
    raw["analyses"] = list(reversed(raw["analyses"]))
    del raw["analyses"][-5]  # tbl0003's first entry, once reversed
    _write_raw(corpus, raw)
    entries = parse_keys.load(Paper(study_id=STUDY, root=corpus).parse)
    after = dict(zip(parse_keys.parse_keys(entries), (e["name"] for e in entries)))
    assert len(after) == len(before) - 1
    assert all(before[key] == name for key, name in after.items())


def test_listing_prints_parse_keys_and_demands_them(corpus):
    doc = TableParse.load(Paper(study_id=STUDY, root=corpus).parse).document
    block = render.stage1_block(doc, {"tbl0003": "tbl3", "tbl0004": "tbl4", "tbl0005": "tbl5"})
    printed = [k for k in re.findall(r"\[parse key: ([^\]]+)\]", block) if k != "..."]
    assert printed == [a["key"] for a in _raw(corpus)["analyses"]]
    assert render.demandable_keys(doc) == set(printed)


def test_listing_shows_a_proposed_non_result_role(corpus):
    raw = _raw(corpus)
    raw["analyses"][4].update(role="anchor", anchor_kind="seed")
    _write_raw(corpus, raw)
    doc = TableParse.load(Paper(study_id=STUDY, root=corpus).parse).document
    block = render.stage1_block(doc, {"tbl0003": "tbl3"})
    line = next(line for line in block.splitlines() if "tbl0003#0e7ce757a988" in line)
    assert "role anchor (seed), proposed" in line


def test_record_links_resolve_to_parse_keys_end_to_end(corpus):
    paper = Paper(study_id=STUDY, root=corpus)
    body = {
        "analyses": [
            {"local_id": "a1", "name": "PO > NO", "tables": ["tbl0004"]},
            {
                "local_id": "a2",
                "name": "PO > Sil",
                "source_table_analysis": _field("text#dcf3d5417a44"),
            },
            {"local_id": "a3", "name": "Made up", "source_table_analysis": "tbl0003#9"},
        ]
    }
    notes = fix.resolve_source_table_analysis(body, paper.parse)
    a1, a2, a3 = body["analyses"]
    assert a1["source_table_analysis"]["value"] == "tbl0004#e10d99f119d5"
    assert a2["source_table_analysis"]["value"] == "text#dcf3d5417a44"
    assert "source_table_analysis" not in a3
    assert any("'tbl0003#9' names no parsed row group" in n for n in notes)


PO_SIL, PC_SIL = "text#dcf3d5417a44", "text#a83e27154bee"
PO_NO = "tbl0003#b67d45b7c84f"


def test_stage1_keys_in_an_existing_record_map_to_the_parse(corpus):
    paper = Paper(study_id=STUDY, root=corpus)
    body = {
        "analyses": [
            # stage 1's third tbl0003 entry is "PO > NO"
            {"local_id": "a1", "name": "x", "source_table_analysis": _field("tbl0003#3")},
            # "PO > Sil" and "PC > Sil": one sentence, one peak, two analyses
            {"local_id": "a2", "name": "y", "source_table_analysis": "text#1"},
            {"local_id": "a3", "name": "z", "source_table_analysis": "text#3"},
        ],
        "coordinate_sets": [{"local_id": "tbl0003#3"}],
    }
    fix.map_legacy_keys(body, paper.parse)
    a1, a2, a3 = body["analyses"]
    assert a1["source_table_analysis"]["value"] == PO_NO
    assert body["coordinate_sets"][0]["local_id"] == PO_NO
    assert (a2["source_table_analysis"], a3["source_table_analysis"]) == (PO_SIL, PC_SIL)


def test_a_stage1_key_for_an_analysis_the_parse_lacks_is_left_and_reported(corpus):
    raw = _raw(corpus)
    raw["analyses"] = [a for a in raw["analyses"] if a["key"] != PC_SIL]
    _write_raw(corpus, raw)
    body = {"analyses": [{"local_id": "a3", "source_table_analysis": "text#3"}]}
    notes = fix.map_legacy_keys(body, Paper(study_id=STUDY, root=corpus).parse)
    assert body["analyses"][0]["source_table_analysis"] == "text#3"
    assert any("'text#3' left: no analysis of the parse has its table and name" in n for n in notes)


def test_a_parse_key_the_record_already_holds_is_not_given_to_a_second(corpus):
    body = {
        "analyses": [
            {"local_id": "a1", "source_table_analysis": PO_NO},
            {"local_id": "a2", "source_table_analysis": "tbl0003#3"},
        ],
        "coordinate_sets": [{"local_id": PO_NO}, {"local_id": "tbl0003#3"}],
    }
    notes = fix.map_legacy_keys(body, Paper(study_id=STUDY, root=corpus).parse)
    assert body["analyses"][1]["source_table_analysis"] == "tbl0003#3"
    assert [c["local_id"] for c in body["coordinate_sets"]] == [PO_NO, "tbl0003#3"]
    assert sum("ambiguous" in n for n in notes) == 2


def test_two_same_named_stage1_entries_never_share_one_parse_key():
    stage1 = [
        {"table_id": "tbl0002", "name": "Patients > Controls",
         "points": [{"coordinates": [10, 20, 30]}]},
        {"table_id": "tbl0002", "name": "Patients > Controls",
         "points": [{"coordinates": [-40, 2, 8]}]},
    ]
    parse = [{"key": "tbl0002#aaaaaaaaaaaa", "table_id": "tbl0002",
              "name": "Patients > Controls", "points": [{"coordinates": [0, 0, 0]}]}]
    match = coordinate_parse.match_legacy_keys(stage1, parse)
    assert match.mapping == {}
    assert set(match.unmapped) == {"tbl0002#1", "tbl0002#2"}
    parse[0]["points"] = [{"coordinates": [10, 20, 30]}]
    match = coordinate_parse.match_legacy_keys(stage1, parse)
    assert match.mapping == {"tbl0002#1": "tbl0002#aaaaaaaaaaaa"}
    assert "1 parse analyses share its table and name" in match.unmapped["tbl0002#2"]
    # Equal points as well: neither can be told apart, so neither maps.
    stage1[1]["points"] = [{"coordinates": [10, 20, 30]}]
    assert coordinate_parse.legacy_key_map(stage1, parse) == {}


def test_two_same_named_sets_in_a_record_keep_two_local_ids(tmp_path):
    study = tmp_path / "S"
    (study / "parse").mkdir(parents=True)
    (study / "stage1").mkdir()
    shutil.copy(FIXTURE / "parse" / "coordinate_parse.json", study / "parse")
    raw = json.loads((study / "parse" / "coordinate_parse.json").read_text())
    stage1 = {"analyses": [
        {"table_id": "tbl0003", "name": "PO > NO", "points": [{"coordinates": [1, 2, 3]}]},
        {"table_id": "tbl0003", "name": "PO > NO", "points": [{"coordinates": [4, 5, 6]}]},
    ]}
    (study / "stage1" / "analyses.json").write_text(json.dumps(stage1))
    assert any(a["key"] == PO_NO for a in raw["analyses"])
    body = {
        "analyses": [{"local_id": "a1", "source_table_analysis": "tbl0003#1"},
                     {"local_id": "a2", "source_table_analysis": "tbl0003#2"}],
        "coordinate_sets": [{"local_id": "tbl0003#1"}, {"local_id": "tbl0003#2"}],
    }
    fix.map_legacy_keys(body, study / "parse" / "coordinate_parse.json")
    ids = [c["local_id"] for c in body["coordinate_sets"]]
    assert len(set(ids)) == 2 and PO_NO not in ids


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
            {
                "local_id": "a_po_no",
                "name": _field("PO > NO"),
                "source_table_analysis": _field(original["key"]),
            }
        ]
    }
    notes = fix.mirror_withheld(body, paper.parse)
    mirrored = [a for a in body["analyses"] if a.get("mirror_of") == "a_po_no"]
    assert mirrored, notes
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


def _stage1_keyed_payloads(tmp_path: Path) -> Path:
    payloads = tmp_path / "payloads" / STUDY
    payloads.mkdir(parents=True)
    (payloads / "demands.json").write_text(json.dumps({
        "analyses": [{
            "local_id": "a_tbl0003_3",
            "name": _field("PO > NO"),
            "source_table_analysis": _field("tbl0003#3"),
        }],
        "coordinate_sets": [{"local_id": "tbl0003#3"}],
    }))
    return payloads


def test_a_stored_stage1_keyed_payload_is_remapped_when_built_on_the_parse(corpus, tmp_path):
    from pondie.extraction.record import builder

    text = tmp_path / "text.txt"
    text.write_text("PO > NO in Table 3.\n")
    paper = Paper(study_id=STUDY, root=corpus)
    record, report = builder.build(
        STUDY, text, _stage1_keyed_payloads(tmp_path), "m", "v", "2026-10-10",
        stage1=paper.parse,
    )
    body = record.get("study") or record
    assert body["analyses"][0]["source_table_analysis"]["value"] == PO_NO
    assert [c["local_id"] for c in body["coordinate_sets"]] == [PO_NO]


def test_a_new_parse_makes_every_pass_stale(corpus, tmp_path):
    from pondie import pipeline
    from pondie.extraction.stages import Build, Demands, Single

    settings = Settings(payloads=tmp_path / "p", records=tmp_path / "r", model="m")
    paper = Paper(study_id=STUDY, root=corpus)
    stages = (Demands(), Single(), Build())
    before = [pipeline.digest_of(s.depends_on(paper, settings)) for s in stages]
    assert all(
        s.depends_on(paper, settings)["parse"]["parse_id"] == _raw(corpus)["parse_id"]
        for s in stages
    )
    raw = _raw(corpus)
    raw["parse_id"] = "0" * 64
    _write_raw(corpus, raw)
    after = [pipeline.digest_of(s.depends_on(paper, settings)) for s in stages]
    assert all(b != a for b, a in zip(before, after))
    # The stage-1 path's fingerprint is what it was before the parse existed.
    shutil.rmtree(corpus / STUDY / "parse")
    assert "parse" not in Demands().depends_on(paper, settings)


def test_the_id_map_follows_a_stage1_record_to_its_parse_ids(corpus, tmp_path):
    from pondie import cli

    record = {
        "analyses": [
            {"local_id": "a_tbl0003_3", "source_table_analysis": _field("tbl0003#3")},
            {"local_id": "a_text_1", "source_table_analysis": _field("text#1")},
            {"local_id": "a_text_3", "source_table_analysis": _field("text#3")},
            {"local_id": "a_tbl0009_1", "source_table_analysis": _field("tbl0009#1")},
        ]
    }
    records = tmp_path / "records"
    records.mkdir()
    (records / f"{STUDY}.extraction.json").write_text(json.dumps(record))
    out = tmp_path / "maps"
    assert cli.main(["id-map", "--records", str(records), "--corpus", str(corpus),
                     "--out", str(out)]) == 0
    found = json.loads((out / f"{STUDY}.id-map.json").read_text())
    assert found["parse_id"] == _raw(corpus)["parse_id"]
    assert found["keys"] == {"tbl0003#3": PO_NO, "text#1": PO_SIL, "text#3": PC_SIL}
    assert found["ids"] == {
        "a_tbl0003_3": "a_tbl0003_b67d45b7c84f",
        "a_text_1": "a_text_dcf3d5417a44",
        "a_text_3": "a_text_a83e27154bee",
    }
    assert found["unmapped"] == {"tbl0009#1": "not a key of the stage-1 document"}


def test_select_joins_a_record_on_the_parse_key(corpus, tmp_path, monkeypatch):
    from pondie import paths
    from pondie.query.engine import Selection, select

    parse = corpus / STUDY / "parse" / "coordinate_parse.json"
    monkeypatch.setattr(paths, "coordinate_parse", lambda study, corpus=None: parse)
    monkeypatch.setattr(
        paths, "stage1", lambda study, corpus=None: corpus_stage1
    )
    corpus_stage1 = corpus / STUDY / "stage1" / "analyses.json"
    record = {"analyses": [{
        "local_id": "a1",
        "name": _field("PO > NO"),
        "source_table_analysis": _field(PO_NO),
        "coordinate_space": _field("MNI"),
        "spatial_scope": _field("whole_brain"),
    }]}
    written = tmp_path / f"{STUDY}.extraction.json"
    written.write_text(json.dumps(record))
    outcome = select(Selection(records=(str(written),)))
    assert outcome.rows, dict(outcome.lost)
    expected = next(a for a in _raw(corpus)["analyses"] if a["key"] == PO_NO)
    assert outcome.rows[0]["points"] == [p["coordinates"] for p in expected["points"]]


def test_the_four_mirrored_roi_centres_are_anchors_not_results(corpus):
    by_name = {}
    for entry in TableParse.load(Paper(study_id=STUDY, root=corpus).parse).document["analyses"]:
        if entry["table_id"] == parse_keys.PROSE_TABLE_ID:
            by_name[entry["name"]] = (entry["role"], entry["anchor_kind"])
    assert by_name == {
        name: ("anchor", "roi") for name in ("PO > Sil", "NO > Sil", "PC > Sil", "NC > Sil")
    }


def test_one_unreadable_parse_drops_its_study_and_the_others_continue(corpus, tmp_path, monkeypatch):
    from pondie import paths
    from pondie.query.engine import Selection, select

    broken = tmp_path / "BAD" / "parse"
    broken.mkdir(parents=True)
    raw = _raw(corpus)
    raw["analyses"][0]["points"] = "not a list"
    (broken / "coordinate_parse.json").write_text(json.dumps(raw))
    good = corpus / STUDY / "parse" / "coordinate_parse.json"
    monkeypatch.setattr(
        paths,
        "analyses",
        lambda study, corpus=None: good if study == STUDY else broken / "coordinate_parse.json",
    )
    record = {"analyses": [{
        "local_id": "a1",
        "name": _field("PO > NO"),
        "source_table_analysis": _field(PO_NO),
        "coordinate_space": _field("MNI"),
        "spatial_scope": _field("whole_brain"),
    }]}
    paths_written = []
    for study in ("BAD", STUDY):  # the broken one sorts first
        path = tmp_path / f"{study}.extraction.json"
        path.write_text(json.dumps(record))
        paths_written.append(str(path))
    outcome = select(Selection(records=tuple(paths_written)))
    assert outcome.lost["parse unreadable"] == 1
    assert [r["study"] for r in outcome.rows] == [STUDY], dict(outcome.lost)


def test_backfill_writes_the_parse_key(corpus, tmp_path):
    from pondie.benchmark.backfill import SLOT, backfill

    reference = tmp_path / "reference"
    reference.mkdir()
    path = reference / f"{STUDY}.extraction.json"
    path.write_text(json.dumps({"analyses": [{"local_id": "a1", "name": _field("PC > NC")}]}))
    backfill(reference, corpus, write=True)
    joined = json.loads(path.read_text())["analyses"][0][SLOT]["value"]
    assert joined == "tbl0003#0b0e7dfc137c"


def test_a_repair_prompt_names_a_parse_keyed_analysis_by_table_alone():
    from pondie.extraction.repair.stage import _located

    record = {"tables": [{"local_id": "tbl3", "table_number": _field("3")}]}
    analysis = {"tables": ["tbl3"], "source_table_analysis": _field(PO_NO)}
    assert _located(record, analysis) == " It is from Table 3."
    analysis["source_table_analysis"] = _field("tbl0003#3")
    assert _located(record, analysis) == " It is row group 3 of Table 3."


def test_sign_split_is_never_done_on_the_parse_path(corpus, tmp_path):
    settings = Settings(payloads=tmp_path / "p", records=tmp_path / "r", model="m")
    assert SignSplit().done(Paper(study_id=STUDY, root=corpus), settings) is False
