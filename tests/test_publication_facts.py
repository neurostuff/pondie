"""A record carries no language or publication type; a criterion on them reads the parse."""

import importlib.util
import json
from pathlib import Path

from pondie.formats.coordinate_parse import publication_facts

CORPUS = Path(__file__).parent / "fixtures" / "coordinate_parse"
STUDY = "gx2zzUbbydf9"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "query_metaanalyses.py"

spec = importlib.util.spec_from_file_location("query_metaanalyses", SCRIPT)
queries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queries)


def test_a_papers_language_and_types_come_from_its_parsed_paper():
    assert publication_facts(STUDY, CORPUS) == {
        "study_type": ["Journal Article", "Research Support, Non-U.S. Gov't"],
        "language": ["eng"],
    }


def test_a_paper_with_no_parsed_paper_has_no_facts():
    assert publication_facts("nosuchpaper", CORPUS) == {}


def test_english_answers_once_the_parse_is_joined_and_cannot_say_without_it():
    record = {"local_id": STUDY}
    assert queries.english(record) is None
    assert queries.english({**record, **publication_facts(STUDY, CORPUS)}) is True
    assert queries.english({**record, **publication_facts("nosuchpaper", CORPUS)}) is None


def test_main_joins_the_parse_onto_each_record(tmp_path, capsys, monkeypatch):
    run = tmp_path / "proj"
    run.mkdir()
    (run / f"{STUDY}.extraction.json").write_text(json.dumps({"local_id": STUDY}))
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "included_studies.csv").write_text("meta_pmid,study_pmid\n")
    seen = {}
    monkeypatch.setattr(queries, "QUERIES", {"1": ("proj", [("English", lambda r: seen.update(r))])})
    monkeypatch.setattr("sys.argv", ["q", "--records", str(run / "*.extraction.json"),
                                     "--bench", str(bench), "--corpus", str(CORPUS)])
    queries.main()
    assert seen["language"] == ["eng"]
