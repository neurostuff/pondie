"""A record carries no language or publication type; a criterion on them reads the parse."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from pondie.formats.coordinate_parse import publication_facts

CORPUS = Path(__file__).parent / "fixtures" / "coordinate_parse"
STUDY = "gx2zzUbbydf9"
#: A corpus is named by study id and a record file by pmid; the join must use the former.
PMID = "34411741"
ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "query_metaanalyses", ROOT / "scripts" / "query_metaanalyses.py"
)
queries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queries)

sys.path.insert(0, str(ROOT / "experiments" / "stage-ablation"))
import score  # noqa: E402


def test_a_papers_language_and_types_come_from_its_parsed_paper():
    assert publication_facts(STUDY, CORPUS) == {
        "study_type": ["Journal Article", "Research Support, Non-U.S. Gov't"],
        "language": ["eng"],
    }


def test_a_paper_with_no_parsed_paper_raises():
    with pytest.raises(FileNotFoundError):
        publication_facts("nosuchpaper", CORPUS)


def test_a_bibliography_of_an_older_contract_raises(tmp_path):
    shutil.copytree(CORPUS / STUDY, tmp_path / STUDY)
    path = tmp_path / STUDY / "parse" / "parsed_paper.json"
    raw = json.loads(path.read_text())
    raw["bibliography"].pop("license")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="license"):
        publication_facts(STUDY, tmp_path)


def test_english_answers_from_the_joined_language():
    record = {"local_id": STUDY}
    assert queries.english(record) is None
    assert queries.english({**record, **publication_facts(STUDY, CORPUS)}) is True


def test_main_joins_by_the_records_local_id_not_its_file_name(tmp_path, monkeypatch):
    run = tmp_path / "proj"
    run.mkdir()
    (run / f"{PMID}.extraction.json").write_text(json.dumps({"local_id": STUDY}))
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "included_studies.csv").write_text("meta_pmid,study_pmid\n")
    seen = {}
    monkeypatch.setattr(queries, "QUERIES", {"1": ("proj", [("English", seen.update)])})
    monkeypatch.setattr("sys.argv", ["q", "--records", str(run / "*.extraction.json"),
                                     "--bench", str(bench), "--corpus", str(CORPUS)])
    queries.main()
    assert seen["language"] == ["eng"]


def test_the_scorer_loads_each_record_with_its_papers_facts(tmp_path):
    run = tmp_path / "run"
    (run / "records").mkdir(parents=True)
    (run / "records" / f"{PMID}.extraction.json").write_text(json.dumps({"local_id": PMID}))
    shutil.copytree(CORPUS / STUDY, run / "corpus" / PMID)
    record = score.load(run, raw=False)[PMID]
    assert record["language"] == ["eng"] and "Journal Article" in record["study_type"]


def test_the_scorer_refuses_a_run_whose_corpus_has_no_parse(tmp_path):
    run = tmp_path / "run"
    (run / "records").mkdir(parents=True)
    (run / "records" / f"{PMID}.extraction.json").write_text(json.dumps({"local_id": PMID}))
    (run / "corpus" / PMID / "stage1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        score.load(run, raw=False)
