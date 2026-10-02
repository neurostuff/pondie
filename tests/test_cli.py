"""The CLI's own parsing, which no stage sees and no other test covered."""

import pytest

from pondie.cli import _papers
from pondie.extraction.models import Flavour


def test_a_header_comment_is_not_a_paper(tmp_path):
    """`# pmid<TAB>neurostore_id<TAB>axis` has two tabs and a non-empty second field.

    Every pmids file in `data/selection/` carries that header, so parsing it as a study
    sent a run looking for a paper called `neurostore_id` -- and paid a model to do it.
    """
    ids = tmp_path / "five.pmids"
    ids.write_text(
        "# a comment\n"
        "# pmid\tneurostore_id\tdefect\n"
        "22184615\trxs2yUUZs7gt\tA:dx_prose_dump\n"
        "\n"
        "30770788\tzHJcyWNszdRf\tA:task_is_modality\n",
        encoding="utf-8",
    )
    assert [p.study_id for p in _papers(tmp_path, ids, Flavour.pubget)] == [
        "rxs2yUUZs7gt",
        "zHJcyWNszdRf",
    ]


def test_a_file_of_bare_ids_is_refused_rather_than_run_empty(tmp_path):
    """The check that already existed: nothing parsed is an error, not a silent no-op."""
    ids = tmp_path / "bare.pmids"
    ids.write_text("rxs2yUUZs7gt\nzHJcyWNszdRf\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _papers(tmp_path, ids, Flavour.pubget)


def test_a_comment_only_file_is_refused_too(tmp_path):
    """Skipping comments must not turn "nothing here" into a successful empty run."""
    ids = tmp_path / "comments.pmids"
    ids.write_text("# pmid\tneurostore_id\tdefect\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        _papers(tmp_path, ids, Flavour.pubget)


def test_flavour_best_takes_each_paper_on_the_render_it_has(tmp_path):
    """The default must not silently drop a paper that simply has a different render.

    `--flavour pubget` was the default, and `driver.run` reports a paper with no text for
    the named flavour as not-ready. Over the 100-paper defect set that skipped the 51
    elsevier-only papers -- a little over half the run -- with no error.
    """
    corpus = tmp_path / "corpus"
    for study, flavour in (("has_pubget", "pubget"), ("has_elsevier", "elsevier")):
        text = corpus / study / "processed" / flavour
        text.mkdir(parents=True)
        (text / "text.txt").write_text("a paper", encoding="utf-8")
    ids = tmp_path / "two.pmids"
    ids.write_text("1\thas_pubget\t\n2\thas_elsevier\t\n", encoding="utf-8")

    named = _papers(corpus, ids, Flavour.pubget)
    assert [p.text.is_file() for p in named] == [True, False], "pubget drops the other one"

    best = _papers(corpus, ids, "best")
    assert [p.flavour for p in best] == [Flavour.pubget, Flavour.elsevier]
    assert all(p.text.is_file() for p in best)


def test_flavour_best_still_yields_a_paper_when_there_is_no_text(tmp_path):
    """Not-ready is the driver's report to make, not an exception out of argument parsing."""
    corpus = tmp_path / "corpus"
    (corpus / "bare").mkdir(parents=True)
    ids = tmp_path / "one.pmids"
    ids.write_text("1\tbare\t\n", encoding="utf-8")
    papers = _papers(corpus, ids, "best")
    assert len(papers) == 1 and not papers[0].text.is_file()


def test_the_derived_fills_can_be_run_from_the_cli(tmp_path, capsys):
    """The fills were the one part of `normalization` with no way to be measured on real
    records short of writing a script, and the two defects that measurement found were
    both invisible to the per-field reports -- those only ever see values that exist."""
    import json

    from pondie.cli import main

    record = {"groups": [{
        "name": {"value": "controls", "extraction_status": "extracted"},
        "medical_condition": {"value": ["healthy"], "extraction_status": "extracted"},
        "age_mean": {"value": 24.0, "extraction_status": "extracted"},
        "age_unit": {"value": "years", "extraction_status": "extracted"},
    }]}
    path = tmp_path / "12345678.extraction.json"
    path.write_text(json.dumps({"study": record}), encoding="utf-8")

    assert main(["normalize", "derived", "--records", str(tmp_path / "*.json")]) == 0
    out = capsys.readouterr().out
    assert "1 records" in out
    assert "is_healthy" in out and "age_unit" in out


def test_derived_is_not_mistaken_for_a_field():
    """`normalize derived` runs the fills; every other choice reports one field. The two
    cannot collide -- `fields()` returns module names and there is no `derived` module."""
    from pondie import normalization
    from pondie.cli import DERIVED, _normalizable

    assert DERIVED not in normalization.fields()
    assert DERIVED in _normalizable()
