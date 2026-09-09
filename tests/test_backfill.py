"""Tying a reference analysis to the stage-1 entry it came from.

This writes into `benchmarks/reference`, which the gold direction tables are keyed to, so the
two ways it can be wrong are both silent: a provenance that names the wrong entry, and one
shaped so nothing reads it. Both happened while it was being written, and both are pinned
here.
"""

from __future__ import annotations

import json

from pondie.benchmark.backfill import SLOT, _by_name, backfill, entry_id, resolve

ENTRIES = [
    {"table_id": "t2", "name": "Visuotactile > visuomotor"},
    {"table_id": "t2", "name": "Visuomotor > visuotactile"},
    {"table_id": "t3", "name": "Synchronous > asynchronous"},
    {"table_id": "ts1", "name": "sz > nc"},
    {"table_id": "ts2", "name": "sz > nc"},
    {"table_id": "prose", "name": ""},
]


def analysis(name, tables=()):
    return {
        "local_id": "a1",
        "name": {"extraction_status": "extracted", "value": name},
        "tables": list(tables),
    }


def test_the_ordinal_counts_within_the_table_and_not_across_the_parse():
    """`t3#1` is the first entry of table 3, which is the parse's third. Numbering globally
    produced `t3#3` -- an id no candidate writes, so a join meant to be exact would have
    matched nothing while reporting success."""
    assert entry_id(ENTRIES, 0) == "t2#1"
    assert entry_id(ENTRIES, 1) == "t2#2"
    assert entry_id(ENTRIES, 2) == "t3#1"


def test_a_name_the_parse_uses_once_is_joined():
    assert resolve(analysis("Visuotactile > visuomotor"), _by_name(ENTRIES)) == 0


def test_a_name_the_parse_repeats_is_left_alone():
    """`JzsUUQbDr2bm` reports the same contrasts in four supplementary tables, one per
    diffusion parameter. Narrowing those by the table the analysis cites was tried and agreed
    with the candidate's own recorded provenance 12 times in 24 -- so it is not a route, and
    a repeated name yields nothing."""
    assert resolve(analysis("sz > nc", ["tbl1"]), _by_name(ENTRIES)) is None


def test_an_unnamed_analysis_matches_no_entry():
    """The parse carries unnamed `prose` entries, which an unnamed analysis matches all of."""
    assert resolve(analysis(""), _by_name(ENTRIES)) is None


def test_what_is_written_is_an_extracted_value_and_not_a_bare_string():
    """`source_table_analysis` has range `ExtractedString`, and `flatten` yields wrappers. A
    bare string is well-formed JSON that the scorer cannot see: the first run of this wrote
    54 of them and the alignment did not move."""
    assert resolve(analysis("Synchronous > asynchronous"), _by_name(ENTRIES)) == 2


def test_the_backfill_adds_one_key_and_changes_nothing_else(tmp_path):
    reference, corpus = tmp_path / "reference", tmp_path / "corpus"
    (corpus / "S1" / "stage1").mkdir(parents=True)
    (corpus / "S1" / "stage1" / "analyses.json").write_text(json.dumps({"analyses": ENTRIES}))
    record = {
        "local_id": "S1",
        "analyses": [analysis("Visuotactile > visuomotor"), analysis("sz > nc")],
    }
    reference.mkdir()
    (reference / "S1.extraction.json").write_text(json.dumps(record))

    tally = backfill(reference, corpus, write=True)
    assert (tally["joined"], tally["repeated_name"]) == (1, 1)

    after = json.loads((reference / "S1.extraction.json").read_text())
    first, second = after["analyses"]
    assert first[SLOT] == {
        "extraction_status": "extracted",
        "value": "t2#1",
        "value_source": "generated",
        "evidence": {"status": "not_found"},
    }
    assert SLOT not in second, "a repeated name must be left without a provenance"
    for key in ("local_id", "name", "tables"):
        assert first[key] == record["analyses"][0][key]


def test_a_dry_run_writes_nothing(tmp_path):
    reference, corpus = tmp_path / "reference", tmp_path / "corpus"
    (corpus / "S1" / "stage1").mkdir(parents=True)
    (corpus / "S1" / "stage1" / "analyses.json").write_text(json.dumps({"analyses": ENTRIES}))
    reference.mkdir()
    body = json.dumps({"local_id": "S1", "analyses": [analysis("Visuotactile > visuomotor")]})
    (reference / "S1.extraction.json").write_text(body)

    assert backfill(reference, corpus, write=False)["joined"] == 1
    assert (reference / "S1.extraction.json").read_text() == body
