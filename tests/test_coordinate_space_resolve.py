"""The precedence cascade in `coordinate_space.resolve`, which had no test of its own.

It was reached only through `pondie select`, which is how it returned a `Decision` where a
dict was expected for as long as it did. The cases below are the ones it got wrong: two
spellings of one space read as a conflict, and a space no rule matched reported as though
the coordinate parse had answered.
"""

from __future__ import annotations

from pondie.normalization.coordinate_space import resolve

ANALYSIS = {"coordinate_space": None, "tables": ["t1"], "source_table_analysis": "k"}
NO_TABLES: dict = {"tables": []}


def points(*spaces: str | None) -> dict:
    return {"k": [{"space": s} for s in spaces]}


def test_the_analysis_field_outranks_the_tables_and_the_parse():
    analysis = dict(ANALYSIS, coordinate_space={"value": "Talairach"})
    record = {"tables": [{"local_id": "t1", "coordinate_space": {"value": "MNI"}}]}
    assert resolve(analysis, record, points("MNI")).value == "TAL"


def test_a_table_answers_where_the_analysis_is_silent():
    record = {"tables": [{"local_id": "t1", "coordinate_space": {"value": "MNI"}}]}
    decided = resolve(ANALYSIS, record, points("TALAIRACH"))
    assert (decided.value, decided.reason) == ("MNI", "tables agree")


def test_two_spellings_of_one_space_are_not_a_conflict():
    """Stage 1 writes "MNI" for one sentence and "MNI152" for the next."""
    decided = resolve(ANALYSIS, NO_TABLES, points("MNI", "MNI152", "ICBM"))
    assert (decided.value, decided.reason) == ("MNI", "parsed coordinates")


def test_two_spaces_are_a_conflict_and_the_transform_must_not_run():
    assert resolve(ANALYSIS, NO_TABLES, points("MNI", "TALAIRACH")).value == "UNKNOWN"


def test_a_space_no_rule_matched_does_not_claim_the_parse_answered():
    """`reason` is what the query funnel prints, and "parsed coordinates" would be a lie.

    Carrying the token is what the missing rule gets written from.
    """
    decided = resolve(ANALYSIS, NO_TABLES, points("REFERENCE ATLAS"))
    assert decided.value == "UNKNOWN"
    assert (decided.reason, decided.text) == ("unmatched", "REFERENCE ATLAS")


def test_nothing_anywhere_is_unknown_and_says_why():
    decided = resolve(ANALYSIS, NO_TABLES, points())
    assert (decided.value, decided.reason) == ("UNKNOWN", "empty")


def test_the_deriver_reads_the_same_lexicon_as_the_resolver(tmp_path):
    """The build-time filler and the query-time reader must not disagree about a space.

    Comparing the parser's raw tokens made "MNI" and "MNI152" in one table a mixed paper,
    which the deriver declines to fill -- and the query then had nothing to read either.
    """
    import json

    from pondie.extraction.record.fix.derive import derive_coordinate_spaces

    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps(
            {
                "analyses": [
                    {
                        "table_id": "T1",
                        "points": [{"space": "MNI"}, {"space": "MNI152"}, {"space": "ICBM"}],
                    }
                ]
            }
        )
    )
    table_map = tmp_path / "tables.json"
    table_map.write_text(json.dumps({"T1": "t1"}))

    body = {"analyses": [{"tables": ["t1"], "coordinate_space": {"extraction_status": "absent"}}]}
    assert derive_coordinate_spaces(body, stage1, table_map) == ["analyses[0] -> MNI"]
    assert body["analyses"][0]["coordinate_space"]["value"] == "MNI"


def test_the_deriver_writes_nothing_for_a_space_no_rule_matched(tmp_path):
    """An unrecognised token in the authoritative slot is a wrong answer, not a gap."""
    import json

    from pondie.extraction.record.fix.derive import derive_coordinate_spaces

    stage1 = tmp_path / "stage1.json"
    stage1.write_text(
        json.dumps({"analyses": [{"table_id": "T1", "points": [{"space": "REFERENCE ATLAS"}]}]})
    )
    table_map = tmp_path / "tables.json"
    table_map.write_text(json.dumps({"T1": "t1"}))

    body = {"analyses": [{"tables": ["t1"], "coordinate_space": {"extraction_status": "absent"}}]}
    assert derive_coordinate_spaces(body, stage1, table_map) == []
    assert body["analyses"][0]["coordinate_space"] == {"extraction_status": "absent"}
