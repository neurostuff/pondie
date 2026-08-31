"""The contracts exist to fail early. These check that they do."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from pondie.contracts import Cost, Paper, Settings, StageName


def test_a_misspelled_setting_is_an_error_not_a_setting_that_does_nothing():
    with pytest.raises(ValidationError, match="efort"):
        Settings(payloads=Path("/tmp/p"), records=Path("/tmp/r"), model="m", efort="low")


def test_build_without_its_inputs_is_refused_before_any_call_is_made(tmp_path):
    with pytest.raises(ValidationError, match="does not exist"):
        Settings(payloads=tmp_path / "absent", records=tmp_path, model="m",
                 stages=(StageName.build,))


def test_cost_sums_rather_than_tallies():
    total = Cost(input_tokens=5, calls=1) + Cost(input_tokens=7, calls=1)
    assert (total.input_tokens, total.calls) == (12, 2)


def test_a_paper_knows_where_its_inputs_are_without_a_stage_being_told(tmp_path):
    paper = Paper(study_id="S1", root=tmp_path)
    assert paper.text.name == "text.txt"
    assert not paper.ready(), "a paper with no text is not ready, and says so"
