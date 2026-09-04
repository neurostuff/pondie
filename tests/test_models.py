"""The pipeline models exist to fail early. These check that they do."""

import collections
from pathlib import Path

import pytest
from pydantic import ValidationError

from pondie.extraction.models import (
    Cost,
    Paper,
    PaperOutcome,
    RunReport,
    Settings,
    StageName,
    StageOutcome,
    Workflow,
)
from pondie.extraction.stages import sequence


def test_a_misspelled_setting_is_an_error_not_a_setting_that_does_nothing():
    with pytest.raises(ValidationError, match="efort"):
        Settings(payloads=Path("/tmp/p"), records=Path("/tmp/r"), model="m", efort="low")


def test_build_without_its_inputs_is_refused_before_any_call_is_made(tmp_path):
    with pytest.raises(ValidationError, match="does not exist"):
        Settings(
            payloads=tmp_path / "absent",
            records=tmp_path,
            model="m",
            stages=(StageName.build,),
        )


def test_cost_sums_rather_than_tallies():
    total = Cost(input_tokens=5, calls=1) + Cost(input_tokens=7, calls=1)
    assert (total.input_tokens, total.calls) == (12, 2)


def test_a_paper_knows_where_its_inputs_are_without_a_stage_being_told(tmp_path):
    paper = Paper(study_id="S1", root=tmp_path)
    assert paper.text.name == "text.txt"
    assert not paper.ready(), "a paper with no text is not ready, and says so"


def test_a_workflow_the_stages_do_not_implement_is_refused(tmp_path):
    """`entity_first` was accepted, stored, and then never read.

    A run asked for it, got the demand-driven ordering, and reported success -- the exact
    failure `extra="forbid"` exists to prevent, reached by a correctly spelled value.
    """

    with pytest.raises(ValidationError, match="not implemented"):
        Settings(
            payloads=tmp_path, records=tmp_path, model="m", workflow=Workflow.entity_first
        )


def test_the_implemented_workflow_is_still_accepted(tmp_path):
    settings = Settings(payloads=tmp_path, records=tmp_path, model="m")
    assert settings.workflow is Workflow.demand_driven
    assert [stage.name.value for stage in sequence(settings)] == [
        "tables",
        # Appends prose-stated coordinates to the parse. Before "split", which rewrites
        # the whole document, so one stage owns the analyses list at a time.
        "prose",
        "split",
        "demands",
        "satisfy",
        "evidence",
        "build",
        "repair",
    ]


def test_costs_add(tmp_path):
    """A run total is one addition, not a tally scraped back off logging."""
    total = Cost(input_tokens=10, output_tokens=2, calls=1) + Cost(
        input_tokens=5, output_tokens=1, calls=1
    )
    assert (total.input_tokens, total.output_tokens, total.calls) == (15, 3, 2)


def test_a_report_totals_what_its_stages_spent(tmp_path):
    report = RunReport(
        papers=tuple(
            PaperOutcome(
                study_id=study,
                outcomes=(
                    StageOutcome(
                        stage=StageName.demands,
                        study_id=study,
                        cost=Cost(input_tokens=100, calls=1),
                    ),
                    StageOutcome(stage=StageName.build, study_id=study),
                ),
            )
            for study in ("A", "B")
        )
    )
    assert report.cost.input_tokens == 200
    assert report.cost.calls == 2


def test_evidence_devices_are_spread_and_reproducible(tmp_path):
    """crc32 and not hash: Python randomises string hashing per process, so a resumed run
    would assign differently from the one that wrote the payloads, and a spread that cannot
    be reproduced cannot be debugged."""
    settings = Settings(
        payloads=tmp_path,
        records=tmp_path,
        model="m",
        reranker_devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    )
    papers = [Paper(study_id=f"study{i}", root=tmp_path) for i in range(40)]
    spread = collections.Counter(settings.device_for(p) for p in papers)
    assert len(spread) == 4 and max(spread.values()) - min(spread.values()) <= 2
    assert settings.device_for(papers[0]) == settings.device_for(papers[0])
