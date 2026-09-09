"""The pipeline models exist to fail early. These check that they do."""

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


def test_the_pipeline_is_one_ordering(tmp_path):
    """There is one workflow, so the order is a property of `DEMAND_DRIVEN` and not of a
    setting. It is pinned because the order is the design: `demands` before `satisfy` so the
    analyses declare their terms first, `fill` after `satisfy` because it finishes what that
    pass left open, and `evidence` after `fill` so a value the loop adds gets a quote."""
    settings = Settings(payloads=tmp_path, records=tmp_path, model="m")
    assert [stage.name.value for stage in sequence(settings)] == [
        "tables",
        "prose",
        "split",
        "demands",
        "satisfy",
        "fill",
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


def test_a_run_that_lost_its_tables_says_so_even_though_nothing_failed():
    """The goldbench run reported "15 paper(s), 1 failed" while fourteen of them held zero
    Table records: `data/corpus-rev` ships the text and the stage-1 parse but no
    `tables.jsonl`, so `Tables` wrote an empty list and noted it, and nothing above the stage
    repeated the note. Polarity coverage fell from 54% to 38% on a run that reported success.
    """
    report = RunReport(
        papers=(
            PaperOutcome(
                study_id="P1",
                outcomes=(
                    StageOutcome(
                        stage=StageName.tables,
                        study_id="P1",
                        notes=("no tables.jsonl beside the pubget text; no Table records",),
                    ),
                ),
            ),
            PaperOutcome(
                study_id="P2",
                outcomes=(StageOutcome(stage=StageName.tables, study_id="P2"),),
            ),
        )
    )
    assert report.starved() == ("P1",)
    assert "WARNING" in report.summary() and "P1" in report.summary()
    assert "P2" not in report.summary()


def test_a_run_with_every_manifest_present_warns_about_nothing():
    report = RunReport(
        papers=(
            PaperOutcome(
                study_id="P1", outcomes=(StageOutcome(stage=StageName.tables, study_id="P1"),)
            ),
        )
    )
    assert report.starved() == ()
    assert "WARNING" not in report.summary()
