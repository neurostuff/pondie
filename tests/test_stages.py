"""Stages are functions, so they are called directly and the model is substituted."""

import json

from pondie.extraction import plan, run
from pondie.extraction.models import Cost, ModelReply, Paper, Settings, StageName


class Recorder:
    """A Caller that records what it was asked and returns a fixed payload."""

    def __init__(self):
        self.calls = []

    def __call__(self, call, *, paper, stage):
        self.calls.append((paper, stage))
        return ModelReply(payload={"stage": stage}, cost=Cost(input_tokens=10, calls=1))


def _paper(tmp_path):
    root = tmp_path / "texts"
    (root / "S1" / "processed" / "pubget").mkdir(parents=True)
    (root / "S1" / "stage1").mkdir(parents=True)
    (root / "S1" / "processed" / "pubget" / "text.txt").write_text("a paper")
    (root / "S1" / "stage1" / "analyses.json").write_text('{"analyses": []}')
    (root / "S1" / "stage1" / "table-map.json").write_text(
        '{"t1": {"table_number": "1", "caption": "Peaks", "footer": ""}}'
    )
    return Paper(study_id="S1", root=root)


def _settings(tmp_path, **kw):
    return Settings(
        payloads=tmp_path / "payloads", records=tmp_path / "records", model="test-model", **kw
    )


def test_tables_takes_no_model_call(tmp_path):
    caller = Recorder()
    report = run([_paper(tmp_path)], _settings(tmp_path, stages=(StageName.tables,)), caller)
    assert report.failures == ()
    assert caller.calls == [], "the manifest is copied, never retyped through a model"
    produced = report.papers[0].outcomes[0].produced[0]
    assert json.loads(produced.read_text())["tables"][0]["caption"] == "Peaks"


def test_a_missing_stage_one_parse_stops_the_paper_with_a_reason(tmp_path):
    paper = Paper(study_id="absent", root=tmp_path)
    report = run([paper], _settings(tmp_path, stages=(StageName.tables,)), Recorder())
    assert report.failures, "stage 1 is an input; the pipeline does not regenerate it"
    assert "stage-1" in report.papers[0].failed.reason


def test_a_produced_stage_is_skipped_and_plan_says_so_without_spending(tmp_path):
    settings = _settings(tmp_path, stages=(StageName.tables,))
    paper = _paper(tmp_path)
    run([paper], settings, Recorder())
    assert plan([paper], settings) == {"S1": ["skip:tables"]}


def test_evidence_asks_about_the_fields_that_exist_not_about_the_paper(tmp_path, monkeypatch):
    """The quote pass is driven by earlier payloads, and batches its questions."""
    from pondie.extraction.stages import Evidence

    paper = _paper(tmp_path)
    settings = _settings(tmp_path, stages=(StageName.evidence,))
    payload = settings.payloads / "S1" / "satisfy"
    payload.mkdir(parents=True)
    (payload / "payload.json").write_text(
        json.dumps(
            {"groups": [{"name": {"extraction_status": "extracted", "value": "patients"}}]}
        )
    )

    caller = Recorder()
    outcome = Evidence().run(paper, settings, caller)
    assert not outcome.skipped and outcome.ok
    assert [stage for _p, stage in caller.calls] == ["evidence"]


def test_evidence_is_skipped_rather_than_failed_when_turned_off(tmp_path):
    from pondie.extraction.stages import Evidence

    settings = _settings(tmp_path, stages=(StageName.evidence,), retrieve_evidence=False)
    outcome = Evidence().run(_paper(tmp_path), settings, Recorder())
    assert outcome.skipped and outcome.ok and "disabled" in outcome.reason
