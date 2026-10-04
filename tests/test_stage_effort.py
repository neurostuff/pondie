"""Reasoning effort is set per stage: judgement gets more than transcription."""

import json

from pondie.extraction.models import Cost, Flavour, ModelReply, Paper, Settings, StageName
from pondie.extraction.stages import Fill, Single


def _settings(tmp_path, **kw):
    return Settings(payloads=tmp_path / "p", records=tmp_path / "r", model="m", **kw)


def test_the_default_map_puts_judgement_above_transcription(tmp_path):
    s = _settings(tmp_path)
    assert s.effort_for(StageName.single) == "medium"
    assert s.effort_for(StageName.repair) == "medium"
    assert s.effort_for(StageName.fill) == "low"
    assert s.effort_for(StageName.evidence) == "low"


def test_a_stage_the_map_does_not_name_takes_the_run_effort(tmp_path):
    assert _settings(tmp_path, effort="high").effort_for(StageName.demands) == "high"


def test_an_explicit_map_replaces_the_default(tmp_path):
    s = _settings(tmp_path, effort="low", stage_effort={})
    assert s.effort_for(StageName.single) == "low"


def _paper(tmp_path):
    root = tmp_path / "corpus"
    (root / "S1" / "processed" / "local").mkdir(parents=True)
    (root / "S1" / "stage1").mkdir(parents=True)
    (root / "S1" / "processed" / "local" / "text.tables.txt").write_text("a paper")
    (root / "S1" / "stage1" / "analyses.json").write_text(json.dumps({"analyses": []}))
    return Paper(study_id="S1", root=root, flavour=Flavour.local)


def test_each_stage_calls_the_model_at_its_own_effort(tmp_path):
    seen = {}
    record = {"analyses": [{"local_id": "ana_1", "effect": {"cells": []},
                            "name": {"extraction_status": "extracted", "value": "x",
                                     "value_source": "reported"}}]}

    def caller(call, *, paper, stage):
        seen.setdefault(stage.rstrip("0123456789"), call.effort)
        return ModelReply(payload=json.loads(json.dumps(record if stage == "single" else {})),
                          cost=Cost(calls=1))

    paper, settings = _paper(tmp_path), _settings(tmp_path, stages=(StageName.single,
                                                                     StageName.fill))
    Single().run(paper, settings, caller)
    Fill().run(paper, settings, caller)
    assert seen["single"] == "medium"
    assert seen["fill"] == "low"


def test_the_cache_key_changes_with_the_stage_effort(tmp_path):
    paper = _paper(tmp_path)
    low = Single().depends_on(paper, _settings(tmp_path, stage_effort={StageName.single: "low"}))
    medium = Single().depends_on(paper, _settings(tmp_path))
    assert low["effort"] == "low" and medium["effort"] == "medium"


def test_the_adjudicator_takes_the_effort_it_is_given():
    """It hardcoded `low`, whatever the run asked of `repair`."""
    from pondie.extraction.record.validate import EXTRACTION_SCHEMA
    from pondie.extraction.repair import stage
    from pondie.schema import reader

    record = {
        "analyses": [{"local_id": "ana_1", "regions": ["reg_acc"],
                      "spatial_scope": {"extraction_status": "extracted",
                                        "value": "whole_brain", "value_source": "reported"}}],
        "regions": [{"local_id": "reg_acc"}],
    }
    seen = []

    def caller(call, *, paper, stage):
        seen.append(call.effort)
        return ModelReply(payload={"resolutions": []}, cost=Cost(calls=1))

    stage.adjudicate(record, reader.load(EXTRACTION_SCHEMA), "a paper", caller,
                     study_id="S1", model="m", report=stage.Report(), effort="medium")
    assert seen == ["medium"]
