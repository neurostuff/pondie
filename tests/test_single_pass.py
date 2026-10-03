"""The single-pass alternative to demands -> satisfy: one call, the whole record.

Measured in experiments/stage-ablation/JOURNAL.md. These pin what the pass is held to,
because its post-conditions are what replace the protection the two-pass split gave.
"""

from pondie.extraction.models import Settings, StageName
from pondie.extraction.prompt import render
from pondie.extraction.stages import SINGLE_PASS, Single, sequence


def _wrapped(value):
    return {"extraction_status": "extracted", "value": value, "value_source": "reported"}


def _analysis(**refs):
    return {"local_id": "ana_1", "name": _wrapped("PTSD < controls"),
            "effect": {"cells": []}, **refs}


def test_the_prompt_carries_both_halves_of_the_schema():
    prompt = render.build_prompt("A paper.", "single", False, "")
    assert "### Analysis" in prompt.system and "### Group" in prompt.system
    assert "### ModelEstimation" in prompt.system
    assert "ONLY extraction pass" in prompt.user


def test_the_prompt_offers_every_entity_list_and_omitted_but_not_tables():
    system = render.build_prompt("A paper.", "single", False, "").system
    lists = system.split("TOP LEVEL of the object and nowhere else:", 1)[1].split("\n", 1)[0]
    for key in ("analyses", "groups", "model_estimations", "omitted"):
        assert key in lists
    assert "tables" not in lists


def test_an_empty_reply_is_a_retry():
    assert "no analyses were emitted" in render.postcondition_failures({"groups": []}, "single")


def test_a_reference_the_reply_never_declares_is_a_retry():
    """19538748: analyses citing a model and a measure the reply emitted as empty lists."""
    payload = {"analyses": [_analysis(model_estimation="mod_vbm", measure="mea_gmv")],
               "model_estimations": [], "measures": []}
    failures = render.postcondition_failures(payload, "single")
    assert any("mod_vbm" in f and "mea_gmv" in f for f in failures)


def test_a_table_the_tables_stage_made_is_not_dangling():
    payload = {"analyses": [_analysis(tables=["tbl2"])]}
    assert render.postcondition_failures(payload, "single", existing=["tbl2"]) == []
    assert render.postcondition_failures(payload, "single", existing=[]) != []


def test_a_listing_entry_left_unaccounted_for_is_a_retry():
    payload = {"analyses": [_analysis()]}
    failures = render.postcondition_failures(payload, "single", listing={"t1#1"})
    assert any("t1#1" in f for f in failures)


def test_a_run_that_names_single_takes_the_single_pass_sequence(tmp_path):
    settings = Settings(payloads=tmp_path, records=tmp_path, model="m",
                        stages=(StageName.tables, StageName.single, StageName.build))
    names = [s.name for s in sequence(settings)]
    assert names == [StageName.tables, StageName.single, StageName.build]
    assert StageName.demands not in [s.name for s in SINGLE_PASS]


def test_the_default_run_is_still_demand_driven(tmp_path):
    names = [s.name for s in sequence(Settings(payloads=tmp_path, records=tmp_path, model="m"))]
    assert StageName.demands in names and StageName.single not in names


def test_the_single_pass_runs_both_halves_payload_repairs():
    assert set(Single().repair_stage) == {"shape", "demands", "satisfy"}


def test_a_retry_that_comes_back_empty_does_not_replace_a_good_answer(tmp_path):
    """The pass keeps its best attempt, not its last.

    The first reply has an analysis and one fault (a reference it does not declare), so it
    is re-asked; both retries return nothing. Writing the last reply turned 6 of 55
    single-pass records empty on the PTSD benchmark.
    """
    import json

    from pondie.extraction.models import Cost, Flavour, ModelReply, Paper

    root = tmp_path / "corpus"
    (root / "S1" / "processed" / "local").mkdir(parents=True)
    (root / "S1" / "stage1").mkdir(parents=True)
    (root / "S1" / "processed" / "local" / "text.tables.txt").write_text("a paper")
    (root / "S1" / "stage1" / "analyses.json").write_text(json.dumps({"analyses": []}))
    paper = Paper(study_id="S1", root=root, flavour=Flavour.local)
    good = {"analyses": [_analysis(measure="mea_missing")]}
    replies = iter([good, {"groups": []}, {"groups": []}])

    def caller(call, *, paper, stage):
        return ModelReply(payload=json.loads(json.dumps(next(replies))), cost=Cost(calls=1))

    settings = Settings(payloads=tmp_path / "p", records=tmp_path / "r", model="m",
                        stages=(StageName.single,))
    outcome = Single().run(paper, settings, caller)
    written = json.loads(outcome.produced[0].read_text())
    assert [a["local_id"] for a in written["analyses"]] == ["ana_1"]
    assert any("mea_missing" in n for n in outcome.notes)
