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


def test_the_default_run_is_the_single_pass(tmp_path):
    names = [s.name for s in sequence(Settings(payloads=tmp_path, records=tmp_path, model="m"))]
    assert StageName.single in names and StageName.demands not in names


def test_naming_demands_and_satisfy_still_runs_the_split(tmp_path):
    settings = Settings(payloads=tmp_path, records=tmp_path, model="m",
                        stages=(StageName.demands, StageName.satisfy, StageName.build))
    names = [s.name for s in sequence(settings)]
    assert names == [StageName.demands, StageName.satisfy, StageName.build]


def test_the_single_pass_runs_both_halves_payload_repairs():
    assert set(Single().repair_stage) == {"shape", "demands", "satisfy"}


def test_a_retry_that_comes_back_empty_does_not_replace_a_good_answer(tmp_path, caplog):
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
    with caplog.at_level("INFO", logger="pondie"):
        outcome = Single().run(paper, settings, caller)
    written = json.loads(outcome.produced[0].read_text())
    assert [a["local_id"] for a in written["analyses"]] == ["ana_1"]
    assert any("mea_missing" in n for n in outcome.notes)
    # Each failed attempt is logged as it happens, not only in the notes at the end.
    assert "S1/single attempt 1/3 failed" in caplog.text and "mea_missing" in caplog.text


def _staged(tmp_path):
    import json

    from pondie.extraction.models import Flavour, Paper

    root = tmp_path / "corpus"
    (root / "S1" / "processed" / "local").mkdir(parents=True)
    (root / "S1" / "stage1").mkdir(parents=True)
    (root / "S1" / "processed" / "local" / "text.tables.txt").write_text("a paper")
    (root / "S1" / "stage1" / "analyses.json").write_text(json.dumps({"analyses": []}))
    return Paper(study_id="S1", root=root, flavour=Flavour.local)


def _scripted(replies, seen):
    import json

    from pondie.extraction.models import Cost, ModelReply

    replies = iter(replies)

    def caller(call, *, paper, stage):
        seen.append(stage)
        return ModelReply(payload=json.loads(json.dumps(next(replies))), cost=Cost(calls=1))

    return caller


def test_completion_asks_only_for_the_missing_entities(tmp_path):
    """31887311: a correct contrast whose cohorts were never emitted, in every attempt."""
    import json

    dangling = {"analyses": [_analysis(groups=[{"group": "grp_bvftd"}])], "groups": []}
    completion = {"groups": [{"local_id": "grp_bvftd", "name": _wrapped("bvFTD")}]}
    seen = []
    settings = Settings(payloads=tmp_path / "p", records=tmp_path / "r", model="m",
                        stages=(StageName.single,), complete_references=True)
    outcome = Single().run(_staged(tmp_path), settings,
                           _scripted([dangling, dangling, dangling, completion], seen))
    written = json.loads(outcome.produced[0].read_text())
    assert [g["local_id"] for g in written["groups"]] == ["grp_bvftd"]
    assert seen == ["single", "single", "single", "single-complete"]
    assert any("1 of 1 references now resolve" in n for n in outcome.notes)


def test_completion_can_be_turned_off(tmp_path):
    dangling = {"analyses": [_analysis(groups=[{"group": "grp_bvftd"}])], "groups": []}
    seen = []
    settings = Settings(payloads=tmp_path / "p", records=tmp_path / "r", model="m",
                        stages=(StageName.single,), complete_references=False)
    Single().run(_staged(tmp_path), settings, _scripted([dangling] * 3, seen))
    assert seen == ["single"] * 3


def test_omitted_filed_under_study_is_hoisted_to_the_top_level():
    """23021615: the reply nested `omitted` in `study`. It then reached the record as an
    undeclared Study attribute, and the listing check, which reads the top level, saw the
    pass decline nothing."""
    reply = {"study": {"omitted": [{"key": "t1#1", "reason": "seed_coordinate"}]},
             "analyses": [_analysis()]}
    payload, notes = render.normalize(reply, "single")
    assert payload["omitted"] == [{"key": "t1#1", "reason": "seed_coordinate"}]
    assert "omitted" not in payload.get("study", {})
    assert any("hoisted 'omitted'" in n for n in notes)
    assert not render.unconsumed_listing(payload, {"t1#1"})


def test_hoisting_omitted_keeps_what_was_already_at_the_top_level():
    reply = {"omitted": [{"key": "t1#1", "reason": "localizer"}],
             "study": {"omitted": [{"key": "t1#2", "reason": "seed_coordinate"}]},
             "analyses": [_analysis()]}
    payload, _ = render.normalize(reply, "single")
    assert [o["key"] for o in payload["omitted"]] == ["t1#1", "t1#2"]


def test_a_shopping_list_filed_under_study_still_reaches_satisfy():
    reply = {"study": {"required_entities": [{"local_id": "grp_a", "kind": "Group"}]},
             "analyses": [{**_analysis(), "effect": {"cells": [{}]}}]}
    payload, _ = render.normalize(reply, "demands")
    assert payload["required_entities"] == [{"local_id": "grp_a", "kind": "Group"}]


def test_omitted_never_reaches_the_record(tmp_path):
    import json

    from pondie.extraction.record.builder import merge_payloads

    reply = {"study": {"omitted": [{"key": "t1#1", "reason": "seed_coordinate"}]},
             "analyses": [_analysis()]}
    payload, _ = render.normalize(reply, "single")
    (tmp_path / "single.json").write_text(json.dumps(payload))
    body, _ = merge_payloads(tmp_path)
    assert "omitted" not in body


def _two_models(term_b: str):
    """Two analyses, each on its own model, both models declaring a group term."""
    def model(mid, tid):
        return {"local_id": mid, "terms": [{"local_id": tid, "name": _wrapped("group"),
                "type": _wrapped("categorical"),
                "levels": [{"level": _wrapped("PTSD")}, {"level": _wrapped("control")}]}]}
    def analysis(aid, mid, tid):
        return {"local_id": aid, "name": _wrapped(aid), "model_estimation": mid,
                "effect": {"cells": [{"term": tid, "level": _wrapped("PTSD"),
                                      "direction": _wrapped("negative")}]}}
    return {"analyses": [analysis("ana_1", "mod_a", "trm_group"),
                         analysis("ana_2", "mod_b", term_b)],
            "model_estimations": [model("mod_a", "trm_group"), model("mod_b", "trm_group")]}


def _calls_for(tmp_path, reply):
    import json

    from pondie.extraction.models import Cost, ModelReply

    calls = []

    def caller(call, *, paper, stage):
        calls.append(stage)
        return ModelReply(payload=json.loads(json.dumps(reply)), cost=Cost(calls=1))

    settings = Settings(payloads=tmp_path / "p", records=tmp_path / "r", model="m",
                        stages=(StageName.single,), complete_references=False)
    Single().run(_staged(tmp_path), settings, caller)
    return calls


def test_a_fault_a_repair_fixes_does_not_cost_a_retry(tmp_path):
    """16199014: `trm_group` declared once per model read as a duplicated id and paid for a
    second full attempt, though `scope_duplicate_terms` scopes it at no cost."""
    assert render.postcondition_failures(_two_models("trm_group"), "single"), "raw: a fault"
    assert _calls_for(tmp_path, _two_models("trm_group")) == ["single"]


def test_a_fault_no_repair_fixes_is_still_retried(tmp_path):
    assert _calls_for(tmp_path, _two_models("trm_nowhere")) == ["single"] * 3


def test_an_omission_without_a_listing_is_no_fault():
    """11950456 has no listing, recorded omissions saying so, and was re-asked."""
    payload = {"analyses": [_analysis()],
               "omitted": [{"key": "stage-1 table listing", "reason": "none was supplied"}]}
    assert render.postcondition_failures(payload, "single", listing=set()) == []
    assert render.postcondition_failures(payload, "single", listing={"t1#1"}) != []
