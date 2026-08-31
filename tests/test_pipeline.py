"""What the pipeline package must not get wrong.

The package exists to make four questions answerable without reading a log: what will
run, what did it cost, which repairs fired, and why did a stage not run. Each of those is
a test here. The stages that call a model are exercised through fakes -- what is checked
is the sequencing and the accounting, not the prompts, which have their own suites.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

import pytest


from pipeline import repairs as repair_module  # noqa: E402
from pipeline.driver import plan, run_paper  # noqa: E402
from pipeline.kinds import (DONE, FAILED, NOT_REQUESTED, SKIPPED, Cost,  # noqa: E402
                            Paper, PaperOutcome, RunReport, StageOutcome, TableParse)
from pipeline.stages import Settings, SignSplit, Stage  # noqa: E402


def _paper(tmp_path: Path, study: str = "S1", flavours=(("local", "text.tables.txt"),)):
    for flavour, name in flavours:
        target = tmp_path / study / "processed" / flavour
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text("Methods\n\nTwenty patients were recruited.\n")
    stage1 = tmp_path / study / "stage1"
    stage1.mkdir(parents=True, exist_ok=True)
    (stage1 / "analyses.json").write_text(json.dumps({"analyses": []}))
    (stage1 / "table-map.json").write_text("{}")
    return Paper(study, tmp_path)


def _settings(tmp_path: Path) -> Settings:
    return Settings(payloads=tmp_path / "p", records=tmp_path / "r",
                    key_file=tmp_path / ".env", model="m")


# --- Paper ------------------------------------------------------------------

def test_the_table_bearing_flavour_wins(tmp_path):
    # A locator searching a table-free text cannot find the sentence a group size was
    # read from, so this preference is not cosmetic.
    paper = _paper(tmp_path, flavours=(("pubget", "text.txt"), ("local", "text.tables.txt")))
    assert paper.text_path.name == "text.tables.txt"
    assert paper.flavour == "local"


def test_a_paper_with_no_text_says_so_before_anything_is_spent(tmp_path):
    (tmp_path / "S9" / "stage1").mkdir(parents=True)
    (tmp_path / "S9" / "stage1" / "analyses.json").write_text("{}")
    ready, why = Paper("S9", tmp_path).is_ready()
    assert not ready and "no text" in why


def test_a_paper_with_no_table_parse_is_not_ready(tmp_path):
    paper = _paper(tmp_path)
    paper.stage1_path.unlink()
    ready, why = paper.is_ready()
    assert not ready and "table parse" in why


# --- TableParse and the withheld halves -------------------------------------

def test_described_and_withheld_are_separable(tmp_path):
    path = tmp_path / "analyses.json"
    path.write_text(json.dumps({"analyses": [
        {"name": "A > B"},
        {"name": "A > B (reversed)", "withhold": True, "mirror_of": "A > B"}]}))
    parse = TableParse.load(path)
    assert [a.name for a in parse.described()] == ["A > B"]
    assert [a.mirror_of for a in parse.withheld()] == ["A > B"]


def test_the_split_stage_withholds_the_reversed_half(tmp_path):
    paper = _paper(tmp_path)
    paper.stage1_path.write_text(json.dumps({"analyses": [{"name": "A > B", "points": [
        {"values": [{"kind": "t", "value": 3.1}]},
        {"values": [{"kind": "t", "value": -2.9}]}]}]}))
    outcome = SignSplit().run(paper, _settings(tmp_path))
    assert outcome.status == DONE
    parse = TableParse.load(paper.stage1_path)
    assert len(parse.described()) == 1 and len(parse.withheld()) == 1
    assert parse.sign_split_applied


def test_the_split_stage_does_not_re_split_what_it_already_partitioned(tmp_path):
    # Idempotence is what lets a resumed run re-enter this stage safely.
    paper = _paper(tmp_path)
    paper.stage1_path.write_text(json.dumps({"analyses": [{"name": "A > B", "points": [
        {"values": [{"kind": "t", "value": 3.1}]},
        {"values": [{"kind": "t", "value": -2.9}]}]}]}))
    settings = _settings(tmp_path)
    SignSplit().run(paper, settings)
    before = paper.stage1_path.read_text()
    assert SignSplit().run(paper, settings).status == SKIPPED
    assert paper.stage1_path.read_text() == before


# --- accounting -------------------------------------------------------------

def test_costs_add():
    total = Cost(10, 2, calls=1) + Cost(5, 1, calls=1)
    assert (total.prompt_tokens, total.completion_tokens, total.calls) == (15, 3, 2)


def test_a_free_stage_says_free():
    assert Cost().render() == "free"


def test_the_report_totals_by_stage():
    report = RunReport()
    for study in ("A", "B"):
        outcome = PaperOutcome(study)
        outcome.stages.append(StageOutcome("demands", study, DONE, Cost(100, 10, calls=1)))
        outcome.stages.append(StageOutcome("build", study, DONE, Cost()))
        report.papers.append(outcome)
    assert report.by_stage()["demands"].prompt_tokens == 200
    assert report.cost.calls == 2
    assert "demands" in report.explain()


# --- sequencing -------------------------------------------------------------

class _Fake(Stage):
    def __init__(self, name, fails=False):
        self.name, self.fails = name, fails
        self.ran = 0

    def produces(self, paper, settings):
        return settings.payloads / paper.study_id / f"{self.name}.done"

    def perform(self, paper, settings):
        self.ran += 1
        if self.fails:
            raise RuntimeError("boom")
        target = self.produces(paper, settings)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok")
        return StageOutcome(self.name, paper.study_id, DONE, Cost(7, 1, calls=1))


def test_a_failed_stage_stops_the_paper_and_the_rest_are_recorded(tmp_path):
    # `satisfy` reads what `demands` wrote; running it anyway produces a second, more
    # confusing failure that hides the first.
    paper = _paper(tmp_path)
    first, second, third = _Fake("one"), _Fake("two", fails=True), _Fake("three")
    outcome = run_paper(paper, _settings(tmp_path), (first, second, third))
    assert [s.status for s in outcome.stages] == [DONE, FAILED, NOT_REQUESTED]
    assert third.ran == 0
    assert not outcome.ok


def test_a_stage_whose_artefact_exists_is_skipped(tmp_path):
    paper, settings = _paper(tmp_path), _settings(tmp_path)
    stage = _Fake("one")
    run_paper(paper, settings, (stage,))
    outcome = run_paper(paper, settings, (stage,))
    assert outcome.stages[0].status == SKIPPED
    assert stage.ran == 1


def test_redo_overrides_the_skip(tmp_path):
    paper, settings = _paper(tmp_path), _settings(tmp_path)
    stage = _Fake("one")
    run_paper(paper, settings, (stage,))
    settings.redo = True
    run_paper(paper, settings, (stage,))
    assert stage.ran == 2


def test_a_paper_that_cannot_run_is_reported_not_raised(tmp_path):
    (tmp_path / "S9" / "stage1").mkdir(parents=True)
    outcome = run_paper(Paper("S9", tmp_path), _settings(tmp_path), (_Fake("one"),))
    assert outcome.stages[0].status == FAILED and not outcome.ok


def test_the_plan_says_what_would_run_without_running_it(tmp_path):
    paper, stage = _paper(tmp_path), _Fake("one")
    text = plan([paper], _settings(tmp_path), (stage,))
    assert "one=run" in text and stage.ran == 0


# --- the repair sequence ----------------------------------------------------

def test_the_declared_repair_order_holds():
    assert repair_module.check_order(repair_module.build_sequence()) == []


def test_an_order_that_violates_its_own_constraint_is_refused():
    late = repair_module.Repair("a", "", lambda body, ctx: [], after="b")
    early = repair_module.Repair("b", "", lambda body, ctx: [])
    assert repair_module.check_order((late, early))
    with pytest.raises(ValueError):
        repair_module.apply_all({}, repair_module.Context(classes={}), (late, early))


def test_the_log_says_which_repairs_fired():
    sequence = (repair_module.Repair("noisy", "", lambda body, ctx: ["did a thing"]),
                repair_module.Repair("quiet", "", lambda body, ctx: []))
    log = repair_module.apply_all({}, repair_module.Context(classes={}), sequence)
    assert log.fired() == ["noisy"]
    assert log.total == 1
    assert "did a thing" in log.explain()


def test_the_mirror_runs_after_the_direction_fill():
    # The mirror copies the corrected contrast; taking it before the fill would copy a
    # cell the builder was about to sign.
    names = [r.name for r in repair_module.build_sequence()]
    assert names.index("mirrored") > names.index("directions")
    assert names.index("directions") > names.index("cell_levels")


# --- adopting a corpus split by the earlier rule -----------------------------

from parse_tables import adopt_withholding  # noqa: E402


def _part(parent, direction, points=1):
    return {"name": f"{parent} ({direction})", "split_from": parent,
            "split_direction": direction, "split_rule": "sign-of-directional-statistic",
            "points": [{"values": [{"kind": "t", "value": 1.0}]}] * points}


def test_an_old_pair_becomes_a_described_half_and_a_withheld_one():
    analyses, converted = adopt_withholding(
        [_part("A > B", "positive"), _part("A > B", "negative")])
    described = [a for a in analyses if not a.get("withhold")]
    withheld = [a for a in analyses if a.get("withhold")]
    # The described half takes the paper's own name back: the prompt tells the model to
    # quote the parsed name verbatim, and "(positive)" is not what the paper called it.
    assert described[0]["name"] == "A > B"
    assert withheld[0]["mirror_of"] == "A > B"
    assert converted


def test_an_entry_split_on_something_else_as_well_is_left_alone():
    # Three parts sharing a parent means a band or a session was also split on, and
    # which one the paper describes is not answerable from the sign.
    parts = [_part("A > B", "positive"), _part("A > B", "negative"),
             _part("A > B", "positive")]
    analyses, converted = adopt_withholding(parts)
    assert converted == []
    assert not any(a.get("withhold") for a in analyses)


def test_an_unsplit_analysis_is_untouched():
    entry = {"name": "plain", "points": []}
    analyses, converted = adopt_withholding([entry])
    assert converted == [] and analyses == [entry]


def test_adoption_is_idempotent():
    parts = [_part("A > B", "positive"), _part("A > B", "negative")]
    once, _ = adopt_withholding(parts)
    twice, _ = adopt_withholding(once)
    assert [a["name"] for a in once] == [a["name"] for a in twice]
    assert sum(1 for a in twice if a.get("withhold")) == 1


# --- resume safety ----------------------------------------------------------

from pipeline.stages import Evidence  # noqa: E402


def test_a_started_but_unfinished_evidence_stage_is_not_done(tmp_path):
    # `noev/` is the pre-evidence backup and exists before the work. Seventeen papers
    # crashed after that point and a resume skipped every one, building records with no
    # evidence at all.
    paper = _paper(tmp_path)
    settings = _settings(tmp_path)
    payloads = settings.payload_dir(paper)
    (payloads / "noev").mkdir(parents=True)
    (payloads / "analyses.json").write_text(json.dumps(
        {"analyses": [{"name": {"extraction_status": "extracted", "value": "x"}}]}))
    assert Evidence().is_done(paper, settings) is False


def test_evidence_is_done_once_a_payload_carries_it(tmp_path):
    paper = _paper(tmp_path)
    settings = _settings(tmp_path)
    payloads = settings.payload_dir(paper)
    (payloads / "noev").mkdir(parents=True)
    (payloads / "analyses.json").write_text(json.dumps(
        {"analyses": [{"name": {"extraction_status": "extracted", "value": "x",
                                "evidence": {"status": "present"}}}]}))
    assert Evidence().is_done(paper, settings) is True


def test_devices_are_spread_and_reproducible(tmp_path):
    # crc32 and not hash: Python randomises string hashing per process, so a resumed run
    # would assign differently from the one that wrote the payloads.
    settings = Settings(payloads=tmp_path / "p", records=tmp_path / "r",
                        key_file=tmp_path / ".env", model="m",
                        reranker_devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"))
    papers = [Paper(f"study{i}", tmp_path) for i in range(40)]
    spread = collections.Counter(settings.device_for(p) for p in papers)
    assert len(spread) == 4 and max(spread.values()) - min(spread.values()) <= 2
    assert settings.device_for(papers[0]) == settings.device_for(papers[0])


# --- the deterministic repairs added for validator findings -------------------

import build_record as br  # noqa: E402


@pytest.fixture(scope="module")
def classes():
    import schema_utils
    from build_record import EXTRACTION_SCHEMA
    return schema_utils.load_imported_classes(EXTRACTION_SCHEMA)


def test_a_wrapper_in_a_reference_slot_is_unwrapped(classes):
    # The model has just written twenty wrappers and writes a twenty-first into a slot
    # that holds a bare local_id. The wrapper's own value is the answer.
    body = {"analyses": [{"local_id": "a1",
                          "model_estimation": {"extraction_status": "extracted",
                                               "value": "m1", "evidence": {"status": "present"}}}]}
    changed = br.unwrap_plain_slots(body, classes)
    assert body["analyses"][0]["model_estimation"] == "m1"
    assert changed


def test_a_wrapper_in_an_evidence_slot_is_left_alone(classes):
    # An evidence slot is supposed to hold a wrapper; unwrapping it destroys the value
    # and the span that warrants it.
    body = {"groups": [{"local_id": "g1",
                        "name": {"extraction_status": "extracted", "value": "controls",
                                 "evidence": {"status": "present"}}}]}
    br.unwrap_plain_slots(body, classes)
    assert isinstance(body["groups"][0]["name"], dict)
    assert body["groups"][0]["name"]["value"] == "controls"


def test_a_numeric_string_becomes_the_number_its_slot_declares(classes):
    body = {"acquisitions": [{"local_id": "acq1",
                              "acquisition_duration_seconds": {
                                  "extraction_status": "extracted", "value": "252 s",
                                  "evidence": {"status": "present"}}}]}
    changed = br.coerce_numeric_values(body, classes)
    assert body["acquisitions"][0]["acquisition_duration_seconds"]["value"] == 252.0
    assert changed


def test_a_value_that_is_not_a_number_is_left_for_the_validator(classes):
    # Inventing a number is worse than reporting a string.
    body = {"acquisitions": [{"local_id": "acq1",
                              "acquisition_duration_seconds": {
                                  "extraction_status": "extracted", "value": "not stated",
                                  "evidence": {"status": "present"}}}]}
    assert br.coerce_numeric_values(body, classes) == []
    assert body["acquisitions"][0]["acquisition_duration_seconds"]["value"] == "not stated"


def test_a_table_written_as_a_study_attribute_is_rehomed(classes):
    # The stray key is dropped on load, so every analysis pointing at it loses the table
    # its coordinates join through and the paper contributes nothing.
    body = {"analyses": [{"local_id": "a1", "tables": ["tab4"]}],
            "tab4": {"table_number": {"extraction_status": "extracted", "value": 4}}}
    moved = br.rehome_stray_tables(body, classes)
    assert "tab4" not in body
    assert [t["local_id"] for t in body["tables"]] == ["tab4"]
    assert moved


def test_an_unreferenced_stray_key_is_left_reported(classes):
    body = {"analyses": [], "somethingElse": {"x": 1}}
    assert br.rehome_stray_tables(body, classes) == []
    assert "somethingElse" in body


def _scoped(cell_term):
    return {
        "model_estimations": [
            {"local_id": "m_low", "terms": [{"local_id": "t_low", "name": _wrapped("group")}]},
            {"local_id": "m_high", "inputs_from": ["m_low"],
             "terms": [{"local_id": "t_high", "name": _wrapped("condition")}]},
            {"local_id": "m_other", "terms": [{"local_id": "t_other", "name": _wrapped("group")}]},
        ],
        "analyses": [{"local_id": "a1", "model_estimation": "m_high",
                      "effect": {"cells": [{"term": cell_term, "level": _wrapped("x")}]}}]}


def test_a_cell_is_repointed_to_the_same_named_term_in_scope():
    # `t_other` names "group" but sits in a model the analysis does not reach; `t_low`
    # names "group" and is reachable through inputs_from.
    body = _scoped("t_other")
    changed = br.repoint_out_of_scope_terms(body)
    assert body["analyses"][0]["effect"]["cells"][0]["term"] == "t_low"
    assert changed


def test_a_cell_already_in_scope_is_untouched():
    body = _scoped("t_high")
    assert br.repoint_out_of_scope_terms(body) == []
    assert body["analyses"][0]["effect"]["cells"][0]["term"] == "t_high"


def test_no_same_named_term_in_scope_is_left_reported():
    body = _scoped("t_other")
    # Rename the reachable term so nothing in scope matches.
    body["model_estimations"][0]["terms"][0]["name"]["value"] = "timepoint"
    assert br.repoint_out_of_scope_terms(body) == []
    assert body["analyses"][0]["effect"]["cells"][0]["term"] == "t_other"


def test_two_same_named_terms_in_scope_are_not_guessed_between():
    body = _scoped("t_other")
    body["model_estimations"][1]["terms"].append(
        {"local_id": "t_dup", "name": _wrapped("group")})
    assert br.repoint_out_of_scope_terms(body) == []


def test_a_valueless_wrapper_in_a_reference_slot_is_dropped(classes):
    # Every valueless wrapper in the corpus says `not_reported`, which is correct in an
    # evidence slot and meaningless in a reference slot: a reference has no wrapper form,
    # so "not reported" is simply absence.
    body = {"model_estimations": [{"local_id": "m1", "terms": [
        {"local_id": "t1", "name": {"value": "age"},
         "assessment": {"extraction_status": "not_reported",
                        "evidence": {"status": "not_applicable"}}}]}]}
    changed = br.unwrap_plain_slots(body, classes)
    assert "assessment" not in body["model_estimations"][0]["terms"][0]
    assert changed and "dropped" in changed[0]


def test_a_bare_scalar_in_an_evidence_slot_is_wrapped(classes):
    body = {"analyses": [{"local_id": "a1", "effect": {"cells": [
        {"term": "t1", "level": {"value": "x"}, "direction": "held"}]}}]}
    changed = br.unwrap_plain_slots(body, classes)
    cell = body["analyses"][0]["effect"]["cells"][0]
    assert cell["direction"]["value"] == "held"
    assert cell["direction"]["extraction_status"] == "extracted"
    # No span was offered, so the evidence is honestly not_found rather than invented.
    assert cell["direction"]["evidence"]["status"] == "not_found"
    assert changed


# --- the Tables stage, and the regression that motivated it -------------------

from pipeline.stages import Tables  # noqa: E402


def _paper_with_manifest(tmp_path, rows, flavour="ace", name="text.txt"):
    target = tmp_path / "S1" / "processed" / flavour
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_text("Methods\n\nTwenty patients were recruited.\n")
    (target / "tables.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    stage1 = tmp_path / "S1" / "stage1"
    stage1.mkdir(parents=True, exist_ok=True)
    (stage1 / "analyses.json").write_text(json.dumps({"analyses": []}))
    return Paper("S1", tmp_path)


def test_tables_are_copied_from_the_flavour_the_text_came_from(tmp_path):
    # The earlier implementation hardcoded processed/pubget/tables.jsonl, which no paper
    # staged from `ace` or `elsevier` has -- and this corpus is mostly those.
    paper = _paper_with_manifest(tmp_path, [
        {"table_id": "t0035", "table_number": 3, "caption": "Between-group differences.",
         "footer": "L, left."}])
    settings = _settings(tmp_path)
    outcome = Tables().run(paper, settings)
    assert outcome.status == DONE
    written = json.loads((settings.payload_dir(paper) / "tables.json").read_text())
    assert [t["local_id"] for t in written["tables"]] == ["t0035"]
    assert written["tables"][0]["caption"]["value"] == "Between-group differences."
    # The id map keeps the identity the staging wrote, so `Analysis.tables` references
    # into the parse still resolve.
    assert json.loads(paper.table_map_path.read_text()) == {"t0035": "t0035"}


def test_a_table_with_no_caption_is_not_reported_rather_than_empty(tmp_path):
    paper = _paper_with_manifest(tmp_path, [{"table_id": "t1", "table_number": 1}])
    settings = _settings(tmp_path)
    Tables().run(paper, settings)
    written = json.loads((settings.payload_dir(paper) / "tables.json").read_text())
    assert written["tables"][0]["caption"]["extraction_status"] == "not_reported"
    assert written["tables"][0]["table_number"]["value"] == "Table 1"


def test_table_number_is_not_used_as_the_identifier(tmp_path):
    # One paper in the corpus carries two tables numbered 1; keying on the number would
    # collapse them into one record.
    paper = _paper_with_manifest(tmp_path, [
        {"table_id": "a", "table_number": 1}, {"table_id": "b", "table_number": 1}])
    settings = _settings(tmp_path)
    Tables().run(paper, settings)
    written = json.loads((settings.payload_dir(paper) / "tables.json").read_text())
    assert [t["local_id"] for t in written["tables"]] == ["a", "b"]


def test_a_paper_with_no_manifest_is_not_a_failure(tmp_path):
    paper = _paper_with_manifest(tmp_path, [])
    (paper.text_path.parent / "tables.jsonl").unlink()
    outcome = Tables().run(paper, _settings(tmp_path))
    assert outcome.status == DONE and "no tables.jsonl" in outcome.notes[0]


def test_the_baseline_runs_tables_before_demands():
    # The analyses pass is told the local_ids this stage assigns, so it cannot run after.
    from pipeline.stages import BASELINE
    names = [s.name for s in BASELINE]
    assert names.index("tables") < names.index("demands")
    assert "tables" in dict((s.name, s.needs) for s in BASELINE)["demands"]


# --- the link from an analysis to its parsed row group ------------------------

def _wrapped(value):
    """A realistic ExtractedValue. `_is_field` keys on `extraction_status`, so a fixture
    of bare `{"value": ...}` never gets unwrapped and tests nothing."""
    return {"extraction_status": "extracted", "value": value, "value_source": "reported",
            "evidence": {"status": "not_applicable"}}


def _stage1(tmp_path, entries):
    path = tmp_path / "analyses.json"
    path.write_text(json.dumps({"analyses": entries}))
    return path


def test_a_present_key_that_names_a_row_group_is_left_alone(tmp_path):
    stage1 = _stage1(tmp_path, [{"table_id": "t1", "name": "SZ > HC"},
                                {"table_id": "t1", "name": "HC > SZ"}])
    body = {"analyses": [{"local_id": "a1", "tables": ["t1"],
                          "name": _wrapped("HC > SZ"),
                          "source_table_analysis": _wrapped("t1#2")}]}
    assert br.resolve_source_table_analysis(body, stage1) == []
    assert body["analyses"][0]["source_table_analysis"]["value"] == "t1#2"


def test_an_invented_key_is_dropped(tmp_path):
    # A key that resolves to nothing is worse than none: it looks like a working join.
    stage1 = _stage1(tmp_path, [{"table_id": "t1", "name": "SZ > HC"}])
    body = {"analyses": [{"local_id": "a1", "tables": ["t1"],
                          "name": _wrapped("something else"),
                          "source_table_analysis": _wrapped("t9#7")}]}
    notes = br.resolve_source_table_analysis(body, stage1)
    assert "source_table_analysis" not in body["analyses"][0]
    assert notes and "names no parsed row group" in notes[0]


def test_a_missing_key_is_filled_from_a_unique_name_match(tmp_path):
    stage1 = _stage1(tmp_path, [{"table_id": "t1", "name": "SZ > HC"},
                                {"table_id": "t1", "name": "HC > SZ"}])
    body = {"analyses": [{"local_id": "a1", "tables": ["t1"],
                          "name": _wrapped("HC>SZ")}]}
    notes = br.resolve_source_table_analysis(body, stage1)
    assert body["analyses"][0]["source_table_analysis"]["value"] == "t1#2"
    # Filled, not read off the page, so the provenance says so.
    assert body["analyses"][0]["source_table_analysis"]["value_source"] == "generated"
    assert notes


def test_an_ambiguous_name_leaves_the_analysis_honestly_unjoinable(tmp_path):
    # Two row groups with the same name: the join is a guess, and the slot stays empty.
    stage1 = _stage1(tmp_path, [{"table_id": "t1", "name": "SZ > HC"},
                                {"table_id": "t2", "name": "SZ > HC"}])
    body = {"analyses": [{"local_id": "a1", "tables": ["t1", "t2"],
                          "name": _wrapped("SZ > HC")}]}
    assert br.resolve_source_table_analysis(body, stage1) == []
    assert "source_table_analysis" not in body["analyses"][0]


def test_the_key_is_scoped_to_the_tables_the_analysis_cites(tmp_path):
    stage1 = _stage1(tmp_path, [{"table_id": "t1", "name": "SZ > HC"},
                                {"table_id": "t2", "name": "SZ > HC"}])
    body = {"analyses": [{"local_id": "a1", "tables": ["t2"],
                          "name": _wrapped("SZ > HC")}]}
    br.resolve_source_table_analysis(body, stage1)
    assert body["analyses"][0]["source_table_analysis"]["value"] == "t2#1"


def test_no_parse_means_nothing_is_invented(tmp_path):
    body = {"analyses": [{"local_id": "a1", "name": _wrapped("x")}]}
    assert br.resolve_source_table_analysis(body, tmp_path / "missing.json") == []
    assert "source_table_analysis" not in body["analyses"][0]


# --- derived analysis ids -----------------------------------------------------

def test_an_analysis_id_is_derived_from_its_parse_key():
    # A model-chosen id is unstable: over the same 16 papers extracted twice, only four
    # produced identical analysis ids.
    body = {"analyses": [{"local_id": "a_independent_component_spatial_maps",
                          "source_table_analysis": _wrapped("t0035#2")}]}
    notes = br.derive_analysis_ids(body)
    assert body["analyses"][0]["local_id"] == "a_t0035_2"
    assert notes and "t0035#2" in notes[0]


def test_deriving_ids_is_idempotent():
    body = {"analyses": [{"local_id": "a_x", "source_table_analysis": _wrapped("t1#1")}]}
    br.derive_analysis_ids(body)
    first = body["analyses"][0]["local_id"]
    assert br.derive_analysis_ids(body) == []
    assert body["analyses"][0]["local_id"] == first


def test_analyses_sharing_a_key_are_numbered_apart():
    # A SPLIT emits several analyses against one listing entry, so they share a key.
    body = {"analyses": [
        {"local_id": "a1", "source_table_analysis": _wrapped("t1#1")},
        {"local_id": "a2", "source_table_analysis": _wrapped("t1#1")}]}
    br.derive_analysis_ids(body)
    assert [a["local_id"] for a in body["analyses"]] == ["a_t1_1", "a_t1_1_2"]


def test_an_analysis_with_no_key_keeps_the_models_id():
    # 25% of analyses cannot be tied to a row group; inventing a stable-looking id for
    # them would claim the parse identifies something it does not.
    body = {"analyses": [{"local_id": "a_hand_named", "name": _wrapped("x")}]}
    assert br.derive_analysis_ids(body) == []
    assert body["analyses"][0]["local_id"] == "a_hand_named"


def test_mirror_of_is_repointed_to_the_new_id():
    # `mirror_of` is the only pointer at an analysis anywhere in the record.
    body = {"analyses": [
        {"local_id": "a_described", "source_table_analysis": _wrapped("t1#1")},
        {"local_id": "a_described-reversed", "mirror_of": "a_described",
         "source_table_analysis": _wrapped("t1#2")}]}
    br.derive_analysis_ids(body)
    assert body["analyses"][0]["local_id"] == "a_t1_1"
    assert body["analyses"][1]["mirror_of"] == "a_t1_1"


def test_a_derived_id_already_taken_leaves_both_alone():
    # Collapsing two analyses into one id is worse than an unstable id.
    body = {"analyses": [
        {"local_id": "a_t1_1", "name": _wrapped("already answers to it")},
        {"local_id": "a_other", "source_table_analysis": _wrapped("t1#1")}]}
    notes = br.derive_analysis_ids(body)
    assert body["analyses"][1]["local_id"] == "a_other"
    assert notes and "already taken" in notes[0]
