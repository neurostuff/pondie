"""Stages are functions, so they are called directly and the model is substituted."""

import json

import pytest

from pondie.extraction import plan, run
from pondie.extraction.models import (
    Cost,
    EvidenceCounts,
    Flavour,
    ModelReply,
    Paper,
    Settings,
    StageName,
    StageOutcome,
)
from pondie.extraction.prompt import render


class Recorder:
    """A Caller that records what it was asked and returns a fixed payload."""

    def __init__(self):
        self.calls = []

    def __call__(self, call, *, paper, stage):
        self.calls.append((paper, stage))
        return ModelReply(payload={"stage": stage}, cost=Cost(input_tokens=10, calls=1))


def _paper(tmp_path, manifest=({"table_id": "t1", "table_number": 1, "caption": "Peaks"},)):
    """A staged paper: a text, the stage-1 parse, and the table manifest beside the text.

    `table-map.json` is deliberately not written. It is an *output* of the Tables stage --
    the map `Analysis.tables` references resolve through -- and a fixture that supplies it
    hides the stage failing to write one.
    """
    root = tmp_path / "texts"
    (root / "S1" / "processed" / "pubget").mkdir(parents=True)
    (root / "S1" / "stage1").mkdir(parents=True)
    (root / "S1" / "processed" / "pubget" / "text.txt").write_text("a paper")
    (root / "S1" / "stage1" / "analyses.json").write_text('{"analyses": []}')
    if manifest is not None:
        (root / "S1" / "processed" / "pubget" / "tables.jsonl").write_text(
            "\n".join(json.dumps(row) for row in manifest)
        )
    return Paper(study_id="S1", root=root)


def _settings(tmp_path, **kw):
    return Settings(
        payloads=tmp_path / "payloads", records=tmp_path / "records", model="test-model", **kw
    )


def test_tables_takes_no_model_call(tmp_path):
    caller = Recorder()
    paper = _paper(tmp_path)
    report = run([paper], _settings(tmp_path, stages=(StageName.tables,)), caller)
    assert report.failures == ()
    assert caller.calls == [], "the manifest is copied, never retyped through a model"
    produced = report.papers[0].outcomes[0].produced[0]
    written = json.loads(produced.read_text())["tables"]
    assert [entry["local_id"] for entry in written] == ["t1"]
    # Wrapped, because the slot is an ExtractedValue. A bare string here is a repair the
    # builder then has to make, on a value that was never uncertain.
    assert written[0]["caption"]["value"] == "Peaks"
    assert written[0]["table_number"]["value"] == "Table 1"


def test_tables_writes_the_id_map_analysis_references_resolve_through(tmp_path):
    """Omitting this is the regression that motivated the stage.

    155 of 156 records ended up with no tables declared while 1,076 of 1,084 analyses
    referenced one. Direction scoring never noticed -- polarity reads the parse, not the
    Table entity -- so it stayed invisible until a coordinate query asked for the join.
    """
    paper = _paper(tmp_path)
    run([paper], _settings(tmp_path, stages=(StageName.tables,)), Recorder())
    assert json.loads(paper.table_map.read_text()) == {"t1": "t1"}


def test_a_table_with_no_caption_is_not_reported_rather_than_empty(tmp_path):
    """An absent caption is a claim about the manifest, not a caption that is blank."""
    paper = _paper(tmp_path, manifest=({"table_id": "t1", "table_number": 1},))
    report = run([paper], _settings(tmp_path, stages=(StageName.tables,)), Recorder())
    written = json.loads(report.papers[0].outcomes[0].produced[0].read_text())["tables"]
    assert written[0]["caption"]["extraction_status"] == "not_reported"
    assert "value" not in written[0]["caption"]


def test_table_number_is_not_used_as_the_identifier(tmp_path):
    """One paper in the corpus carries two tables numbered 1; keying on the number
    collapses them into a single record."""
    paper = _paper(
        tmp_path,
        manifest=(
            {"table_id": "a", "table_number": 1},
            {"table_id": "b", "table_number": 1},
        ),
    )
    report = run([paper], _settings(tmp_path, stages=(StageName.tables,)), Recorder())
    written = json.loads(report.papers[0].outcomes[0].produced[0].read_text())["tables"]
    assert [entry["local_id"] for entry in written] == ["a", "b"]


def test_a_paper_with_no_manifest_is_not_a_failure(tmp_path):
    """Most of the corpus is ace, which ships no table manifest at all."""
    paper = _paper(tmp_path, manifest=None)
    report = run([paper], _settings(tmp_path, stages=(StageName.tables,)), Recorder())
    outcome = report.papers[0].outcomes[0]
    assert report.failures == ()
    assert "no tables.jsonl" in outcome.notes[0]


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
    settings = _settings(tmp_path, stages=(StageName.evidence,), union=False)
    payload = settings.payloads / "S1" / "satisfy.json"
    payload.parent.mkdir(parents=True)
    payload.write_text(
        json.dumps(
            {"groups": [{"name": {"extraction_status": "extracted", "value": "patients"}}]}
        )
    )

    caller = Recorder()
    monkeypatch.setattr(
        "pondie.extraction.evidence.quote.apply_evidence",
        lambda payload, quotes, reranker=None, units=(): EvidenceCounts(),
    )
    outcome = Evidence().run(paper, settings, caller)
    assert not outcome.skipped and outcome.ok
    assert [stage for _p, stage in caller.calls] == ["evidence"]


def test_evidence_writes_its_blocks_into_the_payloads(tmp_path):
    """`evidence` is REQUIRED on every ExtractedValue, so the blocks go on the fields.

    A stage that collected quotes into a file of its own left every field without a block,
    which the builder leaves untouched -- so the record failed validation at the end of the
    run instead of here, where the reason is still visible.
    """
    from pondie.extraction.stages import Evidence

    class Quoter:
        def __call__(self, call, *, paper, stage):
            return ModelReply(
                payload={"groups[0].name": "Twenty patients were recruited."}, cost=Cost()
            )

    paper = _paper(tmp_path)
    settings = _settings(tmp_path, stages=(StageName.evidence,), union=False)
    payload = settings.payloads / "S1" / "satisfy.json"
    payload.parent.mkdir(parents=True)
    payload.write_text(
        json.dumps(
            {"groups": [{"name": {"extraction_status": "extracted", "value": "patients"}}]}
        )
    )

    outcome = Evidence().run(paper, settings, Quoter())
    assert outcome.ok and not outcome.skipped
    written = json.loads(payload.read_text())
    assert written["groups"][0]["name"]["evidence"]["status"] == "present"
    # The pre-evidence copy lets evidence run again without rerunning
    # the pass that produced the values.
    backup = json.loads((settings.payloads / "S1" / "noev" / "satisfy.json").read_text())
    assert "evidence" not in backup["groups"][0]["name"]


def test_a_started_but_unfinished_evidence_stage_is_not_done(tmp_path):
    """`noev/` is made before any evidence is written, so it proves the stage began.

    Reading it as "done" cost seventeen papers their evidence: the stage died loading its
    reranker, the backup was already on disk, and every resume skipped it and built records
    with no evidence at all.
    """
    from pondie.extraction.stages import Evidence

    paper = _paper(tmp_path)
    settings = _settings(tmp_path, stages=(StageName.evidence,))
    (settings.payloads / "S1" / "noev").mkdir(parents=True)
    (settings.payloads / "S1" / "satisfy.json").write_text(
        json.dumps({"groups": [{"name": {"extraction_status": "extracted", "value": "x"}}]})
    )
    assert Evidence().done(paper, settings) is False


def test_evidence_is_skipped_rather_than_failed_when_turned_off(tmp_path):
    from pondie.extraction.stages import Evidence

    settings = _settings(tmp_path, stages=(StageName.evidence,), retrieve_evidence=False)
    outcome = Evidence().run(_paper(tmp_path), settings, Recorder())
    assert outcome.skipped and outcome.ok and "disabled" in outcome.reason


def test_split_withholds_the_reversed_half_of_a_two_signed_table(tmp_path):
    """The sign rule runs before any pass reads the parse, because it changes the ask.

    A parse entry reporting both signs is two contrasts and the paper describes one. If
    `split` ran after `demands`, the model would already have been shown the merged entry
    and the withheld half would have nothing to mirror.
    """
    paper = _paper(tmp_path)
    paper.parse.write_text(
        json.dumps(
            {
                "analyses": [
                    {
                        "name": "patients > controls",
                        "table_id": "t1",
                        "points": [
                            {"x": 1, "y": 2, "z": 3, "values": [{"kind": "t", "value": 4.0}]},
                            {
                                "x": -1,
                                "y": -2,
                                "z": -3,
                                "values": [{"kind": "t", "value": -4.0}],
                            },
                        ],
                    }
                ]
            }
        )
    )
    report = run([paper], _settings(tmp_path, stages=(StageName.sign_split,)), Recorder())

    assert report.failures == ()
    document = json.loads(paper.parse.read_text())
    assert document["sign_split_applied"] is True
    withheld = [a for a in document["analyses"] if a.get("withhold")]
    assert len(withheld) == 1, "one half is extracted, the other is rebuilt by arithmetic"
    assert withheld[0].get("mirror_of"), "the withheld half must say what it mirrors"


def test_split_leaves_an_already_partitioned_parse_alone(tmp_path):
    """Resuming a run must not re-partition parts that each already hold one sign."""
    paper = _paper(tmp_path)
    paper.parse.write_text(json.dumps({"analyses": [], "sign_split_applied": True}))
    settings = _settings(tmp_path, stages=(StageName.sign_split,))
    report = run([paper], settings, Recorder())
    assert report.papers[0].outcomes[0].skipped


def test_build_writes_a_record_that_can_be_validated(tmp_path, monkeypatch):
    """Merging the payloads is not building a record.

    The three things only the builder supplies are the three a validator needs: `local_id`,
    the `source_text_hash` every evidence offset is relative to, and the section index those
    offsets are resolved against. A Build that merged payloads and stopped produced a body
    no consumer could check, which is worse than a Build that failed.
    """
    paper = _paper(tmp_path)
    payloads = tmp_path / "payloads" / paper.study_id
    payloads.mkdir(parents=True)
    (payloads / "satisfy.json").write_text(
        json.dumps({"study": {"description": "a study"}, "groups": [{"local_id": "g1"}]})
    )
    settings = _settings(tmp_path, stages=(StageName.build,))

    report = run([paper], settings, Recorder())
    assert report.failures == (), report.papers[0].failed

    record = json.loads((settings.records / "S1.extraction.json").read_text())
    # Flat, the shape every consumer and every fixture already reads. The merge-only Build
    # wrapped the body in {"study": ...}, which no reader of a record expects.
    assert record["local_id"] == "S1"
    metadata = record["extraction_metadata"]
    assert metadata["source_text_hash"], "the hash every offset is relative to"
    assert metadata["extractor_model"] == "test-model"
    assert metadata["extractor_version"], "a record says which pipeline made it"
    assert "paper_sections" in metadata

    # The body itself also arrived. Checking only metadata passes on a Build that
    # merged nothing: the merge globs `<payloads>/*.json`, so a stage writing
    # `<stage>/payload.json` produces a record of metadata and no content.
    # Wrapped on the way through: a bare scalar in an ExtractedValue slot is one of the
    # repairs the builder applies, so seeing the wrapper proves both that the payload was
    # merged and that the repair sequence ran over it.
    assert record["description"]["value"] == "a study"
    assert [group["local_id"] for group in record["groups"]] == ["g1"]


# --- what the driver guarantees ------------------------------------------------


class _Fake:
    """A stage that records how often it ran, so skipping is observable."""

    def __init__(self, name, fails=False):
        self.name, self.fails, self.ran = name, fails, 0

    def produces(self, paper, settings):
        return settings.payloads / paper.study_id / f"{self.name.value}.json"

    def done(self, paper, settings):
        return self.produces(paper, settings).is_file() and not settings.redo

    def run(self, paper, settings, caller):
        if self.done(paper, settings):
            return StageOutcome(stage=self.name, study_id=paper.study_id, skipped=True)
        self.ran += 1
        if self.fails:
            raise RuntimeError("boom")
        target = self.produces(paper, settings)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
        return StageOutcome(
            stage=self.name, study_id=paper.study_id, cost=Cost(input_tokens=7, calls=1)
        )


def test_a_failed_stage_stops_the_paper_rather_than_running_the_next(tmp_path, monkeypatch):
    """`satisfy` reads what `demands` wrote.

    Running it after `demands` failed produces a second, more confusing failure that hides
    the first, against inputs that do not exist.
    """
    first = _Fake(StageName.tables)
    second = _Fake(StageName.demands, fails=True)
    third = _Fake(StageName.satisfy)
    monkeypatch.setattr("pondie.extraction.driver.sequence", lambda s: (first, second, third))

    report = run([_paper(tmp_path)], _settings(tmp_path), Recorder())
    assert report.failures, "a raising stage is a reported failure, not an exception"
    assert third.ran == 0


def test_redo_overrides_the_skip(tmp_path, monkeypatch):
    stage = _Fake(StageName.tables)
    monkeypatch.setattr("pondie.extraction.driver.sequence", lambda s: (stage,))
    paper = _paper(tmp_path)
    run([paper], _settings(tmp_path), Recorder())
    run([paper], _settings(tmp_path), Recorder())
    assert stage.ran == 1, "an artefact on disk is not rebuilt"
    run([paper], _settings(tmp_path, redo=True), Recorder())
    assert stage.ran == 2


def test_a_paper_that_cannot_run_is_reported_not_raised(tmp_path):
    (tmp_path / "S9" / "stage1").mkdir(parents=True)
    report = run([Paper(study_id="S9", root=tmp_path)], _settings(tmp_path), Recorder())
    assert report.failures and not report.papers[0].failed.ok


def test_the_best_available_flavour_wins(tmp_path):
    """A locator searching a table-free text cannot find the sentence a group size came
    from, so the preference is measured rather than cosmetic."""
    root = tmp_path / "texts"
    for flavour, name in (("ace", "text.txt"), ("local", "text.tables.txt")):
        (root / "S1" / "processed" / flavour).mkdir(parents=True)
        (root / "S1" / "processed" / flavour / name).write_text("a paper")
    assert Paper.best("S1", root).flavour is Flavour.local


def test_a_study_with_no_text_at_all_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no text"):
        Paper.best("S9", tmp_path)


def test_the_withheld_half_is_mirrored_back_into_the_record(tmp_path):
    """`split` withholds a half; `build` rebuilds it. Neither is useful without the other.

    This fired zero times under the previous runner and the reason was not the rule: the
    runner never re-ran stage 1, so the parses on disk had been partitioned before the
    withhold flag existed and carried no withheld entries at all. `split` runs inside the
    pipeline now, which is what makes the mirror reachable -- so the two are tested as one
    chain rather than separately.
    """
    paper = _paper(tmp_path)
    paper.parse.write_text(
        json.dumps(
            {
                "analyses": [
                    {
                        "name": "FESZ > NC",
                        "table_id": "t1",
                        "points": [
                            {"x": 1, "y": 2, "z": 3, "values": [{"kind": "t", "value": 4.0}]},
                            {
                                "x": -1,
                                "y": -2,
                                "z": -3,
                                "values": [{"kind": "t", "value": -4.0}],
                            },
                        ],
                    }
                ]
            }
        )
    )
    run([paper], _settings(tmp_path, stages=(StageName.sign_split,)), Recorder())

    # What `satisfy` would have produced for the half the paper actually describes.
    payloads = tmp_path / "payloads" / paper.study_id
    payloads.mkdir(parents=True)
    (payloads / "satisfy.json").write_text(
        json.dumps(
            {
                "analyses": [
                    {
                        "local_id": "a1",
                        "name": {
                            "extraction_status": "extracted",
                            "value": "FESZ > NC",
                            "evidence": {"status": "not_applicable"},
                        },
                        "effect": {
                            "cells": [
                                {
                                    "term": "group",
                                    "level": "FESZ",
                                    "direction": {
                                        "extraction_status": "extracted",
                                        "value": "positive",
                                        "evidence": {"status": "not_applicable"},
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        )
    )
    settings = _settings(tmp_path, stages=(StageName.build,))
    report = run([paper], settings, Recorder())
    assert report.failures == (), report.papers[0].failed

    record = json.loads((settings.records / "S1.extraction.json").read_text())
    mirrored = [a for a in record["analyses"] if a.get("mirror_of")]
    assert len(mirrored) == 1, "the withheld half is one analysis, rebuilt by arithmetic"

    reversed_half = mirrored[0]
    described = next(a for a in record["analyses"] if not a.get("mirror_of"))
    # The described half's id as the *builder* left it, not as the payload wrote it: the
    # mirror runs last, after `derived_ids` has renamed it, because it is taken from the
    # contrast the model settled on including everything the repairs changed about it.
    assert reversed_half["mirror_of"] == described["local_id"]
    # The direction is flipped, and marked `generated` -- no sentence of the paper warrants
    # a reversal the paper never describes.
    direction = reversed_half["effect"]["cells"][0]["direction"]
    assert direction["value"] == "negative"
    assert direction["value_source"] == "generated"
    # Its own name, not the described half's. A mirror called "FESZ > NC" whose cells say
    # the opposite collides with the real one and makes a correct extraction score as a
    # sign flip.
    assert reversed_half["name"]["value"] == "FESZ > NC (reversed)"
    # It addresses its own row group instead of carrying flipped points inline.
    assert "points" not in reversed_half
    assert reversed_half["source_table_analysis"]["value"]

    # The build says it happened, rather than leaving it to be discovered in the record.
    notes = " ".join(report.papers[0].outcomes[0].notes)
    assert "mirrored" in notes


def test_the_paper_reaches_the_model(tmp_path):
    """The one thing a model pass must not get wrong, and the one nothing checked.

    `build_prompt` returns two halves and `stages` unpacked them as `(prompt, schema_name)`,
    so the half carrying the paper went into a `ModelCall` field the caller ignores. Every
    stage reported success, every test passed -- a fake `Caller` records what it was asked
    and does not read it -- and the model was sent instructions about a paper it had never
    been shown. `Prompt` is a named pair now, so the two cannot be swapped; this asserts the
    consequence rather than the shape.
    """
    seen = []

    class Reader:
        def __call__(self, call, *, paper, stage):
            seen.append(call)
            return ModelReply(payload={}, cost=Cost())

    root = tmp_path / "texts"
    (root / "S1" / "processed" / "pubget").mkdir(parents=True)
    (root / "S1" / "stage1").mkdir(parents=True)
    (root / "S1" / "processed" / "pubget" / "text.txt").write_text(
        "Methods\n\nWe scanned SEVENTEEN-UNLIKELY-SENTINEL volunteers.\n"
    )
    (root / "S1" / "stage1" / "analyses.json").write_text('{"analyses": []}')
    paper = Paper(study_id="S1", root=root)

    run([paper], _settings(tmp_path, stages=(StageName.demands,)), Reader())

    assert seen, "the demands pass makes a call"
    call = seen[0]
    assert "SEVENTEEN-UNLIKELY-SENTINEL" in call.prompt, "the paper is in the user turn"
    assert call.system, "the instructions are sent as their own message"
    assert (
        "SEVENTEEN-UNLIKELY-SENTINEL" not in call.system
    ), "the system half must not vary per paper, or no prompt cache can ever hit it"


def test_evidence_already_in_the_payloads_does_not_make_the_stage_look_done(tmp_path):
    """No payload can answer "did this stage run", so none is allowed to.

    `retrieve_evidence` also selects the prompt rule that has the extraction passes emit
    `evidence` inline, so on default settings every extracted field already carries a block
    in exactly the shape this pass writes. Add the `noev/` backup -- made *before* the work
    -- and a resume after a crash finds both and skips. That is the seventeen-paper bug.
    """
    from pondie.extraction.stages import Evidence

    paper = _paper(tmp_path)
    settings = _settings(tmp_path, stages=(StageName.evidence,))
    payloads = settings.payloads / paper.study_id
    (payloads / "noev").mkdir(parents=True)
    (payloads / "tables.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "local_id": "t1",
                        "caption": {
                            "extraction_status": "extracted",
                            "value": "Peaks",
                            "evidence": {"status": "not_applicable"},
                        },
                    }
                ]
            }
        )
    )
    # Exactly what a model writes when asked for inline evidence.
    (payloads / "satisfy.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "name": {
                            "extraction_status": "extracted",
                            "value": "patients",
                            "evidence": {
                                "status": "present",
                                "sets": [{"quotes": ["twenty patients"]}],
                            },
                        }
                    }
                ]
            }
        )
    )
    assert Evidence().done(paper, settings) is False


def test_the_evidence_pass_does_not_re_ask_what_the_table_manifest_already_answered(tmp_path):
    """A caption is warranted by construction; asking a model to quote it can only lose.

    The field is `not_applicable` -- there is no sentence to quote -- and a quote pass that
    finds none writes `not_found`, which the record reports as a defect for a reviewer. Thus,
    every table manufactured about three spurious defects per paper, and paid output tokens
    to do it.
    """
    from pondie.extraction.stages import Evidence

    paper = _paper(tmp_path)
    settings = _settings(tmp_path, stages=(StageName.evidence,), union=False)
    payloads = settings.payloads / paper.study_id
    payloads.mkdir(parents=True)
    tables = payloads / "tables.json"
    tables.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "local_id": "t1",
                        "caption": {
                            "extraction_status": "extracted",
                            "value": "Peaks",
                            "evidence": {"status": "not_applicable"},
                        },
                    }
                ]
            }
        )
    )
    (payloads / "satisfy.json").write_text(
        json.dumps({"groups": [{"name": {"extraction_status": "extracted", "value": "x"}}]})
    )
    before = tables.read_text()

    Evidence().run(paper, settings, Recorder())
    assert tables.read_text() == before, "the deterministic payload is left exactly as it was"


def test_the_evidence_pass_sends_its_instructions_on_the_cacheable_half(tmp_path):
    """45% of the pipeline's input tokens go through this pass.

    Concatenating the instructions into the user turn leaves the `system` half empty, so the
    one part of the prompt that is identical across every paper -- the only part a whole-
    prompt cache can ever hit -- is not there to hit.
    """
    from pondie.extraction.stages import Evidence

    seen = []

    class Reader:
        def __call__(self, call, *, paper, stage):
            seen.append(call)
            return ModelReply(payload={}, cost=Cost())

    paper = _paper(tmp_path)
    settings = _settings(tmp_path, stages=(StageName.evidence,), union=False)
    payloads = settings.payloads / paper.study_id
    payloads.mkdir(parents=True)
    (payloads / "satisfy.json").write_text(
        json.dumps({"groups": [{"name": {"extraction_status": "extracted", "value": "x"}}]})
    )

    Evidence().run(paper, settings, Reader())
    assert seen, "the pass makes a call"
    assert seen[0].system, "the instructions are on the system half"
    assert "# Paper" in seen[0].prompt and "# Paper" not in seen[0].system


def test_the_demands_pass_is_told_the_parse_key_and_the_zero_foci_rule(tmp_path):
    """The stage-1 parse is rendered as instructions, not handed over as JSON.

    `source_table_analysis` is the only exact join from an analysis back to its coordinates,
    and the model can only return a key it was shown. The zero-foci rule -- an entry with no
    coordinates is a tested effect that found nothing -- is worth +16 points paired with
    this ordering. Dumping `analyses.json` into the prompt sends neither.
    """
    seen = []

    class Reader:
        def __call__(self, call, *, paper, stage):
            seen.append(call)
            return ModelReply(payload={}, cost=Cost())

    paper = _paper(tmp_path)
    paper.parse.write_text(
        json.dumps(
            {
                "analyses": [
                    {"name": "A > B", "table_id": "t1", "points": []},
                    {"name": "C > D", "table_id": "t1", "points": [{"x": 1, "y": 2, "z": 3}]},
                ]
            }
        )
    )
    paper.table_map.write_text(json.dumps({"t1": "t1"}))

    run([paper], _settings(tmp_path, stages=(StageName.demands,)), Reader())
    assert seen, "the demands pass makes a call"
    prompt = seen[0].prompt

    assert "parse key" in prompt, "the key the analysis must carry back"
    assert "t1#1" in prompt or "t1#2" in prompt, "the keys themselves, not just the word"
    # The rule's own text, not "0 foci": `stage1_block` prints the foci count for every
    # entry whether the rule is on or not, so asserting on that passes with the rule off.
    assert render.ZERO_FOCI_RULE.strip() in prompt, "the +16-point rule itself"


def test_the_zero_foci_rule_can_be_turned_off(tmp_path):
    """Measured -25 unpaired and +16 paired with demand-driven ordering, so it is a setting
    rather than a constant -- and a setting nothing can observe is not one."""
    seen = []

    class Reader:
        def __call__(self, call, *, paper, stage):
            seen.append(call)
            return ModelReply(payload={}, cost=Cost())

    paper = _paper(tmp_path)
    paper.parse.write_text(
        json.dumps({"analyses": [{"name": "A > B", "table_id": "t1", "points": []}]})
    )
    settings = _settings(tmp_path, stages=(StageName.demands,), zero_foci_rule=False)
    run([paper], settings, Reader())
    assert render.ZERO_FOCI_RULE.strip() not in seen[0].prompt


def test_the_satisfy_pass_is_given_the_shopping_list_as_a_contract(tmp_path):
    """Asking the analyses first only pays if the entity pass is held to what they asked.

    Handed the demands payload as raw JSON, the pass is free to mint its own local_ids and
    every cell pointing at a declared one dangles.
    """
    seen = []

    class Reader:
        def __call__(self, call, *, paper, stage):
            seen.append(call)
            return ModelReply(payload={}, cost=Cost())

    paper = _paper(tmp_path)
    payloads = tmp_path / "payloads" / paper.study_id
    payloads.mkdir(parents=True)
    (payloads / "demands.json").write_text(
        json.dumps(
            {
                "required_entities": [
                    {"local_id": "grp_patients", "kind": "Group", "label": "the patient group"}
                ]
            }
        )
    )
    settings = _settings(tmp_path, stages=(StageName.satisfy,))
    run([paper], settings, Reader())

    prompt = seen[0].prompt
    assert "shopping list" in prompt
    assert "grp_patients" in prompt, "the id the pass must reuse verbatim"
    assert "EXACTLY the local_id given" in prompt


def test_the_build_validates_and_reports_rather_than_failing(tmp_path):
    """A run printed "0 failed" for records the validator would reject.

    `Validator` was constructed only by its own CLI and by tests, so dangling references,
    directions outside their vocabulary and spans that do not verify were all invisible to
    a run. They are notes, not a `reason`: the record is written either way, and treating a
    defect as a failure is what made five of sixteen papers read as lost when all sixteen
    had been built and scored.
    """
    paper = _paper(tmp_path)
    payloads = tmp_path / "payloads" / paper.study_id
    payloads.mkdir(parents=True)
    # A cell pointing at a term nothing declares: a defect a reviewer has to settle.
    (payloads / "satisfy.json").write_text(
        json.dumps(
            {
                "analyses": [
                    {
                        "local_id": "a1",
                        "model_estimation": "m_absent",
                        "effect": {"cells": [{"term": "t_absent", "direction": "positive"}]},
                    }
                ]
            }
        )
    )
    settings = _settings(tmp_path, stages=(StageName.build,))
    report = run([paper], settings, Recorder())

    assert report.failures == (), "a defective record is still written and still reported"
    notes = " ".join(report.papers[0].outcomes[0].notes)
    assert "validation" in notes or "cross-reference" in notes, notes
    assert (settings.records / "S1.extraction.json").is_file()


def test_a_malformed_reply_is_retried_rather_than_killing_the_paper(tmp_path):
    """The one fault the retry loop existed for was the one it could not see.

    `_as_json` ran in `GatewayCaller.__call__`'s `return` statement, outside the try, so a
    `JSONDecodeError` escaped the network loop, escaped this post-condition loop, and
    escaped the accounting -- a paper that spent 40,000 tokens logged `calls: 0`. Four of
    five papers in a real run died on a single `}` the model wrote where a `]` belonged.
    """
    from pondie.extraction.llm import MalformedReply
    from pondie.extraction.stages import Demands

    class Stutterer:
        """Malformed once, then fine -- the stochastic failure the loop is built for."""

        def __init__(self):
            self.calls = 0

        def __call__(self, call, *, paper, stage):
            self.calls += 1
            if self.calls == 1:
                raise MalformedReply(
                    "reply was not valid JSON: Expecting ',' delimiter",
                    body='{"groups": [{"name": "x"}}}',
                    cost=Cost(input_tokens=40_000, output_tokens=3_000, calls=1),
                )
            return ModelReply(
                payload={
                    # `effect` is what makes an Analysis one: `normalize` files an
                    # effect-less entry under required_entities instead.
                    "analyses": [
                        {
                            "local_id": "a1",
                            "name": "Patients > controls",
                            "effect": {"kind": "contrast"},
                        }
                    ],
                    "required_entities": [{"local_id": "g1", "kind": "Group"}],
                },
                cost=Cost(input_tokens=10, calls=1),
            )

    caller = Stutterer()
    outcome = Demands().run(_paper(tmp_path), _settings(tmp_path), caller)

    assert caller.calls == 2, "the malformed reply was re-asked, not surfaced"
    assert outcome.ok
    # The tokens the rejected call burned are the paper's, whatever its body said.
    assert outcome.cost.input_tokens == 40_010
    assert outcome.cost.calls == 2
    assert any("malformed" in note for note in outcome.notes), outcome.notes


def test_a_reply_that_never_parses_fails_instead_of_writing_an_empty_payload(tmp_path):
    """`{}` is well formed, legally empty, and builds a record about no study at all.

    That is the silent failure the post-condition was added to stop, so exhausting the
    attempts on unparseable JSON has to raise rather than fall through to `_write`.
    """
    from pondie.extraction.llm import MalformedReply
    from pondie.extraction.stages import Demands

    class Garbage:
        def __init__(self):
            self.calls = 0

        def __call__(self, call, *, paper, stage):
            self.calls += 1
            raise MalformedReply("not JSON", body="{oh no", cost=Cost(calls=1))

    caller = Garbage()
    settings = _settings(tmp_path, attempts=3)
    with pytest.raises(MalformedReply):
        Demands().run(_paper(tmp_path), settings, caller)

    assert caller.calls == 3, "every attempt was spent before giving up"
    assert not (settings.payloads / "S1" / "demands.json").exists(), "no empty payload"


def test_the_caller_treats_an_unparseable_body_as_a_failed_attempt(tmp_path):
    """Parsing belongs inside the loop, so `attempts` covers a bad body as well as a 500."""
    from pondie.extraction.llm import GatewayCaller, MalformedReply
    from pondie.extraction.models import ModelCall

    class Reply:
        def __init__(self, body):
            self.choices = [
                type(
                    "C",
                    (),
                    {"message": type("M", (), {"content": body})(), "finish_reason": "stop"},
                )()
            ]
            self.usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 22})()

    class Raw:
        headers: dict = {}

        def __init__(self, body):
            self._body = body

        def parse(self):
            return Reply(self._body)

    class Client:
        def __init__(self, bodies):
            self.bodies = list(bodies)
            self.chat = type(
                "Chat",
                (),
                {"completions": type("Cmp", (), {"with_raw_response": self})()},
            )()

        def create(self, **_):
            return Raw(self.bodies.pop(0))

    caller = GatewayCaller()
    client = Client(['{"broken": ', '{"ok": 1}'])
    caller._client = lambda paper, stage: client

    call = ModelCall(model="m", system="s", prompt="p", max_output_tokens=10, attempts=2)
    reply = caller(call, paper="S1", stage="demands")
    assert reply.payload == {"ok": 1}, "the second attempt's good body is returned"

    caller._client = lambda paper, stage: Client(['{"broken": '])
    with pytest.raises(MalformedReply) as raised:
        caller(
            ModelCall(model="m", system="s", prompt="p", max_output_tokens=10, attempts=1),
            paper="S1",
            stage="demands",
        )
    # The cost rides on the exception, which is how the stage bills a rejected reply.
    assert raised.value.cost.input_tokens == 11 and raised.value.cost.calls == 1
