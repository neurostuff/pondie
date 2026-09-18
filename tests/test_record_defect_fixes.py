"""The five defect classes measured in docs/record-defects.md, and their fixes.

Every case is a shape observed in the 1,817-record corpus, with the count it occurred at in
the docstring of the thing it tests. The counts are what make these regression tests rather
than examples: if a fix stops firing, the corpus number is the thing to re-measure.
"""

from __future__ import annotations

import pytest

from pondie import schema
from pondie.extraction.record import builder, fix, rules
from pondie.schema import reader


class Sink:
    def __init__(self):
        self.errors, self.warnings = [], []

    def error(self, path, message):
        self.errors.append((path, message))

    def warn(self, path, message):
        self.warnings.append((path, message))


def field(value, source="reported", evidence="present"):
    return {
        "value": value,
        "extraction_status": "extracted",
        "value_source": source,
        "evidence": {"status": evidence},
    }


# --- 1. check_table_purpose errored on derive_table_effects' own answer ---------------


def test_a_cited_table_marked_reported_effect_is_not_a_contradiction():
    """799 errors over 461 papers, a quarter of every error in the corpus.

    `derive_table_effects` writes `reported_effect` on every cited table by design, and the
    check read any truthy purpose as a contradiction -- so the message refuted itself:
    "says this table reports 'reported_effect' rather than an effect".
    """
    record = {
        "analyses": [{"local_id": "a1", "tables": ["tbl1"]}],
        "tables": [{"local_id": "tbl1", "purpose": field("reported_effect", "generated")}],
    }
    sink = Sink()
    rules.check_table_purpose(record, sink)
    assert sink.errors == []


def test_a_cited_table_marked_something_else_is_still_a_contradiction():
    """The check's actual purpose, which the exemption must not remove."""
    record = {
        "analyses": [{"local_id": "a1", "tables": ["tbl1"]}],
        "tables": [{"local_id": "tbl1", "purpose": field("roi_definition", "reported")}],
    }
    sink = Sink()
    rules.check_table_purpose(record, sink)
    assert len(sink.errors) == 1
    assert "roi_definition" in sink.errors[0][1]


def test_an_uncited_unmarked_table_still_warns():
    record = {"analyses": [{"local_id": "a1"}], "tables": [{"local_id": "tbl1"}]}
    sink = Sink()
    rules.check_table_purpose(record, sink)
    assert len(sink.warnings) == 1


# --- 2. a level naming an entity the record declares ----------------------------------


def _level_record(level_name, **entities):
    record = {
        "model_estimations": [
            {
                "local_id": "mod1",
                "terms": [
                    {
                        "local_id": "trm1",
                        "name": field("condition"),
                        "type": field("categorical"),
                        "levels": [{"level": field(level_name)}],
                    }
                ],
            }
        ],
    }
    record.update(entities)
    return record


def test_a_level_links_to_the_condition_it_names(extraction_schema):
    """715 of 1,713 unjoined levels fold to an entity the same record declares."""
    record = _level_record(
        "smoking cue",
        tasks=[{"local_id": "tsk1", "conditions": [
            {"local_id": "cond_smoking_cue", "name": field("Smoking Cue")}]}],
    )
    changed = fix.link_entities_by_name(record, extraction_schema)
    level = record["model_estimations"][0]["terms"][0]["levels"][0]
    assert level["conditions"] == ["cond_smoking_cue"]
    assert changed and "cond_smoking_cue" in changed[0]


def test_a_level_links_to_the_timepoint_it_names(extraction_schema):
    record = _level_record(
        "predose",
        design={"timepoints": [{"local_id": "tp_predose", "name": field("predose")}]},
    )
    fix.link_entities_by_name(record, extraction_schema)
    assert record["model_estimations"][0]["terms"][0]["levels"][0]["timepoints"] == ["tp_predose"]


def test_a_name_matching_both_an_arm_and_a_cohort_writes_both(extraction_schema):
    """112 of the 122 two-kind matches. The cohort was allocated to the arm; `Group.arm`
    exists to say so, and writing one and not the other would lose half the fact."""
    record = _level_record(
        "Exercise",
        groups=[{"local_id": "grp_exercise", "name": field("Exercise")}],
        design={"arms": [{"local_id": "arm_exercise", "name": field("Exercise")}]},
    )
    fix.link_entities_by_name(record, extraction_schema)
    level = record["model_estimations"][0]["terms"][0]["levels"][0]
    assert level["groups"] == ["grp_exercise"]
    assert level["arms"] == ["arm_exercise"]


def test_two_candidates_of_one_kind_are_left_alone(extraction_schema):
    record = _level_record(
        "controls",
        groups=[{"local_id": "grp_a", "name": field("controls")},
                {"local_id": "grp_b", "name": field("Controls")}],
    )
    assert fix.link_entities_by_name(record, extraction_schema) == []
    assert "groups" not in record["model_estimations"][0]["terms"][0]["levels"][0]


def test_a_relation_slot_is_never_filled_from_a_name(extraction_schema):
    """The guard the whole design rests on.

    `interaction_with` means *crossed with*. Matched on a name it proposes 708 links of
    which 97% are a term named `group` in one model matching a term named `group` in
    another -- fabricating interactions `check_crossings` then reports as unrecorded.
    """
    record = {
        "model_estimations": [
            {"local_id": "mod1", "terms": [{"local_id": "trm_a", "name": field("group")}]},
            {"local_id": "mod2", "terms": [{"local_id": "trm_b", "name": field("group")}]},
        ]
    }
    assert fix.link_entities_by_name(record, extraction_schema) == []
    assert "interaction_with" not in record["model_estimations"][0]["terms"][0]


def test_an_entity_is_never_linked_to_itself(extraction_schema):
    """9,151 self-links without this: a name trivially matches its own owner."""
    record = {"analyses": [{"local_id": "a1", "name": field("A > B")}]}
    assert fix.link_entities_by_name(record, extraction_schema) == []
    assert "mirror_of" not in record["analyses"][0]


def test_the_written_shape_is_the_one_multivalued_declares(extraction_schema):
    """A reference is a bare id or a bare list of them, never an ExtractedValue.

    The first prototype wrapped them, which would have written 838 malformed fields.
    """
    record = _level_record(
        "patients",
        groups=[{"local_id": "grp_p", "name": field("patients")}],
        design={"arms": [{"local_id": "arm_p", "name": field("patients")}]},
    )
    fix.link_entities_by_name(record, extraction_schema)
    level = record["model_estimations"][0]["terms"][0]["levels"][0]
    assert level["groups"] == ["grp_p"]          # multivalued -> bare list
    assert record["groups"][0]["arm"] == "arm_p"  # scalar -> bare string


def test_a_slot_that_already_holds_a_reference_is_not_touched(extraction_schema):
    record = _level_record(
        "patients", groups=[{"local_id": "grp_p", "name": field("patients")}]
    )
    record["model_estimations"][0]["terms"][0]["levels"][0]["groups"] = ["grp_other"]
    fix.link_entities_by_name(record, extraction_schema)
    assert record["model_estimations"][0]["terms"][0]["levels"][0]["groups"] == ["grp_other"]


# --- 3. a scalar enum slot holding a one-item list ------------------------------------


def test_a_one_item_list_in_a_scalar_wrapper_is_unwrapped(extraction_schema):
    """21,701 fields. `values.read` returns the list, so a filter on `spatial_scope`
    matches nothing on 4,470 analyses."""
    record = {"analyses": [{"local_id": "a1", "spatial_scope": field(["whole_brain"])}]}
    changed = fix.unwrap_singleton_lists(record, extraction_schema)
    assert record["analyses"][0]["spatial_scope"]["value"] == "whole_brain"
    assert changed


def test_a_longer_list_in_a_scalar_wrapper_is_left_for_the_check(extraction_schema):
    """730 of them, and `spatial_scope: ['whole_brain', 'roi']` 31 times -- the two exclude
    each other, so picking one would be deciding which."""
    record = {"analyses": [{"local_id": "a1", "spatial_scope": field(["whole_brain", "roi"])}]}
    assert fix.unwrap_singleton_lists(record, extraction_schema) == []
    assert record["analyses"][0]["spatial_scope"]["value"] == ["whole_brain", "roi"]


def test_a_multivalued_wrapper_keeps_its_list(extraction_schema):
    record = {"groups": [{"local_id": "g1", "medical_condition": field(["obesity"])}]}
    assert fix.unwrap_singleton_lists(record, extraction_schema) == []
    assert record["groups"][0]["medical_condition"]["value"] == ["obesity"]


@pytest.mark.parametrize("value,expect", [(["whole_brain"], "1-item list"), ("whole_brain", None)])
def test_the_validator_reports_a_list_in_a_scalar_wrapper(value, expect):
    """22,431 violations passed validation in silence, which is why this went unseen.

    The membership loop iterated `[value]` for a scalar slot and skipped anything that was
    not a string, so a one-item list was neither unwrapped nor reported.
    """
    from pondie.extraction.record.validate import Validator

    validator = Validator(reader.load(schema.EXTRACTION), None)
    validator.check_record({"local_id": "S1", "analyses": [
        {"local_id": "a1", "spatial_scope": field(value)}]})
    said = [e for e in validator.errors if "value must be a single" in e]
    assert bool(said) is bool(expect)
    if expect:
        assert expect in said[0]


# --- 4. derived values labelled reported ----------------------------------------------


def test_a_conclusion_with_no_sentence_is_relabelled_generated(extraction_schema):
    """`ModelTerm.type` 3,037 of 3,041. No paper writes down that a term is continuous."""
    record = {"model_estimations": [{"local_id": "m1", "terms": [
        {"local_id": "t1", "type": field("continuous", "reported", "not_found")}]}]}
    changed = fix.relabel_conclusions(record, extraction_schema)
    assert record["model_estimations"][0]["terms"][0]["type"]["value_source"] == "generated"
    assert changed


def test_a_conclusion_with_a_sentence_keeps_reported(extraction_schema):
    """"the interaction was negative" is a direction read off prose, and the label is
    earned. Relabelling it would throw away the only case where it is."""
    record = {"analyses": [{"local_id": "a1", "effect": {"cells": [
        {"term": "t1", "direction": field("negative", "reported", "present")}]}}]}
    assert fix.relabel_conclusions(record, extraction_schema) == []
    cells = record["analyses"][0]["effect"]["cells"]
    assert cells[0]["direction"]["value_source"] == "reported"


def test_a_slot_a_paper_could_have_stated_is_left_for_the_warning(extraction_schema):
    """`tfce_used` at 57% and `Statistic.family` at 31% are things a results section
    prints, so an unevidenced one is a reviewer's business rather than a relabel."""
    record = {"inference_settings": [
        {"local_id": "i1", "tfce_used": field(True, "reported", "not_found")}]}
    assert fix.relabel_conclusions(record, extraction_schema) == []
    assert record["inference_settings"][0]["tfce_used"]["value_source"] == "reported"


# --- 5. a cell naming a level on a continuous term ------------------------------------


def _continuous(level, direction=None):
    cell = {"term": "trm_bmi", "level": field(level)}
    if direction is not None:
        cell["direction"] = field(direction)
    return {
        "analyses": [{"local_id": "a1", "model_estimation": "mod1",
                      "effect": {"cells": [cell]}}],
        "model_estimations": [{"local_id": "mod1", "terms": [
            {"local_id": "trm_bmi", "name": field("BMI"), "type": field("continuous")}]}],
    }


def test_a_level_restating_its_own_term_is_dropped():
    """547 of 1,185 (46%): `BMI` on term `BMI`, `age` on `age`, `pack-years` on itself."""
    record = _continuous("BMI")
    changed = fix.drop_redundant_cell_levels(record)
    assert "level" not in record["analyses"][0]["effect"]["cells"][0]
    assert changed and "restated" in changed[0]


def test_a_level_duplicating_its_direction_is_dropped():
    """185 of 1,185. The sign is already in the slot that holds signs."""
    record = _continuous("positive", "positive")
    fix.drop_redundant_cell_levels(record)
    cell = record["analyses"][0]["effect"]["cells"][0]
    assert "level" not in cell
    assert cell["direction"]["value"] == "positive"


def test_a_level_that_is_the_only_sign_is_moved_not_dropped():
    record = _continuous("higher")
    fix.drop_redundant_cell_levels(record)
    cell = record["analyses"][0]["effect"]["cells"][0]
    assert "level" not in cell
    assert cell["direction"]["value"] == "positive"
    assert cell["direction"]["value_source"] == "generated"


def test_a_level_contradicting_its_direction_is_left_for_the_check():
    """26 cells. Dropping the level resolves a contradiction about the sign of an effect by
    picking a side silently, and the sign decides which map a coordinate enters."""
    record = _continuous("positive", "negative")
    assert fix.drop_redundant_cell_levels(record) == []
    assert record["analyses"][0]["effect"]["cells"][0]["level"]["value"] == "positive"
    sink = Sink()
    rules.check_cell_level_polarity(record, sink)
    assert len(sink.errors) == 1
    assert "increase or decrease map" in sink.errors[0][1]


def test_a_categorical_level_on_a_mistyped_term_is_left_alone():
    """424 of 1,185. Flipping `type` means synthesising the levels the term should have
    declared, which is a claim about the model rather than a tidy-up."""
    record = _continuous("bvFTD")
    assert fix.drop_redundant_cell_levels(record) == []
    assert record["analyses"][0]["effect"]["cells"][0]["level"]["value"] == "bvFTD"


def test_a_sign_word_another_term_declares_as_a_level_is_a_level():
    """29935441, and the only record the corpus sweep showed this repair breaking.

    `level: 'negative'` sits on the product column `PCL-by-emotion-by-task`, and `negative`
    is one of `trm_emotion`'s declared levels beside `positive` and `neutral` -- an emotion,
    not a sign. Read as a direction it became `direction: negative`, turning a factor level
    into a polarity, and `check_crossings` reported a signed cell on a product column.
    """
    record = _continuous("negative")
    record["model_estimations"][0]["terms"].append(
        {"local_id": "trm_emotion", "name": field("emotion"), "type": field("categorical"),
         "levels": [{"level": field("negative")}, {"level": field("positive")},
                    {"level": field("neutral")}]}
    )
    assert fix.drop_redundant_cell_levels(record) == []
    cell = record["analyses"][0]["effect"]["cells"][0]
    assert cell["level"]["value"] == "negative"
    assert "direction" not in cell


def test_an_undirected_cell_is_not_overwritten():
    """`undirected` is an answer, not a gap. Overwriting it replaces a stated fact with an
    inference, which is how the product-column case above went wrong."""
    record = _continuous("positive", "undirected")
    assert fix.drop_redundant_cell_levels(record) == []
    assert record["analyses"][0]["effect"]["cells"][0]["direction"]["value"] == "undirected"


def test_a_level_on_a_term_that_declares_levels_is_left_alone():
    record = _continuous("BMI")
    record["model_estimations"][0]["terms"][0]["levels"] = [{"level": field("high")}]
    assert fix.drop_redundant_cell_levels(record) == []


# --- the guard that keeps one broken rule from hiding the others -----------------------


def test_a_rule_that_raises_becomes_a_finding_rather_than_an_abort():
    """`check_crossings` raises on 3 of 1,817 records, where a wrapper carries `value` and
    no `extraction_status`, so those records silently lost the 17 checks after it."""
    record = {
        "analyses": [
            {
                "local_id": "a1",
                "model_estimation": "mod1",
                "name": field("x"),
                "effect": {"cells": [{"term": "t1", "direction": {"value": "positive"}}]},
            }
        ],
        "model_estimations": [
            {
                "local_id": "mod1",
                "terms": [{"local_id": "t1", "name": field("g"), "type": field("categorical")}],
            }
        ],
    }
    sink = Sink()
    rules.check_all(record, sink)          # must not raise
    raised = [m for _p, m in sink.errors if "check raised" in m]
    assert len(raised) <= 1, "only the crashing rule should report a crash"


# --- the two faults `not_found` used to collapse ---------------------------------------


def _text():
    return "Participants were 24 right-handed volunteers recruited from the community."


@pytest.mark.parametrize(
    "quote,expect_status,expect_unlocated",
    [
        ("24 right-handed volunteers", "present", None),
        ("participants were 24", "present", None),          # case only -- third pass
        ("two dozen right handers", "not_found", 1),        # a paraphrase
    ],
)
def test_build_records_which_evidence_fault_occurred(quote, expect_status, expect_unlocated):
    """`status: not_found` said only that one of two opposite faults happened.

    A value with no quote is a recall failure; a value whose quote was rejected is a
    fidelity failure. Over 1,817 records 32,500 fields carried the first status and no way
    to tell which fault it was, so the warning firing on 98.6% of papers was unactionable.
    """
    from pondie.extraction.record import spans
    from pondie.extraction.evidence.warrant import Warrant, _resolve_field

    text = _text()
    node = {
        "extraction_status": "extracted", "value": "x", "value_source": "reported",
        "evidence": {"status": "present", "sets": [{"quotes": [quote]}]},
    }
    report = Warrant()
    _resolve_field(node, text, spans.fold(text), "Study.x", report)
    assert node["evidence"]["status"] == expect_status
    assert node["evidence"].get("unlocated_quotes") == expect_unlocated


def test_a_field_that_offered_no_quote_carries_no_marker():
    """The slot's presence is the claim, so silence must stay silent."""
    from pondie.extraction.record import spans
    from pondie.extraction.evidence.warrant import Warrant, _resolve_field

    text = _text()
    node = {"extraction_status": "extracted", "value": "x", "value_source": "reported",
            "evidence": {"status": "not_found"}}
    report = Warrant()
    _resolve_field(node, text, spans.fold(text), "Study.y", report)
    assert "unlocated_quotes" not in node["evidence"]
    assert report.unlocated == 0


def test_the_honesty_warning_says_which_fault_it_is():
    record = {"analyses": [
        {"local_id": "a1", "name": {
            "extraction_status": "extracted", "value": "A > B", "value_source": "reported",
            "evidence": {"status": "not_found", "unlocated_quotes": 2}}},
        {"local_id": "a2", "name": {
            "extraction_status": "extracted", "value": "C > D", "value_source": "reported",
            "evidence": {"status": "not_found"}}},
    ]}
    sink = Sink()
    rules.check_value_source_honesty(record, sink)
    said = " | ".join(m for _p, m in sink.warnings)
    assert "2 proposed quote(s) could not be placed" in said
    assert "no supporting sentence was proposed at all" in said


# --- the case-insensitive third pass ---------------------------------------------------


@pytest.mark.parametrize(
    "text,quote,how",
    [
        (_text(), "24 right-handed volunteers", "exact"),
        (_text().replace("handed volunteers", "handed\nvolunteers"),
         "right-handed volunteers", "tolerant"),
        (_text(), "participants were 24", "cased"),
        (_text(), "RIGHT-HANDED VOLUNTEERS", "cased"),
        (_text(), "two dozen right handers", None),
    ],
)
def test_resolve_tries_exact_then_whitespace_then_case(text, quote, how):
    """Case is the one perturbation a model makes mechanically, and `fold` cannot absorb it
    -- casefolding is not length-preserving for all of Unicode and `fold` promises length.
    Matching case-insensitively against the folded original sidesteps that."""
    from pondie.extraction.record import spans

    if how is None:
        with pytest.raises(spans.SpanResolutionError):
            spans.resolve(text, quote)
        return
    found = spans.resolve(text, quote)
    assert (found.exact, found.cased) == {
        "exact": (True, False),
        "tolerant": (False, False),
        "cased": (False, True),
    }[how]
    # The invariant the whole design rests on: offsets address the document, and the span
    # text is the document's substring rather than the quote that located it.
    assert text[found.start_char:found.end_char] == found.text


def test_the_case_insensitive_pass_cannot_move_an_existing_match():
    """A separate pass rather than a flag on the second, so a case-sensitive hit at a later
    offset is never displaced by a case-different one earlier in the document."""
    from pondie.extraction.record import spans

    text = "The GROUP was scanned. Later the group was rescanned."
    found = spans.resolve(text, "the group")
    assert found.cased is False
    assert text[found.start_char:found.end_char] == "the group"
    assert found.start_char > 20


# --- elided quotes: two citations written as one ---------------------------------------


def _elided_text():
    return ("Our design is fully factorial. It allows us to identify significant effects "
            "of heroin injection and salient visual stimuli separately.")


def test_an_elided_quote_resolves_as_the_spans_it_cites():
    """176 of the 2,952 unplaceable quotes on the 903-paper run are this shape, and for 169
    every fragment resolves alone. An `EvidenceSet` already holds several spans, so an
    elided quote is two citations written as one rather than a malformed quote."""
    from pondie.extraction.record import spans
    from pondie.extraction.evidence.warrant import Warrant, _resolve_field

    text = _elided_text()
    node = {"extraction_status": "extracted", "value": "x", "value_source": "reported",
            "evidence": {"status": "present", "sets": [
                {"quotes": ["Our design ... allows us to identify significant effects"]}]}}
    report = Warrant()
    _resolve_field(node, text, spans.fold(text), "Study.x", report)
    placed = node["evidence"]["sets"][0]["spans"]
    assert node["evidence"]["status"] == "present"
    assert [span["text"] for span in placed] == [
        "Our design", "allows us to identify significant effects"]
    assert report.elided == 2
    for span in placed:
        assert text[span["start_char"]:span["end_char"]] == span["text"]


def test_an_elided_quote_with_one_invented_fragment_is_a_drop():
    """All or nothing: half the support offered is not the support offered."""
    from pondie.extraction.record import spans
    from pondie.extraction.evidence.warrant import Warrant, _resolve_field

    text = _elided_text()
    node = {"extraction_status": "extracted", "value": "x", "value_source": "reported",
            "evidence": {"status": "present", "sets": [
                {"quotes": ["Our design ... proves causation"]}]}}
    report = Warrant()
    _resolve_field(node, text, spans.fold(text), "Study.x", report)
    assert node["evidence"]["status"] == "not_found"
    assert node["evidence"]["unlocated_quotes"] == 1
    assert report.elided == 0


def test_a_partly_lost_field_records_the_loss_while_staying_present():
    """A field offering two quotes and keeping one reads as fully evidenced to any consumer.
    Recording the count only on total failure would measure unevidenced FIELDS while
    claiming to measure dropped QUOTES."""
    from pondie.extraction.record import spans
    from pondie.extraction.evidence.warrant import Warrant, _resolve_field

    text = _elided_text()
    node = {"extraction_status": "extracted", "value": "x", "value_source": "reported",
            "evidence": {"status": "present", "sets": [
                {"quotes": ["Our design is fully factorial", "a claim the paper never makes"]}]}}
    report = Warrant()
    _resolve_field(node, text, spans.fold(text), "Study.x", report)
    assert node["evidence"]["status"] == "present"
    assert node["evidence"]["unlocated_quotes"] == 1
    assert report.partly_unlocated == 1
    assert report.unlocated == 0


# --- the shared schema-guided walk ------------------------------------------------------


def test_the_walk_finds_what_the_schema_declares(extraction_schema):
    """Ten functions in `builder` each walked the record, and they diverged. The point of
    one walk is that `declared_ids` descends -- `repair_references` swept the top-level
    lists and missed 10,867 declared ids over the corpus, every ModelTerm and Condition
    among them, while `validate.index_ids` descended and disagreed with it."""
    from pondie.extraction.record import walk

    record = {
        "local_id": "S1",
        "tasks": [{"local_id": "tsk1", "name": field("go/no-go"),
                   "conditions": [{"local_id": "cond_go", "name": field("go")}]}],
        "model_estimations": [{"local_id": "mod1", "terms": [
            {"local_id": "trm1", "name": field("condition")}]}],
    }
    ids = walk.declared_ids(record, extraction_schema)
    assert ids["cond_go"] == "Condition", "a Condition nests under tasks[].conditions"
    assert ids["trm1"] == "ModelTerm", "a ModelTerm nests under model_estimations[].terms"
    assert ids["tsk1"] == "Task"


def test_the_walk_yields_a_slots_own_declaration(extraction_schema):
    """`attribute.range` on a field is the wrapper, not the value. Five repairs needed the
    inner declaration and each reached for it differently."""
    from pondie.extraction.record import walk

    record = {"local_id": "S1", "groups": [
        {"local_id": "g1", "medical_condition": field(["obesity"]), "age_mean": field(31.0)}]}
    by_key = {slot.key: slot for slot in walk.fields(record, extraction_schema)}
    assert by_key["medical_condition"].declared_value(extraction_schema).multivalued is True
    assert by_key["age_mean"].declared_value(extraction_schema).multivalued in (None, False)


def test_a_caller_may_mutate_while_walking(extraction_schema):
    """Every repair assigns to or deletes the slot it is looking at, so the walk
    materialises each node's items before yielding."""
    from pondie.extraction.record import walk

    record = {"local_id": "S1", "analyses": [
        {"local_id": "a1", "spatial_scope": field(["whole_brain"]),
         "prespecification": field(["preregistered"])}]}
    for slot in walk.fields(record, extraction_schema):
        del slot.owner[slot.key]
    assert record["analyses"][0] == {"local_id": "a1"}


def test_references_are_yielded_with_their_ids(extraction_schema):
    from pondie.extraction.record import walk

    record = {"local_id": "S1", "analyses": [
        {"local_id": "a1", "tables": ["tbl1", "tbl2"], "inference_settings": "inf1"}]}
    found = {
        slot.key: walk.ids_of(slot.value) for slot in walk.references(record, extraction_schema)
    }
    assert found["tables"] == ["tbl1", "tbl2"]
    assert found["inference_settings"] == ["inf1"]


# --- repointing a dangling reference: the three conditions ------------------------------


def _with_tasks(*tasks):
    return {
        "local_id": "S1",
        "analyses": [{"local_id": "a1", "tasks": ["tsk_missing"]}],
        "tasks": [{"local_id": lid, "name": field(name)} for lid, name in tasks],
    }


def test_a_transcription_slip_is_repaired_outright(extraction_schema):
    """Nothing is decided: the id differs only in case and punctuation."""
    record = _with_tasks(("tsk_cue_exposure", "cue exposure"))
    record["analyses"][0]["tasks"] = ["tsk_Cue-Exposure"]
    assert fix.repair_references(record, extraction_schema)
    assert record["analyses"][0]["tasks"] == ["tsk_cue_exposure"]


def test_the_only_candidate_is_taken_when_the_names_agree(extraction_schema):
    record = _with_tasks(("tsk_cue_exposure_fmri", "cue exposure fMRI task"))
    record["analyses"][0]["tasks"] = ["tsk_cue_exposure"]
    assert fix.repair_references(record, extraction_schema)
    assert record["analyses"][0]["tasks"] == ["tsk_cue_exposure_fmri"]


def test_the_only_candidate_is_refused_when_the_names_do_not(extraction_schema):
    """`asm_mini` onto `asm_ftnd` -- a psychiatric interview onto a nicotine scale -- is
    what taking the sole candidate on its own did."""
    record = _with_tasks(("tsk_fear_conditioning", "fear conditioning"))
    record["analyses"][0]["tasks"] = ["tsk_resting_state"]
    assert fix.repair_references(record, extraction_schema) == []
    assert record["analyses"][0]["tasks"] == ["tsk_resting_state"]


def test_an_initialism_of_the_target_name_agrees(extraction_schema):
    """8% of the references this resolves. `asm_scid` shares no word with "Structured
    Clinical Interview for DSM-V" and plainly means it."""
    record = {
        "local_id": "S1",
        "groups": [{"local_id": "g1", "diagnostic_instrument": ["asm_scid"]}],
        "assessments": [{"local_id": "asm_structured_clinical",
                         "name": field("Structured Clinical Interview for DSM-V")}],
    }
    assert fix.repair_references(record, extraction_schema)
    assert record["groups"][0]["diagnostic_instrument"] == ["asm_structured_clinical"]


def test_two_differently_named_references_do_not_collapse_onto_one_target(extraction_schema):
    """A record declaring one term whose cells name three is a record missing two, and an
    interaction term shares a word with the main effect inside it -- so
    `trm_smoking_opportunity_cue` and `trm_quitting_motivation_cue` both pass the name test
    against a term called "cue"."""
    record = {
        "local_id": "S1",
        "analyses": [{"local_id": "a1", "model_estimation": "mod1", "effect": {"cells": [
            {"term": "trm_smoking_opportunity_cue"}, {"term": "trm_quitting_motivation_cue"}]}}],
        "model_estimations": [{"local_id": "mod1", "terms": [
            {"local_id": "trm_cue_1", "name": field("cue")}]}],
    }
    assert fix.repair_references(record, extraction_schema) == []
    terms = [cell["term"] for cell in record["analyses"][0]["effect"]["cells"]]
    assert terms == ["trm_smoking_opportunity_cue", "trm_quitting_motivation_cue"]


def test_one_reference_repeated_across_analyses_is_not_a_collapse(extraction_schema):
    """The guard counts distinct NAMES, not occurrences: the same dangling id in four
    analyses is one thing named once."""
    record = _with_tasks(("tsk_cue_exposure_fmri", "cue exposure fMRI task"))
    record["analyses"] = [
        {"local_id": f"a{n}", "tasks": ["tsk_cue_exposure"]} for n in range(4)]
    assert len(fix.repair_references(record, extraction_schema)) == 4
    assert all(a["tasks"] == ["tsk_cue_exposure_fmri"] for a in record["analyses"])


def test_a_reference_is_never_repointed_at_its_own_owner(extraction_schema):
    record = {"local_id": "S1", "analyses": [
        {"local_id": "a1", "name": field("A > B"), "mirror_of": "a_missing"}]}
    assert fix.repair_references(record, extraction_schema) == []


def test_declared_ids_are_read_schema_guided(extraction_schema):
    """Sweeping the Study-level lists misses every ModelTerm and Condition -- 10,867 ids
    over 1,817 records -- so a `Cell.term` reference looked dangling and was repaired by
    guesswork or not at all."""
    record = {
        "local_id": "S1",
        "analyses": [{"local_id": "a1", "model_estimation": "mod1", "effect": {"cells": [
            {"term": "trm_Condition"}]}}],
        "model_estimations": [{"local_id": "mod1", "terms": [
            {"local_id": "trm_condition", "name": field("condition")}]}],
    }
    assert fix.repair_references(record, extraction_schema)
    assert record["analyses"][0]["effect"]["cells"][0]["term"] == "trm_condition"
