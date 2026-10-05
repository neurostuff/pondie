"""The strict reply schemas: valid for Structured Outputs, and faithful to the extraction schema."""

from pondie import schema
from pondie.extraction import llm
from pondie.extraction.models import ModelCall
from pondie.extraction.prompt import render, reply_schema
from pondie.schema import reader


def _sch():
    return reader.load(schema.EXTRACTION)


def _objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


def test_every_object_is_closed_and_requires_every_property():
    """Strict mode refuses a schema where either fails, so the call would be rejected."""
    for evidence in reply_schema.EVIDENCE_FORMATS:
        for node in _objects(reply_schema.single(_sch(), evidence, silence=True)):
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])


def test_the_top_level_is_the_lists_the_prompt_states():
    single = reply_schema.single(_sch(), "quotes")
    lists = [k for k in render.payload_keys("single") if k in _sch().classes_by_container()]
    assert sorted(single["properties"]) == sorted(lists + ["study", "omitted"])
    assert list(single["properties"])[0] == "analyses", "SINGLE_NOTE asks for analyses first"
    assert "language" not in single["properties"], "a slot code fills is not asked for"


def test_a_closed_vocabulary_is_an_enum_and_an_open_one_a_string():
    defs = reply_schema.single(_sch(), "none")["$defs"]
    kind = defs["ExtractedEffectKind__wrapper"]["anyOf"][0]["properties"]["value"]
    assert "contrast" in kind["enum"]
    species = defs["ExtractedSpecies__wrapper"]["anyOf"][0]["properties"]["value"]
    assert species["type"] == "string" and "enum" not in species
    assert "human" in species["description"], "an open vocabulary still names its terms"


def test_a_type_designated_slot_offers_each_subclass_with_its_name_fixed():
    defs = reply_schema.single(_sch(), "none")["$defs"]
    assert defs["MRI"]["properties"]["acquisition_type"] == {"type": "string", "enum": ["MRI"]}
    assert defs["PET"]["properties"]["acquisition_type"] == {"type": "string", "enum": ["PET"]}


def test_fill_answers_each_asked_id_with_a_typed_value_or_a_reason():
    rows = [{"id": "groups[g].age_mean", "range": "float", "ranges": ["float"],
             "multivalued": False}]
    answer = reply_schema.fill(_sch(), rows)["properties"]["groups[g].age_mean"]["anyOf"]
    assert answer[0]["properties"]["value"] == {"type": "number"}
    assert "silent_default" in answer[1]["properties"]["unreported_reason"]["enum"]


def test_a_call_with_a_schema_asks_for_it_strictly_and_reads_null_as_absent():
    call = ModelCall(model="m", prompt="p", json_schema={"type": "object"})
    assert llm._format(call)["json_schema"]["strict"] is True
    assert llm._format(ModelCall(model="m", prompt="p")) == {"type": "json_object"}
    assert llm._without_nulls({"a": None, "b": [{"c": None, "d": 1}]}) == {"b": [{"d": 1}]}


def test_an_omission_is_written_the_way_the_listing_check_reads_it():
    """It was `entry`; `unconsumed_listing` reads `key`, so every structured omission read as
    an entry ignored, and `single` was asked again."""
    item = reply_schema.single(_sch(), "quotes")["properties"]["omitted"]["items"]
    assert list(item["properties"]) == ["key", "reason"]
    payload = {"analyses": [], "omitted": [{"key": "t1#1", "reason": "seed_coordinate"}]}
    assert render.unconsumed_listing(payload, ["t1#1"]) == []
