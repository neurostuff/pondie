"""pondie-astra: the prompt and template the ns extraction sent, and the record they restore to."""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/pondie-astra/src"))
from pondie_astra import compact, prompt

#: sha256 of the template every body of the 2026-10-28 ns shards carries (vista-out/ns/shards-20261028).
SENT_TEMPLATE = "cd79462cd7b93159101a6df3fdcd89ba0499473ead74489d03fa64215eb6b9f9"

ENTRY = {"key": "analysis#0123456789abcdef", "origin": "table", "table_id": "T1", "name": "A > B",
         "coordinate_space": "MNI", "caption": "Peaks",
         "points": [{"coordinates": [-42.0, 18, 6.5], "values": [{"kind": "t", "value": 5.1}],
                     "cluster_size": 120}]}


def test_the_template_is_the_one_the_ns_extraction_sent():
    assert hashlib.sha256(prompt.template().encode()).hexdigest() == SENT_TEMPLATE


def test_a_parse_entry_reads_as_the_model_saw_it():
    assert prompt.parse_block(ENTRY) == (
        "Coordinate set: analysis#0123456789abcdef\nOrigin: table (table T1)\nName: A > B\n"
        "Coordinate space: MNI\nCaption: Peaks\nPoints: 1\n-42 18 6.5  [t=5.1; k=120]")
    text = prompt.prompt_of("Body.", {ENTRY["key"]: ENTRY})
    assert text.startswith(prompt.INSTR + "\n\nText:\nBody.\n\nCoordinate sets:\nCoordinate set: analysis#")


def test_the_27b_takes_the_template_in_its_prompt_and_nuextract_through_its_chat_template():
    content, kwargs = prompt.user_content("P", "T", "q27")
    assert content == "P" + prompt.TEMPLATE_SUFFIX + "T" and kwargs == {"enable_thinking": False}
    assert prompt.user_content("P", "T", "nu") == ("P", {"enable_thinking": False, "template": "T"})


def test_captions_are_trimmed_as_training_s_were():
    tables = [{"table_id": "T1", "caption": "c" * 400, "footer": " f "}]
    assert prompt.captions(tables, [ENTRY]) == {ENTRY["key"]: {"caption": "c" * 300, "footer": "f"}}


def test_a_compact_record_restores_against_the_packaged_schema():
    restored = compact.from_compact({"coordinate_sets": [{"local_id": ENTRY["key"]}]},
                                    prompt.schema_index(), "sid1")
    assert restored["local_id"] == "sid1"
    assert compact.to_compact(restored)["coordinate_sets"] == [{"local_id": ENTRY["key"]}]


def test_ev1_cites_units_and_the_merge_puts_them_back_as_spans():
    from pondie_astra import evidence

    text = "Methods.\nWe scanned 24 healthy adults. Each lay still.\nRow\t1"
    units = evidence.units(text)
    assert [text[a:b] for a, b in units] == ["Methods.", "We scanned 24 healthy adults. Each lay still.",
                                             "Row\t1"]
    record = compact.from_compact({"groups": [{"local_id": "g1", "enrolled_count": 24}]}, prompt.schema_index(), "sid1")
    request, _ = evidence.prompt_of(record, text)
    assert "\n\nText:\n[S1] Methods.\n[S2] We scanned" in request and request.startswith(evidence.INSTR)
    counts = evidence.merge(record, text, {"groups": [{"local_id": "g1", "enrolled_count": [2]}]})
    assert counts["cited_values"] == 1 and counts["roundtrip"] == "ok"
    (span,) = record["groups"][0]["enrolled_count"]["evidence"]["sets"][0]["spans"]
    assert span["text"] == text[span["start_char"]:span["end_char"]] == \
        "We scanned 24 healthy adults. Each lay still."
