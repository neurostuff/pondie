"""Evidence written once per sentence, read back into per-field quotes."""

from pondie.extraction.evidence import cited
from pondie.extraction.record import spans

TEXT = ("## Methods\nTwelve patients (8 females) had PTSD. Their mean age was 34.6 years.\n"
        "Controls were matched for age.")


def _v(value, **extra):
    return {"extraction_status": "extracted", "value": value, "value_source": "reported",
            **extra}


def test_numbered_marks_every_sentence_and_keeps_the_text():
    shown = cited.numbered(TEXT)
    assert "[S2] Twelve patients (8 females) had PTSD. [S3] Their mean age" in shown
    assert shown.replace("[S1] ", "").replace("[S2] ", "").replace("[S3] ", "") \
        .replace("[S4] ", "") == TEXT


def test_a_cited_number_becomes_the_exact_sentence_and_resolves():
    age = _v(34.6, evidence=[3])
    payload = {"groups": [{"local_id": "grp_ptsd", "age_mean": age, "n": _v(12, evidence=[99])}]}
    notes = cited.expand(payload, "indexed", TEXT)
    quote = age["evidence"]["sets"][0]["quotes"][0]
    assert quote == "Their mean age was 34.6 years."
    assert spans.resolve(TEXT, quote).exact
    assert "evidence" not in payload["groups"][0]["n"], "a number naming no sentence cites nothing"
    assert any("name no sentence" in n for n in notes)


def test_one_sentence_listed_once_supports_every_field_it_names():
    payload = {
        "groups": [{"local_id": "grp_ptsd", "name": _v("PTSD"), "n": _v(12)}],
        "analyses": [{"local_id": "ana_1", "effect": {"cells": [{"term": "trm_g",
                                                                 "direction": _v("negative")}]}}],
        "study": {"design": {"allocation": _v("not_applicable")}},
        "support": [{"sentence": "Twelve patients (8 females) had PTSD.",
                     "fields": ["grp_ptsd.name", "grp_ptsd.n", "ana_1.effect.cells[0].direction",
                                "study.design.allocation", "grp_ptsd.nothing"]}],
    }
    notes = cited.expand(payload, "inverted", TEXT)
    assert "support" not in payload
    for field in (payload["groups"][0]["n"], payload["study"]["design"]["allocation"],
                  payload["analyses"][0]["effect"]["cells"][0]["direction"]):
        assert field["evidence"]["sets"][0]["quotes"] == ["Twelve patients (8 females) had PTSD."]
    assert any("name no field" in n for n in notes)


def test_silent_default_becomes_plain_not_reported():
    field = {"extraction_status": "not_reported", "unreported_reason": "silent_default"}
    kept = {"extraction_status": "not_reported", "unreported_reason": "outside_text"}
    cited.expand({"groups": [{"local_id": "g", "a": field, "b": kept}]}, "quotes", TEXT)
    assert field == {"extraction_status": "not_reported"}
    assert kept["unreported_reason"] == "outside_text"
