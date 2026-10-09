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
    notes = cited.expand(payload, TEXT)
    quote = age["evidence"]["sets"][0]["quotes"][0]
    assert quote == "Their mean age was 34.6 years."
    assert spans.resolve(TEXT, quote).exact
    assert "evidence" not in payload["groups"][0]["n"], "a number naming no sentence cites nothing"
    assert any("name no sentence" in n for n in notes)


def test_a_fill_answer_cites_numbers_and_the_slot_gets_the_sentences():
    from pondie.extraction.prompt import fill

    payload = {"groups": [{"local_id": "grp_ptsd", "age_mean": None}]}
    answers = {"groups[grp_ptsd].age_mean": {"value": 34.6, "evidence": [3, 77]}}
    cited.quote_answers(answers, TEXT)
    fill.apply_fill(payload, answers, ["groups[grp_ptsd].age_mean"])
    age = payload["groups"][0]["age_mean"]
    assert age["value"] == 34.6
    assert age["evidence"]["sets"][0]["quotes"] == ["Their mean age was 34.6 years."]


#: "Table 4" is a sentence of its own (the table's label line) and also occurs earlier,
#: inside a longer one. Searching for the cited sentence's text finds the earlier one.
REPEATED = ("## Results\nPeaks are listed in Table 4 below.\nTable 4\n"
            "Peaks of the group contrast.")
LABEL = REPEATED.index("\nTable 4\n") + 1


def _placed(field):
    from pondie.extraction.evidence import warrant

    warrant.warrant({"field": field}, REPEATED)
    return [(s["start_char"], s["text"]) for st in field["evidence"]["sets"] for s in st["spans"]]


def test_a_cited_sentence_is_placed_where_it_was_cited_not_where_its_text_first_occurs():
    """23QRyA7agtip cited [S202], the "Table 4" label line, and its span landed on the
    "(Table 4)" inside an earlier sentence; 22fv2dADxHqR's "ALFF" heading landed in the
    abstract. The number says which occurrence, and that has to survive becoming a quote."""
    key = _v("t4#1", evidence=[3])
    cited.expand({"analyses": [{"local_id": "a", "source_table_analysis": key}]}, REPEATED)
    assert _placed(key) == [(LABEL, "Table 4")]


def test_a_fill_answer_is_placed_where_its_sentence_was_cited():
    from pondie.extraction.prompt import fill

    payload = {"analyses": [{"local_id": "a", "source_table_analysis": None}]}
    answers = {"analyses[a].source_table_analysis": {"value": "t4#1", "evidence": [3]}}
    cited.quote_answers(answers, REPEATED)
    fill.apply_fill(payload, answers, ["analyses[a].source_table_analysis"])
    assert _placed(payload["analyses"][0]["source_table_analysis"]) == [(LABEL, "Table 4")]


def test_results_sentences_about_the_brain_that_no_analysis_cites_are_candidates():
    """19538748's fMRI contrasts were cited as two cell labels' wording and belonged to no
    analysis. Demographic tests in the same section are not candidates."""
    text = ("## Methods\nTwelve patients were scanned.\n## Results\n"
            "Patients had less gray matter in the insula (T = 4.6).\n"
            "Controls showed greater activation in the left insula during encoding.\n"
            "The groups did not differ in age (t = 0.5, p = 0.6).")
    vbm = "Patients had less gray matter in the insula (T = 4.6)."
    payload = {"analyses": [{"local_id": "ana_vbm", "name": _v("VBM",
               evidence={"status": "present", "sets": [{"quotes": [vbm]}]})}]}
    found = cited.unanalysed_results(payload, text)
    shown = [text[a:b] for n, (a, b) in enumerate(cited.sentence_spans(text), 1) if n in found]
    assert shown == ["Controls showed greater activation in the left insula during encoding."]


def test_a_recheck_without_sentence_numbers_is_refused():
    """It needs the citations to know which Results sentences an analysis covers."""
    import pytest

    from pondie.extraction.models import Settings

    with pytest.raises(ValueError, match="indexed"):
        Settings(payloads="p", records="r", model="m", recheck_results=True)
    Settings(payloads="p", records="r", model="m", recheck_results=True,
             evidence_format="indexed")
