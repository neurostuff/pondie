"""A group at risk of a condition does not have it.

Every value here is one ns-pond's production catalog held (2026-11-27 survey of 100,519
groups). The reading needs no vocabulary release: it is triage.
"""

from __future__ import annotations

import pytest
from study_schema.models.storage import ConditionRole, RiskBasis

from pondie.normalization import medical_condition as mc
from pondie.normalization.is_healthy import derive
from pondie.vocabularies.phrases import AT_RISK, NO_CONDITION, triage


def read(value: str) -> tuple:
    """(present heads, {at-risk head: bases}, bases of a risk naming no condition)."""
    t = triage(value)
    risky = {h: t.bases(i) for i, h in enumerate(t.heads) if t.bases(i) is not None}
    return t.present, risky, t.unnamed


@pytest.mark.parametrize(
    "value, condition, bases",
    [
        ("Clinical high risk for psychosis", "psychosis", ("clinical",)),
        ("Ultra-high risk for psychosis at baseline", "psychosis", ("clinical",)),
        ("At-risk mental state", "psychosis", ("clinical",)),
        ("Attenuated psychotic symptoms", "psychosis", ("clinical",)),
        ("Genetic risk and deterioration syndrome", "psychosis", ("familial", "clinical")),
        ("Familial high risk for schizophrenia", "schizophrenia", ("familial",)),
        ("Unaffected first-degree relatives of patients with schizophrenia", "schizophrenia",
         ("familial",)),
        ("Unaffected siblings of schizophrenia patients", "schizophrenia", ("familial",)),
        ("Family history of alcohol use disorder", "alcohol use disorder", ("familial",)),
        ("High risk for depression defined by maternal depression history", "depression",
         ("familial",)),
        ("Premanifest Huntington's disease", "Huntington's disease", ("genetic",)),
        ("Presymptomatic familial Alzheimer's disease mutation carrier", "Alzheimer's disease",
         ("familial", "genetic")),
        # prodromal is early signs, a clinical risk (James, 2026-11-27)
        ("Prodromal Alzheimer's disease", "Alzheimer's disease", ("clinical",)),
        ("Preclinical Alzheimer's disease", "Alzheimer's disease", ("biomarker",)),
        ("Cognitive vulnerability to depression", "depression", ("trait",)),
        # no basis stated: no basis entry
        ("At risk for bipolar disorder", "bipolar disorder", ()),
    ],
)
def test_a_risk_links_the_condition_and_says_the_group_lacks_it(value, condition, bases) -> None:
    present, risky, unnamed = read(value)
    assert (present, risky, unnamed) == ((), {condition: bases}, None)


def test_the_kinds_of_risk_one_clause_names_are_all_kept() -> None:
    """Several bases hold at once: "genetic or clinical high risk" is both."""
    _, risky, _ = read("Genetic or clinical high risk for psychosis or bipolar disorder")
    assert risky == {"psychosis": ("genetic", "clinical"),
                     "bipolar disorder": ("genetic", "clinical")}


def test_the_condition_s_own_words_are_not_its_basis() -> None:
    """Substance use is the condition here, not the evidence."""
    assert read("Familial risk for substance use disorder")[1] == {
        "substance use disorder": ("familial",)}


@pytest.mark.parametrize(
    "value, gene",
    [
        ("APOE ε4 carrier", "APOE ε4"),
        ("Presymptomatic GRN mutation carrier", "GRN"),
        ("Asymptomatic LRRK2 G2019S mutation carriers", "LRRK2 G2019S"),
        ("Non-manifesting DYT1 mutation carrier status", "DYT1"),
    ],
)
def test_a_carrier_is_at_risk_of_a_condition_the_value_does_not_name(value, gene) -> None:
    t = triage(value)
    assert (t.kind, t.heads, t.unnamed, gene in t.qualifiers) == (AT_RISK, (), ("genetic",), True)


@pytest.mark.parametrize(
    "value",
    [
        "No personal or family history of psychiatric disorders",
        "No family history of developmental dyslexia or reading difficulties",
        "Healthy; no personal or family history of psychiatric disorder",
        "No attenuated psychotic symptoms",
        "APOE4 non-carrier status",
        "Healthy non-carriers of the G2019S mutation",
        "Low familial risk for major depressive disorder",
        "Low-risk for psychosis",
    ],
)
def test_a_screened_or_low_risk_control_is_an_absence(value) -> None:
    """223 production items deny a family history; none may come out at risk."""
    t = triage(value)
    assert (t.kind, t.heads) == (NO_CONDITION, ())


@pytest.mark.parametrize(
    "value, present",
    [
        # a diagnosis in carriers, not a risk
        ("Definite Parkinson's disease in four homozygous carriers", 1),
        # a present behavior (James, 2026-11-27)
        ("Binge drinking", 1),
        # the group has them; they are risk factors, not a risk
        ("Cardiovascular risk factors", 1),
        # a familial form of a disease is the disease
        ("Familial Alzheimer's disease", 1),
    ],
)
def test_a_condition_the_group_has_is_not_a_risk(value, present) -> None:
    t = triage(value)
    at_risk = any(r is not None for r in t.risk)
    assert (len(t.present), t.unnamed, at_risk) == (present, None, False)


def test_a_condition_beside_a_risk_keeps_its_own_role() -> None:
    assert read("Depressive symptoms; at risk for depression")[:2] == (
        ("Depressive symptoms",), {"depression": ()})
    # some of the group have it, some have a relative who does
    assert read("Current or family history of diabetes")[:2] == (
        ("diabetes",), {"diabetes": ("familial",)})


def test_a_group_only_at_risk_is_healthy() -> None:
    """study_schema's `is_healthy`: true when every entry is an absence or a risk."""

    def group(*items: str) -> dict:
        return {"medical_condition": {"value": list(items), "extraction_status": "extracted"}}

    assert derive(group("Unaffected first-degree relatives of schizophrenia patients")) is True
    assert derive(group("Healthy; familial risk for depression")) is True
    assert derive(group("APOE ε4 carrier")) is True
    assert derive(group("Familial high risk for schizophrenia", "Obesity")) is False


def test_an_at_risk_row_writes_its_role_and_one_entry_per_basis() -> None:
    row = mc.Link(study="s", group="g", text="Presymptomatic familial AD mutation carrier",
                  head="AD", mondo="MONDO:0004975", label="Alzheimer disease", method="lexical",
                  score=1.0, at_risk=True, bases=("familial", "genetic"))
    written = [(e["vocabulary"], e.get("value"), e.get("head")) for e in mc.entries(row, "tool")]
    assert written == [
        ("MONDO", "MONDO:0004975", "AD"),
        ("ConditionRole", "at_risk", "AD"),
        ("RiskBasis", "familial", "AD"),
        ("RiskBasis", "genetic", "AD"),
    ]
    roles = {e["value"] for e in mc.entries(row, "tool") if e["vocabulary"] == mc.ROLE}
    bases = {e["value"] for e in mc.entries(row, "tool") if e["vocabulary"] == mc.BASIS}
    assert roles <= {r.value for r in ConditionRole}
    assert bases <= {b.value for b in RiskBasis}


def test_a_risk_naming_no_condition_is_an_unmapped_row_with_its_role() -> None:
    row = mc.Link(study="s", group="g", text="APOE ε4 carrier", sentinel=AT_RISK,
                  qualifiers=("APOE ε4",), at_risk=True, bases=("genetic",))
    written = [(e["vocabulary"], e["status"], e.get("value")) for e in mc.entries(row, "tool")]
    assert written == [("MONDO", "unmapped", None), ("ConditionRole", "mapped", "at_risk"),
                       ("RiskBasis", "mapped", "genetic")]


def test_a_condition_the_group_has_writes_no_role() -> None:
    """Primary or comorbid is the caller's to say; only `at_risk` is decided here."""
    row = mc.Link(study="s", group="g", text="Schizophrenia", mondo="MONDO:0005090",
                  label="schizophrenia", method="lexical", score=1.0)
    assert [e["vocabulary"] for e in mc.entries(row, "tool")] == ["MONDO"]
