"""The VBM-of-PTSD criteria as predicates over a record, at the ANALYSIS level.

Pankey 2022 (36100907), quoted from neurometabench's meta_datasets.csv:

  inclusion  "peer-reviewed MRI studies, reporting results among adult humans, written in the
             English language, focused on gray matter structural differences, and included
             original data"
  exclusion  "trauma or stressful life event studies not measuring PTSD, other
             non-voxel-based morphometry methods, treatment and longitudinal effects, papers
             reporting a priori regions of interest (ROIs), within-group effects, null
             effects, overlapping samples to previous studies, and studies that did not
             report coordinate-based results"
  analysis   "non-PTSD > PTSD"

Every predicate answers True, False or None ("the record cannot say"). A paper is selected
STRICTLY when every study criterion is True and some analysis has every analysis criterion
True; PERMISSIVELY when nothing is False (None counts as a pass).

The criteria are split by where they live. Study-level: original research, English, adult,
a PTSD cohort. Analysis-level: structural MRI, a grey-matter voxel-based measure, whole
brain, and a PTSD effect -- a between-subject contrast of a PTSD cohort against a non-PTSD
one, or a between-subject regression on PTSD severity in a sample that is not PTSD-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

PTSD = re.compile(r"ptsd|post.?traumatic stress|posttraumatic stress", re.I)
#: A cohort whose name or condition says it is the comparison: non-PTSD, trauma-exposed
#: without PTSD, healthy controls.
NEGATED = re.compile(r"non.?ptsd|without (a )?(current |lifetime )?ptsd|\bno ptsd|ptsd.?neg|"
                     r"(did not|never|not) develop(ed)? (a )?ptsd|"
                     r"\bcontrols?\b|healthy|\bhc\b|\btec\b|non.?traumati|resilient|"
                     r"comparison|unaffected", re.I)
#: The same, for a factor level label with no group reference behind it.
NOT_PTSD = re.compile(NEGATED.pattern + r"|trauma.?exposed", re.I)
SEVERITY = re.compile(r"caps|pcl|ptsd (symptom|severity|score)|clinician.?administered ptsd|"
                      r"ptsd checklist|symptom severity|impact of event|\bies\b|davidson", re.I)
GREY = re.compile(r"gr[ae]y.?matter", re.I)
NOT_ORIGINAL = {"review", "systematic review", "meta-analysis", "editorial", "letter",
                "comment", "case reports"}


def val(node: Any) -> Any:
    """The value of an ExtractedValue, None for anything not extracted."""
    if isinstance(node, dict) and "extraction_status" in node:
        if node.get("extraction_status") != "extracted":
            return None
        return val(node.get("value"))
    if isinstance(node, dict) and set(node) == {"value"}:
        return val(node["value"])
    return node


def strs(node: Any) -> list[str]:
    v = val(node)
    if v is None:
        return []
    items = v if isinstance(v, list) else [v]
    return [str(x) for x in items if isinstance(x, (str, int, float)) and str(x).strip()]


def num(node: Any) -> float | None:
    v = val(node)
    if isinstance(v, list):
        v = v[0] if v else None
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def refs(x: Any) -> list[str]:
    """A reference slot as id strings, however the model shaped it."""
    out = []
    for item in as_list(val(x)):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and isinstance(item.get("local_id"), str):
            out.append(item["local_id"])
    return out


def as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


class Index:
    """local_id -> entity, for every class a predicate needs to follow a reference to."""

    def __init__(self, record: dict, spec: "Spec | None" = None):
        self.record = record
        self.spec = spec or SPECS["36100907"]
        self.groups = {g.get("local_id"): g for g in record.get("groups") or []
                       if isinstance(g, dict)}
        self.measures = {m.get("local_id"): m for m in record.get("measures") or []
                         if isinstance(m, dict)}
        self.acqs = {a.get("local_id"): a for a in record.get("acquisitions") or []
                     if isinstance(a, dict)}
        self.models = {m.get("local_id"): m for m in record.get("model_estimations") or []
                       if isinstance(m, dict)}
        self.assessments = {a.get("local_id"): a for a in record.get("assessments") or []
                            if isinstance(a, dict)}
        self.terms = {}
        for m in self.models.values():
            for t in m.get("terms") or []:
                if isinstance(t, dict):
                    self.terms[t.get("local_id")] = t

    def group_text(self, gid: str) -> str:
        g = self.groups.get(gid) or {}
        return " ".join(strs(g.get("name")) + strs(g.get("medical_condition"))
                        + strs(g.get("description")))

    def is_case(self, gid: str) -> bool | None:
        g = self.groups.get(gid)
        return None if g is None else self.spec.status(g)

    # the PTSD name, kept for callers written against it
    is_ptsd_group = is_case


def cohort_status(group: dict, case: re.Pattern, comparison: re.Pattern) -> bool | None:
    """True for a case cohort, False for a comparison cohort, None for anything else.

    A comparison must READ as one -- controls, healthy, without the condition -- or be
    derived healthy. A cohort with some other diagnosis is neither: "PTSD vs OCD" is not
    the contrast a PTSD meta-analysis pools, and "bvFTD vs Alzheimer's" is not bvFTD < HC.
    """
    if val(group.get("is_healthy")) is True:
        return False
    # The label first; the description only where the label decides nothing -- rep2's
    # "non-symptomatic (NS)" has no condition and says "did not develop PTSD" only there.
    for slots in (("name", "medical_condition"), ("description",)):
        text = " ".join(x for slot in slots for x in strs(group.get(slot)))
        if comparison.search(text):
            return False
        if case.search(text):
            return True
    return None


def ptsd_status(group: dict) -> bool | None:
    return cohort_status(group, PTSD, NOT_PTSD)


# ------------------------------------------------------------------ study level

def original(record: dict, ix: Index) -> bool | None:
    types = [t.lower() for t in strs(record.get("study_type"))]
    if not types:
        return None
    return not any(t in NOT_ORIGINAL for t in types)


def english(record: dict, ix: Index) -> bool | None:
    codes = [str(c).lower() for c in as_list(val(record.get("language")))]
    return any(c.startswith("en") for c in codes) if codes else None


MINORS = re.compile(r"child(?!hood)(?!.{0,3}(abuse|maltreat|trauma|sexual|neglect))|"
                    r"adolescen|youth|pediatric|paediatric|juvenile|teen|minors?\b|"
                    r"school.?age|boys|girls", re.I)
ADULTS = re.compile(r"adult|veteran|soldier|military|police|firefighter|parent|mother|father|"
                    r"widow|university student|college student|\bmen\b|\bwomen\b|"
                    r"(aged?|ages?)\D{0,12}(1[89]|[2-9]\d)\s*(-|–|to)\s*\d\d", re.I)


def adult(record: dict, ix: Index) -> bool | None:
    """"reporting results among adult humans" -- what it excludes is children.

    Numbers first: any cohort whose minimum (or, lacking one, mean) age is under 18 is a
    False. Then words: a cohort described as children or adolescents is a False, one
    described as adults, veterans or parents, or with an adult age range in its inclusion
    criteria, is a True. Ages usually live in a demographics table, and for a third of
    these papers that table never reached the text -- so a stated population is what the
    record has, and it is what a meta-analyst reads too.
    """
    numbers = []
    for g in ix.groups.values():
        low, mean = num(g.get("age_minimum")), num(g.get("age_mean"))
        if low is not None:
            numbers.append(low >= 18)
        elif mean is not None:
            numbers.append(mean >= 18)
    if False in numbers:
        return False
    words = " ".join(x for g in ix.groups.values() for slot in
                     ("name", "description", "inclusion_criteria", "population_characteristics",
                      "sample_source") for x in strs(g.get(slot)))
    words += " " + " ".join(strs((record.get("study") or record).get("description")))
    if MINORS.search(words) and not numbers:
        return False
    if numbers:
        return True
    return True if ADULTS.search(words) else None


def search_window(record: dict, ix: Index) -> bool | None:
    """The search's date range, against PubMed's pubdate, to the month where the criterion
    gives one. PTSD: "from 2002 to 2020"; dementia: "- 5/2020" (32353756, July 2020, is
    outside it, and reading the bound as a year let it in)."""
    when = record.get("_pubdate")  # (year, month), month 0 when PubMed gives none
    if not when:
        return None
    low, high = ix.spec.window
    low = low if isinstance(low, tuple) else (low, 0)
    high = high if isinstance(high, tuple) else (high, 12)
    return low <= tuple(when) <= high


def reports_coordinates(record: dict, ix: Index) -> bool:
    """"studies that did not report coordinate-based results" excluded.

    From the inputs, not the model: parsed table points or prose coordinates, or -- with
    `score.py --gold-coords` -- the benchmark's own foci for an included paper.
    """
    return bool(record.get("_input_coordinates") or record.get("_gold_coordinates"))


def no_declared_overlap(record: dict, ix: Index) -> bool | None:
    """"overlapping samples to previous studies" -- the half a record can state on its own.

    A cohort the paper says was reported before is `Group.sample_source:
    previously_reported`. The undeclared half needs two records; see `overlap.py`.
    """
    sources = [s for g in ix.groups.values() for s in strs(g.get("sample_source"))]
    if not sources:
        return None
    return "previously_reported" not in sources


def case_cohort(record: dict, ix: Index) -> bool | None:
    verdicts = [ix.is_case(g) for g in ix.groups]
    if any(v is True for v in verdicts):
        return True
    # a single mixed cohort described as containing cases ("13 met criteria for PTSD")
    if any(ix.spec.case.search(" ".join(strs(g.get("description")))) for g in ix.groups.values()):
        return True
    return False if verdicts else None


ptsd_cohort = case_cohort


PTSD_STUDY: list[tuple[str, Callable]] = [
    ("original research", original),
    ("2002-2020", search_window),
    ("English", english),
    ("adult", adult),
    ("PTSD cohort", ptsd_cohort),
    ("reports coordinates", reports_coordinates),
]


# --------------------------------------------------------------- analysis level

def structural(a: dict, record: dict, ix: Index) -> bool | None:
    acqs = [ix.acqs.get(x) for x in refs(a.get("acquisitions"))]
    acqs = [x for x in acqs if x] or list(ix.acqs.values())
    mods = [s for x in acqs for s in strs(x.get("modality"))]
    if not mods:
        return None
    return any(m in ("sMRI", "MRI") or re.search(r"struct|t1", m, re.I) for m in mods)


def grey_voxelwise(a: dict, record: dict, ix: Index) -> bool | None:
    """Grey matter, measured voxel-wise ("other non-voxel-based morphometry" excluded)."""
    m = next((ix.measures[t] for t in refs(a.get("measure")) if t in ix.measures), None)
    kinds = (strs(m.get("type")) + strs(m.get("source_label")) + strs(m.get("specific_metric"))
             if m else [])
    if not kinds:
        return None
    if not any(GREY.search(k) for k in kinds):
        return False
    model = next((ix.models[t] for t in refs(a.get("model_estimation")) if t in ix.models), None)
    unit = strs(model.get("spatial_unit")) if model else []
    if unit and not any(u == "voxel" for u in unit):
        return False
    return True


def whole_brain(a: dict, record: dict, ix: Index) -> bool | None:
    scope = strs(a.get("spatial_scope"))
    if not scope:
        return None
    return "whole_brain" in scope


def _cell_terms(a: dict) -> list[dict]:
    return [c for c in ((a.get("effect") or {}).get("cells") or []) if isinstance(c, dict)]


def _level_of(term: dict | None, label: str) -> dict | None:
    """The FactorLevel a cell's `level` string names."""
    if term is None or not label:
        return None
    for lv in term.get("levels") or []:
        if isinstance(lv, dict) and label in strs(lv.get("level")):
            return lv
    return None


def _cell_cohort(cell: dict, ix: Index) -> tuple[bool | None, dict | None, dict | None]:
    """(is the contrasted level a PTSD cohort?, the term, the level) for one cell."""
    term = next((ix.terms.get(t) for t in refs(cell.get("term")) if t in ix.terms), None)
    label = " ".join(strs(cell.get("level")))
    level = _level_of(term, label)
    groups = refs(level.get("groups")) if level else []
    verdicts = {ix.is_ptsd_group(g) for g in groups} - {None}
    if True in verdicts and False not in verdicts:
        return True, term, level
    if False in verdicts and True not in verdicts:
        return False, term, level
    if label and not groups:
        if ix.spec.comparison.search(label):
            return False, term, level
        if ix.spec.case.search(label):
            return True, term, level
    return None, term, level


def ptsd_effect(a: dict, record: dict, ix: Index) -> bool | None:
    """A between-subject PTSD vs non-PTSD contrast, or a PTSD-severity regression across a
    sample that is not PTSD-only.

    Only the levels the cells CONTRAST count -- a three-level group term naming PTSD does
    not make "ASD < controls" a PTSD effect. "within-group effects" and "treatment and
    longitudinal effects" are excluded: a within-subject term, a level that is a timepoint
    or an arm, or a severity slope fitted inside the PTSD cohort alone.
    """
    effect = a.get("effect") or {}
    kind = " ".join(strs(effect.get("kind")))
    cells = _cell_terms(a)
    if not kind and not cells:
        return None
    signed = [c for c in cells if " ".join(strs(c.get("direction"))) in ("positive", "negative")]
    cohorts, slopes = set(), []
    for cell in signed:
        verdict, term, level = _cell_cohort(cell, ix)
        if term is not None and " ".join(strs(term.get("variation_level"))) == "within_subject":
            continue
        if level is not None and (level.get("timepoints") or level.get("arms")):
            continue
        if term is not None and " ".join(strs(term.get("type"))) == "continuous":
            slopes.append(term)
            continue
        cohorts.add(verdict)
    if True in cohorts and False in cohorts:
        return True
    for term in slopes if ix.spec.severity is not None else ():
        name = " ".join(strs(term.get("name")))
        asm = next((ix.assessments[t] for t in refs(term.get("assessment"))
                    if t in ix.assessments), {})
        name += " " + " ".join(strs(asm.get("name")))
        if not (ix.spec.severity.search(name) or ix.spec.case.search(name)):
            continue
        # A slope fitted inside the case cohort alone is a within-group effect; one across a
        # sample holding anyone else (23021615's veterans with and without PTSD) is not.
        sample = [x.get("group") for x in a.get("groups") or [] if isinstance(x, dict)]
        sample = [g for g in sample if isinstance(g, str)] or list(ix.groups)
        if any(ix.is_case(g) is not True for g in sample):
            return True
    if not signed and "contrast" in kind:
        return None
    return False


def reported_foci(a: dict, record: dict, ix: Index) -> bool | None:
    """"null effects" excluded.

    `Analysis.outcome` where the record has it; otherwise zero foci under every parse entry
    the analysis cites (`score.py` puts the count on `_n_foci` from the run's own parse).
    """
    outcome = strs(a.get("outcome"))
    if "no_significant_effect" in outcome:
        return False
    if "significant_effect" in outcome:
        return True
    n = a.get("_n_foci")
    return None if n is None else n > 0


def ptsd_decrease(a: dict, record: dict, ix: Index) -> bool | None:
    """The direction the meta-analysis pooled: "non-PTSD > PTSD", less grey matter in PTSD.

    The PTSD cohort's cell is negative, or the comparison cohort's is positive, or a PTSD
    severity slope is negative.
    """
    seen = False
    for cell in _cell_terms(a):
        direction = " ".join(strs(cell.get("direction")))
        if direction not in ("positive", "negative"):
            continue
        term = next((ix.terms.get(t) for t in refs(cell.get("term")) if t in ix.terms), None)
        level = " ".join(strs(cell.get("level")))
        groups = []
        if term is not None:
            for lv in term.get("levels") or []:
                if isinstance(lv, dict) and level and level in strs(lv.get("level")):
                    groups += refs(lv.get("groups"))
        verdicts = {ix.is_case(g) for g in groups}
        case, comparison = ix.spec.case, ix.spec.comparison
        if True in verdicts or (not groups and case.search(level) and not comparison.search(level)):
            seen = True
            if direction == "negative":
                return True
        elif False in verdicts or (not groups and comparison.search(level)):
            seen = True
            if direction == "positive":
                return True
        elif (ix.spec.severity is not None and term is not None
              and " ".join(strs(term.get("type"))) == "continuous"):
            seen = True
            if direction == "negative":
                return True
    return False if seen else None


PTSD_ANALYSIS: list[tuple[str, Callable]] = [
    ("structural MRI", structural),
    ("grey matter, voxel-wise", grey_voxelwise),
    ("whole brain", whole_brain),
    ("PTSD effect", ptsd_effect),
    ("reported foci", reported_foci),
]

#: Not a criterion for the study, a criterion for WHICH of its analyses is pooled.
PTSD_POOLED = [("PTSD decrease", ptsd_decrease)]


#: What a selected analysis must positively be. Everything else vetoes only on False.
PTSD_REQUIRED = ("structural MRI", "grey matter, voxel-wise", "PTSD effect")


def evaluate(record: dict, meta: str = "36100907") -> dict:
    """Per-criterion answers, the qualifying analyses, and the strict/permissive verdicts."""
    spec = SPECS[meta]
    ix = Index(record, spec)
    STUDY, ANALYSIS, POOLED, REQUIRED = spec.study, spec.analysis, spec.pooled, spec.required
    study = {name: f(record, ix) for name, f in STUDY}
    analyses = []
    for a in record.get("analyses") or []:
        if not isinstance(a, dict):
            continue
        answers = {name: f(a, record, ix) for name, f in ANALYSIS}
        pooled = all(f(a, record, ix) is True for _, f in POOLED)
        analyses.append({"local_id": a.get("local_id"), "name": " ".join(strs(a.get("name"))),
                         "answers": answers, "pooled": pooled,
                         "source": refs(a.get("source_table_analysis"))})
    strict_hits = [x for x in analyses if all(v is True for v in x["answers"].values())]
    # veto: the defining criteria must be answered yes; the rest exclude only on a no
    veto_hits = [x for x in analyses
                 if all(x["answers"][k] is True for k in REQUIRED)
                 and not any(v is False for v in x["answers"].values())]
    perm_hits = [x for x in analyses if not any(v is False for v in x["answers"].values())]
    # per analysis-criterion: the best answer any analysis gives (True > None > False)
    best = {}
    for name, _ in ANALYSIS:
        vals = [x["answers"][name] for x in analyses]
        best[name] = True if True in vals else (None if None in vals or not vals else False)
    return {
        "study": study, "analysis_best": best, "analyses": analyses,
        "strict": all(v is True for v in study.values()) and bool(strict_hits),
        "permissive": not any(v is False for v in study.values()) and bool(perm_hits or not analyses),
        "veto": not any(v is False for v in study.values()) and bool(veto_hits),
        "hits": [x["local_id"] for x in strict_hits],
        "veto_hits": [x["local_id"] for x in veto_hits],
        "pooled_hits": [x["local_id"] for x in veto_hits if x["pooled"]],
    }


# ------------------------------------------------------- dementia (Kamalian 2022, 35664889)
#
# Written from the criteria text alone, before any dementia record was read, so the first
# score is a held-out number. Quoted from neurometabench's meta_datasets.csv:
#
#   (1) clinically diagnosed bvFTD patients with no concurrent psychiatric diagnosis (e.g.,
#   major depressive disorder and bipolar mood disorder), other forms of dementia or
#   neurological symptoms, and no history of alcohol and substance abuse; (2) at least six
#   participants in either the patient or healthy group; (3) VBM, fMRI (resting-state or
#   task based), and FDG-PET as the imaging modality; (4) reported the coordinates of
#   between-group contrasts in a defined stochastic space (MNI or Talairach); and (5)
#   performed a whole-brain analysis.  Exclusion: 1) ROI 2) No Control.  dates: - 5/2020.
#   Analysis: bvFTD < HC.

BVFTD = re.compile(r"bvftd|behaviou?ral[- ]variant|frontotemporal dementia|\bftd\b|"
                   r"frontal[- ]variant|pick'?s", re.I)
HEALTHY_CONTROL = re.compile(r"healthy|\bcontrols?\b|\bhcs?\b|\bncs?\b|normal|"
                             r"cognitively (normal|unimpaired|intact)|non.?demented", re.I)
COMORBID = re.compile(r"depress|bipolar|schizophren|psychos|alzheimer|\bad\b|lewy|"
                      r"vascular dementia|\bals\b|amyotrophic|motor neuron|\bmnd\b|parkinson|"
                      r"progressive supranuclear|corticobasal|huntington|alcohol|substance|"
                      r"semantic dementia|primary progressive aphasia|\bppa\b|svppa|nfvppa", re.I)


def bvftd_without_comorbidity(record: dict, ix: Index) -> bool | None:
    """(1): the bvFTD cohort carries no other diagnosis. Read off the case cohorts' own
    `medical_condition` entries, each one separately: "bvFTD" is clean, "bvFTD with
    motor neuron disease" is not."""
    cases = [g for gid, g in ix.groups.items() if ix.is_case(gid)]
    if not cases:
        return None
    entries = [x for g in cases for x in strs(g.get("medical_condition"))]
    if not entries:
        return None
    return not any(COMORBID.search(e) for e in entries)


def at_least_six(record: dict, ix: Index) -> bool | None:
    """(2): "at least six participants in either the patient or healthy group"."""
    sizes = []
    for gid, g in ix.groups.items():
        if ix.is_case(gid) is None:
            continue
        n = num(g.get("acquired_count"))
        if n is None:
            n = num(g.get("enrolled_count"))
        if n is not None:
            sizes.append(n)
    return max(sizes) >= 6 if sizes else None


GREY_TYPES = {"gray_matter_volume", "gray_matter_density"}
BOLD_TYPES = {"bold_response", "bold_derived_metric", "connectivity_strength",
              "component_loading"}
PET_TYPES = {"metabolic_rate", "tracer_uptake", "receptor_binding"}
FDG = re.compile(r"fdg|fluoro.?deoxy|glucose", re.I)
OTHER_TRACER = re.compile(r"amyloid|pib|florbetapir|florbetaben|flutemetamol|tau|av.?1451|"
                          r"dopamin|raclopride|flumazenil|hmpao|ecd|spect", re.I)


def vbm_fmri_or_fdg(a: dict, record: dict, ix: Index) -> bool | None:
    """(3): VBM, fMRI (resting-state or task) or FDG-PET."""
    m = next((ix.measures[t] for t in refs(a.get("measure")) if t in ix.measures), None)
    kinds = set(strs(m.get("type"))) if m else set()
    label = " ".join(strs(m.get("source_label")) + strs(m.get("specific_metric"))) if m else ""
    acqs = [ix.acqs.get(x) for x in refs(a.get("acquisitions"))]
    acqs = [x for x in acqs if x] or list(ix.acqs.values())
    mods = {x for acq in acqs for x in strs(acq.get("modality"))}
    acq_text = " ".join(x for acq in acqs for x in strs(acq.get("acquisition_type")))
    if not kinds and not mods:
        return None
    # VBM's quantity is often typed `structural_morphometry_other` with the label
    # "grey matter intensity" (26401935's bvFTD-vs-controls atrophy analysis), so the
    # label decides as well as the type -- as `grey_voxelwise` already does for PTSD.
    if kinds & GREY_TYPES or (GREY.search(label) and not kinds & (BOLD_TYPES | PET_TYPES)):
        model = next((ix.models[t] for t in refs(a.get("model_estimation")) if t in ix.models), None)
        unit = strs(model.get("spatial_unit")) if model else []
        return not unit or "voxel" in unit
    if kinds & BOLD_TYPES:
        return not mods or bool(mods & {"fMRI", "MRI"})
    if kinds & PET_TYPES or "PET" in mods or "SPECT" in mods:
        text = f"{label} {acq_text}"
        if FDG.search(text):
            return True
        if OTHER_TRACER.search(text) or "SPECT" in mods:
            return False
        return None
    return False if kinds else None


DEMENTIA_STUDY: list[tuple[str, Callable]] = [
    ("original research", original),
    ("by May 2020", search_window),
    ("bvFTD cohort", case_cohort),
    ("no comorbidity", bvftd_without_comorbidity),
    ("a group of six", at_least_six),
    ("reports coordinates", reports_coordinates),
]
DEMENTIA_ANALYSIS: list[tuple[str, Callable]] = [
    ("VBM, fMRI or FDG-PET", vbm_fmri_or_fdg),
    ("whole brain", whole_brain),
    ("bvFTD vs control", ptsd_effect),
    ("reported foci", reported_foci),
]
DEMENTIA_POOLED = [("bvFTD < HC", ptsd_decrease)]
DEMENTIA_REQUIRED = ("VBM, fMRI or FDG-PET", "bvFTD vs control")


# ------------------------------------------------------------------ the registry

@dataclass(frozen=True)
class Spec:
    """One meta-analysis's criteria. `case`/`comparison` decide a cohort's side."""

    case: re.Pattern
    comparison: re.Pattern
    severity: re.Pattern | None
    window: tuple
    study: list
    analysis: list
    required: tuple
    pooled: list

    def status(self, group: dict) -> bool | None:
        return cohort_status(group, self.case, self.comparison)


SPECS: dict[str, Spec] = {
    "36100907": Spec(PTSD, NOT_PTSD, SEVERITY, (2002, 2020), PTSD_STUDY, PTSD_ANALYSIS,
                     PTSD_REQUIRED, PTSD_POOLED),
    "35664889": Spec(BVFTD, HEALTHY_CONTROL, None, (1900, (2020, 5)), DEMENTIA_STUDY,
                     DEMENTIA_ANALYSIS, DEMENTIA_REQUIRED, DEMENTIA_POOLED),
}
