#!/usr/bin/env python3
"""Can the records answer the queries a meta-analyst actually runs?

Each published meta-analysis in neurometabench states its inclusion and exclusion criteria in
prose. This decomposes them into deterministic predicates over the extraction schema, runs
them against the records, and scores the result against the benchmark's own included set.

    python scripts/query_metaanalyses.py --records '<dir>/*/*.extraction.json' \
        --bench <neurometabench>/data

THE POINT IS THE THIRD ANSWER. A predicate returns True, False, or **None for "this record
cannot answer"**, and the difference between the last two is the whole measurement: a paper
excluded because it used an ROI is a correct exclusion, and a paper excluded because no
analysis says what its scope was is a hole in the record. Collapsed into a boolean -- which
is what a SQL `WHERE` clause does -- the two are one number and the schema looks like it
works.

Criteria are quoted from `meta_datasets.csv` in the docstring of each query so a reader can
check the translation rather than trust it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

UNANSWERABLE = None

MNI_TALAIRACH = re.compile(r"mni|talairach|tal\b|icbm|mni152|asym", re.I)
ROI_WORDS = re.compile(r"\broi\b|region of interest|small volume|\bsvc\b", re.I)


def read(node: object) -> object:
    """Unwrap an ExtractedValue the way a consumer would, without importing pondie."""
    if not isinstance(node, dict):
        return node
    if "extraction_status" not in node:
        return node
    if node.get("extraction_status") != "extracted":
        return UNANSWERABLE
    return node.get("value")


#: Set by --literal. A query written against the schema compares a SCALAR slot to a string;
#: flattening a one-item list is a kindness this harness does and a `WHERE` clause does not.
#: Toggling it measures what the enum-list defect costs a real query.
#:
#: Applied only where the storage schema declares the slot scalar. A list in
#: `Group.medical_condition` is the slot working as declared, and treating it as broken
#: would attribute a defect to a field that has none -- which an earlier version of this
#: harness did, making every dementia query unanswerable for the wrong reason.
LITERAL = False
_SCALAR_SLOTS: set[str] = set()


def strings(node: object, slot: str = "") -> list[str]:
    value = read(node)
    if value is UNANSWERABLE or value is None:
        return []
    if LITERAL and isinstance(value, list) and slot in _SCALAR_SLOTS:
        # A scalar slot holding a list matches nothing, which is the defect, not an error.
        return []
    items = value if isinstance(value, list) else [value]
    return [str(x) for x in items if isinstance(x, (str, int, float)) and str(x).strip()]


def scalar_slots() -> set[str]:
    """`Class.slot` names the storage schema declares single-valued."""
    from pondie import schema
    from pondie.schema import reader

    storage = reader.load(schema.STORAGE)
    out = set()
    for cls in ("Analysis", "Acquisition", "MRI", "PET", "EEG", "Task", "Group", "Measure",
                "InferenceSettings", "ModelEstimation", "Cell", "ModelTerm", "Region"):
        for name, attribute in (storage.attributes(cls) or {}).items():
            if not attribute.multivalued:
                out.add(f"{cls}.{name}")
                out.add(name)
    return out


def number(node: object) -> float | None:
    value = read(node)
    if isinstance(value, list):
        value = value[0] if value else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# ---------------------------------------------------------------- predicates

def any_modality(record: dict, pattern: str) -> bool | None:
    """`modality: fMRI` / `Structural` / `PET`. Reads `Acquisition.modality`."""
    seen = [s for a in record.get("acquisitions") or [] if isinstance(a, dict)
            for s in strings(a.get("modality"), "modality")]
    if not seen:
        return UNANSWERABLE
    return any(re.search(pattern, s, re.I) for s in seen)


def whole_brain(record: dict) -> bool | None:
    """"only studies reporting whole-brain analysis outcomes ... as ROI analyses violate the
    ALE null-hypothesis" -- every meta-analysis here states it. `Analysis.spatial_scope`."""
    seen = [s for a in record.get("analyses") or [] if isinstance(a, dict)
            for s in strings(a.get("spatial_scope"), "spatial_scope")]
    if not seen:
        return UNANSWERABLE
    return any(s.lower() == "whole_brain" for s in seen)


def standard_space(record: dict) -> bool | None:
    """"reporting foci as 3D coordinates (X, Y, Z) in Talairach or Montreal Neurological
    Institute (MNI) stereotaxic space". `Analysis.coordinate_space`."""
    seen = [s for a in record.get("analyses") or [] if isinstance(a, dict)
            for s in strings(a.get("coordinate_space"), "coordinate_space")]
    if not seen:
        return UNANSWERABLE
    return any(MNI_TALAIRACH.search(s) for s in seen)


def between_group(record: dict) -> bool | None:
    """"a between-subjects contrast comparing smokers to matched nonsmoking participants".

    Two cohorts on one term's levels, which is the only encoding a query can traverse.
    """
    answerable = False
    for model in record.get("model_estimations") or []:
        if not isinstance(model, dict):
            continue
        for term in model.get("terms") or []:
            if not isinstance(term, dict):
                continue
            levels = [lv for lv in (term.get("levels") or []) if isinstance(lv, dict)]
            if len(levels) < 2:
                continue
            answerable = True
            if sum(1 for lv in levels if strings(lv.get("groups"))) >= 2:
                return True
    return False if answerable else UNANSWERABLE


def group_condition(record: dict, pattern: str) -> bool | None:
    """"included clinically diagnosed bvFTD patients" / "substance users and controls".
    `Group.medical_condition`."""
    seen = [s for g in record.get("groups") or [] if isinstance(g, dict)
            for s in strings(g.get("medical_condition"))]
    if not seen:
        return UNANSWERABLE
    return any(re.search(pattern, s, re.I) for s in seen)


def a_healthy_cohort(record: dict) -> bool | None:
    """"healthy adults with no prior report of neurological, medical, or psychiatric
    disorders ... Articles including patients were only selected if they reported results
    for a control group separately, and only the latter group was included here."

    ANY cohort, not every cohort. An earlier version of this predicate required all of them
    and vetoed 16 of emotion regulation's 87 gold papers -- a mistranslation, not a record
    defect: the criterion explicitly admits a paper with patients so long as a healthy group
    is reported separately, which is most of this literature. Derived from
    `medical_condition` by `pondie.normalization.is_healthy`.
    """
    from pondie.normalization.is_healthy import derive

    groups = [g for g in record.get("groups") or [] if isinstance(g, dict)]
    if not groups:
        return UNANSWERABLE
    verdicts = [derive(g) for g in groups]
    if any(v is True for v in verdicts):
        return True
    return UNANSWERABLE if all(v is None for v in verdicts) else False


def adults(record: dict, floor: float = 18.0) -> bool | None:
    """"age 18-60" / "studies in children/adolescent (<18 y)" excluded.
    `Group.age_minimum`, falling back to `age_mean`."""
    answered = []
    for group in record.get("groups") or []:
        if not isinstance(group, dict):
            continue
        low = number(group.get("age_minimum"))
        if low is None:
            low = number(group.get("age_mean"))
        if low is not None:
            answered.append(low >= floor)
    if not answered:
        return UNANSWERABLE
    return all(answered)


def min_group_size(record: dict, floor: int) -> bool | None:
    """"included at least six participants in either the patient or healthy group" /
    "studies with less than seven subjects" excluded. `Group.acquired_count`."""
    counts = [number(g.get("acquired_count")) for g in record.get("groups") or []
              if isinstance(g, dict)]
    counts = [c for c in counts if c is not None]
    if not counts:
        return UNANSWERABLE
    return max(counts) >= floor


def visual_stimuli(record: dict) -> bool | None:
    """"cue-reactivity using visual stimuli ... other sensory cues (e.g., gustatory,
    olfactory, tactile) were not considered". `Task.stimulus_modality` where present,
    otherwise the prose in `Task.stimuli`."""
    modalities = [s for t in record.get("tasks") or [] if isinstance(t, dict)
                  for s in strings(t.get("stimulus_modality"))]
    if modalities:
        return any(re.search(r"visual", s, re.I) for s in modalities)
    prose = [s for t in record.get("tasks") or [] if isinstance(t, dict)
             for s in strings(t.get("stimuli")) + strings(t.get("description"))]
    if not prose:
        return UNANSWERABLE
    return any(re.search(r"visual|image|picture|photograph|video|film", s, re.I) for s in prose)


def has_direction(record: dict) -> bool | None:
    """"users < non-users ; users > non-users" / "bvFTD < HC" -- the contrast has a sign.
    `Cell.direction`."""
    seen = [s for a in record.get("analyses") or [] if isinstance(a, dict)
            for c in ((a.get("effect") or {}).get("cells") or []) if isinstance(c, dict)
            for s in strings(c.get("direction"), "direction")]
    if not seen:
        return UNANSWERABLE
    return any(s.lower() in ("positive", "negative") for s in seen)


def no_pharmacological(record: dict) -> bool | None:
    """"presence of pharmacological manipulations" excluded. `StudyDesign.allocation`.

    Read off `allocation` and not off whether an Arm is declared. An earlier version used
    arm presence and vetoed 7 of substance use's gold against 12 non-gold -- barely better
    than chance -- because the records invent Arms for diagnostic cohorts: 47 of the 164
    `parallel`-with-arms records have every arm name identical to a cohort name. That is the
    defect `AssignmentStructure.observational_cohorts` was added for, and it reaches a query
    here. `allocation: not_applicable` is glossed "Nothing was administered", which is the
    criterion itself.
    """
    allocation = strings((record.get("design") or {}).get("allocation"), "allocation")
    if not allocation:
        return UNANSWERABLE
    return all(a.lower() in ("not_applicable", "single_arm") for a in allocation)


def whole_brain_correction(record: dict) -> bool | None:
    """"we excluded studies using region of interest (ROI) or small volume correction (SVC)".
    `InferenceSettings.correction_scope` -- the correction's own domain, not the analysis's."""
    seen = [s for i in record.get("inference_settings") or [] if isinstance(i, dict)
            for s in strings(i.get("correction_scope"), "correction_scope")]
    if not seen:
        return UNANSWERABLE
    return any(s.lower() == "whole_brain" or not ROI_WORDS.search(s) for s in seen)


Predicate = Callable[[dict], "bool | None"]

#: meta_pmid -> (project directory, [(name, predicate)]). The criteria each one is
#: translated from are quoted in the predicate docstrings above.
QUERIES: dict[str, tuple[str, list[tuple[str, Predicate]]]] = {
    "36100907": ("vbm_of_ptsd", [
        ("structural modality", lambda r: any_modality(r, r"structural|vbm|smri|\bMRI\b|T1")),
        ("whole brain", whole_brain),
        ("standard space", standard_space),
        ("PTSD cohort", lambda r: group_condition(r, r"PTSD|post.?traumatic")),
        ("between-group contrast", between_group),
        ("adults", lambda r: adults(r, 18)),
        ("signed contrast", has_direction),
    ]),
    "35664889": ("dementia", [
        ("fMRI, structural or PET", lambda r: any_modality(r, r"fMRI|structural|VBM|PET|MRI")),
        ("whole brain", whole_brain),
        ("standard space", standard_space),
        ("bvFTD cohort", lambda r: group_condition(r, r"bvFTD|frontotemporal|FTD")),
        ("a group of at least six", lambda r: min_group_size(r, 6)),
        ("between-group contrast", between_group),
        ("signed contrast", has_direction),
    ]),
    "34400176": ("cue_reactivity", [
        ("fMRI", lambda r: any_modality(r, r"fMRI|functional")),
        ("whole brain", whole_brain),
        ("standard space", standard_space),
        ("visual cues", visual_stimuli),
        ("whole-brain correction", whole_brain_correction),
    ]),
    "36115222": ("vbm_of_substance_use", [
        ("structural modality", lambda r: any_modality(r, r"structural|vbm|smri|\bMRI\b|T1")),
        ("whole brain", whole_brain),
        ("standard space", standard_space),
        ("substance-use cohort", lambda r: group_condition(
            r, r"alcohol|nicotine|tobacco|smok|cocaine|cannabis|opioid|heroin|"
               r"methamphetamine|substance|depend|abuse|addict")),
        ("between-group contrast", between_group),
        ("no pharmacological arm", no_pharmacological),
    ]),
    "35413444": ("emotion_regulation_2022", [
        ("fMRI", lambda r: any_modality(r, r"fMRI|functional")),
        ("whole brain", whole_brain),
        ("standard space", standard_space),
        ("a healthy cohort", a_healthy_cohort),
        ("adults", lambda r: adults(r, 18)),
    ]),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--repair", action="store_true",
                    help="apply the deterministic repairs first, to measure what they buy "
                         "a query rather than what they buy a check")
    ap.add_argument("--literal", action="store_true",
                    help="compare an enum slot as a scalar, the way a SQL WHERE clause "
                         "would, instead of flattening a one-item list")
    args = ap.parse_args()

    global LITERAL, _SCALAR_SLOTS
    LITERAL = args.literal
    if LITERAL:
        _SCALAR_SLOTS = scalar_slots()

    gold: dict[str, set[str]] = defaultdict(set)
    with (args.bench / "included_studies.csv").open() as fh:
        for row in csv.DictReader(fh):
            gold[row["meta_pmid"]].add(row["study_pmid"])

    records: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(args.records)):
        body = json.loads(Path(path).read_text())
        body = body.get("study") or body
        if args.repair:
            from pondie import schema
            from pondie.extraction.record import builder as br
            from pondie.schema import reader
            global _SCH
            try:
                _SCH
            except NameError:
                _SCH = reader.load(schema.EXTRACTION)
            br.unwrap_singleton_lists(body, _SCH)
            br.link_entities_by_name(body, _SCH)
            br.drop_redundant_cell_levels(body)
        records[Path(path).parent.name][Path(path).name.split(".")[0]] = body

    for meta_pmid, (project, predicates) in QUERIES.items():
        here = records.get(project) or {}
        if not here:
            continue
        want = gold[meta_pmid] & set(here)
        print("=" * 84)
        print(f"{project}  (meta {meta_pmid})   {len(here)} records screened, "
              f"{len(want)} of the benchmark's {len(gold[meta_pmid])} included studies present")
        print(f"\n  {'predicate':28} {'true':>6} {'false':>6} {'CANNOT SAY':>11}   "
              f"{'true on gold':>13} {'unanswerable on gold':>21}")
        verdicts: dict[str, dict[str, bool | None]] = {}
        for name, predicate in predicates:
            counts: Counter = Counter()
            per_record = {}
            for pmid, body in here.items():
                try:
                    answer = predicate(body)
                except Exception:
                    answer = UNANSWERABLE
                per_record[pmid] = answer
                counts[answer] += 1
            verdicts[name] = per_record
            on_gold = [per_record[p] for p in want]
            print(f"  {name:28} {counts[True]:6d} {counts[False]:6d} "
                  f"{counts[UNANSWERABLE]:11d}   {sum(1 for a in on_gold if a is True):13d} "
                  f"{sum(1 for a in on_gold if a is UNANSWERABLE):21d}")

        # The query as a meta-analyst would write it, three ways.
        for label, treat_unknown in (("strict (unanswerable excludes)", False),
                                     ("permissive (unanswerable passes)", True)):
            selected = {
                pmid for pmid in here
                if all(
                    (verdicts[name][pmid] is True)
                    or (treat_unknown and verdicts[name][pmid] is UNANSWERABLE)
                    for name, _p in predicates
                )
            }
            hit = len(selected & want)
            recall = hit / len(want) if want else 0.0
            precision = hit / len(selected) if selected else 0.0
            print(f"\n  {label:34} selects {len(selected):4d}   recall {recall:5.1%}   "
                  f"precision {precision:5.1%}")
        lost = {
            pmid for pmid in want
            if any(verdicts[name][pmid] is UNANSWERABLE for name, _p in predicates)
        }
        wrong = {
            pmid for pmid in want
            if any(verdicts[name][pmid] is False for name, _p in predicates)
        } - lost
        print(f"  gold papers the record cannot answer for: {len(lost)}/{len(want)}"
              f"   contradicted by the record: {len(wrong)}/{len(want)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
