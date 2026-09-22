#!/usr/bin/env python3
"""Which analysis in a record is the contrast a published map pooled?

Screening picks papers; a coordinate meta-analysis pools *contrasts*, and the benchmark
names them: "non-PTSD > PTSD", "bvFTD < HC (all & by modality)", "drug cue > neutral cue".
This is the one walk that answers that question, shared by the two scripts that ask it --
`query_analysis_selection.py`, which scores it against the annotation gold, and
`query_workflow.py`, which runs it behind the screening query to get a pooled map.

It was written twice before it was written here, which is the duplication
`docs/meta-analysis-queries.md` names as the largest simplification available: every one of
the sixteen published criteria asks for this walk and the schema makes each caller do it.

THE THIRD ANSWER IS THE POINT, as it is in the screening query. An analysis gets True,
False, or **None for "this record cannot say"**: no cells to read, a level that reaches no
entity, a cell with no direction. Strict selection takes the True ones and permissive takes
the None ones too, and the gap between them is the record's silence rather than the
query's judgement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

UNANSWERABLE = None

PTSD = r"PTSD|post.?traumatic"
FTD = r"bvFTD|frontotemporal|FTD"
SUBSTANCE = (r"alcohol|nicotine|tobacco|smok|cocaine|cannabis|opioid|heroin|"
             r"methamphetamine|substance|depend|abuse|addict")
#: Read off the gold analysis names under each cue key: drug is `A>N`, `CO>N`, `NI>N`,
#: `H>C`, `CAN>N`; natural is `F>N` and `S>N` and nothing else.
DRUG_CUE = (r"drug|alcohol|beer|wine|liquor|cocaine|nicotine|cigarett|smok|tobacco|heroin|"
            r"opiat|opioid|cannabis|marijuana|methamphetamine|stimulant|substance|craving")
NATURAL_CUE = (r"food|eat|appetit|palatab|sweet|chocolat|snack|meal|calor|"
               r"sex|erotic|porn|nude|romantic")


@dataclass(frozen=True)
class Contrast:
    """The contrast one published map pooled, as a record can state it.

    `cohort` is a between-group map -- the patient or user group, which by the convention
    of all four of these meta-analyses is the LOWER side. `cue` is a within-subject map --
    the condition that is the HIGHER side.

    Each names ONE side and the other side is whatever else the contrast resolved to. The
    cue keys are written `drug_neutral`, but their gold contains `CO>food`, `CA>nature`
    and `CA>food`: the map pools the cue against whatever the paper contrasted it with,
    so a neutral-cue lexicon would drop them.

    `signed=False` admits the contrast in either direction. It is set where the gold key
    does: dementia's `all`, `structural` and `functional` sets contain five analyses its
    `decrease` set does not, so those three keys are not direction-restricted and
    `decrease` is.
    """

    cohort: str | None = None
    cue: str | None = None
    family: str | None = None
    signed: bool = True


#: (project, the annotation's own key) -> the contrast it names. The keys are the
#: `note_keys` of `nimads/<project>/merged/nimads_annotation.json`.
KEYS: dict[tuple[str, str], Contrast] = {
    ("vbm_of_ptsd", "nonptsdgtptsd_merged"): Contrast(cohort=PTSD),
    ("dementia", "all"): Contrast(cohort=FTD, signed=False),
    ("dementia", "decrease"): Contrast(cohort=FTD),
    ("dementia", "structural"): Contrast(cohort=FTD, family=r"morphometry|structural",
                                         signed=False),
    ("dementia", "functional"): Contrast(
        cohort=FTD, family=r"bold|functional|perfusion|molecular", signed=False),
    ("vbm_of_substance_use", "all_drug_classes"): Contrast(cohort=SUBSTANCE),
    # The `_wbonly` half of each cue key, because that is the half the project's own
    # `nmb_mappings.json` points autonima's annotations at, and scoring the two selectors
    # against different gold would make the comparison meaningless. It is also the half
    # the criterion describes: ROI studies are excluded.
    #
    # `1_reward` is exactly `2_drug` union `3_natural` in the annotation, so it is written
    # as that union rather than as a third vocabulary.
    ("cue_reactivity", "1_reward_neutral_2020_wbonly"): Contrast(
        cue=f"{DRUG_CUE}|{NATURAL_CUE}"),
    ("cue_reactivity", "2_drug_neutral_2020_wbonly"): Contrast(cue=DRUG_CUE),
    ("cue_reactivity", "3_natural_neutral_2020_wbonly"): Contrast(cue=NATURAL_CUE),
}


def read(node: object) -> object:
    """An `ExtractedValue`'s value, or None where the record did not extract one."""
    if not isinstance(node, Mapping) or "extraction_status" not in node:
        return node
    return node.get("value") if node.get("extraction_status") == "extracted" else None


def texts(node: object) -> list[str]:
    value = read(node)
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(x) for x in items if str(x).strip()]


def ids(node: object) -> list[str]:
    value = read(node)
    items = value if isinstance(value, list) else [value]
    return [x for x in items if isinstance(x, str) and x]


def _conditions(record: Mapping[str, Any]) -> dict[str, str]:
    """`Condition.local_id` -> the words it can be recognised by."""
    out: dict[str, str] = {}
    for task in record.get("tasks") or []:
        if not isinstance(task, Mapping):
            continue
        for condition in task.get("conditions") or []:
            if isinstance(condition, Mapping) and isinstance(condition.get("local_id"), str):
                out[condition["local_id"]] = " ".join(
                    texts(condition.get("name")) + texts(condition.get("description")))
    return out


def _levels(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """`<term>|<level>` -> what that level names: its own label, its groups, its conditions.

    The key is the one a `Cell` writes, and the fold is exact: `spans.fold_label` already
    argues that `AD` and `AD group` are not the same level and that calling them equal
    hides a join failure rather than fixing one.
    """
    out: dict[str, dict[str, Any]] = {}
    for model in record.get("model_estimations") or []:
        if not isinstance(model, Mapping):
            continue
        for term in model.get("terms") or []:
            if not isinstance(term, Mapping):
                continue
            for level in term.get("levels") or []:
                if not isinstance(level, Mapping):
                    continue
                name = read(level.get("level"))
                if not isinstance(name, str):
                    continue
                slot = out.setdefault(f"{term.get('local_id')}|{name}",
                                      {"label": name, "groups": set(), "conditions": set()})
                slot["groups"].update(ids(level.get("groups")))
                slot["conditions"].update(ids(level.get("conditions")))
    return out


def _measure_blurb(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        m["local_id"]: " ".join(texts(m.get("family")) + texts(m.get("type")))
        for m in record.get("measures") or []
        if isinstance(m, Mapping) and isinstance(m.get("local_id"), str)
    }


def _cells(analysis: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    effect = analysis.get("effect")
    if not isinstance(effect, Mapping):
        return []
    return [c for c in effect.get("cells") or [] if isinstance(c, Mapping)]


def judge_analyses(record: Mapping[str, Any], spec: Contrast) -> dict[str, bool | None]:
    """`Analysis.local_id` -> True, False, or None for "the record cannot say"."""

    groups = {
        g["local_id"]: " ".join(texts(g.get("name")) + texts(g.get("medical_condition")))
        for g in record.get("groups") or []
        if isinstance(g, Mapping) and isinstance(g.get("local_id"), str)
    }
    conditions = _conditions(record)
    levels = _levels(record)
    blurbs = _measure_blurb(record)
    analyses = [a for a in record.get("analyses") or [] if isinstance(a, Mapping)]

    #: The cohort half of the join, decided once per record. No group carrying any text is
    #: "cannot say"; groups that carry text and none of it the cohort's is a No.
    cohort_side: set[str] = set()
    cohort_answer: bool | None = True
    if spec.cohort is not None:
        if not any(blurb.strip() for blurb in groups.values()):
            cohort_answer = UNANSWERABLE
        else:
            cohort_side = {gid for gid, blurb in groups.items()
                           if re.search(spec.cohort, blurb, re.I)}
            cohort_answer = bool(cohort_side)

    def side_of(cell: Mapping[str, Any]) -> tuple[str | None, bool]:
        """Which side of the contrast this cell is on, and whether it resolved at all.

        `(side, resolved)`: `cohort` or `cue` is the side the map names, `other` is
        whatever it is contrasted against, and None is a level that reaches no entity
        and no words -- a join failure, which is not the same as the other side.
        """
        name = read(cell.get("level"))
        key = f"{cell.get('term')}|{name}" if isinstance(name, str) else ""
        level = levels.get(key)
        if level is None:
            return None, False
        if spec.cohort is not None:
            reached = level["groups"]
            if not reached:
                return None, False
            return ("cohort" if reached & cohort_side else "other"), True
        if level["groups"] and not level["conditions"]:
            # A cohort level, on a term a cue map does not read. It resolved -- the record
            # is not silent -- it is just not a side of this contrast: without this,
            # "cocaine users > comparison subjects" is selected for the drug-cue map
            # because the level is spelled with the word "cocaine".
            return None, True
        blurb = " ".join([level["label"]]
                         + [conditions.get(c, "") for c in level["conditions"]])
        if not blurb.strip():
            return None, False
        return ("cue" if re.search(spec.cue or "", blurb, re.I) else "other"), True

    high, low = ("other", "cohort") if spec.cohort is not None else ("cue", "other")

    verdicts: dict[str, bool | None] = {}
    for analysis in analyses:
        local_id = str(analysis.get("local_id"))

        if spec.family is not None:
            blurb = blurbs.get(str(analysis.get("measure") or ""), "")
            if not blurb.strip():
                verdicts[local_id] = UNANSWERABLE
                continue
            if not re.search(spec.family, blurb, re.I):
                verdicts[local_id] = False
                continue

        if cohort_answer is not True:
            verdicts[local_id] = False if cohort_answer is False else UNANSWERABLE
            continue

        cells = _cells(analysis)
        if not cells:
            verdicts[local_id] = UNANSWERABLE
            continue

        seen: dict[tuple[str, str], bool] = {}
        resolved = False
        unsigned = False
        for cell in cells:
            side, ok = side_of(cell)
            resolved = resolved or ok
            direction = str(read(cell.get("direction")) or "").lower()
            if direction not in ("positive", "negative"):
                unsigned = unsigned or ok
                continue
            if side is not None:
                seen[(side, direction)] = True

        if not resolved:
            verdicts[local_id] = UNANSWERABLE
            continue
        as_published = seen.get((high, "positive")) and seen.get((low, "negative"))
        reversed_ = seen.get((low, "positive")) and seen.get((high, "negative"))
        if as_published or (not spec.signed and reversed_):
            verdicts[local_id] = True
        elif unsigned:
            verdicts[local_id] = UNANSWERABLE
        else:
            verdicts[local_id] = False
    return verdicts


def selected_analyses(record: Mapping[str, Any], spec: Contrast,
                      permissive: bool = False) -> list[str]:
    """The analyses the query pools: the Trues, and under `permissive` the silences too."""
    verdicts = judge_analyses(record, spec)
    return [k for k, v in verdicts.items()
            if v is True or (permissive and v is UNANSWERABLE)]
