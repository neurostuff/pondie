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

    `whole_brain_only` is the key's own restriction, not a criterion of the review: cue
    reactivity's gold comes in a plain and a `_wbonly` flavour and the maps are built from
    the second, so the analyses pooled for it are the whole-brain ones.

    `signed=False` admits the contrast in either direction. It is set where the gold key
    does: dementia's `all`, `structural` and `functional` sets contain five analyses its
    `decrease` set does not, so those three keys are not direction-restricted and
    `decrease` is.
    """

    cohort: str | None = None
    cue: str | None = None
    family: str | None = None
    signed: bool = True
    whole_brain_only: bool = False


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
    # The meta-analysis publishes a map per drug class as well as the pooled one -- "1) All
    # substances b) Alcohol c) Nicotine d) Alcohol - Nicotine" -- and the figures compare
    # arms column by column, so a key the query cannot answer drops that column for every
    # arm. Each is the same contrast over a narrower cohort.
    ("vbm_of_substance_use", "alcohol"): Contrast(
        cohort=r"alcohol|\bAUD\b|ethanol|drink|beer|wine|liquor"),
    ("vbm_of_substance_use", "nicotine"): Contrast(
        cohort=r"nicotine|tobacco|smok|cigarett|\bFTND\b"),
    ("vbm_of_substance_use", "opioids"): Contrast(
        cohort=r"opioid|opiate|heroin|morphine|methadone|buprenorphine|fentanyl|oxycod"),
    ("vbm_of_substance_use", "stimulants"): Contrast(
        cohort=r"cocaine|crack|amphetamine|methamphetamine|stimulant|\bMDMA\b|ecstasy"),
    ("vbm_of_substance_use", "cannabis"): Contrast(
        cohort=r"cannabis|marijuana|marihuana|\bTHC\b|cannabinoid"),
    # The `_wbonly` half of each cue key, because that is the half the project's own
    # `nmb_mappings.json` points autonima's annotations at, and scoring the two selectors
    # against different gold would make the comparison meaningless. It is also the half
    # the criterion describes: ROI studies are excluded.
    #
    # `1_reward` is exactly `2_drug` union `3_natural` in the annotation, so it is written
    # as that union rather than as a third vocabulary.
    ("cue_reactivity", "1_reward_neutral_2020_wbonly"): Contrast(
        cue=f"{DRUG_CUE}|{NATURAL_CUE}", whole_brain_only=True),
    ("cue_reactivity", "2_drug_neutral_2020_wbonly"): Contrast(
        cue=DRUG_CUE, whole_brain_only=True),
    ("cue_reactivity", "3_natural_neutral_2020_wbonly"): Contrast(
        cue=NATURAL_CUE, whole_brain_only=True),
}


#: Words that turn a cohort mention into its own control group. `non` covers the forms a
#: parser does not see, because they are morphology rather than syntax: `nonsmoking`,
#: `cannabis non-consuming`, `non-use of marijuana`.
_NEGATOR = re.compile(r"^(?:non[-\w]*|never[-\w]*|without|free|na[iï]ve|no|not)$", re.I)
#: What a negation reaches across on its way to the cohort word: function words, and the
#: nouns a clinical negation is built on -- "no HISTORY OF alcohol misuse".
_CARRIES = {"of", "for", "to", "with", "any", "the", "a", "an",
            "history", "current", "prior", "past", "use", "using", "usage"}
#: Hyphenated forms are one token: `non-use`, `cannabis-consuming`, `non-PTSD` each make
#: sense only whole, and splitting them put the negator out of reach of its own word.
_WORD = re.compile(r"[A-Za-z0-9\u00c0-\u024f'-]+")


def names_cohort(blurb: str, pattern: str) -> bool:
    """Does this group's own words put it IN the cohort, rather than opposite it?

    A control group is named after the condition it does not have -- "nonsmoking control
    subjects", "comparison group, non-use of marijuana", "cannabis non-consuming group",
    "healthy controls with no history of alcohol misuse". Matching the cohort pattern
    against those puts both sides of the contrast in the cohort, which leaves the contrast
    with no other side: 13 of the 15 gold papers under substance use's nicotine key were
    lost that way, and 4 more under cannabis.

    The window is one content token either side, reaching backwards across the function
    words and history nouns a clinical negation is built on and no further. Wider was
    measured and is wrong: "Chronic marijuana use; no current or history of psychosis"
    puts a `no` three tokens behind the cohort word and would negate the USER group.
    """
    tokens = list(_WORD.finditer(blurb))
    for match in re.finditer(pattern, blurb, re.I):
        inside = [i for i, t in enumerate(tokens)
                  if t.start() <= match.start() < t.end()]
        if not inside:
            continue
        i = inside[0]
        word = tokens[i].group()
        if _NEGATOR.match(word) and match.start() > tokens[i].start():
            continue                                  # `nonsmoking`, `nonusers`
        j = i - 1
        steps = 0
        while j >= 0 and tokens[j].group().lower() in _CARRIES and steps < 3:
            j -= 1
            steps += 1
        if j >= 0 and _NEGATOR.match(tokens[j].group()):
            continue                                  # `non-use of marijuana`
        if i + 1 < len(tokens) and _NEGATOR.match(tokens[i + 1].group()):
            continue                                  # `cannabis non-consuming group`
        return True
    return False


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
    """`Condition.local_id` -> its NAME, which is the field that names it.

    Not the description. A neutral condition is described by what it is not -- "pictures of
    people not smoking cigarettes", "nonalcoholic beverage pictures matched to the alcohol
    pictures" -- so a cue pattern run over the prose matches both sides of the contrast and
    the contrast loses its other side. That cost 58 of cue reactivity's 118 gold papers.
    """
    out: dict[str, str] = {}
    for task in record.get("tasks") or []:
        if not isinstance(task, Mapping):
            continue
        for condition in task.get("conditions") or []:
            if isinstance(condition, Mapping) and isinstance(condition.get("local_id"), str):
                out[condition["local_id"]] = " ".join(texts(condition.get("name")))
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
            # A control group is named after the condition it does not have: "nonsmoking
            # control subjects", "healthy controls, no history of alcohol misuse", "trauma-
            # exposed non-PTSD". Matched on words alone, both sides of the contrast land in
            # the cohort, the contrast has no other side and the paper is missed -- 13 of
            # the 15 gold papers under substance use's nicotine key went that way.
            # `is_healthy` for the healthy controls and `names_cohort` for the cohorts
            # named after the condition they do not have.
            #
            # These two and not a `Group.role` field. One was built, measured and
            # removed: a role is relative to the question, and a per-group enum cannot be.
            # 22445480 has `Control Smoker` beside `MA-dependent Smoker`, and that cohort
            # is the comparison for the methamphetamine map and the case for the nicotine
            # one. docs/normalization-rationale.md, "group_role - built, measured, removed".
            from pondie.normalization.is_healthy import derive

            entities = {g["local_id"]: g for g in record.get("groups") or []
                        if isinstance(g, Mapping) and isinstance(g.get("local_id"), str)}
            healthy = {gid: derive(g) for gid, g in entities.items()}
            cohort_side = {
                gid for gid, blurb in groups.items()
                if names_cohort(blurb, spec.cohort) and healthy.get(gid) is not True
            }
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
        if not level["conditions"] and not re.search(r"[A-Za-z]{3}", blurb):
            # A level spelled `H` against `O`, with the task's own `foods of high hedonic
            # value` and `neutral nonfood objects` sitting unlinked beside it. The label
            # reads as neither side, and reading it as the other side would report a
            # contradiction where the record is silent.
            return None, False
        # `names_cohort` for the same reason it is used on a group: "non-drug cue" and
        # "never-smoking" name the other side of the contrast, not this one.
        return ("cue" if names_cohort(blurb, spec.cue or "") else "other"), True

    high, low = ("other", "cohort") if spec.cohort is not None else ("cue", "other")

    verdicts: dict[str, bool | None] = {}
    for analysis in analyses:
        local_id = str(analysis.get("local_id"))

        if spec.whole_brain_only:
            scopes = [t.lower() for t in texts(analysis.get("spatial_scope"))]
            if not scopes:
                verdicts[local_id] = UNANSWERABLE
                continue
            if "whole_brain" not in scopes:
                verdicts[local_id] = False
                continue

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
        halves = [seen.get((high, "positive")), seen.get((low, "negative"))]
        other_way = [seen.get((low, "positive")), seen.get((high, "negative"))]
        if all(halves) or (not spec.signed and all(other_way)):
            verdicts[local_id] = True
        elif unsigned or any(halves) or (not spec.signed and any(other_way)):
            # Half a contrast is not the other contrast. A simple effect that carries the
            # cue side and no comparison -- "heroin-related, positive", where the paper's
            # table is a cue-minus-neutral map -- does not say what it was contrasted
            # against, and calling that False asserts something the record does not.
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
