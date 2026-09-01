#!/usr/bin/env python3
"""Select the records and coordinates a meta-analysis should pool, and say what it dropped.

Every meta-analysis in this repo re-implements the same funnel -- species, direction, a
placeable space, a joinable row group -- and each one loses papers somewhere different. This
is that funnel once, with the selection stated as a validated object rather than as a pile of
flags, so a run is reproducible from the object and a typo is an error rather than a filter
that silently matches nothing.

What it does NOT do is decide the science. `Selection` defaults to the choices a coordinate
meta-analysis usually wants -- whole-brain analyses only, human only, primary studies only --
and every one of them is overridable, because each is a judgement rather than a fact.

    from query_engine import Selection, select
    result = select(Selection(diagnosis="schizophrenia", measure_type={"gray_matter_volume"}))
    result.funnel()                 # where papers were lost
    studyset = result.to_studyset() # one analysis per study, ready for NiMARE

Boundary contract: `Selection` is pydantic, so an unknown field or a bad literal fails at
construction. The records themselves are read through `values.value_of`, which takes the
wrapper and the multivalued shape from the LinkML schema -- see
docs/pipeline-architecture.md#the-contract-at-each-seam.
"""

from __future__ import annotations

import glob as globlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pondie import paths
from pondie.formats import parse_keys
from pondie.formats.values import value_of
from pondie.normalization import UNKNOWN, contrasts, coordinate_space  # noqa: E402

#: The two sides of an allocated contrast, named here because `contrasts` speaks of
#: interventions and comparators and the rest of this module of active and control.
ACTIVE, CONTROL = "ACTIVE", "CONTROL"

SpatialScope = Literal["whole_brain", "roi", "searchlight", "other"]
Space = Literal["MNI", "TAL", "OTHER", "UNKNOWN"]
Contrast = Literal["any", "within_subject", "between_group"]


#: Which way an analysis's signal runs with respect to being treated.
Exposure = Literal["increase", "decrease"]


def _exposure_of(
    analysis: Mapping,
    body: Mapping[str, Any],
    arm_sides: Mapping[tuple[str, str], str],
    time_sides: Mapping[tuple[str, str], str],
) -> str | None:
    """`increase`, `decrease`, or None when the analysis is not a treatment contrast.

    Two routes to the same question, because this corpus asks it both ways. A trial
    compares an intervention arm against its comparator; a longitudinal study compares the
    same people before and after. `increase` means the signal is higher under or after
    treatment, whichever the design measured.

    They are not the same estimand and pooling them is a decision, not a detail: a
    between-arm difference controls for time and repetition and a within-subject change
    does not, so a pooled map answers "where does the brain differ around treatment"
    rather than "what does treatment do relative to placebo". The route is recorded on
    every row so the funnel can say how much of a result rests on which.
    """

    positive, negative = _signed_cells(analysis)
    if not (positive and negative):
        return None

    arms_pos = {arm_sides.get(k) for k in positive}
    arms_neg = {arm_sides.get(k) for k in negative}
    if arms_pos == {ACTIVE} and arms_neg == {CONTROL}:
        return "increase"
    if arms_pos == {CONTROL} and arms_neg == {ACTIVE}:
        return "decrease"

    times_pos = {time_sides.get(k) for k in positive}
    times_neg = {time_sides.get(k) for k in negative}
    if {frozenset(times_pos), frozenset(times_neg)} == {
        frozenset({"post_intervention"}),
        frozenset({"pre_intervention"}),
    }:
        # A before/after change measured inside the comparator arm is a placebo or
        # repetition effect wearing a treatment contrast's shape. Excluded only when every
        # cohort is a known control arm: an unallocated or mixed one is kept, because most
        # single-arm longitudinal studies declare no arms at all and dropping those would
        # discard the route's whole contribution.
        if _analysis_arms(analysis, body) <= {CONTROL} != set():
            return None
        return "increase" if times_pos == {"post_intervention"} else "decrease"
    return None


class Selection(BaseModel):
    """What to pool. Unknown fields are an error, not a silently ignored typo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[str, ...] = (str(paths.RUNS / "*" / "records" / "*.extraction.json"),)

    #: Paper level.
    exclude_meta_analyses: bool = True
    species: frozenset[str] | None = frozenset({"human"})

    #: Analysis level. `roi` is excluded by default: a region-restricted search can only
    #: report coordinates inside that region, and pooling it with whole-brain analyses
    #: inflates convergence exactly where studies chose to look.
    spatial_scope: frozenset[SpatialScope] = frozenset({"whole_brain"})
    measure_type: frozenset[str] | None = None
    space: frozenset[Space] = frozenset({"MNI", "TAL"})

    #: Contrast level. A patient-minus-control map is a difference between activations, not
    #: an activation, and convergence over the two is not interpretable as either.
    contrast: Contrast = "any"
    direction: frozenset[str] | None = None
    #: Keep only analyses whose signed cells run this way between an intervention arm and
    #: its comparator. `None` pools regardless of direction, which is what every field
    #: above does; set it and an analysis that does not assert a side is dropped.
    arm_contrast: ArmContrast | None = None
    #: Keep only analyses that contrast being treated against not being treated, by either
    #: route: an intervention arm against its comparator, or after against before. Wider
    #: than `arm_contrast`, which is the between-arm half alone.
    treatment_exposure: Exposure | None = None

    #: Entity level, against the normalized values.
    diagnosis: str | None = None
    task_family: str | None = None

    #: Reporting only -- a warning, never a filter.
    min_studies: int = Field(default=10, ge=1)

    @field_validator("records", mode="before")
    @classmethod
    def _one_or_many(cls, v):
        return (v,) if isinstance(v, str) else v


class Result:
    """The selected rows, the funnel that produced them, and a NiMARE dataset."""

    def __init__(
        self,
        selection: Selection,
        rows: list[dict],
        lost: Counter,
        kept_papers: set[str],
        seen_papers: set[str],
    ):
        self.selection, self.rows, self.lost = selection, rows, lost
        self.kept_papers, self.seen_papers = kept_papers, seen_papers

    @property
    def studies(self) -> set[str]:
        return {r["study"] for r in self.rows}

    def funnel(self) -> str:
        out = [
            f"{len(self.seen_papers)} papers read, {len(self.kept_papers)} contribute",
            f"{len(self.rows)} analyses selected from {len(self.studies)} studies, "
            f"{sum(len(r['points']) for r in self.rows)} foci",
        ]
        if self.lost:
            out.append("lost:")
            out += [f"   {n:5d}  {why}" for why, n in self.lost.most_common()]
        # Weighting provenance, because a pooled result rests on it. A row weighted on a
        # cohort total rather than an analysed count is over-weighted by however many
        # participants that analysis dropped, and nothing downstream can tell.
        inferred = sorted(
            {s for r in self.rows for s in r.get("n_source", ()) if s != "analysis"}
        )
        if inferred:
            rows = sum(
                1 for r in self.rows if any(s != "analysis" for s in r.get("n_source", ()))
            )
            out.append(
                f"NOTE: {rows} of {len(self.rows)} analyses are weighted on a cohort total "
                f"({', '.join(inferred)}) because no per-analysis n was reported; that is "
                f"an upper bound on the number actually analysed"
            )
        routes = Counter(r["route"] for r in self.rows if r.get("route"))
        if routes:
            out.append(
                "routes: "
                + ", ".join(
                    f"{n} {k} ({len({r['study'] for r in self.rows if r.get('route') == k})}"
                    f" studies)"
                    for k, n in routes.most_common()
                )
            )
        if len(self.studies) < self.selection.min_studies:
            out.append(
                f"WARNING: {len(self.studies)} studies is below min_studies="
                f"{self.selection.min_studies}; a coordinate meta-analysis over this "
                f"many converges on whichever paper reports the most foci"
            )
        return "\n".join(out)

    def poolable(self) -> list[dict]:
        """The selected rows as NIMADS studies, minus the ones that cannot be pooled.

        Separate from `to_studyset` because the two are different jobs and only one is
        ours. Deciding what may enter a meta-analysis is this package's call and is where
        the interesting mistake lives; turning NIMADS into a `Studyset` is NiMARE's, and
        testing that would be testing NiMARE.

        Splitting them also means the rule below is checked on every run of the suite
        rather than only where NiMARE is installed -- and this is the rule that was wrong.

        Only the keys NiMARE actually reads to pool coordinates are emitted: a study `id`,
        an analysis `id`, its `sample_sizes`, and its points. `conditions`, `weights`,
        `images` and `annotations` are all optional and none of them mean anything for a
        selection that carries no images and no contrast weights.
        """
        studies: dict[str, dict] = {}
        for row in self.rows:
            # A row with no extracted `AnalysisGroup.n` is dropped, not given one. NiMARE
            # weights studies by sample size, so substituting a number -- this said 30,
            # with no comment and no count -- changes the pooled result and says nothing.
            # Every other module in this package refuses exactly that: "a deriver that
            # guesses is worse than no deriver", "nothing is bucketed silently".
            if not row["n"]:
                self.lost["no sample size, so it cannot be weighted"] += 1
                continue
            study = row["study"]
            held = studies.setdefault(study, {"id": study, "analyses": []})["analyses"]
            held.append(
                {
                    "id": f"{study}-{len(held)}",
                    "metadata": {"sample_sizes": [row["n"]]},
                    "points": [
                        {"coordinates": [x, y, z], "space": "MNI"} for x, y, z in row["points"]
                    ],
                }
            )
        return list(studies.values())

    def to_studyset(self, target: str = "mni152_2mm"):
        """One analysis per study. `Studyset.combine_analyses()` does the pooling.

        A NIMADS `Studyset` rather than a `Dataset`: NiMARE deprecated `Dataset` for
        removal in 1.0, its estimators already accept either, and building one natively
        means this never went through the deprecated class at all. Anything that still
        wants the old object can call `.to_dataset()` on the result.
        """
        from nimare.nimads import Studyset

        return Studyset({"studies": self.poolable()}, target=target).combine_analyses()


def _texts(x) -> list[str]:
    """Every string in a value, whatever shape the slot declares."""
    v = value_of(x, True)
    return [s for s in (value_of(i) for i in v) if isinstance(s, str) and s.strip()]


#: A coordinate meta-analysis indexed as a paper. Its peaks are already a convergence over
#: primary studies, several of which are usually in the same corpus, so leaving it in counts
#: those samples twice and piles the double count on the loci under test.
META_ANALYSIS = re.compile(
    r"meta-?analy|activation likelihood estimation|\bALE\b|\bMKDA\b|"
    r"seed-based d mapping|\bSDM\b|coordinate-based",
    re.I,
)


def _is_meta_analysis(body: dict) -> bool:
    design = body.get("design") if isinstance(body.get("design"), dict) else {}
    blob = " ".join(
        [
            str(value_of(body.get("description")) or ""),
            str(value_of(design.get("description")) or ""),
            *(str(h) for h in (value_of(body.get("hypothesis"), True) or [])),
        ]
    )
    return bool(META_ANALYSIS.search(blob))


def _points(entry: dict, space: str) -> list[list[float]]:
    """Coordinates from one parsed row group, moved into MNI where the space says to."""
    import numpy as np
    from nimare.utils import tal2mni

    raw = [p.get("coordinates") for p in (entry.get("points") or [])]
    raw = [
        c
        for c in raw
        if isinstance(c, list) and len(c) == 3 and all(isinstance(v, (int, float)) for v in c)
    ]
    if not raw:
        return []
    if space == "TAL":
        return tal2mni(np.array(raw, dtype=float)).tolist()
    return raw


#: Which side of an allocated contrast an analysis sits on, for a treatment question.
ArmContrast = Literal["active_over_control", "control_over_active"]


#: `Arm.arm_kind` is a required slot over a closed vocabulary, and `contrasts.role` already
#: maps it. Both halves matter: reading the enum rather than the arm's name, and reusing the
#: mapping rather than writing a second one.
_ARM_ROLE = {"intervention": ACTIVE, "comparator": CONTROL}


def _arm_roles(body: Mapping[str, Any]) -> dict[str, str]:
    """`arm local_id -> ACTIVE | CONTROL | UNKNOWN`, from the arm's declared kind.

    From `arm_kind` and not from `name`. This read the name through a keyword lexicon
    first, which was a mistake with a measurable cost: over 125 arms the lexicon left 24
    unclassified that the enum classifies, and inverted 3 -- `placebo-ketamine` is a
    `pharmacological` arm whose name trips a `placebo` rule. An inverted arm does not
    weaken a pooled map, it puts the foci in the opposite one.

    The enum is populated on 193 of 193 arms across the corpora this was built against,
    with no `not_reported` among them, so there is nothing for a name to fall back to and
    no reason to guess. An arm whose kind does not map is UNKNOWN and its analysis is
    dropped, which is the same refusal the rest of this module makes.
    """

    return {
        arm.get("local_id"): _ARM_ROLE.get(
            contrasts.role(str(value_of(arm.get("arm_kind")))) or "", UNKNOWN
        )
        for arm in ((body.get("design") or {}).get("arms") or [])
        if isinstance(arm, Mapping) and arm.get("local_id")
    }


def _time_sides(body: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """`(term id, level) -> pre_intervention | post_intervention`.

    Read from `Timepoint.relation_to_intervention`, which the schema requires and which is
    populated on every timepoint in the corpus this was built against. Structure rather
    than a pattern over level names: `baseline`, `scan 1`, `T0` and `pre-injection` are the
    same pole and `week eight` is the other, and a regex over those is a guess the record
    does not need us to make.
    """

    relation = {
        t.get("local_id"): str(value_of(t.get("relation_to_intervention")))
        for t in ((body.get("design") or {}).get("timepoints") or [])
        if isinstance(t, Mapping)
    }
    sides: dict[tuple[str, str], str] = {}
    for model in body.get("model_estimations") or []:
        if not isinstance(model, Mapping):
            continue
        for term in model.get("terms") or []:
            if not isinstance(term, Mapping):
                continue
            for level in term.get("levels") or []:
                if not isinstance(level, Mapping):
                    continue
                poles = {relation.get(t) for t in (level.get("timepoints") or [])}
                # `during_intervention` and `single_occasion` are neither pole and are left
                # out rather than folded onto one.
                poles &= {"pre_intervention", "post_intervention"}
                if len(poles) == 1:
                    sides[(term.get("local_id"), str(value_of(level.get("level"))))] = (
                        poles.pop()
                    )
    return sides


def _analysis_arms(analysis: Mapping, body: Mapping[str, Any]) -> set[str]:
    """The arm roles of the cohorts an analysis ran on."""
    roles = _arm_roles(body)
    groups = {
        g.get("local_id"): g
        for g in (body.get("groups") or [])
        if isinstance(g, Mapping) and g.get("local_id")
    }
    return {
        roles.get((groups.get(link.get("group")) or {}).get("arm"))
        for link in (analysis.get("groups") or [])
        if isinstance(link, Mapping)
    }


def _signed_cells(analysis: Mapping) -> tuple[list, list]:
    """The `(term, level)` keys on each side of the contrast, positive first."""
    positive, negative = [], []
    for cell in (analysis.get("effect") or {}).get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        key = (cell.get("term"), str(value_of(cell.get("level"))))
        sign = value_of(cell.get("direction"))
        if sign == "positive":
            positive.append(key)
        elif sign == "negative":
            negative.append(key)
    return positive, negative


def _arm_sides(body: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """`(term id, level) -> ACTIVE | CONTROL` for every factor level that names an arm.

    The join the schema prescribes: a `Cell` names a term and a level, the term's
    `FactorLevel` carries the arms that level stands for, and the arm's own name says
    whether it is the intervention or the comparator. Going through the model rather than
    string-matching the cell's level is what makes `0.5 mg/kg` resolvable -- as a level name
    it says nothing, and only its arm (`arm_ketamine_0_5mgkg`) does.
    """

    roles = _arm_roles(body)
    groups = {
        g.get("local_id"): g
        for g in (body.get("groups") or [])
        if isinstance(g, Mapping) and g.get("local_id")
    }
    sides: dict[tuple[str, str], str] = {}
    for model in body.get("model_estimations") or []:
        if not isinstance(model, Mapping):
            continue
        for term in model.get("terms") or []:
            if not isinstance(term, Mapping):
                continue
            for level in term.get("levels") or []:
                if not isinstance(level, Mapping):
                    continue
                # Both routes the schema names. `FactorLevel.arms` is the crossover case;
                # a parallel-group trial allocates whole cohorts, so its arm reaches the
                # model through `Group.arm` instead and reading only the first route makes
                # every parallel-group trial invisible to this filter.
                named = {roles.get(a) for a in (level.get("arms") or [])}
                for gid in level.get("groups") or []:
                    named.add(roles.get((groups.get(gid) or {}).get("arm")))
                named.discard(None)
                # One arm, one side. A level standing for both an active and a control arm
                # is not a side of a treatment contrast, and neither is one whose arms are
                # all UNKNOWN.
                decided = named - {UNKNOWN}
                if len(decided) == 1:
                    sides[(term.get("local_id"), str(value_of(level.get("level"))))] = (
                        decided.pop()
                    )
    return sides


def _arm_contrast_of(analysis: Mapping, sides: Mapping[tuple[str, str], str]) -> str | None:
    """Which way this analysis's signed cells run, or None when it is not an arm contrast.

    Requires a signed cell on each side and both resolving to a known arm role. An
    unsigned contrast, a contrast between cohorts rather than arms, and one whose arms
    cannot be classified all return None and are dropped -- selecting a direction the
    record does not assert is how a map ends up being its own opposite.
    """

    cells = (analysis.get("effect") or {}).get("cells") or []
    seen: dict[str, set[str]] = {"positive": set(), "negative": set()}
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        sign = value_of(cell.get("direction"))
        if sign not in seen:
            continue
        side = sides.get((cell.get("term"), str(value_of(cell.get("level")))))
        if side:
            seen[sign].add(side)
    # Exactly one role a side, or the contrast is not interpretable as active-vs-control.
    if seen["positive"] == {ACTIVE} and seen["negative"] == {CONTROL}:
        return "active_over_control"
    if seen["positive"] == {CONTROL} and seen["negative"] == {ACTIVE}:
        return "control_over_active"
    return None


def _analysed_n(link: Mapping, groups_by_id: Mapping[str, Any]) -> tuple[int | None, str]:
    """How many participants a cohort contributed to one analysis, and where that came from.

    `AnalysisGroup.n` is the number this analysis actually used and is preferred whenever
    the source gave one. When it did not, the cohort's own reported size is the honest
    upper bound: the schema says `n` should "be smaller than the acquired count when an
    analysis drops participants for motion or missing data", so the group count can only
    over-state, never invent. That distinction is the whole reason this is a fallback and
    not a default -- the engine used to substitute a flat 30 and say nothing, and the rule
    this package works to is that a number a reviewer cannot trace is worse than none.

    So the source is returned with the count and the caller reports it. A run where most
    weights came from `acquired_count` is a run whose weighting a reviewer should check.
    """

    count = value_of(link.get("n"))
    if isinstance(count, (int, float)) and not isinstance(count, bool):
        return int(count), "analysis"
    entity = groups_by_id.get(link.get("group"))
    if isinstance(entity, Mapping):
        # Acquired before enrolled: enrolment precedes the scanner, so it over-states by
        # the dropouts as well as the exclusions.
        for slot in ("acquired_count", "enrolled_count"):
            count = value_of(entity.get(slot))
            if isinstance(count, (int, float)) and not isinstance(count, bool):
                return int(count), slot
    return None, ""


def select(
    selection: Selection, diagnoses: dict | None = None, task_families: dict | None = None
) -> Result:
    """Apply the funnel. `diagnoses` and `task_families` are the normalizer outputs, keyed
    `study|local_id`; without them those two filters cannot be applied and say so."""
    lost: Counter = Counter()
    rows: list[dict] = []
    seen: set[str] = set()
    kept: set[str] = set()

    found = sorted({p for pattern in selection.records for p in globlib.glob(pattern)})
    for path in (Path(p) for p in found if not p.endswith(".raw.json")):
        try:
            body = json.loads(path.read_text())
        except Exception:
            lost["record unreadable"] += 1
            continue
        body = body.get("study") or body
        if not isinstance(body, dict):
            lost["record unreadable"] += 1
            continue
        study = path.name.split(".")[0]
        seen.add(study)

        if selection.exclude_meta_analyses and _is_meta_analysis(body):
            lost["paper is itself a meta-analysis"] += 1
            continue
        if selection.species is not None:
            said = {
                s.lower()
                for g in (body.get("groups") or [])
                if isinstance(g, dict)
                for s in _texts(g.get("species"))
            }
            if said and not (said & {s.lower() for s in selection.species}):
                lost[f"species not in {sorted(selection.species)}"] += 1
                continue

        keyed = {}
        # From `paths`, not by counting `..` from the record. This read
        # `<run>/texts/<id>/stage1/analyses.json`, a directory that has never existed, so
        # `keyed` was always empty: every analysis was dropped as "no joinable row group"
        # -- blaming the extractor for a missing key -- and the parsed-coordinate fallback
        # that answers the space for 11% of analyses could never fire either.
        stage1 = paths.stage1(study)
        if stage1.is_file():
            parsed = json.loads(stage1.read_text()).get("analyses") or []
            keyed = dict(zip(parse_keys.parse_keys(parsed), parsed))
        points_by_key = {k: (v.get("points") or []) for k, v in keyed.items()}
        wants_arms = (
            selection.arm_contrast is not None or selection.treatment_exposure is not None
        )
        arm_sides = _arm_sides(body) if wants_arms else {}
        time_sides = _time_sides(body) if selection.treatment_exposure is not None else {}
        groups_by_id = {
            g.get("local_id"): g
            for g in (body.get("groups") or [])
            if isinstance(g, Mapping) and g.get("local_id")
        }

        for analysis in body.get("analyses") or []:
            if not isinstance(analysis, dict):
                continue
            aid = value_of(analysis.get("local_id"))

            scope = str(value_of(analysis.get("spatial_scope")) or "")
            if scope not in selection.spatial_scope:
                lost[f"spatial_scope={scope or 'unset'}"] += 1
                continue

            if selection.measure_type is not None:
                mid = value_of(analysis.get("measure"))
                kind = next(
                    (
                        str(value_of(m.get("type")) or "")
                        for m in (body.get("measures") or [])
                        if isinstance(m, dict) and value_of(m.get("local_id")) == mid
                    ),
                    "",
                )
                if kind not in selection.measure_type:
                    lost[f"measure_type={kind or 'unset'}"] += 1
                    continue

            groups = analysis.get("groups") or []
            if selection.contrast == "within_subject" and len(groups) > 1:
                lost["between-group contrast"] += 1
                continue
            if selection.contrast == "between_group" and len(groups) <= 1:
                lost["not a between-group contrast"] += 1
                continue

            if selection.diagnosis is not None:
                if diagnoses is None:
                    lost[
                        "diagnosis filter needs `pondie normalize medical_condition` output"
                    ] += 1
                    continue
                if diagnoses.get(f"{study}|{aid}") != selection.diagnosis:
                    lost[f"diagnosis != {selection.diagnosis}"] += 1
                    continue
            if selection.task_family is not None:
                if task_families is None:
                    lost["task filter needs `pondie normalize task` output"] += 1
                    continue
                if task_families.get(f"{study}|{aid}") != selection.task_family:
                    lost[f"task_family != {selection.task_family}"] += 1
                    continue

            route = None
            if selection.treatment_exposure is not None:
                route = _exposure_of(analysis, body, arm_sides, time_sides)
                if route is None:
                    lost["not a treatment-exposure contrast"] += 1
                    continue
                if route != selection.treatment_exposure:
                    lost[f"treatment exposure runs {route}"] += 1
                    continue
                route = (
                    "arm"
                    if any(arm_sides.get(k) for k in sum(_signed_cells(analysis), []))
                    else "time"
                )

            if selection.arm_contrast is not None:
                way = _arm_contrast_of(analysis, arm_sides)
                if way is None:
                    lost["no signed arm contrast"] += 1
                    continue
                if way != selection.arm_contrast:
                    lost[f"arm contrast runs {way}"] += 1
                    continue

            resolved = coordinate_space.resolve(analysis, body, points_by_key)
            if resolved.value not in selection.space:
                # `.reason` is how the space was decided -- the analysis's own field, its
                # tables, or the parsed coordinates. The funnel exists to say why a paper
                # was dropped, and this was throwing that half away.
                lost[f"space={resolved.value} ({resolved.reason})"] += 1
                continue

            entry = keyed.get(value_of(analysis.get("source_table_analysis")))
            if entry is None:
                # Two different problems, and only one is the extraction's fault.
                lost["no stage-1 parse synced" if not keyed else "no joinable row group"] += 1
                continue
            pts = _points(entry, resolved.value)
            if not pts:
                lost["no placeable coordinates"] += 1
                continue

            n = 0
            n_from = set()
            for link in groups:
                if not isinstance(link, dict):
                    continue
                count, source = _analysed_n(link, groups_by_id)
                if count is not None:
                    n += count
                    n_from.add(source)
            rows.append(
                {
                    "study": study,
                    "analysis": aid,
                    "points": pts,
                    "n": n or None,
                    # Which slot each weight came from, so `funnel` can say how many rows
                    # are weighted on a cohort total rather than an analysed count.
                    "n_source": sorted(n_from),
                    #: Which route made this a treatment contrast, when one was asked for.
                    "route": route,
                    "space": resolved.value,
                    "name": str(value_of(analysis.get("name")) or "")[:70],
                }
            )
            kept.add(study)
    return Result(selection, rows, lost, kept, seen)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--records", action="append")
    ap.add_argument("--measure-type", action="append")
    ap.add_argument("--spatial-scope", action="append")
    ap.add_argument("--contrast", default="any")
    ap.add_argument("--include-roi", action="store_true")
    args = ap.parse_args()

    kwargs: dict = {"contrast": args.contrast}
    if args.records:
        kwargs["records"] = tuple(args.records)
    if args.measure_type:
        kwargs["measure_type"] = frozenset(args.measure_type)
    if args.spatial_scope:
        kwargs["spatial_scope"] = frozenset(args.spatial_scope)
    elif args.include_roi:
        kwargs["spatial_scope"] = frozenset({"whole_brain", "roi"})
    result = select(Selection(**kwargs))
    print(result.funnel())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
