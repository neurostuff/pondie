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
    dataset = result.to_dataset()   # one experiment per study, ready for NiMARE

Boundary contract: `Selection` is pydantic, so an unknown field or a bad literal fails at
construction. The records themselves are read through `schema_utils.value_of`, which takes the
wrapper and the multivalued shape from the LinkML schema -- see
docs/pipeline-architecture.md#the-contract-at-each-seam.
"""
from __future__ import annotations

import glob as globlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA = Path(__file__).resolve().parents[2] / "study_schema"
for _path in (SCHEMA, SCHEMA / "review"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from schema_utils import value_of  # noqa: E402

from ..normalization import coordinate_space  # noqa: E402

SpatialScope = Literal["whole_brain", "roi", "searchlight", "other"]
Space = Literal["MNI", "TAL", "OTHER", "UNKNOWN"]
Contrast = Literal["any", "within_subject", "between_group"]


class Selection(BaseModel):
    """What to pool. Unknown fields are an error, not a silently ignored typo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[str, ...] = ("data/runs/*/records/*.extraction.json",)

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

    def __init__(self, selection: Selection, rows: list[dict], lost: Counter,
                 kept_papers: set[str], seen_papers: set[str]):
        self.selection, self.rows, self.lost = selection, rows, lost
        self.kept_papers, self.seen_papers = kept_papers, seen_papers

    @property
    def studies(self) -> set[str]:
        return {r["study"] for r in self.rows}

    def funnel(self) -> str:
        out = [f"{len(self.seen_papers)} papers read, {len(self.kept_papers)} contribute",
               f"{len(self.rows)} analyses selected from {len(self.studies)} studies, "
               f"{sum(len(r['points']) for r in self.rows)} foci"]
        if self.lost:
            out.append("lost:")
            out += [f"   {n:5d}  {why}" for why, n in self.lost.most_common()]
        if len(self.studies) < self.selection.min_studies:
            out.append(f"WARNING: {len(self.studies)} studies is below min_studies="
                       f"{self.selection.min_studies}; a coordinate meta-analysis over this "
                       f"many converges on whichever paper reports the most foci")
        return "\n".join(out)

    def to_dataset(self, target: str = "mni152_2mm"):
        """One experiment per study. `Studyset.combine_analyses()` does the pooling."""
        from nimare.dataset import Dataset
        from nimare.nimads import Studyset
        data: dict[str, dict] = {}
        for row in self.rows:
            held = data.setdefault(row["study"], {"contrasts": {}})
            xs, ys, zs = zip(*row["points"])
            held["contrasts"][str(len(held["contrasts"]))] = {
                "coords": {"space": "MNI", "x": list(xs), "y": list(ys), "z": list(zs)},
                "metadata": {"sample_sizes": [row["n"] or 30]}}
        return Studyset.from_dataset(Dataset(data, target=target)) \
                       .combine_analyses().to_dataset()


def _texts(x) -> list[str]:
    """Every string in a value, whatever shape the slot declares."""
    v = value_of(x, True)
    return [s for s in (value_of(i) for i in v) if isinstance(s, str) and s.strip()]


#: A coordinate meta-analysis indexed as a paper. Its peaks are already a convergence over
#: primary studies, several of which are usually in the same corpus, so leaving it in counts
#: those samples twice and piles the double count on the loci under test.
META_ANALYSIS = re.compile(r"meta-?analy|activation likelihood estimation|\bALE\b|\bMKDA\b|"
                           r"seed-based d mapping|\bSDM\b|coordinate-based", re.I)


def _is_meta_analysis(body: dict) -> bool:
    design = body.get("design") if isinstance(body.get("design"), dict) else {}
    blob = " ".join([str(value_of(body.get("description")) or ""),
                     str(value_of(design.get("description")) or ""),
                     *(str(h) for h in (value_of(body.get("hypothesis"), True) or []))])
    return bool(META_ANALYSIS.search(blob))


def _points(entry: dict, space: str) -> list[list[float]]:
    """Coordinates from one parsed row group, moved into MNI where the space says to."""
    import numpy as np
    from nimare.utils import tal2mni
    raw = [p.get("coordinates") for p in (entry.get("points") or [])]
    raw = [c for c in raw if isinstance(c, list) and len(c) == 3
           and all(isinstance(v, (int, float)) for v in c)]
    if not raw:
        return []
    if space == "TAL":
        return tal2mni(np.array(raw, dtype=float)).tolist()
    return raw


def select(selection: Selection, diagnoses: dict | None = None,
           task_families: dict | None = None) -> Result:
    """Apply the funnel. `diagnoses` and `task_families` are the normalizer outputs, keyed
    `study|local_id`; without them those two filters cannot be applied and say so."""
    import parse_tables

    lost: Counter = Counter()
    rows: list[dict] = []
    seen: set[str] = set()
    kept: set[str] = set()

    paths = sorted({p for pattern in selection.records for p in globlib.glob(pattern)})
    for path in (Path(p) for p in paths if not p.endswith(".raw.json")):
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
            said = {s.lower() for g in (body.get("groups") or []) if isinstance(g, dict)
                    for s in _texts(g.get("species"))}
            if said and not (said & {s.lower() for s in selection.species}):
                lost[f"species not in {sorted(selection.species)}"] += 1
                continue

        keyed = {}
        stage1 = path.parent.parent / "texts" / study / "stage1" / "analyses.json"
        if stage1.is_file():
            parsed = json.loads(stage1.read_text()).get("analyses") or []
            keyed = dict(zip(parse_tables.parse_keys(parsed), parsed))
        points_by_key = {k: (v.get("points") or []) for k, v in keyed.items()}

        for analysis in (body.get("analyses") or []):
            if not isinstance(analysis, dict):
                continue
            aid = value_of(analysis.get("local_id"))

            scope = str(value_of(analysis.get("spatial_scope")) or "")
            if scope not in selection.spatial_scope:
                lost[f"spatial_scope={scope or 'unset'}"] += 1
                continue

            if selection.measure_type is not None:
                mid = value_of(analysis.get("measure"))
                kind = next((str(value_of(m.get("type")) or "")
                             for m in (body.get("measures") or [])
                             if isinstance(m, dict) and value_of(m.get("local_id")) == mid), "")
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
                    lost["diagnosis filter needs normalize_conditions output"] += 1
                    continue
                if diagnoses.get(f"{study}|{aid}") != selection.diagnosis:
                    lost[f"diagnosis != {selection.diagnosis}"] += 1
                    continue
            if selection.task_family is not None:
                if task_families is None:
                    lost["task filter needs normalize_tasks output"] += 1
                    continue
                if task_families.get(f"{study}|{aid}") != selection.task_family:
                    lost[f"task_family != {selection.task_family}"] += 1
                    continue

            resolved = coordinate_space.resolve(analysis, body, points_by_key)
            if resolved["space"] not in selection.space:
                lost[f"space={resolved['space']}"] += 1
                continue

            entry = keyed.get(value_of(analysis.get("source_table_analysis")))
            if entry is None:
                lost["no joinable row group"] += 1
                continue
            pts = _points(entry, resolved["space"])
            if not pts:
                lost["no placeable coordinates"] += 1
                continue

            n = 0
            for link in groups:
                count = value_of(link.get("n")) if isinstance(link, dict) else None
                if isinstance(count, (int, float)):
                    n += int(count)
            rows.append({"study": study, "analysis": aid, "points": pts, "n": n or None,
                         "space": resolved["space"],
                         "name": str(value_of(analysis.get("name")) or "")[:70]})
            kept.add(study)
    return Result(selection, rows, lost, kept, seen)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
