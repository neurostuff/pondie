#!/usr/bin/env python3
"""The gold a coordinate meta-analysis publishes, and how a selector's foci are scored on it.

One module because the two scripts that score the second gate have to agree about the
comparison down to the transform: `query_analysis_selection.py` puts autonima's annotation
next to the deterministic query, and `query_workflow.py` runs the query behind the
screening query. A metric implemented twice is a metric that will read differently for
reasons that are not the selectors.

What it knows:

* the gold -- `nimads/<project>/merged/`, whose annotation carries one boolean per key per
  analysis and whose studyset carries that analysis's coordinates;
* the join from a record's analysis to its coordinates, `Analysis.source_table_analysis`
  into `<pmid>/stage1/analyses.json`;
* the comparison, which happens in MNI at a tolerance, because each side converts its own
  Talairach coordinates and the two transforms disagree by about a millimetre.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from pondie.normalization import coordinate_space
from pondie.query.engine import _points

# `tal2mni` logs a line per call about a 3x3 input being ambiguous. It reads them as row
# vectors, which is what they are, and a table with exactly three foci is common enough
# that the notice drowns the report.
logging.getLogger("nimare").setLevel(logging.ERROR)

#: Coordinates are rounded to this many decimals before they are compared.
PLACES = 1

#: How far apart two foci may be and still be the same focus, in mm.
#:
#: Not a fudge factor. Both sides are moved into MNI -- the parse by
#: `query.engine._points`, the benchmark by whoever built its studysets -- and the two
#: transforms are not the same one: `tal2mni` sends this corpus's (-53, -56, -15) to
#: (-55.5, -59.7, -15.2) where the benchmark's gold holds (-54.7, -59.4, -15.8). That is
#: 1.1mm of disagreement about a transform, on a coordinate both sides agree about.
#: Distinct foci in one table are rarely within 8mm of each other, so 2mm buys the
#: transform and concedes nothing else.
TOLERANCE = 2.0


def read(node: Any) -> Any:
    if not isinstance(node, Mapping) or "extraction_status" not in node:
        return node
    return node.get("value") if node.get("extraction_status") == "extracted" else None


def to_mni(points: list[Mapping[str, Any]]) -> list[tuple]:
    """Coordinates in MNI, moving the Talairach ones and leaving the rest alone."""

    def talairach(point: Mapping[str, Any]) -> bool:
        return str(point.get("space") or "").upper().startswith("TAL")

    usable = [p for p in points if len(p.get("coordinates") or []) == 3]
    moved = _points({"points": [p for p in usable if talairach(p)]}, "TAL")
    moved += [p["coordinates"] for p in usable if not talairach(p)]
    return [tuple(round(float(c), PLACES) for c in xyz) for xyz in moved]


def study_pmids(study: Mapping[str, Any]) -> list[str]:
    """The pmids a merged gold study stands for.

    Dementia merges up to 27 papers into one study and its foci cannot be attributed to
    any one of them. Those studies are dropped by every caller rather than credited to
    every member -- crediting them made each per-paper number meaningless and read 2-6%
    before it was caught.
    """
    metadata = study.get("metadata") or {}
    raw = metadata.get("pmids") or metadata.get("original_study_ids") or study.get("id")
    items = raw.split(",") if isinstance(raw, str) else list(raw or [])
    return [str(x).strip() for x in items if str(x).strip().isdigit()]


def gold_maps(bench: Path, project: str) -> tuple[dict[str, dict[str, list]], int]:
    """key -> pmid -> the coordinates that map pooled, plus the merged studies dropped."""
    merged = bench / "nimads" / project / "merged"
    if not (merged / "nimads_annotation.json").is_file():
        return {}, 0
    annotation = json.loads((merged / "nimads_annotation.json").read_text())
    studyset = json.loads((merged / "nimads_studyset.json").read_text())

    owner: dict[str, str] = {}
    points: dict[str, list] = {}
    dropped = 0
    for study in studyset.get("studies") or []:
        pmids = study_pmids(study)
        if len(pmids) != 1:
            dropped += 1
            continue
        for analysis in study.get("analyses") or []:
            owner[analysis["id"]] = pmids[0]
            points[analysis["id"]] = to_mni(analysis.get("points") or [])

    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for note in annotation.get("notes") or []:
        pmid = owner.get(note.get("analysis"))
        if pmid is None:
            continue
        for key, value in (note.get("note") or {}).items():
            if value:
                out[key][pmid] += points.get(note["analysis"], [])
    return ({k: {p: sorted(set(c)) for p, c in v.items()} for k, v in out.items()}, dropped)


def gold_analysis_counts(bench: Path, project: str) -> dict[str, dict[str, int]]:
    """key -> pmid -> how many analyses that map pooled from that paper."""
    merged = bench / "nimads" / project / "merged"
    if not (merged / "nimads_annotation.json").is_file():
        return {}
    annotation = json.loads((merged / "nimads_annotation.json").read_text())
    studyset = json.loads((merged / "nimads_studyset.json").read_text())
    owner = {
        analysis["id"]: pmids[0]
        for study in studyset.get("studies") or []
        for pmids in (study_pmids(study),)
        if len(pmids) == 1
        for analysis in study.get("analyses") or []
    }
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for note in annotation.get("notes") or []:
        pmid = owner.get(note.get("analysis"))
        if pmid is None:
            continue
        for key, value in (note.get("note") or {}).items():
            if value:
                out[key][pmid] += 1
    return {k: dict(v) for k, v in out.items()}


def gold_spaces(bench: Path, project: str) -> dict[str, set[str]]:
    """pmid -> the spaces the benchmark's own studyset puts that paper's foci in."""
    path = bench / "nimads" / project / "merged" / "nimads_studyset.json"
    if not path.is_file():
        return {}
    out: dict[str, set[str]] = {}
    for study in json.loads(path.read_text()).get("studies") or []:
        pmids = study_pmids(study)
        if len(pmids) == 1:
            out[pmids[0]] = {str(p.get("space") or "").upper()
                             for a in study.get("analyses") or []
                             for p in a.get("points") or []}
    return out


def parse_entries(stage1: Path | None, pmid: str) -> dict[str, dict] | None:
    """`<table>#<n>` -> its parsed row group, or None when the paper has no parse."""
    from pondie.formats import parse_keys

    if stage1 is None:
        return None
    path = stage1 / pmid / "stage1" / "analyses.json"
    if not path.is_file():
        return None
    entries = json.loads(path.read_text()).get("analyses") or []
    return dict(zip(parse_keys.parse_keys(entries), entries))


def foci_of(analysis: Mapping[str, Any], body: Mapping[str, Any],
            entries: Mapping[str, dict]) -> list[tuple] | None:
    """One record analysis's coordinates in MNI, or None where the join reaches none.

    The space is resolved and the transform applied by the same code the query engine
    uses, so a map built here and a map built by `pondie query` place their foci
    identically.
    """
    entry = entries.get(str(read(analysis.get("source_table_analysis")) or ""))
    if entry is None:
        return None
    by_key = {key: e.get("points") for key, e in entries.items()}
    resolved = coordinate_space.resolve(dict(analysis), dict(body), by_key)
    return [tuple(round(float(c), PLACES) for c in point)
            for point in _points(entry, resolved.value)]


def raw_foci_of(analysis: Mapping[str, Any], body: Mapping[str, Any],
                entries: Mapping[str, dict]) -> tuple[list[list[float]], str] | None:
    """One analysis's coordinates AS THE PAPER PUBLISHED THEM, and the space they are in.

    For writing a studyset rather than for scoring one. autonima's own exports carry the
    parsed coordinates untransformed with a `space` label beside them -- 1,527 Talairach
    points sit in cue reactivity's studyset as Talairach -- and an arm that converted
    first would differ from the others by a transform as well as by its selector.
    """
    entry = entries.get(str(read(analysis.get("source_table_analysis")) or ""))
    if entry is None:
        return None
    by_key = {key: e.get("points") for key, e in entries.items()}
    resolved = coordinate_space.resolve(dict(analysis), dict(body), by_key)
    points = [p.get("coordinates") for p in entry.get("points") or []
              if len(p.get("coordinates") or []) == 3]
    return points, resolved.value


def spaces_of(body: Mapping[str, Any], entries: Mapping[str, dict]) -> set[str]:
    """The spaces the record puts this paper's analyses in."""
    by_key = {key: e.get("points") for key, e in entries.items()}
    return {coordinate_space.resolve(dict(a), dict(body), by_key).value
            for a in body.get("analyses") or [] if isinstance(a, Mapping)}


def match(selected: Mapping[str, list], gold: Mapping[str, list],
          tolerance: float = TOLERANCE) -> tuple[float, float, float]:
    """Precision, recall and F1 over coordinates, paper by paper.

    A selected focus counts once, against its nearest unclaimed gold focus in the same
    paper. Nearest-first rather than in order, so two candidates inside the tolerance
    cannot both claim the closer gold focus and leave the further one unmatched.
    """
    hits = 0
    for pmid, points in selected.items():
        want = list(gold.get(pmid) or [])
        pairs = sorted(
            (sum((a - b) ** 2 for a, b in zip(point, target)) ** 0.5, i, j)
            for i, point in enumerate(points)
            for j, target in enumerate(want)
        )
        taken_point: set[int] = set()
        taken_gold: set[int] = set()
        for distance, i, j in pairs:
            if distance > tolerance or i in taken_point or j in taken_gold:
                continue
            taken_point.add(i)
            taken_gold.add(j)
            hits += 1
    chosen = sum(len(v) for v in selected.values())
    wanted = sum(len(v) for v in gold.values())
    precision = hits / chosen if chosen else 0.0
    recall = hits / wanted if wanted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def papers(selected: set, gold: set) -> tuple[float, float]:
    hit = len(selected & gold)
    return (hit / len(selected) if selected else 0.0, hit / len(gold) if gold else 0.0)
