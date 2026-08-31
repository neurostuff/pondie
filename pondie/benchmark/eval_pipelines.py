#!/usr/bin/env python3
"""Score candidate pipelines on the field the goal turns on, and on what they cost.

Direction accuracy against the reviewer gold, coverage, and output size. A pipeline that
saves tokens and moves direction is not a saving -- direction carries 0.45 of the composite
and is the one fact a synthesis cannot recover elsewhere, so it is reported first and the
token column is only read once accuracy holds.

    python eval_pipelines.py

Hypotheses and their rationale: docs/pipeline-hypotheses.md.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes
from derive_fields import derive_cell_direction, same_level, unwrap  # noqa: E402

SIGNED = {"positive", "negative"}


def gold_cells() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in glob.glob(str(ROOT / "benchmarks/gold/direction/*.direction.json")):
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        for cell in doc["cells"]:
            if cell.get("tier") == "silent" or cell.get("disputed"):
                continue
            if cell.get("direction") not in SIGNED:
                continue
            out.setdefault(doc["paper_id"], []).append(cell)
    return out


def record_cell(rec: dict, analysis_id: str, term: str, level: str | None):
    for analysis in rec.get("analyses") or []:
        if analysis.get("local_id") != analysis_id:
            continue
        for cell in ((analysis.get("effect") or {}).get("cells") or []):
            if cell.get("term") != term:
                continue
            if level and not same_level(str(unwrap(cell.get("level")) or ""), level):
                continue
            return analysis, cell
    return None, None


def evaluate() -> dict[str, Counter]:
    gold = gold_cells()
    res: dict[str, Counter] = defaultdict(Counter)
    for paper, cells in gold.items():
        rp = ROOT / f"data/records/{paper}.extraction.json"
        if not rp.is_file():
            continue
        rec = json.loads(rp.read_text(encoding="utf-8"))
        for gc in cells:
            analysis, cell = record_cell(rec, gc["analysis"], gc["term"], gc.get("level"))
            want = gc["direction"]

            # P0: what the deployed extractor said for this cell.
            if cell is not None:
                got = unwrap(cell.get("direction"))
                if got in SIGNED:
                    res["P0 luna (deployed)"]["n"] += 1
                    res["P0 luna (deployed)"]["ok"] += int(got == want)
                else:
                    res["P0 luna (deployed)"]["unsigned"] += 1
            else:
                res["P0 luna (deployed)"]["no_cell"] += 1

            # P1: derive from the contrast name / statistic sign, abstain otherwise.
            derived = (derive_cell_direction(paper, analysis=analysis, level=gc.get("level"))
                       if analysis is not None else None)
            if derived is None:
                res["P1 deterministic-first"]["abstain"] += 1
            else:
                res["P1 deterministic-first"]["n"] += 1
                res["P1 deterministic-first"]["ok"] += int(derived == want)

            # P4: accept only where the deriver and the deployed record agree.
            got = unwrap(cell.get("direction")) if cell is not None else None
            if derived is not None and got in SIGNED and derived == got:
                res["P4 agreement cascade"]["n"] += 1
                res["P4 agreement cascade"]["ok"] += int(derived == want)
            else:
                res["P4 agreement cascade"]["to_model"] += 1
    return res


def token_cost() -> dict[str, int]:
    """Characters of record JSON each pipeline still asks a model to emit."""
    DETERMINISTIC = {
        "acquisitions.magnetic_field_strength_tesla", "groups.species", "groups.age_unit",
        "acquisitions.mr_acquisition_type", "design.blinding", "design.assignment_structure",
        "analyses.effect.statistic.family",
    }
    NUEXTRACT = {
        "groups.enrolled_count", "groups.acquired_count", "groups.excluded_count",
        "groups.age_mean", "groups.age_standard_deviation", "groups.recruitment_method",
        "groups.diagnostic_system", "preprocessings.smoothing_fwhm_mm",
        "preprocessings.software", "model_estimations.software",
        "acquisitions.repetition_time_seconds", "acquisitions.echo_time_seconds",
        "acquisitions.number_of_volumes", "inference_settings.cluster_extent_threshold",
        "inference_settings.permutation_count",
        "inference_settings.multiple_comparison_method", "measures.unit",
        "model_estimations.hrf_model", "tables.table_number", "tables.caption",
        "tables.footer", "model_estimations.terms.type",
        "model_estimations.terms.variation_level", "model_estimations.spatial_unit",
        "analyses.spatial_scope",
        "analyses.effect.statistic.degrees_of_freedom_denominator",
        "analyses.coordinate_space", "devices.model", "devices.manufacturer",
    }
    CELLS = {"analyses.effect.cells.direction", "analyses.effect.cells.level"}

    def walk(node, path=""):
        if isinstance(node, dict):
            if "extraction_status" in node:
                yield path, node
                return
            for k, v in node.items():
                yield from walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v, f"{path}[]")

    total = Counter()
    for f in glob.glob(str(ROOT / "data/records/*.extraction.json")):
        for raw, node in walk(json.loads(Path(f).read_text(encoding="utf-8"))):
            key = re.sub(r"\[\]", "", raw)
            if node.get("extraction_status") != "extracted" or node.get("value") in (None, ""):
                continue
            size = len(json.dumps(node, ensure_ascii=False))
            total["all"] += size
            if key in DETERMINISTIC:
                total["deterministic"] += size
            elif key in NUEXTRACT:
                total["nuextract"] += size
            elif key in CELLS:
                total["cells"] += size
    return dict(total)


def main() -> int:
    res = evaluate()
    print("direction accuracy on the reviewer gold (101 signed cells)")
    print(f"{'pipeline':26s} {'scored':>7s} {'correct':>8s} {'acc':>6s}  deferred/abstained")
    for name in ("P0 luna (deployed)", "P1 deterministic-first", "P4 agreement cascade"):
        c = res[name]
        n, ok = c["n"], c["ok"]
        other = ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("n", "ok"))
        acc = f"{ok / n:.0%}" if n else "  --"
        print(f"{name:26s} {n:7d} {ok:8d} {acc:>6s}  {other}")

    t = token_cost()
    print()
    print("output the model still has to emit (record JSON chars, 16 papers)")
    base = t["all"]
    rows = [
        ("P0 luna (deployed)", base),
        ("P1 deterministic-first", base - t.get("deterministic", 0)),
        ("P3 pre-fill", base - t.get("deterministic", 0) - t.get("nuextract", 0)),
        ("P6 cells-only", t.get("cells", 0)),
    ]
    for name, size in rows:
        print(f"  {name:26s} {size:9d}  {size / base:5.0%} of baseline  ~{size // 4 // 16:6d} tok/paper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
