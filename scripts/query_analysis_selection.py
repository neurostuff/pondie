#!/usr/bin/env python3
"""Selecting the ANALYSIS a meta-analysis pools: autonima's annotation and the query, scored alike.

    python scripts/query_analysis_selection.py --records '<dir>/*/*.extraction.json' \
        --bench <neurometabench>/data --autonima <autonima-results>/projects \
        --stage1 <corpus> [--arms "full text,record + evidence"]

Screening picks papers; a coordinate meta-analysis pools *contrasts*, and the benchmark
names them: "non-PTSD > PTSD", "bvFTD < HC (all & by modality)", "drug cue > neutral cue".
The gold is `nimads/<project>/merged/`, whose annotation carries one boolean per key per
analysis and whose studyset carries that analysis's coordinates.

TWO SELECTORS, ONE SCORER. Autonima annotates the analyses its own coordinate parse found
-- `annotation_results.json`, an `include` per analysis per key, over the arms the
record-vs-full-text experiment ran. The query reads `Effect.cells` on the same paper's
record. Both are put through the same function here, on the same denominator, because the
earlier version of this script read autonima's numbers from a csv built elsewhere and the
query's from the records, and two metrics computed in two places will differ for reasons
that are not the selectors.

Every column is per gold paper except the last three, which pool coordinates:

    >=1        the selector found any analysis for this key
    =count     it found exactly as many as the gold has
    =foci      its analyses carry exactly as many coordinates as the gold's
    P R F1     coordinates, matched paper by paper in MNI within a tolerance

`=count` and `=foci` are the columns the first run of this comparison reported, kept so
the two runs can be read against each other; the pooled precision and recall are what a
map is actually made of, and `docs/meta-analysis-queries.md` records why the count columns
overstate agreement -- two different analyses can carry the same number of foci.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from query_contrasts import KEYS, UNANSWERABLE, judge_analyses  # noqa: E402
from query_foci import (  # noqa: E402
    TOLERANCE,
    foci_of,
    gold_analysis_counts,
    gold_maps,
    match,
    parse_entries,
    to_mni,
)

#: Arm label -> the run directory suffix the record-arms experiment gave it. The stem is
#: the project's `baseline_run` from `experiments/record_arms/arms/<project>.yaml`; a
#: project with a `fulltext_run` names its text arm separately, because that arm was
#: rehomed onto the same provider as the record arms.
ARMS = {
    "full text": {"vbm_of_ptsd": "v1-A1-mini", "dementia": "v3",
                  "cue_reactivity": "v5-gpt-A1-mini", "vbm_of_substance_use": "v2-A1-mini"},
    "record + evidence": {"vbm_of_ptsd": "v1-record-with-evidence",
                          "dementia": "v3-record-with-evidence",
                          "cue_reactivity": "v5-gpt-record-with-evidence",
                          "vbm_of_substance_use": "v2-record-with-evidence"},
    "record, no evidence": {"vbm_of_ptsd": "v1-record-no-evidence",
                            "dementia": "v3-record-no-evidence",
                            "cue_reactivity": "v5-gpt-record-no-evidence",
                            "vbm_of_substance_use": "v2-record-no-evidence"},
}


def load_records(pattern: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        body = json.loads(Path(path).read_text())
        out[Path(path).parent.name][Path(path).name.split(".")[0]] = body.get("study") or body
    return out


def autonima_key(autonima: Path, project: str, manual: str) -> str | None:
    """The annotation autonima ran for this benchmark key, from the project's own map."""
    path = autonima / project / "nmb_mappings.json"
    if not path.is_file():
        return None
    return (json.loads(path.read_text()).get("annotation_mappings") or {}).get(manual)


def autonima_selection(autonima: Path, project: str, run: str,
                       annotation: str) -> dict[str, list] | None:
    """pmid -> the coordinates that arm's annotation pooled for this key.

    `analysis_id` is `<pmid>_analysis_<n>` and `n` indexes the study's entry in the arm's
    own coordinate parse, which is the join autonima's own reports use.
    """
    outputs = autonima / project / run / "outputs"
    notes = outputs / "annotation_results.json"
    parsed = outputs / "coordinate_parsing_results.json"
    if not (notes.is_file() and parsed.is_file()):
        return None

    points: dict[str, list[list[tuple]]] = {}
    for study in json.loads(parsed.read_text()).get("studies") or []:
        points[str(study.get("pmid"))] = [to_mni(a.get("points") or [])
                                          for a in study.get("analyses") or []]

    out: dict[str, list] = defaultdict(list)
    for note in json.loads(notes.read_text()):
        if note.get("annotation_name") != annotation or not note.get("include"):
            continue
        pmid = str(note.get("study_id"))
        index = str(note.get("analysis_id") or "").rsplit("_", 1)[-1]
        here = points.get(pmid) or []
        if index.isdigit() and int(index) < len(here):
            out[pmid] += here[int(index)]
        else:
            out[pmid] += []
    return {p: sorted(set(c)) for p, c in out.items()}


def autonima_saw(autonima: Path, project: str, run: str) -> set[str]:
    """The papers that arm's own coordinate parse found any analysis in."""
    path = autonima / project / run / "outputs" / "coordinate_parsing_results.json"
    if not path.is_file():
        return set()
    return {str(study.get("pmid")) for study in json.loads(path.read_text()).get("studies") or []
            if study.get("analyses")}


def autonima_counts(autonima: Path, project: str, run: str, annotation: str) -> dict[str, int]:
    """pmid -> how many analyses that arm's annotation included for this key."""
    path = autonima / project / run / "outputs" / "annotation_results.json"
    if not path.is_file():
        return {}
    tally: Counter = Counter()
    for note in json.loads(path.read_text()):
        if note.get("annotation_name") == annotation and note.get("include"):
            tally[str(note.get("study_id"))] += 1
    return dict(tally)


def query_selection(records: dict, stage1: Path | None, spec, permissive: bool,
                    pool: set[str]) -> tuple[dict[str, list], dict[str, int]]:
    """pmid -> the coordinates the query pools, and how many analyses it pooled them from."""
    foci: dict[str, list] = {}
    counts: dict[str, int] = {}
    for pmid in pool:
        body = records.get(pmid)
        if body is None:
            continue
        judged = judge_analyses(body, spec)
        picked = {a for a, v in judged.items()
                  if v is True or (permissive and v is UNANSWERABLE)}
        if not picked:
            continue
        counts[pmid] = len(picked)
        entries = parse_entries(stage1, pmid)
        found: list[tuple] = []
        if entries is not None:
            for analysis in body.get("analyses") or []:
                if isinstance(analysis, dict) and str(analysis.get("local_id")) in picked:
                    found += foci_of(analysis, body, entries) or []
        foci[pmid] = sorted(set(found))
    return foci, counts


def report(label: str, foci: dict[str, list], counts: dict[str, int], saw: set[str],
           gold_foci: dict[str, list], gold_counts: dict[str, int], tolerance: float) -> None:
    """One selector's row, over the gold papers -- the denominator every selector shares.

    `saw` is how many of those papers the selector had anything to choose from: an
    annotation cannot include an analysis its own arm never parsed, and the two selectors
    do not lose the same papers upstream, so the column that says so is printed beside
    the ones that assume they do.
    """
    found = sum(1 for p in gold_foci if counts.get(p, 0) >= 1)
    same_count = sum(1 for p in gold_foci if counts.get(p, 0) == gold_counts.get(p, 0))
    same_foci = sum(1 for p in gold_foci if len(foci.get(p) or []) == len(gold_foci[p]))
    n = len(gold_foci)
    precision, recall, f1 = match({p: c for p, c in foci.items() if p in gold_foci},
                                  gold_foci, tolerance)
    print(f"  {label:28} {len(saw & set(gold_foci)):5d} {found / n:6.0%} "
          f"{same_count / n:7.0%} {same_foci / n:6.0%}   "
          f"{precision:7.1%} {recall:7.1%} {f1:7.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--autonima", type=Path,
                    help="<autonima-results>/projects, for each arm's annotation and its "
                         "own coordinate parse -- without it only the query is scored")
    ap.add_argument("--stage1", type=Path,
                    help="the corpus, for <pmid>/stage1/analyses.json, which is how a "
                         "record's analysis reaches its coordinates")
    # Not comma separated: one of the labels is "record, no evidence", and splitting on
    # the comma scored two arms that do not exist and silently dropped the one that does.
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS),
                    help="which autonima arms to score")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    records = load_records(args.records)
    arms = [a for a in args.arms if a in ARMS]

    for (project, manual), spec in KEYS.items():
        here = records.get(project) or {}
        if not here:
            continue
        maps, dropped = gold_maps(args.bench, project)
        counts = gold_analysis_counts(args.bench, project)
        if manual not in maps:
            continue
        # The denominator: gold papers for this key that have a record. Every selector is
        # scored over it, including an arm whose pipeline never reached the paper -- being
        # absent from a selector's own parse is a miss, not an exemption.
        gold_foci = {p: c for p, c in maps[manual].items() if p in here}
        gold_counts = {p: counts.get(manual, {}).get(p, 0) for p in gold_foci}

        print("=" * 84)
        print(f"{project} / {manual}   {len(gold_foci)} gold papers with a record"
              f"   {sum(len(c) for c in gold_foci.values())} gold foci"
              + (f"   ({dropped} merged gold studies dropped)" if dropped else ""))
        print(f"  {'selector':28} {'saw':>5} {'>=1':>6} {'=count':>7} {'=foci':>6}   "
              f"{'coord P':>7} {'coord R':>7} {'F1':>7}")

        annotation = autonima_key(args.autonima, project, manual) if args.autonima else None
        for arm in arms if annotation else []:
            run = ARMS[arm].get(project)
            selection = autonima_selection(args.autonima, project, run, annotation)
            if selection is None:
                continue
            report(f"autonima: {arm}", selection,
                   autonima_counts(args.autonima, project, run, annotation),
                   autonima_saw(args.autonima, project, run),
                   gold_foci, gold_counts, args.tolerance)
        if args.autonima and annotation is None:
            print("  (this key has no autonima annotation in the project's nmb_mappings)")

        saw = {p for p, body in here.items() if body.get("analyses")}
        for label, permissive in (("query, strict", False), ("query, permissive", True)):
            foci, picked = query_selection(here, args.stage1, spec, permissive, set(gold_foci))
            report(label, foci, picked, saw, gold_foci, gold_counts, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
