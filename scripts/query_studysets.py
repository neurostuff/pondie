#!/usr/bin/env python3
"""Write the query's selection as a run directory, so it can be mapped like an autonima arm.

    python scripts/query_studysets.py --records '<dir>/*/*.extraction.json' \
        --bench <neurometabench>/data --stage1 <corpus> \
        --autonima <autonima-results>/projects --out <dir>

The record-arms experiment compares full-text autonima against autonima reading the
record, and it compares them as *maps*: each arm's `outputs/nimads_studyset.json` and
`nimads_annotation.json` go through `code/run_mkda.py`, and `code/meta_r2.py` correlates
the result with the meta-analysis's published map. This writes the same two files for the
deterministic query, so it enters those figures as two more arms rather than as a separate
table.

AN ARM IS END TO END, so the query arm is both gates: the published inclusion criteria
choose the papers and the published contrast chooses their analyses. `query, strict` is
strict at both, `query, permissive` permissive at both -- a paper or an analysis the
record cannot answer for is dropped by the first and kept by the second.

Two conventions are copied from autonima rather than improved on, because the figure is a
comparison of selectors and every other difference is noise in it:

* coordinates go in as the paper published them, with a `space` label beside them, which
  is what autonima's exports hold -- 1,527 Talairach points sit in cue reactivity's
  studyset as Talairach;
* the note keys are autonima's annotation names from the project's `nmb_mappings.json`,
  not the benchmark's own key names, so `run_mkda.py` and `meta_r2.py` find them where
  they already look.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from query_contrasts import KEYS, UNANSWERABLE, judge_analyses  # noqa: E402
from query_foci import parse_entries, raw_foci_of, read  # noqa: E402
from query_metaanalyses import QUERIES  # noqa: E402
from query_workflow import passed, screen  # noqa: E402

MODES = {"query, strict": False, "query, permissive": True}

#: The run directory each mode is written to, under `<out>/<project>/`.
RUNS = {"query, strict": "query-strict", "query, permissive": "query-permissive"}


def load_records(pattern: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        body = json.loads(Path(path).read_text())
        out[Path(path).parent.name][Path(path).name.split(".")[0]] = body.get("study") or body
    return out


def mappings(autonima: Path, project: str) -> dict[str, str]:
    path = autonima / project / "nmb_mappings.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("annotation_mappings") or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--stage1", required=True, type=Path)
    ap.add_argument("--autonima", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    records = load_records(args.records)

    for _meta_pmid, (project, predicates) in QUERIES.items():
        here = records.get(project) or {}
        annotations = mappings(args.autonima, project)
        wanted = {manual: auto for manual, auto in annotations.items()
                  if (project, manual) in KEYS}
        if not here or not wanted:
            continue
        verdicts = screen(here, predicates)

        for label, permissive in MODES.items():
            screened = {p for p in here if passed(verdicts[p], permissive)}
            # analysis id -> (pmid, name, points, space) and the keys it was selected for
            chosen: dict[str, tuple] = {}
            flags: dict[str, dict[str, bool]] = defaultdict(dict)
            for pmid in sorted(screened):
                body = here[pmid]
                entries = parse_entries(args.stage1, pmid)
                if entries is None:
                    continue
                picked: dict[str, dict[str, bool]] = defaultdict(dict)
                for manual, auto in wanted.items():
                    judged = judge_analyses(body, KEYS[(project, manual)])
                    for local_id, verdict in judged.items():
                        picked[local_id][auto] = bool(
                            verdict is True or (permissive and verdict is UNANSWERABLE))
                for analysis in body.get("analyses") or []:
                    if not isinstance(analysis, dict):
                        continue
                    local_id = str(analysis.get("local_id"))
                    if not any(picked.get(local_id, {}).values()):
                        continue
                    found = raw_foci_of(analysis, body, entries)
                    if not found or not found[0]:
                        # Selected and unmappable. An analysis with no coordinates cannot
                        # enter a studyset, and counting it as selected anywhere else while
                        # dropping it here is what makes the two numbers differ.
                        continue
                    points, space = found
                    chosen[f"{pmid}_{local_id}"] = (
                        pmid, str(read(analysis.get("name")) or local_id), points, space)
                    flags[f"{pmid}_{local_id}"] = dict(picked[local_id])

            by_study: dict[str, list] = defaultdict(list)
            for analysis_id, (pmid, name, points, space) in chosen.items():
                by_study[pmid].append({
                    "id": analysis_id,
                    "name": name,
                    "conditions": [{"name": "default", "description": ""}],
                    "weights": [1.0],
                    "images": [],
                    "points": [{"space": space, "coordinates": [float(c) for c in xyz]}
                               for xyz in points],
                    "metadata": {},
                    "study_id": pmid,
                })

            run = args.out / project / RUNS[label] / "outputs"
            run.mkdir(parents=True, exist_ok=True)
            studyset_id = f"query_{project}_{RUNS[label]}"
            (run / "nimads_studyset.json").write_text(json.dumps({
                "id": studyset_id,
                "name": f"{project} {label}",
                "description": "Deterministic query over pondie records: published "
                               "inclusion criteria, then the published contrast.",
                "pmid": None,
                "studies": [
                    {"id": pmid, "name": pmid, "authors": "", "publication": "",
                     "metadata": {"pmids": pmid}, "analyses": analyses}
                    for pmid, analyses in sorted(by_study.items())
                ],
            }, indent=1))
            (run / "nimads_annotation.json").write_text(json.dumps({
                "id": f"annotation_{studyset_id}",
                "name": f"{project} {label} annotations",
                "description": "One boolean per published map, from the contrast spec in "
                               "scripts/query_contrasts.py.",
                "metadata": {"mode": label},
                "note_keys": {auto: "boolean" for auto in sorted(set(wanted.values()))},
                "studyset": studyset_id,
                "notes": [
                    {"analysis": analysis_id,
                     "annotation": f"annotation_{studyset_id}",
                     "note": {auto: bool(flags[analysis_id].get(auto))
                              for auto in sorted(set(wanted.values()))}}
                    for analysis_id in sorted(chosen)
                ],
            }, indent=1))
            counts = {auto: sum(1 for f in flags.values() if f.get(auto))
                      for auto in sorted(set(wanted.values()))}
            print(f"{project:24} {label:18} {len(by_study):4d} studies "
                  f"{len(chosen):5d} analyses   {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
