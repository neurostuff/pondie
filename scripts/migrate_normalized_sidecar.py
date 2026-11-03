#!/usr/bin/env python3
"""Move normalizer output into `normalized` sidecars on records written before them.

    python scripts/migrate_normalized_sidecar.py --records '<dir>/*/*.extraction.json' \\
        [--groups-medical derived/groups_medical.parquet] [--tasks derived/tasks.parquet] \\
        [--write]

study_schema#56 replaced two sibling slots with a sidecar on every extracted value. A record
written before it fails the new extraction models, and its other normalizations live only in
tables beside it. This does three things, and nothing else:

  siblings        `Group.age_unit_normalized` and `CategoryDistribution.category_normalized`
                  are removed, and their value becomes one entry on `age_unit` / `category`.
                  The commit that wrote a sibling is not recorded anywhere, so the entry's
                  `mapping_tool` says `pre-sidecar`; the next `apply_normalized` over the
                  record replaces it, because the normalizer's name is the same. A sibling
                  whose group has no `age_unit` wrapper has nowhere to go and is dropped.
  groups_medical  each row becomes a MONDO entry (and an ONVOC one where the row has it) on
                  its group's `medical_condition`, at `item` = the row's `index`. A row with
                  no index cannot say which item it is about and is skipped.
  tasks           each row's canonical task becomes an entry on that task's `name`.

The table rows are the astra build's, not pondie's: `mapping_tool` names the table and the
directory it was read from. Without `--write` nothing is written and the tally is printed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path

from pondie.normalization import UNKNOWN, _sidecar
from pondie.normalization._records import value_of

PRE_SIDECAR = "pre-sidecar"

#: Sibling owner -> (module that wrote it, vocabulary).
SIBLINGS = {
    "sex_distribution": ("sex_distribution", "SexCategory"),
    "handedness_distribution": ("handedness_distribution", "HandednessCategory"),
}

#: A canonical task is decided from more than the name it is recorded on.
TASK_ALSO_READ = ["description", "conditions"]


def _rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    if path.suffix == ".parquet":
        import pandas as pd  # pyarrow too; neither is a pondie dependency

        frame = pd.read_parquet(path)
        return [
            {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}
            for row in frame.to_dict("records")
        ]
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _blank(value: object) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def _sibling_entry(sibling: dict, module: str, vocabulary: str) -> dict:
    value = value_of(sibling)
    mapping_tool = f"pondie.normalization.{module} {PRE_SIDECAR}"
    method = "moved from " + ("age_unit_normalized" if module == "age_unit" else "category_normalized")
    if not isinstance(value, str) or value == UNKNOWN:
        return _sidecar.entry(vocabulary, _sidecar.UNMAPPED, mapping_tool, method=method)
    return _sidecar.entry(vocabulary, _sidecar.MAPPED, mapping_tool, value=value, method=method)


def move_siblings(body: dict, tally: Counter) -> None:
    for group in body.get("groups") or []:
        if not isinstance(group, dict):
            continue
        sibling = group.pop("age_unit_normalized", None)
        if sibling is not None:
            wrapper = group.get("age_unit")
            if isinstance(wrapper, dict):
                made = _sibling_entry(sibling, "age_unit", "AgeUnit")
                _sidecar.replace(wrapper, made["mapping_tool"], [made])
                tally["age_unit moved"] += 1
            else:
                tally["age_unit dropped, no wrapper"] += 1
        for slot, (module, vocabulary) in SIBLINGS.items():
            for entry in group.get(slot) or []:
                if not isinstance(entry, dict):
                    continue
                sibling = entry.pop("category_normalized", None)
                if sibling is None:
                    continue
                wrapper = entry.get("category")
                if isinstance(wrapper, dict):
                    made = _sibling_entry(sibling, module, vocabulary)
                    _sidecar.replace(wrapper, made["mapping_tool"], [made])
                    tally[f"{slot} moved"] += 1
                else:
                    tally[f"{slot} dropped, no wrapper"] += 1


def medical_entries(row: dict, mapping_tool: str) -> list[dict]:
    """One `groups_medical` row as entries, the same shape `medical_condition.entries` writes."""
    where = {"item": int(row["index"]), "method": row["route"]}
    if row["route"] == "NO_CONDITION":
        return [_sidecar.entry("MONDO", _sidecar.ABSENT, mapping_tool, **where)]
    score = None if _blank(row.get("score")) else min(1.0, max(0.0, float(row["score"])))
    if _blank(row.get("mondo")):
        return [_sidecar.entry("MONDO", _sidecar.UNMAPPED, mapping_tool, confidence=score,
                               **where)]
    out = [_sidecar.entry(
        "MONDO", _sidecar.MAPPED, mapping_tool, value=row["mondo"], label=row.get("label"),
        confidence=score,
        # The candidate a review row was accepted for is the coarser concept the string names.
        rollup=True if row["route"] == "review-contained" else None, **where,
    )]
    if not _blank(row.get("onvoc_id")):
        out.append(_sidecar.entry(
            "ONVOC", _sidecar.MAPPED, mapping_tool, value=row["onvoc_id"],
            label=None if _blank(row.get("onvoc")) else row["onvoc"],
            rollup=True if row.get("query_via") == "rollup" else None, **where,
        ))
    return out


def attach_medical(body: dict, rows: list[dict], mapping_tool: str, tally: Counter) -> None:
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(str(row["group"]), []).append(row)
    for group in body.get("groups") or []:
        if not isinstance(group, dict) or not isinstance(group.get("medical_condition"), dict):
            continue
        written = []
        for row in by_group.get(str(value_of(group.get("local_id")) or ""), []):
            if row["route"] == "empty":
                continue
            if _blank(row.get("index")):
                tally["medical rows skipped, no index"] += 1
                continue
            written += medical_entries(row, mapping_tool)
        if written:
            written.sort(key=lambda e: e["item"])
            _sidecar.replace(group["medical_condition"], mapping_tool, written)
            tally["medical entries"] += len(written)


def attach_tasks(body: dict, rows: list[dict], mapping_tool: str, tally: Counter) -> None:
    by_task = {str(row["task"]): row for row in rows}
    for task in body.get("tasks") or []:
        if not isinstance(task, dict) or not isinstance(task.get("name"), dict):
            continue
        row = by_task.get(str(value_of(task.get("local_id")) or ""))
        if row is None:
            continue
        if _blank(row.get("canonical")):
            made = _sidecar.entry("astra canonical task", _sidecar.UNMAPPED, mapping_tool,
                                  method=row.get("decided_by"), also_read=TASK_ALSO_READ)
        else:
            made = _sidecar.entry(
                "astra canonical task", _sidecar.MAPPED, mapping_tool, value=row["canonical"],
                label=None if _blank(row.get("canonical_name")) else row["canonical_name"],
                # The table's qualifiers are `{text, axis}`; the sidecar holds the words.
                qualifiers=[q["text"] if isinstance(q, dict) else str(q)
                            for q in row.get("qualifiers") or []],
                method=row.get("decided_by"), also_read=TASK_ALSO_READ,
            )
        _sidecar.replace(task["name"], mapping_tool, [made])
        tally["task entries"] += 1


def _tool(table: str, path: Path | None) -> str:
    # `<snapshot>/derived/<table>.parquet`: the snapshot directory is the version.
    return f"astra_norm.{table} {path.resolve().parent.parent.name}" if path else ""


def migrate(
    body: dict,
    study: str,
    medical: dict[str, list[dict]],
    tasks: dict[str, list[dict]],
    medical_tool: str,
    tasks_tool: str,
    tally: Counter,
) -> None:
    move_siblings(body, tally)
    attach_medical(body, medical.get(study, []), medical_tool, tally)
    attach_tasks(body, tasks.get(study, []), tasks_tool, tally)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--groups-medical", type=Path, help="the astra `groups_medical` table")
    ap.add_argument("--tasks", type=Path, help="the astra `tasks` table")
    ap.add_argument("--write", action="store_true", help="without it, nothing is written")
    args = ap.parse_args()

    medical: dict[str, list[dict]] = {}
    for row in _rows(args.groups_medical):
        medical.setdefault(str(row["study"]), []).append(row)
    tasks: dict[str, list[dict]] = {}
    for row in _rows(args.tasks):
        tasks.setdefault(str(row["study"]), []).append(row)
    medical_tool, tasks_tool = _tool("groups_medical", args.groups_medical), _tool(
        "tasks", args.tasks)

    tally: Counter = Counter()
    for path in sorted(glob.glob(args.records)):
        if path.endswith(".raw.json"):
            continue
        document = json.loads(Path(path).read_text())
        body = document.get("study") or document
        before = json.dumps(body, sort_keys=True)
        migrate(body, Path(path).name.split(".")[0], medical, tasks, medical_tool, tasks_tool,
                tally)
        tally["records"] += 1
        if json.dumps(body, sort_keys=True) != before:
            tally["records changed"] += 1
            if args.write:
                Path(path).write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")
    for key, n in sorted(tally.items()):
        print(f"  {key:36} {n}")
    if not args.write:
        print("dry run: nothing written (--write to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
