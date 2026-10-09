#!/usr/bin/env python3
"""Move records onto the slots the schema renamed or retired since they were extracted.

    python scripts/migrate_records.py --records '<glob>' [--records '<glob>' ...] [--write]

Only renames whose meaning carries over one to one, and the removal of slots the schema
dropped with no successor. Nothing is re-read from the paper and nothing is inferred, so a
migrated record says the same things it said before, under today's names. A record whose
facts do not map cleanly is left untouched and reported, never guessed at. Every record it
changes is stamped `repaired_by`, because it is no longer the record the extractor produced.

  Task.response_mode                 -> Task.response_modality. Renamed with the stimulus
                                        modality, same vocabulary, already a list.
  Task.stimuli, a string             -> a one-item list. The slot became multivalued; the
                                        string was one entry, and splitting it would be a
                                        reading of the paper.
  InferenceSettings
    .voxelwise_threshold_value/_type -> .height_threshold_value/_type
    .cluster_forming_threshold_value -> .height_threshold_value, when it is the only height
                                        threshold the record states: the cluster-forming
                                        threshold is the height threshold doing its other
                                        job. Both stated and different is a conflict, and
                                        the record is reported instead.
  ConnectivityDetails
    .parameter_sign, .parameter_change  dropped. Direction lives only in `Effect.cells`.
  ConnectivityEdge.directionality       dropped. An edge's direction is its source and
                                        target.

`.raw.json` files are what the model returned and are never touched; pass globs that match
the `.extraction.json` records only, and anything ending in `.raw.json` is skipped anyway.
Without `--write` nothing is written. A file is rewritten in exactly the JSON style it was
read in, or not at all.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

STAMP = "schema-migrate-1"

THRESHOLD_RENAMES = {
    "voxelwise_threshold_value": "height_threshold_value",
    "voxelwise_threshold_type": "height_threshold_type",
}


class Conflict(Exception):
    """A record whose facts do not map onto the new slots without a judgement."""


def _rename(node: dict, old: str, new: str) -> dict:
    """`node` with `old` renamed to `new`, keeping the key's place."""
    if new in node:
        raise Conflict(f"both {old} and {new} are present")
    return {new if key == old else key: value for key, value in node.items()}


def _stated(wrapper: Any) -> bool:
    return isinstance(wrapper, dict) and wrapper.get("extraction_status") == "extracted"


def migrate_task(task: dict, counts: Counter) -> dict:
    if "response_mode" in task:
        task = _rename(task, "response_mode", "response_modality")
        counts["Task.response_mode -> response_modality"] += 1
    stimuli = task.get("stimuli")
    if isinstance(stimuli, dict) and isinstance(stimuli.get("value"), str):
        stimuli["value"] = [stimuli["value"]]
        counts["Task.stimuli string -> one-item list"] += 1
    return task


def migrate_inference(settings: dict, counts: Counter) -> dict:
    for old, new in THRESHOLD_RENAMES.items():
        if old in settings:
            settings = _rename(settings, old, new)
            counts[f"InferenceSettings.{old} -> {new}"] += 1
    forming = settings.get("cluster_forming_threshold_value")
    if forming is None:
        return settings
    height = settings.get("height_threshold_value")
    if _stated(forming) and _stated(height) and forming.get("value") != height.get("value"):
        raise Conflict(
            f"{settings.get('local_id')}: cluster-forming threshold {forming.get('value')} "
            f"and height threshold {height.get('value')} disagree"
        )
    if _stated(forming) and not _stated(height):
        settings = {
            ("height_threshold_value" if key == "cluster_forming_threshold_value" else key): value
            for key, value in settings.items()
            if key != "height_threshold_value"
        }
        counts["InferenceSettings.cluster_forming_threshold_value -> height_threshold_value"] += 1
    else:
        settings = {k: v for k, v in settings.items() if k != "cluster_forming_threshold_value"}
        counts["InferenceSettings.cluster_forming_threshold_value dropped (said nothing new)"] += 1
    return settings


def migrate_details(details: dict, counts: Counter) -> dict:
    for retired in ("parameter_sign", "parameter_change"):
        if retired in details:
            stated = "stated" if _stated(details[retired]) else "not reported"
            counts[f"ConnectivityDetails.{retired} dropped ({stated})"] += 1
    details = {k: v for k, v in details.items() if k not in ("parameter_sign", "parameter_change")}
    edges = details.get("edges")
    if isinstance(edges, list):
        for index, edge in enumerate(edges):
            if isinstance(edge, dict) and "directionality" in edge:
                edges[index] = {k: v for k, v in edge.items() if k != "directionality"}
                counts["ConnectivityEdge.directionality dropped"] += 1
    return details


def migrate(body: dict, counts: Counter) -> None:
    """Migrate one record in place. Raises Conflict, leaving `counts` to be discarded."""
    if isinstance(body.get("tasks"), list):
        body["tasks"] = [
            migrate_task(task, counts) if isinstance(task, dict) else task
            for task in body["tasks"]
        ]
    if isinstance(body.get("inference_settings"), list):
        body["inference_settings"] = [
            migrate_inference(item, counts) if isinstance(item, dict) else item
            for item in body["inference_settings"]
        ]
    for analysis in body.get("analyses") or []:
        details = analysis.get("details") if isinstance(analysis, dict) else None
        if isinstance(details, dict):
            analysis["details"] = migrate_details(details, counts)


def _style(text: str, data: Any) -> dict | None:
    """The json.dumps arguments that reproduce `text` exactly, if any do."""
    for indent in (2, 1, 4):
        for ensure_ascii in (False, True):
            for trailer in ("\n", ""):
                if json.dumps(data, indent=indent, ensure_ascii=ensure_ascii) + trailer == text:
                    return {"indent": indent, "ensure_ascii": ensure_ascii, "trailer": trailer}
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", action="append", required=True, help="a glob; repeatable")
    ap.add_argument("--write", action="store_true", help="without it, nothing is written")
    args = ap.parse_args(argv)

    paths = sorted(
        {Path(p) for pattern in args.records for p in glob.glob(pattern)}
        - {Path(p) for pattern in args.records for p in glob.glob(pattern) if p.endswith(".raw.json")}
    )
    totals: Counter = Counter()
    changed = conflicts = unstyled = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text)
        style = _style(text, raw)
        body = raw.get("study") or raw
        before = json.dumps(body, sort_keys=True)
        counts: Counter = Counter()
        try:
            migrate(body, counts)
        except Conflict as reason:
            conflicts += 1
            print(f"  left alone: {path}: {reason}")
            continue
        if json.dumps(body, sort_keys=True) == before:
            continue
        if style is None:
            unstyled += 1
            print(f"  left alone: {path}: its JSON style could not be reproduced")
            continue
        changed += 1
        totals.update(counts)
        metadata = body.setdefault("extraction_metadata", {})
        stamped = str(metadata.get("repaired_by") or "")
        if STAMP not in stamped.split("+"):
            metadata["repaired_by"] = f"{stamped}+{STAMP}" if stamped else STAMP
        if args.write:
            path.write_text(
                json.dumps(raw, indent=style["indent"], ensure_ascii=style["ensure_ascii"])
                + style["trailer"],
                encoding="utf-8",
            )

    print(
        f"\n{changed} of {len(paths)} records migrated"
        f"{'' if args.write else '  (dry run -- nothing written)'}"
        f"; {conflicts} left alone for a conflict, {unstyled} for their JSON style"
    )
    for name, count in sorted(totals.items()):
        print(f"  {count:6d}  {name}")
    return 1 if conflicts or unstyled else 0


if __name__ == "__main__":
    raise SystemExit(main())
