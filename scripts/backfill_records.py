#!/usr/bin/env python3
"""Apply the deterministic fills a corpus was extracted before.

    python scripts/backfill_records.py --records '<dir>/*/*.extraction.json' [--write]

THE RECORDS ARE OLDER THAN THE SCHEMA, and a backfilled record is a hybrid: yesterday's
extraction with today's code-filled slots. That is fine for the slots here and not fine in
general, so both are checked rather than assumed:

  `unwrap_singletons`  Reads the CURRENT schema to decide which wrappers are scalar. Safe
                       here because no slot it touches changed cardinality since extraction
                       -- `Region.region_type` and `Region.definition_method`, which hold
                       the most multi-item values, were scalar at extraction too, so the
                       repair fixes a defect that was already one. A slot that had been
                       multivalued then and scalar now would need a different argument.

  `study_type`         Comes from PubMed rather than from the paper, so it does not depend
                       on the schema the record was extracted against at all.

Everything else is left alone. `--all-repairs` runs the rest, and they are not the default
because each carries its own staleness argument and this script should not make five at once.

What the backfill does NOT fix is reported rather than hidden: the modality rename and the
cohort-trait fields landed after these records were written, so a record still says
`response_mode` and still carries no `population_characteristics`. `repaired_by` is stamped
so a record says it is no longer the one the extractor produced -- the reason that field
exists.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from pondie import schema
from pondie.extraction import pubmed
from pondie.extraction.record import builder as br
from pondie.schema import reader

STAMP = "backfill-1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--write", action="store_true", help="without it, nothing is written")
    ap.add_argument("--all-repairs", action="store_true")
    ap.add_argument("--no-pubmed", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.records))
    bodies: dict[Path, dict] = {}
    for path in paths:
        raw = json.loads(Path(path).read_text())
        bodies[Path(path)] = raw

    types: dict[str, list[str]] = {}
    if not args.no_pubmed:
        pmids = [p.name.split(".")[0] for p in bodies]
        types = pubmed.publication_types(pmids)
        print(f"PubMed answered for {len(types):,} of {len(pmids):,} ids")

    sch = reader.load(schema.EXTRACTION)
    changed = Counter()
    touched = 0
    for path, raw in bodies.items():
        body = raw.get("study") or raw
        before = json.dumps(body, sort_keys=True)
        counts = {
            "unwrap_singletons": len(br.unwrap_singleton_lists(body, sch)),
            "study_type": len(pubmed.fill(body, types)),
        }
        if args.all_repairs:
            counts["name_links"] = len(br.link_entities_by_name(body, sch))
            counts["redundant_levels"] = len(br.drop_redundant_cell_levels(body))
            counts["conclusions"] = len(br.relabel_conclusions(body, sch))
        changed.update({k: v for k, v in counts.items() if v})
        if json.dumps(body, sort_keys=True) == before:
            continue
        touched += 1
        metadata = body.setdefault("extraction_metadata", {})
        stamped = str(metadata.get("repaired_by") or "")
        if STAMP not in stamped:
            metadata["repaired_by"] = f"{stamped}+{STAMP}" if stamped else STAMP
        if args.write:
            path.write_text(json.dumps(raw, indent=1, ensure_ascii=False) + "\n")

    print(f"\n{touched:,} of {len(bodies):,} records changed"
          f"{'' if args.write else '  (dry run -- nothing written)'}")
    for name, count in changed.most_common():
        print(f"  {count:8,d}  {name}")

    kinds = Counter()
    for _path, raw in bodies.items():
        body = raw.get("study") or raw
        for group in body.get("groups") or []:
            if isinstance(group, dict) and "population_characteristics" not in group:
                kinds["groups with no population_characteristics"] += 1
        for task in body.get("tasks") or []:
            if isinstance(task, dict) and "response_mode" in task:
                kinds["tasks still using response_mode"] += 1
    print("\nstill stale, because the schema moved after these were extracted:")
    for name, count in kinds.most_common():
        print(f"  {count:8,d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
