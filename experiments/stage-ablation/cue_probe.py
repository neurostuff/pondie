"""Per paper, two runs: analyses whose cue > control the query reads as True, and selection.

    PONDIE_DATA_DIR=... PYTHONPATH=<code>:. python cue_probe.py RUN_A RUN_B PMIDS
"""
import json
import sys
from pathlib import Path

from pondie import paths
import queries as q


def main() -> int:
    a, b, pmids = sys.argv[1], sys.argv[2], Path(sys.argv[3]).read_text().split()
    gold = set(Path("cue.gold.pmids").read_text().split())
    pmids = [p for p in pmids if p.isdigit()]
    for p in pmids:
        row = [p, "gold" if p in gold else "neg"]
        for run in (a, b):
            f = paths.run(run) / "records" / f"{p}.extraction.json"
            if not f.is_file():
                row.append("no record"); continue
            rec = json.loads(f.read_text())
            rec["_input_coordinates"] = rec["_gold_coordinates"] = 1
            rec["_pubdate"] = (2015, 1)
            r = q.evaluate(rec, "34400176")
            true = sum(x["answers"]["cue > control"] is True for x in r["analyses"])
            row.append(f"{true}/{len(r['analyses'])} cue>control, veto={r['veto']}")
        print("  ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
