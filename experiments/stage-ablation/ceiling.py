"""How much of the gold each paper's INPUTS carry, before any model runs.

    python ceiling.py --corpus /data/james/pondie-ablation/corpus --meta 36100907

For every gold analysis: how many of its foci are in the stage-1 parse (within 1 mm), and how
many appear as a number triple anywhere in the text. A focus in neither cannot be recovered by
any arm, so this is the ceiling the ablation is measured under.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--meta", default="36100907")
    args = ap.parse_args()
    totals = [0, 0, 0]
    for pmid, analyses in sorted(gold.gold_studyset(args.meta).items()):
        d = args.corpus / pmid
        if not d.is_dir():
            print(pmid, "NO CORPUS ENTRY")
            continue
        text = (d / "processed/local/text.tables.txt").read_text()
        parse = json.loads((d / "stage1/analyses.orig.json").read_text())
        parsed = [tuple(p["coordinates"]) for a in parse.get("analyses") or []
                  for p in a.get("points") or []]
        for a in analyses:
            n = len(a["points"])
            in_parse = sum(any(gold.near(g, q) for q in parsed) for g in a["points"])
            in_text = sum(gold.in_text(g, text) for g in a["points"])
            totals = [totals[0] + n, totals[1] + in_parse, totals[2] + in_text]
            print(f"{pmid} {a['name'][:28]:28s} n={a['n']} foci={n:3d} parse={in_parse:3d} "
                  f"text={in_text:3d} {a['space']}")
    print(f"TOTAL foci={totals[0]} in_parse={totals[1]} in_text={totals[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
