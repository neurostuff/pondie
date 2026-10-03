"""One row per run: selection, record health, the probe papers, and cost.

    python compare.py --meta 35664889 --negatives neg_dementia.pmids --probes 25797589,... RUN ...

The probes are papers with a known failure mode; the column counts how many of them the
run gets right (selected if gold). Lower-noise than recall alone: a variant aimed at a
failure should move its probes, whatever the rest of the draw does.
"""
import argparse
from pathlib import Path

import score

KEY = {"36100907": "PTSD effect", "35664889": "bvFTD vs control", "36115222": "users vs controls"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--negatives", type=Path, required=True)
    ap.add_argument("--labels", default="benchmark")
    ap.add_argument("--probes", default="")
    args = ap.parse_args()
    score.GOLD_COORDS, score.OVERLAP, score.LABELS = True, True, args.labels
    negatives = set(args.negatives.read_text().split())
    probes = [p for p in args.probes.split(",") if p]
    print(f"{'run':34s} {'veto R':>7s} {'P':>5s} {'strict R':>8s} {'P':>5s} {'empty':>5s} "
          f"{'an/rec':>6s} {'key T':>5s} {'probes':>6s} {'calls':>5s} {'in Mtok':>7s} {'out Mtok':>8s}")
    for run in args.runs:
        out = score.score(run, False, args.meta, negatives, False, quiet=True)
        rows = out["rows"]
        empty = sum(1 for r in rows.values() if r["analyses"] == 0)
        per = sum(r["analyses"] for r in rows.values()) / max(1, len(rows))
        key = out["tally"].get(KEY[args.meta], {}).get("gold:True", 0)
        hit = sum(1 for p in probes if p in rows and rows[p]["veto"] == (rows[p]["label"] == "gold"))
        c = out["cost"]
        print(f"{run:34s} {out['veto']['recall']:>7s} {out['veto']['precision']:>5s} "
              f"{out['strict']['recall']:>8s} {out['strict']['precision']:>5s} {empty:>5d} "
              f"{per:6.1f} {key:>5d} {f'{hit}/{len(probes)}':>6s} {c.get('calls', 0):>5d} "
              f"{c.get('input_tokens', 0) / 1e6:7.2f} {c.get('output_tokens', 0) / 1e6:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
