"""Score replicate draws of each variant on a panel of unstable papers.

    python panel.py --meta 35664889 --negatives neg_dementia.pmids \\
        B=dem_panel_B_r1,dem_panel_B_r2,... inventory=dem_panel_inventory_r1,...

One draw over 55 papers moved recall by four gold papers with identical code, so a variant is
judged by its selection RATE over paper-draws on the papers that flip, pooled across
replicates, not by one draw's recall.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import score
from pondie import paths


def cost_per_paper(run: str) -> tuple[float, float, int]:
    calls = tokens = papers = 0
    path = paths.run(run) / "outcomes.jsonl"
    for line in path.read_text().splitlines() if path.is_file() else []:
        papers += 1
        for o in json.loads(line).get("outcomes") or []:
            calls += (o.get("cost") or {}).get("calls", 0)
            tokens += (o.get("cost") or {}).get("input_tokens", 0)
    return calls, tokens, papers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="NAME=run1,run2,...")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--negatives", type=Path, required=True)
    ap.add_argument("--labels", default="benchmark")
    ap.add_argument("--matrix", action="store_true")
    args = ap.parse_args()
    score.GOLD_COORDS, score.OVERLAP, score.LABELS = True, True, args.labels
    negatives = set(args.negatives.read_text().split())
    per_paper: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, str] = {}
    print(f"{'variant':18s} {'draws':>5s} {'gold sel':>10s} {'rate':>5s} {'neg sel':>8s} "
          f"{'empty':>6s} {'calls/p':>7s} {'Mtok/p':>7s}")
    for group in args.groups:
        name, runs = group.split("=", 1)
        g_hit = g_n = n_hit = n_n = empty = calls = tokens = papers = 0
        for run in runs.split(","):
            if not (paths.run(run) / "records").is_dir():
                continue
            rows = score.score(run, False, args.meta, negatives, False, quiet=True)["rows"]
            for pmid, r in rows.items():
                labels[pmid] = r["label"]
                per_paper[pmid][name].append(r["veto"])
                empty += r["analyses"] == 0
                if r["label"] == "gold":
                    g_hit, g_n = g_hit + r["veto"], g_n + 1
                elif r["label"] == "neg":
                    n_hit, n_n = n_hit + r["veto"], n_n + 1
            c, t, p = cost_per_paper(run)
            calls, tokens, papers = calls + c, tokens + t, papers + p
        print(f"{name:18s} {len(runs.split(',')):>5d} {f'{g_hit}/{g_n}':>10s} "
              f"{g_hit / max(1, g_n):5.2f} {f'{n_hit}/{n_n}':>8s} {empty:>6d} "
              f"{calls / max(1, papers):7.2f} {tokens / max(1, papers) / 1e6:7.3f}")
    if args.matrix:
        names = [g.split("=", 1)[0] for g in args.groups]
        print("\n" + f"{'pmid':10s} {'label':5s} " + " ".join(f"{n[:10]:>10s}" for n in names))
        for pmid in sorted(per_paper):
            cells = []
            for n in names:
                v = per_paper[pmid].get(n, [])
                cells.append(f"{sum(v)}/{len(v)}".rjust(10))
            print(f"{pmid:10s} {labels[pmid][:5]:5s} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
