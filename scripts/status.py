"""One-screen status: arm counts against their baseline, and what extraction has cost."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

EXP = Path("/data/james/pondie-vs-fulltext")
REPO = EXP / "repos" / "autonima-results"

PAIRS = [("vbm_of_ptsd", "v1", "v1-A1-mini")]


def metrics(project: str, run: str) -> None:
    run_dir = REPO / "projects" / project / run
    perf = run_dir / "evaluation" / "performance_metrics.json"
    stats = json.loads((run_dir / "outputs" / "final_results.json").read_text())
    prisma = stats["execution_stats"].get("prisma_stats", {})
    retrieval = stats["execution_stats"].get("retrieval", {})
    line = f"  {run:14}"
    if perf.is_file():
        ft = json.loads(perf.read_text()).get("fulltext", {})
        c, m = ft.get("counts", {}), ft.get("metrics", {})
        line += (f" TP={c.get('true_positives')} FN={c.get('false_negatives')} "
                 f"FP={c.get('false_positives')}"
                 f" recall={m.get('recall_in_search', 0):.3f}"
                 f" precision={m.get('precision', 0):.3f}")
    print(line)
    print(f"  {'':14} prisma={prisma}")
    print(f"  {'':14} retrieval={retrieval}")


def main() -> int:
    for project, base, arm in PAIRS:
        print(f"## {project}")
        for run in (base, arm):
            metrics(project, run)
        print()

    records = EXP / "pondie-data" / "runs" / "pondie-907" / "records"
    usage = EXP / "pondie-data" / "runs" / "pondie-907" / "usage.jsonl"
    done = len(list(records.glob("*.json"))) if records.is_dir() else 0
    print(f"## pondie extraction: {done} / 903 records")
    if usage.is_file():
        rows = [json.loads(l) for l in usage.read_text().splitlines() if l.strip()]
        papers = len({r["paper"] for r in rows})
        total = Counter()
        for row in rows:
            for key in ("input_tokens", "output_tokens", "reasoning_tokens", "calls"):
                total[key] += row[key]
        if papers:
            print(f"  per paper: in={total['input_tokens'] // papers:,} "
                  f"out={total['output_tokens'] // papers:,} "
                  f"reasoning={total['reasoning_tokens'] // papers:,} "
                  f"calls={total['calls'] / papers:.1f}")
            print(f"  projected 903: in={total['input_tokens'] / papers * 903 / 1e6:.0f}M "
                  f"out={total['output_tokens'] / papers * 903 / 1e6:.1f}M")
        by_stage = Counter()
        for row in rows:
            by_stage[row["stage"]] += row["input_tokens"]
        print("  input tokens by stage:",
              {k: f"{v/1e6:.1f}M" for k, v in by_stage.items() if v})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
