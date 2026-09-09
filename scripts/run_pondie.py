"""Run the pondie extractor over the record-arm corpus.

Thin driver, adapted from fdcr/scripts/07_pondie_extract.py. The six stages, their order,
the cost accounting and the record validation are pondie's; this decides which papers run,
on what hardware, and in what size batch.

Two reasons it is not `pondie extract`:

  * the CLI exposes no way to set `reranker_devices`, whose default is `("cpu",)` -- on a
    four-GPU host that leaves the evidence reranker on the CPU and the run takes days;
  * `driver.run` appends `usage.jsonl` only when it returns, so one call over 900 papers
    loses every token count if it dies at paper 800. Dispatching in batches caps that loss
    at one batch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EXP = Path("/data/james/pondie-vs-fulltext")
DEFAULT_MODEL = "@psyc-aid338-ope-333f18/gpt-5.6-luna"


def ready_studies(corpus: Path, require_stage1: bool = True) -> list[tuple[str, str]]:
    """(pmid, study_id) for every corpus paper the extractor can actually read."""
    rows = []
    for study_dir in sorted(p for p in corpus.iterdir() if p.is_dir()):
        if not (study_dir / "processed" / "local" / "text.tables.txt").is_file():
            continue
        if require_stage1 and not (study_dir / "stage1" / "analyses.json").is_file():
            continue
        pmid = study_dir.name
        prov = study_dir / "provenance.json"
        if prov.is_file():
            pmid = str((json.loads(prov.read_text()) or {}).get("pmid") or pmid)
        rows.append((pmid, study_dir.name))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=EXP / "corpus")
    ap.add_argument("--run", required=True, help="names the run directory")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--env", type=Path, default=EXP / "repos/autonima-results/.env")
    ap.add_argument("--effort", default="low",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*", help="restrict to these study ids")
    ap.add_argument("--only-file", type=Path, help="one study id per line")
    ap.add_argument("--tier", choices=["parsed", "empty"],
                    help="restrict to one stage-1 tier")
    ap.add_argument("--stages", nargs="*")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--plan", action="store_true", help="say what would run, spend nothing")
    ap.add_argument("--service-tier", default="", choices=["", "flex", "default", "priority"],
                    help="the provider's tier; `flex` trades latency for price")
    # `--devices` reaches the evidence reranker and nothing else. The repair stage places
    # its own two models from `visible_devices`, which defaults to leaving the environment
    # alone -- so restricting the reranker to three cards still let MiniCheck and NuExtract
    # load onto the fourth, and 18823721 died of OOM on a card another process owned.
    ap.add_argument("--proposer-kind", default="local", choices=["local", "model"],
                    help="who answers the repair sweep: NuExtract on a card, or the "
                         "extraction model over the network")
    ap.add_argument("--visible-devices", default="",
                    help="CUDA_VISIBLE_DEVICES for the repair models, e.g. 0,1,2")
    ap.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3",
                    help="evidence reranker device pool; pondie's own default is cpu")
    args = ap.parse_args()

    rows = ready_studies(args.corpus)
    if args.only or args.only_file:
        keep = set(args.only or [])
        if args.only_file:
            keep |= {l.strip() for l in args.only_file.read_text().splitlines() if l.strip()}
        rows = [r for r in rows if r[1] in keep]
    if args.tier:
        rows = [
            r for r in rows
            if json.loads((args.corpus / r[1] / "provenance.json").read_text())
            .get("stage1_tier") == args.tier
        ]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no corpus papers are ready", file=sys.stderr)
        return 2

    # Set before importing pondie: `pondie.paths` reads PONDIE_DATA_DIR at import time.
    os.environ.setdefault("PONDIE_DATA_DIR", str(EXP / "pondie-data"))

    from pondie import paths
    from pondie.extraction import GatewayCaller, load_env, plan, run
    from pondie.extraction.models import Paper, Settings, StageName, Workflow

    if args.env and args.env.is_file():
        load_env(args.env)

    run_dir = paths.run(args.run)
    settings = Settings(
        payloads=run_dir / "payloads",
        records=run_dir / "records",
        model=args.model,
        workflow=Workflow.demand_driven,
        stages=tuple(StageName(s) for s in args.stages) if args.stages else tuple(StageName),
        effort=args.effort,
        retrieve_evidence=not args.no_evidence,
        redo=args.redo,
        reranker_devices=tuple(d.strip() for d in args.devices.split(",") if d.strip()),
        visible_devices=args.visible_devices,
        proposer_kind=args.proposer_kind,
        service_tier=args.service_tier,
    )
    papers = [Paper(study_id=study, root=args.corpus, flavour=paths.Flavour.local)
              for _pmid, study in rows]
    print(f"{len(papers)} papers | model={args.model} effort={args.effort} "
          f"workers={args.workers} devices={settings.reranker_devices} "
          f"tier={settings.service_tier or 'unset'} repair={settings.repair}")

    if args.plan:
        for study, steps in plan(papers, settings).items():
            print(f"  {study}  {' '.join(steps)}")
        return 0

    # Papers already done are skipped inside `run` by each stage's `done()`, so the bar
    # counts what was dispatched rather than what was billed -- it is a progress bar, not an
    # accounting of spend, and `usage.jsonl` remains the ledger.
    from tqdm import tqdm

    failures = []
    bar = tqdm(total=len(papers), unit="paper", dynamic_ncols=True,
               desc=f"{args.run} [{args.service_tier or 'default'}]")
    for start in range(0, len(papers), args.batch):
        chunk = papers[start : start + args.batch]
        report = run(chunk, settings, GatewayCaller(), workers=args.workers)
        bar.update(len(chunk))
        bar.set_postfix_str(f"{len(failures)} failed")
        # `write`, not `print`: a bare print interleaves with the bar and leaves a half-drawn
        # line in the log every batch.
        bar.write(f"[{start + len(chunk)}/{len(papers)}] {report.summary()}")
        for paper in report.failures:
            bar.write(f"  FAILED {paper.study_id}: {paper.failed.reason}")
            failures.append((paper.study_id, paper.failed.reason))
    bar.close()

    print(f"\ndone: {len(papers) - len(failures)} ok, {len(failures)} failed")
    (run_dir / "failures.json").write_text(json.dumps(failures, indent=1) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
