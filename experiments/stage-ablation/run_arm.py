"""Run one arm over a list of papers, on flex, papers in parallel.

    PONDIE_DATA_DIR=/data/james/pondie-ablation PYTHONPATH=<code>:. \\
      python run_arm.py --arm mono --pmids gold_ptsd.pmids --run mono-v1 --workers 22

Writes under $PONDIE_DATA_DIR/runs/<run>/: corpus/ (private copy), payloads/, records/
(built), records_raw/ (merged payloads, no repair -- the mono arms only), outcomes.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pondie import paths, pipeline
from pondie.extraction.llm import GatewayCaller, load_env
from pondie.extraction.models import Flavour, Paper, Settings
from pondie.extraction.record.builder import merge_payloads
from pondie.extraction.stages import DEMAND_DRIVEN, SINGLE_PASS

import arms

ENV = Path("/data/james/pondie-vs-fulltext/repos/autonima-results/.env")


def stage_objects(names):
    by_name = {s.name: s for s in (*DEMAND_DRIVEN, *SINGLE_PASS)}
    return [by_name[n] for n in names]


def stamp(stage, paper, settings, seconds):
    """Stamp a pondie stage's output the way `pipeline._run_step` does.

    The harness calls `stage.run` directly to keep the `StageOutcome` (cost, notes), which
    skips the scheduler -- and the scheduler is what writes the freshness stamp. Without it
    a seeded run sees every pondie stage as stale and re-runs it.
    """
    if isinstance(stage, arms.Mono):
        return
    step = stage.as_step(settings)
    output = step.produces(paper)
    if output is None or not output.exists():
        return
    pipeline.Stamp(
        step=step.name,
        digest=pipeline.digest_of(step.depends_on(paper), step.name),
        parts=dict(step.depends_on(paper)),
        produced_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seconds=seconds,
    ).write(output)


def run_paper(pmid, arm, settings, corpus, caller):
    before, custom, after = arms.ARMS[arm]
    paper = Paper(study_id=pmid, root=corpus, flavour=Flavour.local)
    sequence = stage_objects(before) + ([custom] if custom else []) + stage_objects(after)
    outcomes = []
    for stage in sequence:
        started = time.time()
        try:
            outcome = stage.run(paper, settings, caller)
        except Exception as error:  # noqa: BLE001
            outcomes.append({"stage": getattr(stage, "name", "?").value
                             if hasattr(getattr(stage, "name", None), "value") else "?",
                             "error": f"{type(error).__name__}: {str(error)[:500]}"})
            break
        outcomes.append({
            "stage": "mono" if isinstance(stage, arms.Mono) else stage.name.value,
            "ok": outcome.ok, "skipped": outcome.skipped, "reason": outcome.reason[:500],
            "notes": [n[:300] for n in outcome.notes][:20],
            "cost": outcome.cost.model_dump(), "wall": round(time.time() - started, 1)})
        if not outcome.ok:
            break
        stamp(stage, paper, settings, time.time() - started)
    if custom is not None:
        payloads = settings.payloads / pmid
        if payloads.is_dir():
            body, _ = merge_payloads(payloads)
            body["local_id"] = pmid
            raw = settings.records.parent / "records_raw" / f"{pmid}.extraction.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n")
    return {"pmid": pmid, "arm": arm, "outcomes": outcomes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(arms.ARMS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--pmids", required=True)
    ap.add_argument("--corpus", type=Path, default=paths.DATA / "corpus")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--model", default=arms.MODEL)
    ap.add_argument("--seed-from", help="copy this run's corpus and mono.json payloads first")
    args = ap.parse_args()
    load_env(ENV)
    os.environ.setdefault("PONDIE_RUN_ID", args.run)

    pmids = [p for p in Path(args.pmids).read_text().split() if p]
    run_dir = paths.run(args.run)
    (run_dir / "payloads").mkdir(parents=True, exist_ok=True)
    if args.seed_from:
        import shutil
        seed = paths.run(args.seed_from)
        for pmid in pmids:
            if not (run_dir / "corpus" / pmid).is_dir() and (seed / "corpus" / pmid).is_dir():
                shutil.copytree(seed / "corpus" / pmid, run_dir / "corpus" / pmid)
            src = seed / "payloads" / pmid
            if src.is_dir() and not (run_dir / "payloads" / pmid).is_dir():
                shutil.copytree(src, run_dir / "payloads" / pmid,
                                ignore=shutil.ignore_patterns("noev", "fill.json"))
    corpus = arms.private_corpus(args.corpus, run_dir, pmids)
    before, custom, after = arms.ARMS[args.arm]
    settings = Settings(
        payloads=run_dir / "payloads", records=run_dir / "records", model=args.model,
        stages=tuple(before) + tuple(after) or ("build",), effort=args.effort,
        service_tier="flex", max_output_tokens=64_000,
        repair="repair" in [s.value for s in after], adjudicate="repair" in [s.value for s in after],
    )
    (run_dir / "settings.json").write_text(json.dumps(
        {"arm": args.arm, "model": args.model, "effort": args.effort, "pmids": pmids,
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    caller = GatewayCaller()
    log = open(run_dir / "outcomes.jsonl", "a")
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_paper, p, args.arm, settings, corpus, caller): p for p in pmids}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                result = {"pmid": futures[future], "error": repr(error)[:500]}
            log.write(json.dumps(result) + "\n")
            log.flush()
            last = (result.get("outcomes") or [{}])[-1]
            print(f"[{done}/{len(pmids)}] {result['pmid']} "
                  f"{last.get('stage')} ok={last.get('ok')} {last.get('reason') or last.get('error') or ''}"[:300],
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
