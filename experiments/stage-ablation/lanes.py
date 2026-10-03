"""Run a list of panel draws a few at a time, so concurrency stays under the flex budget.

    python lanes.py --lanes 2 --workers 6 --pmids panel_dementia.pmids \\
        dem_panel_B_r2: dem_panel_inventory_r2:inventory dem_panel_no_worked_r1:no_worked

Each job is RUN:VARIANT[+VARIANT...][@EFFORT]. A finished run (one record per pmid) is skipped.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from pondie import paths


def done(run: str, pmids: list[str]) -> bool:
    records = paths.run(run) / "records"
    return records.is_dir() and all((records / f"{p}.extraction.json").is_file() for p in pmids)


def command(job: str, pmids_file: str, workers: int) -> list[str]:
    run, _, rest = job.partition(":")
    variants, _, effort = rest.partition("@")
    cmd = [sys.executable, "run_arm.py", "--arm", "pondie_single", "--run", run,
           "--pmids", pmids_file, "--workers", str(workers)]
    if variants:
        cmd += ["--variant", *variants.split("+")]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--pmids", required=True)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    pmids = Path(args.pmids).read_text().split()
    pending = [j for j in args.jobs if not done(j.partition(":")[0], pmids)]
    running: list[tuple[str, subprocess.Popen]] = []
    while pending or running:
        running = [(j, p) for j, p in running if p.poll() is None]
        while pending and len(running) < args.lanes:
            job = pending.pop(0)
            log = open(paths.DATA / "logs" / f"{job.partition(':')[0]}.log", "w")
            running.append((job, subprocess.Popen(command(job, args.pmids, args.workers),
                                                  stdout=log, stderr=subprocess.STDOUT)))
            print(time.strftime("%H:%M:%S"), "started", job, flush=True)
        time.sleep(20)
    print(time.strftime("%H:%M:%S"), "all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
