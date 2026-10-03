"""Apply `Single.complete` to an existing run's replies, as a paired test of the completion.

    python complete_existing.py --from dem_p_B --to dem_p_B+complete --pmids all_dementia.pmids

Copies the run, then for every paper whose `single.json` still has dangling references asks
only for the missing entities, writes the merged payload back and rebuilds the record. Every
other paper is the same draw untouched, so the difference between the two runs is the
completion and nothing else.
"""
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pondie import paths
from pondie.extraction.llm import GatewayCaller, load_env
from pondie.extraction.models import Cost, Flavour, Paper, Settings, StageName
from pondie.extraction.prompt import render
from pondie.extraction.stages import Build, Single, _missing_ids

import arms
from run_arm import ENV


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", required=True)
    ap.add_argument("--to", required=True)
    ap.add_argument("--pmids", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    load_env(ENV)
    src, dst = paths.run(args.source), paths.run(args.to)
    if not dst.exists():
        shutil.copytree(src, dst)
    settings = Settings(payloads=dst / "payloads", records=dst / "records", model=arms.MODEL,
                        service_tier="flex", max_output_tokens=64_000, redo=True,
                        stages=(StageName.single, StageName.build), complete_references=True)
    caller = GatewayCaller()
    single = Single()

    def one(pmid: str) -> dict:
        paper = Paper(study_id=pmid, root=dst / "corpus", flavour=Flavour.local)
        target = settings.payloads / pmid / "single.json"
        if not target.is_file():
            return {"pmid": pmid, "skipped": "no single.json"}
        payload = json.loads(target.read_text())
        missing = _missing_ids(payload, single.existing(paper, settings))
        if not missing:
            return {"pmid": pmid, "missing": 0}
        failures = render.postcondition_failures(
            payload, "single", (), single.listing(paper, settings),
            single.listing_foci(paper, settings), single.existing(paper, settings))
        merged, after, cost, notes = single.complete(
            paper, settings, caller, payload, failures, None)
        target.write_text(json.dumps(merged, indent=1, ensure_ascii=False) + "\n")
        Build().run(paper, settings, None)
        return {"pmid": pmid, "missing": len(missing), "notes": notes,
                "cost": (cost or Cost()).model_dump()}

    pmids = Path(args.pmids).read_text().split()
    with ThreadPoolExecutor(args.workers) as pool, open(dst / "completion.jsonl", "w") as log:
        for result in pool.map(one, pmids):
            log.write(json.dumps(result) + "\n")
            if result.get("missing"):
                print(result["pmid"], result["missing"], result.get("notes"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
