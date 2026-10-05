"""Re-stamp a run's existing stage outputs under the current code, without re-running them.

    python restamp.py --run pondie_single-v4 --pmids all_ptsd.pmids --stages tables prose split single

For when a stage's `depends_on` changed but its output did not -- the `tables` fix in
545896c -- so a seeded run can reuse the payloads instead of re-asking the model.
"""
import argparse
from pathlib import Path

from pondie import paths
from pondie.extraction.models import Flavour, Paper, Settings, StageName
from pondie.extraction.stages import DEMAND_DRIVEN, SINGLE_PASS

import arms
from run_arm import stamp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pmids", required=True)
    ap.add_argument("--stages", nargs="+", required=True)
    args = ap.parse_args()
    run_dir = paths.run(args.run)
    settings = Settings(payloads=run_dir / "payloads", records=run_dir / "records",
                        model=arms.MODEL, service_tier="flex", max_output_tokens=64_000,
                        stages=tuple(StageName(s) for s in args.stages))
    by_name = {s.name: s for s in (*DEMAND_DRIVEN, *SINGLE_PASS)}
    for pmid in Path(args.pmids).read_text().split():
        paper = Paper(study_id=pmid, root=run_dir / "corpus", flavour=Flavour.local)
        for name in args.stages:
            stamp(by_name[StageName(name)], paper, settings, 0.0)
    print("restamped", args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
