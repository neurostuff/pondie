"""Run only repair's adjudicator over a run's built records, into a new run.

    PONDIE_DATA_DIR=... PYTHONPATH=<code>:. python adjudicate_only.py SRC DST [--workers 8]

Copies SRC's records to DST/records, then puts each record's `contradictions` to the model
in one call (no proposer sweep), at `repair`'s effort, on flex. Prints what was settled.
"""
import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pondie import paths
from pondie.extraction.llm import GatewayCaller, load_env
from pondie.extraction.models import Paper, Settings, StageName
from pondie.extraction.record.validate import EXTRACTION_SCHEMA
from pondie.extraction.repair import stage
from pondie.formats import text_index
from pondie.schema import reader

from run_arm import ENV

MODEL = "@psyc-aid338-ope-333f18/gpt-6-luna"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_env(ENV)
    src, dst = paths.run(args.src), paths.run(args.dst)
    (dst / "records").mkdir(parents=True, exist_ok=True)
    if not (dst / "corpus").exists():
        shutil.copytree(src / "corpus", dst / "corpus")
    sch = reader.load(EXTRACTION_SCHEMA)
    effort = Settings(payloads=dst, records=dst, model=MODEL).effort_for(StageName.repair)
    caller = GatewayCaller()

    def one(path):
        pmid = path.name.split(".")[0]
        record = json.loads(path.read_text())
        text, _digest, _sections = text_index.load(Paper.best(pmid, dst / "corpus").text)
        report = stage.Report()
        cases = len(stage.contradictions(record, sch))
        reply = stage.adjudicate(record, sch, text, caller, study_id=pmid, model=MODEL,
                                 report=report, service_tier="flex", effort=effort,
                                 abbreviations=stage._abbreviations(text, pmid))
        (dst / "records" / path.name).write_text(json.dumps(record, indent=1))
        return pmid, cases, report, reply

    tally, calls = Counter(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        for pmid, cases, report, reply in pool.map(one, sorted((src / "records").glob("*.extraction.json"))):
            calls += reply is not None
            for line in report.adjudicated:
                tally[line.split(": ", 1)[1].split(",")[0].split(" ")[0]] += 1
            print(f"{pmid}: {cases} case(s); " + "; ".join(report.adjudicated)[:400], flush=True)
            for refusal in report.refused:
                print(f"    refused {refusal.slot}: {refusal.why}", flush=True)
    print(f"\n{calls} call(s); outcomes: {dict(tally)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
