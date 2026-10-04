"""Compare arms run over the same papers: what the model wrote, what the record became, cost.

    PONDIE_DATA_DIR=... PYTHONPATH=<code>:. python structured_compare.py RUN [RUN ...]

Per run:
- the raw `single` reply (`payloads/<id>/raw/single.json`, saved before normalizing or
  repair), replayed through what `single` does to it -- `render.normalize`, then the
  payload-local repairs -- with every change counted by repair. These are the faults the
  repair code exists for; a run whose replies need none of them does not need that code.
- the built records' validation errors, against the paper text.
- tokens, calls and seconds per stage, from `usage.jsonl`.
"""
import copy
import json
import sys
from collections import Counter, defaultdict

from pondie import paths, schema
from pondie.extraction.models import Paper
from pondie.extraction.prompt import render
from pondie.extraction.record import fix
from pondie.extraction.record.validate import Validator
from pondie.extraction.stages import Single
from pondie.formats import text_index
from pondie.schema import reader


def replay(run_dir, pmid) -> Counter:
    raw_path = run_dir / "payloads" / pmid / "raw" / "single.json"
    if not raw_path.is_file():
        return Counter({"(no raw reply)": 1})
    payload, notes = render.normalize(copy.deepcopy(json.loads(raw_path.read_text())), "single")
    changes = Counter({"normalize": len(notes)} if notes else {})
    paper = Paper.best(pmid, run_dir / "corpus")
    log = fix.apply_all(
        payload,
        fix.Context(schema=reader.load(schema.STORAGE),
                    stage1=paper.parse if paper.parse.is_file() else None),
        stage=Single().repair_stage,
    )
    for name, lines in log.entries:
        if lines:
            changes[name] += len(lines)
    return changes


def main() -> int:
    sch = reader.load(schema.EXTRACTION)
    for run in sys.argv[1:]:
        run_dir = paths.run(run)
        pmids = sorted(p.name for p in (run_dir / "payloads").iterdir() if p.is_dir())
        faults, per_paper, errors, valid, missing = Counter(), {}, 0, 0, []
        for pmid in pmids:
            changes = replay(run_dir, pmid)
            faults += changes
            per_paper[pmid] = sum(changes.values())
            record_path = run_dir / "records" / f"{pmid}.extraction.json"
            if not record_path.is_file():
                missing.append(pmid)
                continue
            text, _, _ = text_index.load(Paper.best(pmid, run_dir / "corpus").text)
            validator = Validator(sch, text)
            validator.check_record(json.loads(record_path.read_text()))
            errors += len(validator.errors)
            valid += not validator.errors
        cost = defaultdict(Counter)
        for line in (run_dir / "usage.jsonl").read_text().splitlines():
            row = json.loads(line)
            for key in ("calls", "input_tokens", "cached_tokens", "output_tokens",
                        "reasoning_tokens", "seconds"):
                cost[row["stage"]][key] += row.get(key) or 0
        print(f"\n=== {run}: {len(pmids)} papers, {len(missing)} without a record {missing}")
        print(f"  raw single replies: {sum(faults.values())} changes before use, in "
              f"{sum(1 for n in per_paper.values() if n)} of {len(pmids)} papers")
        for name, n in faults.most_common():
            print(f"      {n:4d}  {name}")
        print(f"  records: {valid}/{len(pmids) - len(missing)} valid, {errors} validation errors")
        for stage, c in cost.items():
            if c["calls"]:
                print(f"  {stage:9s} {c['calls']:3d} calls  in {c['input_tokens'] / 1e3:7.0f}k "
                      f"(cached {c['cached_tokens'] / 1e3:6.0f}k)  out {c['output_tokens'] / 1e3:6.0f}k "
                      f"(reasoning {c['reasoning_tokens'] / 1e3:5.0f}k)  {c['seconds'] / 60:5.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
