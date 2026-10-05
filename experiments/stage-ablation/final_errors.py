"""Validate a run's final records the way `build` does, and tally the error patterns.

    PONDIE_DATA_DIR=... PYTHONPATH=<code>:. python final_errors.py RUN [RUN ...] [--records DIR] [--detail]

The run log's "records valid" counts `build`'s outcomes, before `repair`. This reads what the
run left in `records/` (after repair), or `--records` (e.g. `unrepaired`).
"""
import argparse
import json
from collections import Counter

from pondie import paths, schema
from pondie.extraction.models import Paper, error_pattern
from pondie.extraction.record import validate
from pondie.schema import reader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--records", default="records")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()
    sch = reader.load(schema.EXTRACTION)
    for run in args.runs:
        root = paths.run(run)
        counts, valid, total = Counter(), 0, 0
        for path in sorted((root / args.records).glob("*.extraction.json")):
            pmid = path.name.split(".")[0]
            text = Paper.best(pmid, root / "corpus").text.read_text(encoding="utf-8", errors="replace")
            validator = validate.Validator(sch, text)
            validator.check_record(json.loads(path.read_text()))
            total += 1
            valid += not validator.errors
            counts.update(error_pattern(e) for e in validator.errors)
            if args.detail:
                for error in validator.errors:
                    print(f"  {pmid}  {error[:220]}")
        print(f"=== {run}/{args.records}: valid {valid}/{total}, {sum(counts.values())} error(s)")
        for pattern, count in counts.most_common():
            print(f"  {count:5d}  {pattern[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
