"""Validate a run's final records the way `build` does, and tally the error patterns.

    PONDIE_DATA_DIR=... PYTHONPATH=<code>:. python final_errors.py RUN [RUN ...] [--records DIR] [--detail]

The run log's "records valid" counts `build`'s outcomes, before `repair`. This reads what the
run left in `records/` (after repair), or `--records` (e.g. `unrepaired`).

The validator does not report dangling or doubly-declared references (`check_local_ids` does,
in the build report), so they are counted here too: a record is "clean" only with neither.
"""
import argparse
import json
from collections import Counter

from pondie import paths, schema
from pondie.extraction.models import Paper, error_pattern
from pondie.extraction.record import validate
from pondie.extraction.record.fix import link
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
        counts, refs, valid, clean, total = Counter(), Counter(), 0, 0, 0
        for path in sorted((root / args.records).glob("*.extraction.json")):
            pmid = path.name.split(".")[0]
            text = Paper.best(pmid, root / "corpus").text.read_text(encoding="utf-8", errors="replace")
            record = json.loads(path.read_text())
            validator = validate.Validator(sch, text)
            validator.check_record(record)
            problems = link.check_local_ids(record, sch)
            total += 1
            valid += not validator.errors
            clean += not validator.errors and not problems
            counts.update(error_pattern(e) for e in validator.errors)
            refs.update(error_pattern(e) for e in problems)
            if args.detail:
                for error in validator.errors + problems:
                    print(f"  {pmid}  {error[:220]}")
        print(
            f"=== {run}/{args.records}: valid {valid}/{total}, clean {clean}/{total}; "
            f"{sum(counts.values())} error(s), {sum(refs.values())} reference problem(s)"
        )
        for pattern, count in counts.most_common() + refs.most_common():
            print(f"  {count:5d}  {pattern[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
