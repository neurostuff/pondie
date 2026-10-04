"""Build the same payloads under two code trees and say what changed, record by record.

    PYTHONPATH=<code>:. python compare_builds.py build RUN [RUN ...] --out NAME [--drop-strays]
    python compare_builds.py diff BEFORE AFTER

`build` runs pondie's `Build` stage in-process over each run's saved payloads (no model
calls, no PubMed) into `$PONDIE_DATA_DIR/compare/NAME/<run>/`. `--drop-strays` makes the
shape repairs discard a stray that contradicts the value already in place, instead of
leaving it reported.

`diff` compares two such builds: validation errors, entities per list, and every leaf value
outside evidence -- so a value that disappears from a record is listed, wherever it went.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from pondie import paths, schema
from pondie.extraction.models import Paper, Settings
from pondie.extraction.record.validate import Validator
from pondie.schema import reader

MODEL = "@psyc-aid338-ope-333f18/gpt-6-luna"


def build(runs: list[str], out: str, drop_strays: bool) -> None:
    from pondie.extraction.stages import Build

    if drop_strays:
        from pondie.extraction.record.fix import shape

        def settle(target, key, value):
            if shape._empty(target.get(key)):
                target[key] = value
            return True

        shape._settle = settle
    for run in runs:
        source = paths.run(run)
        records = paths.DATA / "compare" / out / run
        records.mkdir(parents=True, exist_ok=True)
        settings = Settings(
            payloads=source / "payloads", records=records, model=MODEL, pubmed=False
        )
        for payload in sorted(p for p in (source / "payloads").iterdir() if p.is_dir()):
            Build().run(Paper.best(payload.name, source / "corpus"), settings, None)
        print(run, len(list(records.glob("*.extraction.json"))), "records", flush=True)


def _leaves(node, out: Counter) -> Counter:
    if isinstance(node, dict):
        for key, value in node.items():
            if key not in ("evidence", "extraction_metadata"):
                _leaves(value, out)
    elif isinstance(node, list):
        for value in node:
            _leaves(value, out)
    elif node not in (None, ""):
        out[str(node)] += 1
    return out


def _lists(record: dict) -> Counter:
    return Counter({k: len(v) for k, v in record.items() if isinstance(v, list)})


def diff(before: str, after: str) -> None:
    sch = reader.load(schema.EXTRACTION)
    root = paths.DATA / "compare"
    totals = Counter()
    for run_dir in sorted((root / before).iterdir()):
        for old_path in sorted(run_dir.glob("*.extraction.json")):
            new_path = root / after / run_dir.name / old_path.name
            old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
            errors = []
            for record in (old, new):
                validator = Validator(sch, None)
                validator.check_record(record)
                errors.append(len(validator.errors))
            totals["errors before"] += errors[0]
            totals["errors after"] += errors[1]
            totals["valid before"] += errors[0] == 0
            totals["valid after"] += errors[1] == 0
            totals["records"] += 1
            pmid = old_path.name.split(".")[0]
            shrunk = {k: (n, _lists(new)[k]) for k, n in _lists(old).items() if _lists(new)[k] < n}
            grown = {k: (n, _lists(new)[k]) for k, n in _lists(new).items() if _lists(old)[k] < n}
            lost = _leaves(old, Counter()) - _leaves(new, Counter())
            totals["values lost"] += sum(lost.values())
            if shrunk or grown or lost:
                print(f"{run_dir.name}/{pmid}: errors {errors[0]} -> {errors[1]}")
                if shrunk:
                    print(f"    LISTS SHRANK {shrunk}")
                if grown:
                    print(f"    lists grew {grown}")
                if lost:
                    print(f"    values gone {dict(list(lost.items())[:8])}")
    print("\n" + ", ".join(f"{k}: {v}" for k, v in totals.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("runs", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--drop-strays", action="store_true")
    d = sub.add_parser("diff")
    d.add_argument("before")
    d.add_argument("after")
    args = ap.parse_args()
    if args.command == "build":
        build(args.runs, args.out, args.drop_strays)
    else:
        diff(args.before, args.after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
