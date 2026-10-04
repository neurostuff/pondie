"""How well-formed a run's records are: pondie's validator, references, and shape.

    python check_records.py RUN [RUN ...] [--examples 3]

For each record: the extraction-schema validator (with the paper text, so every quote span
is checked against the text it claims to address), dangling or duplicated local_ids, and
basic shape. Messages are grouped by pattern -- ids, numbers and quoted values blanked -- so
a systematic fault shows as one large group rather than many lines.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from pondie import paths, schema
from pondie.extraction.record.fix.link import check_local_ids
from pondie.extraction.record.validate import Validator
from pondie.schema import reader


def pattern(message: str) -> str:
    m = re.sub(r"'[^']*'", "'…'", message)
    m = re.sub(r"\[\d+\]", "[i]", m)
    m = re.sub(r"\b\d+(\.\d+)?\b", "N", m)
    return m[:150]


def text_of(run_dir: Path, pmid: str) -> str:
    for base in (run_dir / "corpus", paths.DATA / "corpus"):
        f = base / pmid / "processed" / "local" / "text.tables.txt"
        if f.is_file():
            return f.read_text(encoding="utf-8", errors="replace")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()
    ext = reader.load(schema.EXTRACTION)
    for run in args.runs:
        run_dir = paths.run(run)
        records = sorted((run_dir / "records").glob("*.extraction.json"))
        errors, warnings, refs = Counter(), Counter(), Counter()
        examples = defaultdict(list)
        per_record_errors, clean = [], 0
        shape = Counter()
        for f in records:
            pmid = f.name.split(".")[0]
            record = json.loads(f.read_text())
            v = Validator(ext, text_of(run_dir, pmid) or None)
            v.check_record(record)
            per_record_errors.append(len(v.errors))
            clean += not v.errors
            for kind, bucket in (("E", v.errors), ("W", v.warnings)):
                for msg in bucket:
                    key = pattern(msg)
                    (errors if kind == "E" else warnings)[key] += 1
                    if len(examples[key]) < args.examples:
                        examples[key].append(f"{pmid}: {msg[:220]}")
            body = {k: v_ for k, v_ in record.items() if k != "extraction_metadata"}
            for problem in check_local_ids(body, ext):
                refs[pattern(problem)] += 1
            shape["records"] += 1
            shape["no analyses"] += not record.get("analyses")
            shape["analyses"] += len(record.get("analyses") or [])
            shape["repaired_by set"] += bool((record.get("extraction_metadata") or {}).get("repaired_by"))
            shape["has source_text_hash"] += bool((record.get("extraction_metadata") or {}).get("source_text_hash"))
        n = len(records)
        print(f"\n=== {run}: {n} records, {clean} with no validation errors, "
              f"errors per record median {sorted(per_record_errors)[n // 2] if n else 0}, "
              f"max {max(per_record_errors) if n else 0}")
        print("   shape:", dict(shape))
        print(f"   validation errors: {sum(errors.values())} in {len(errors)} patterns")
        for key, count in errors.most_common(12):
            print(f"     {count:5d}  {key}")
            for e in examples[key][:args.examples]:
                print(f"            e.g. {e}")
        print(f"   validation warnings: {sum(warnings.values())} in {len(warnings)} patterns")
        for key, count in warnings.most_common(8):
            print(f"     {count:5d}  {key}")
        print(f"   reference problems: {sum(refs.values())}")
        for key, count in refs.most_common(6):
            print(f"     {count:5d}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
