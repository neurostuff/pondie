"""One paper across arms: what each `single` reply holds, how well its evidence resolves, cost.

    PONDIE_DATA_DIR=... PYTHONPATH=<code>:. python arm_review.py PMID RUN [RUN ...]

Per run: entity counts and analyses in the raw reply; wrapper statuses; in the built record,
the share of extracted fields whose evidence resolved to a span of the paper; validation
errors; and the `single` call's tokens.
"""
import json
import sys
from collections import Counter

from pondie import paths, schema
from pondie.extraction.models import Paper
from pondie.extraction.record.validate import Validator
from pondie.formats import text_index
from pondie.schema import reader


def wrappers(node, out):
    if isinstance(node, dict):
        if "extraction_status" in node:
            out.append(node)
            return out
        for value in node.values():
            wrappers(value, out)
    elif isinstance(node, list):
        for value in node:
            wrappers(value, out)
    return out


def main() -> int:
    pmid, runs = sys.argv[1], sys.argv[2:]
    sch = reader.load(schema.EXTRACTION)
    for run in runs:
        run_dir = paths.run(run)
        raw_path = run_dir / "payloads" / pmid / "raw" / "single.json"
        raw = json.loads(raw_path.read_text()) if raw_path.is_file() else \
            json.loads((run_dir / "payloads" / pmid / "single.json").read_text())
        counts = {k: len(v) for k, v in raw.items() if isinstance(v, list) and v}
        statuses = Counter((w["extraction_status"], w.get("unreported_reason"))
                           for w in wrappers(raw, []))
        print(f"\n=== {run} ({'raw' if raw_path.is_file() else 'repaired payload'})")
        print(f"  entities {counts}")
        print(f"  analyses {[a.get('local_id') for a in raw.get('analyses') or []]}")
        print(f"  wrappers {dict(statuses)}")
        if "support" in raw:
            print(f"  support: {len(raw['support'])} sentences, "
                  f"{sum(len(s.get('fields') or []) for s in raw['support'])} field paths")
        record_path = run_dir / "records" / f"{pmid}.extraction.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            extracted = [w for w in wrappers(record, []) if w["extraction_status"] == "extracted"]
            resolved = sum(1 for w in extracted
                           if (w.get("evidence") or {}).get("status") == "present")
            text, _, _ = text_index.load(Paper.best(pmid, run_dir / "corpus").text)
            validator = Validator(sch, text)
            validator.check_record(record)
            print(f"  record: {len(extracted)} extracted fields, {resolved} with evidence "
                  f"resolved to the paper ({resolved / max(len(extracted), 1):.0%}); "
                  f"{len(validator.errors)} validation errors")
        usage = run_dir / "usage.jsonl"
        if usage.is_file():
            for line in usage.read_text().splitlines():
                row = json.loads(line)
                if row["paper"] == pmid and row["stage"] == "single" and row.get("calls"):
                    print(f"  single: {row['calls']} call(s), out {row['output_tokens']} "
                          f"(reasoning {row['reasoning_tokens']}), {row['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
