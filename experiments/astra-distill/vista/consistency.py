"""Inconsistencies in records, before and after pondie's deterministic fixes.

    python consistency.py ROOT LABEL RECORDS_DIR   (ROOT holds corpus/<sid>/{parse,processed})

For each record: pondie's Validator on the extraction schema (structure, the storage schema's
class rules, and the domain rules of `rules.check_all`), with no text (these records carry no
evidence spans); then every repair of `fix.build_sequence()` (all stages) with the paper's
coordinate parse and text, and the Validator again. Table repairs see no table map.
"""
import collections
import copy
import json
import re
import sys
from pathlib import Path

from pondie import schema
from pondie.extraction.record import fix
from pondie.extraction.record.validate import Validator
from pondie.formats import text_index
from pondie.schema import reader

root, label, records = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sch = reader.load(schema.EXTRACTION)


def kind(message: str) -> str:
    """A message with its paths, ids, quotes and numbers folded, so one fault counts once per kind."""
    path, _, text = message.partition(": ")
    path = re.sub(r"\[[^\]]*\]", "[]", path)
    path = ".".join(p for p in path.split(".")[-2:])
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "'…'", text)
    text = re.sub(r"\d+(\.\d+)?", "N", text)
    return f"{path}: {text[:110]}"


def check(rec):
    v = Validator(sch, None)
    v.check_record(rec)
    return v.errors, v.warnings


tot = collections.Counter()
kinds_before, kinds_after = collections.Counter(), collections.Counter()
warn_before, warn_after = collections.Counter(), collections.Counter()
repairs = collections.Counter()
repair_papers = collections.Counter()
introduced = collections.Counter()
for path in sorted(records.glob("*.extraction.json")):
    sid = path.name.split(".")[0]
    rec = json.loads(path.read_text())
    errors, warnings = check(rec)
    tot["records"] += 1
    tot["errors_before"] += len(errors)
    tot["warnings_before"] += len(warnings)
    tot["records_with_error_before"] += bool(errors)
    kinds_before.update(kind(e) for e in errors)
    warn_before.update(kind(w) for w in warnings)
    tot["records_with_warning_before"] += bool(warnings)
    textf = root / "corpus" / sid / "processed" / "pubget" / "text.txt"
    normalized = text_index.normalize(textf.read_text(encoding="utf-8")) if textf.is_file() else ""
    parse = root / "corpus" / sid / "parse" / "coordinate_parse.json"
    body = copy.deepcopy(rec)
    try:
        log = fix.apply_all(body, fix.Context(schema=sch, parse=parse if parse.is_file() else None,
                                              table_map=None, text=normalized))
    except Exception as e:  # noqa: BLE001
        tot["repair_crashed"] += 1
        print(sid, "repairs failed:", type(e).__name__, str(e)[:200], file=sys.stderr)
        continue
    for name, lines in log.entries:
        if lines:
            repairs[name] += len(lines)
            repair_papers[name] += 1
    for name in log.introduced:
        introduced[name] += 1
    errors2, warnings2 = check(body)
    tot["errors_after"] += len(errors2)
    tot["warnings_after"] += len(warnings2)
    tot["records_with_error_after"] += bool(errors2)
    kinds_after.update(kind(e) for e in errors2)
    warn_after.update(kind(w) for w in warnings2)
    tot["records_with_warning_after"] += bool(warnings2)

out = {"label": label, "totals": dict(tot),
       "errors_before_by_kind": kinds_before.most_common(25),
       "errors_after_by_kind": kinds_after.most_common(25),
       "warnings_before_by_kind": warn_before.most_common(30),
       "warnings_after_by_kind": warn_after.most_common(30),
       "repairs_changes": repairs.most_common(), "repairs_papers": dict(repair_papers),
       "repairs_introducing_dangling": dict(introduced)}
(root / f"consistency-{label}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out["totals"]))
