"""What a repair pass did to a record, measured rather than counted.

`repairs/<pmid>.json` reports how many fields were written. That number says nothing about
whether the record improved: on 18823721 repair wrote 156 fields, and 26 of them replaced a
verified span with `not_found` while leaving the value alone. A pass can write a great deal
and subtract.

Four measures, all from diffing the pre- and post-repair records, none needing a model:

  M1  span delta          spans gained minus spans destroyed. The primary gate.
  M2  provenance downgrade  fields that went `reported` -> `generated` with the value
                            unchanged. A pass that keeps a value has not earned the right
                            to withdraw its warrant.
  M3  introduced findings   `Validator.diff`, post minus pre. Damage the pass caused rather
                            than inherited.
  M5  fill yield            absent-or-empty -> a value. The counterweight to M1: a pass
                            that writes nothing scores perfectly on M1 and zero here.

M4 -- whether the values are RIGHT -- cannot be computed from the record alone and lives in
`benchmarks/repair_truth/`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from pondie import schema
from pondie.extraction.record.validate import Validator
from pondie.formats import values
from pondie.schema import reader


def fields(node, path: str = ""):
    """Every wrapper, keyed by `local_id` rather than by list index.

    `values.iter_fields` numbers list entries positionally, so a pass that reorders
    `groups` makes every field of every group look changed. The record addresses entities
    by `local_id` everywhere else; the measurement has to as well or it will one day report
    a large delta for a no-op.
    """
    if isinstance(node, Mapping):
        if values.MARKER in node:
            yield path, node
            return
        for key, value in node.items():
            yield from fields(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            label = str(values.read(item.get("local_id"))) if isinstance(item, Mapping) \
                and item.get("local_id") is not None else str(index)
            yield from fields(item, f"{path}[{label}]")


def spans(node: dict) -> int:
    return sum(len(s.get("spans") or [])
               for s in ((node.get("evidence") or {}).get("sets") or []))


def status(node: dict) -> str:
    return str((node.get("evidence") or {}).get("status") or "none")


def empty(node: dict) -> bool:
    return values.read(node) in (None, "", [])


def measure(before: dict, after: dict, sch, text: str | None) -> Counter:
    """One record's M1, M2, M3, M5, keyed for a table."""
    was = dict(fields(before))
    now = dict(fields(after))
    out: Counter = Counter()

    for path, node in now.items():
        old = was.get(path)
        gained, lost = spans(node), spans(old) if old else 0
        out["spans_after"] += gained
        if old is None:
            out["fields_new"] += 1
            if not empty(node):
                out["filled"] += 1
            out["spans_gained"] += gained
            continue
        out["spans_gained"] += max(0, gained - lost)
        out["spans_destroyed"] += max(0, lost - gained)
        # A downgrade is only a downgrade when the pass kept the value: replacing a wrong
        # value with a better-sourced one may legitimately lose the old citation.
        # Only a downgrade when there was a warrant to withdraw. `build` can emit
        # `reported` with `not_found`, which is already dishonest, and a pass relabelling
        # that to `generated` is correcting the record rather than damaging it.
        same = str(values.read(old)) == str(values.read(node))
        if same and status(old) == "present" \
                and old.get("value_source") == "reported" \
                and node.get("value_source") == "generated":
            out["provenance_downgrade"] += 1
        if empty(old) and not empty(node):
            out["filled"] += 1

    for path, node in was.items():
        if path not in now:
            out["fields_dropped"] += 1
            out["spans_destroyed"] += spans(node)
        out["spans_before"] += spans(node)

    out["introduced"] = len(Validator(sch, text).diff(before, after))
    out["m1_span_delta"] = out["spans_gained"] - out["spans_destroyed"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, nargs="?",
                    help="a run dir holding records/ and unrepaired/")
    ap.add_argument("--before", type=Path, help="dir of pre-repair records")
    ap.add_argument("--after", type=Path, help="dir of post-repair records")
    ap.add_argument("--corpus", type=Path,
                    default=Path("/data/james/pondie-vs-fulltext/corpus"),
                    help="for the paper text the validator checks spans against")
    args = ap.parse_args()

    # The schema `repair` itself validates against, so M3 here and the pass's own
    # `introduced` count are the same number rather than two plausible ones.
    sch = reader.load(schema.EXTRACTION)
    if args.before and args.after:
        pairs = []
        for before_path in sorted(args.before.glob("*.json")):
            pmid = before_path.name.split(".")[0]
            after_path = next((c for c in (args.after / f"{pmid}.repaired.json",
                                           args.after / f"{pmid}.extraction.json",
                                           args.after / before_path.name) if c.is_file()), None)
            if after_path:
                pairs.append((pmid, before_path, after_path))
    else:
        pairs = [(p.name.split(".")[0], args.run / "unrepaired" / p.name, p)
                 for p in sorted((args.run / "records").glob("*.extraction.json"))
                 if (args.run / "unrepaired" / p.name).is_file()]

    rows = []
    for pmid, before_path, after_path in pairs:
        text_file = args.corpus / pmid / "processed" / "local" / "text.tables.txt"
        text = text_file.read_text(errors="replace") if text_file.is_file() else None
        rows.append((pmid, measure(json.loads(before_path.read_text()),
                                   json.loads(after_path.read_text()), sch, text)))

    if not rows:
        print("no before/after record pairs found")
        return 1

    cols = [("M1 span delta", "m1_span_delta"), ("gained", "spans_gained"),
            ("destroyed", "spans_destroyed"), ("M2 downgrade", "provenance_downgrade"),
            ("M3 introduced", "introduced"), ("M5 filled", "filled")]
    print(f"{'pmid':12}" + "".join(f"{label:>15}" for label, _k in cols))
    total: Counter = Counter()
    for pmid, m in rows:
        print(f"{pmid:12}" + "".join(f"{m[key]:>15}" for _l, key in cols))
        total.update(m)
    print("-" * (12 + 15 * len(cols)))
    print(f"{'TOTAL':12}" + "".join(f"{total[key]:>15}" for _l, key in cols))
    print(f"\n{len(rows)} record(s). Gates: M1 >= 0, M2 == 0, M3 == 0.")
    failed = [p for p, m in rows
              if m["m1_span_delta"] < 0 or m["provenance_downgrade"] or m["introduced"]]
    print(f"records failing a gate: {len(failed)}" + (f" -> {failed[:10]}" if failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
