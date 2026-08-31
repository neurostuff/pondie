"""Can a contrast's directions be read off its statistics instead of asked for?

`parse_tables.split_opposite_signs` already uses the sign of a row's statistic to split
a table before the model sees it, and that ships. This asks the harder question: given
the sign, is the *cell* direction determined -- so the pass could stop asking.

It is not the same question, and the gold says why. In `analysis-connectivity-decreased`
the ASD cell is negative and the TD cell positive: one statistic, two cells, opposite
directions. A sign fixes the contrast's polarity, not which side of it a level sits on.
So three things are measured separately:

  signed      does the analysis carry a statistic with an unambiguous sign at all
  shape       do its cells form the +/- pair a signed two-level contrast implies
  assignment  given the sign, can the right cell be given the right direction

Only the third would let the pass stop asking, and it is the one with a free parameter:
which level is the contrast's reference.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

from parse_tables import _point_sign  # noqa: E402



def unwrap(node):
    return node.get("value") if isinstance(node, dict) and "value" in node else node


#: The parse writes `coordinates` with a flat `statistic_value`; `_point_sign` reads the
#: richer `values` shape. Bridged here rather than in the parser, which has its own
#: contract.
def _as_point(coordinate: dict) -> dict:
    kind = (coordinate.get("statistic_type") or "").lower()
    return {"values": [{"kind": kind, "value": coordinate.get("statistic_value")}]}


def analysis_sign(parsed: dict) -> int | None:
    """The one sign every point of this parsed analysis agrees on, or None."""
    signs = {sign for coordinate in (parsed.get("coordinates") or [])
             if (sign := _point_sign(_as_point(coordinate))) is not None}
    return signs.pop() if len(signs) == 1 else None


def match_analysis(name: str, parsed_list: list[dict]) -> dict | None:
    """The parsed analysis a gold analysis id refers to.

    The record's local_id is derived from the parse's `name`, so they are matched on a
    normalised form of it rather than on identity.
    """
    def fold(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    target = fold(name)
    for parsed in parsed_list:
        if fold(parsed.get("name")) == target:
            return parsed
    for parsed in parsed_list:
        folded = fold(parsed.get("name"))
        if folded and (folded in target or target in folded):
            return parsed
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=ROOT / "benchmarks/gold/direction")
    parser.add_argument("--records", type=Path, default=ROOT / "data/records")
    parser.add_argument("--parsed", type=Path, default=ROOT / "data/tables",
                        help="parse_tables output, if present")
    args = parser.parse_args()

    stats = Counter()
    shapes = Counter()
    examples = []

    for file in sorted(args.gold.glob("*.direction.json")):
        gold = json.loads(file.read_text(encoding="utf-8"))
        paper = gold["paper_id"]
        record_file = args.records / f"{paper}.extraction.json"
        if not record_file.is_file():
            stats["no record"] += 1
            continue
        record = json.loads(record_file.read_text(encoding="utf-8"))
        by_id = {unwrap(a.get("local_id")): a for a in record.get("analyses", [])}

        # The statistics are in the table parse, not the record: the record keeps the
        # contrast and drops the rows it was read from.
        parsed_list = []
        for path in sorted(glob.glob(str(ROOT / f"data/texts/{paper}/processed/*/analyses.jsonl"))):
            parsed_list += [json.loads(line) for line in
                            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
            break
        if not parsed_list:
            stats["no table parse"] += 1

        cells_by_analysis: dict[str, list] = {}
        for cell in gold["cells"]:
            if cell.get("direction") and cell.get("tier") != "silent":
                cells_by_analysis.setdefault(cell["analysis"], []).append(cell)

        for analysis_id, cells in cells_by_analysis.items():
            stats["analyses"] += 1
            analysis = by_id.get(analysis_id)
            if analysis is None:
                stats["analysis not in record"] += 1
                continue
            parsed = match_analysis(unwrap(analysis.get("name")), parsed_list)
            if parsed is None:
                stats["no matching table parse"] += 1
                continue
            sign = analysis_sign(parsed)
            directions = [c["direction"] for c in cells]
            shapes[tuple(sorted(Counter(directions).items()))] += 1
            if sign is None:
                stats["no usable sign"] += 1
                continue
            stats["signed"] += 1
            positive = sum(1 for d in directions if d == "positive")
            negative = sum(1 for d in directions if d == "negative")
            if positive == 1 and negative == 1:
                stats["signed and a clean +/- pair"] += 1
                if len(examples) < 6:
                    examples.append((paper, analysis_id, sign,
                                     [(c["level"], c["direction"]) for c in cells]))
            elif positive and not negative:
                stats["signed, all positive"] += 1
            elif negative and not positive:
                stats["signed, all negative"] += 1
            else:
                stats["signed, other shape"] += 1

    print(f"{stats['analyses']} gold analyses with a reviewed direction\n")
    for key in ("analysis not in record", "no table parse", "no matching table parse",
                "no usable sign", "signed",
                "signed and a clean +/- pair", "signed, all positive",
                "signed, all negative", "signed, other shape"):
        if stats[key]:
            print(f"  {key:34s} {stats[key]:4d}"
                  f"  ({stats[key]*100/max(stats['analyses'],1):.0f}%)")
    print("\ncell-direction shapes across all gold analyses:")
    for shape, count in shapes.most_common(8):
        rendered = ", ".join(f"{n}x{d}" for d, n in shape)
        print(f"  {rendered:34s} {count}")
    print("\nexamples of signed +/- pairs:")
    for paper, analysis_id, sign, cells in examples:
        print(f"  {paper} {analysis_id[:34]:34s} sign={sign:+d}  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
