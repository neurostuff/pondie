#!/usr/bin/env python3
"""How much of each extracted field is recoverable from the paper's surface, and where.

One measurement, over every field the pipeline fills. For each value the extractor wrote,
ask whether that value can be *located* in the paper text at all -- and if so, how many
places it could have come from, and which section the match sits in.

    python audit_field_extraction.py --records 'data/records/*.extraction.json'

Why this bounds every surface method. String matching, a regex, spaCy's NER and a
transformer tagger all read the same surface: they differ in how they find a span, not in
whether the span is there. So a value absent from the text is out of reach for all of them
equally, and the fraction present is a ceiling on any of them rather than a score for one.
A value present but matching in nine places is a *scoping* problem, which is the finding
that governs most of this schema -- see docs/deterministic-fields.md.

Sections come from `extraction_metadata.paper_sections`, which the pipeline already writes
with character offsets, so "look only in Methods" is testable rather than assumed.
"""

from __future__ import annotations

import argparse
import collections
import glob as globlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[3]
TEXTS = ROOT / "data" / "texts"

#: Sections are grouped rather than used verbatim: papers name them differently and the
#: question is only whether a value lives where the methods are described.
SECTION_GROUPS = (
    (
        "methods",
        r"method|material|participant|procedure|acquisition|preprocess|analys|"
        r"statistic|measure|assessment|subject|sample|design",
    ),
    ("results", r"result|finding"),
    ("intro", r"introduction|background|objective"),
    ("discussion", r"discussion|conclusion|limitation"),
    ("abstract", r"abstract|summary"),
    ("tables", r"^#*\s*tables?\b"),
)


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.]+", " ", folded.lower())).strip()


def section_of(offset: int, sections: list[Mapping]) -> str:
    for entry in sections:
        start, end = entry.get("start_char"), entry.get("end_char")
        if isinstance(start, int) and isinstance(end, int) and start <= offset < end:
            title = str(entry.get("title") or "")
            for label, pattern in SECTION_GROUPS:
                if re.search(pattern, title, re.I):
                    return label
            return "other"
    return "unsectioned"


def walk(node: Any, path: str = "") -> Iterator[tuple[str, Mapping]]:
    if isinstance(node, Mapping):
        if "extraction_status" in node:
            yield path, node
            return
        for key, value in node.items():
            yield from walk(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item, f"{path}[]")


def find_number(value: float, text: str) -> list[int]:
    """Offsets where this number appears, in any spelling the corpus uses.

    A field in seconds is written in milliseconds on the page, and an integer count is
    written without its `.0`, so the search is over plausible renderings rather than over
    `str(value)`. Without that, every duration and every threshold reads as unrecoverable.
    """
    renderings = set()
    for candidate in (value, value * 1000, value / 1000, value * 100):
        if candidate != int(candidate):
            renderings.add(f"{candidate:g}")
        else:
            renderings.add(str(int(candidate)))
        renderings.add(f"{candidate:g}")
    hits = []
    for rendering in renderings:
        if not rendering or rendering in {"0"}:
            continue
        hits += [
            m.start() for m in re.finditer(rf"(?<![\w.]){re.escape(rendering)}(?![\w])", text)
        ]
    return sorted(set(hits))


def find_string(value: str, text: str, lowered: str) -> tuple[list[int], str]:
    """Offsets and how the match was made: verbatim, normalised, or not at all."""
    if len(value.strip()) < 2:
        return [], "too_short"
    hits = [m.start() for m in re.finditer(re.escape(value), text)]
    if hits:
        return hits, "verbatim"
    folded = normalize(value)
    if folded and len(folded) >= 3:
        hits = [m.start() for m in re.finditer(re.escape(folded), lowered)]
        if hits:
            return hits, "normalised"
    # Token containment: the value's words all present in one window. Catches a reordering
    # or an inserted qualifier without claiming a match on a single shared word.
    tokens = [t for t in folded.split() if len(t) > 2]
    if len(tokens) >= 2 and all(re.search(rf"\b{re.escape(t)}", lowered) for t in tokens):
        first = min(
            m.start() for t in tokens for m in [re.search(rf"\b{re.escape(t)}", lowered)] if m
        )
        return [first], "tokens_present"
    return [], "absent"


def audit(paths: list[Path]) -> dict[str, dict]:
    stats: dict[str, dict] = collections.defaultdict(
        lambda: {
            "n": 0,
            "how": collections.Counter(),
            "cands": [],
            "section": collections.Counter(),
            "kind": collections.Counter(),
        }
    )
    for path in paths:
        paper = path.name.split(".")[0]
        record = json.loads(path.read_text(encoding="utf-8"))
        text_path = TEXTS / paper / "processed" / "local" / "text.tables.txt"
        if not text_path.is_file():
            continue
        text = text_path.read_text(encoding="utf-8")
        lowered = normalize(text)
        sections = (record.get("extraction_metadata") or {}).get("paper_sections") or []

        for raw_path, node in walk(record):
            field = re.sub(r"\[\]", "", raw_path)
            if node.get("extraction_status") != "extracted":
                continue
            value = node.get("value")
            if value in (None, ""):
                continue
            for item in (value if isinstance(value, list) else [value]):
                entry = stats[field]
                entry["n"] += 1
                if isinstance(item, bool):
                    entry["kind"]["boolean"] += 1
                    entry["how"]["not_surface"] += 1
                    continue
                if isinstance(item, (int, float)):
                    entry["kind"]["number"] += 1
                    hits = find_number(float(item), text)
                    how = "number" if hits else "absent"
                elif isinstance(item, str):
                    entry["kind"]["string"] += 1
                    hits, how = find_string(item, text, lowered)
                else:
                    entry["kind"]["other"] += 1
                    entry["how"]["not_surface"] += 1
                    continue
                entry["how"][how] += 1
                if hits:
                    entry["cands"].append(len(hits))
                    entry["section"][section_of(hits[0], sections)] += 1
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--records", default="data/records/*.extraction.json")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--min-n", type=int, default=1)
    args = ap.parse_args(argv)

    paths = sorted(Path(p) for p in globlib.glob(args.records))
    stats = audit(paths)
    rows = []
    for field, entry in stats.items():
        if entry["n"] < args.min_n:
            continue
        # `tokens_present` is a real surface match and a weak one, so it is counted as
        # locatable but reported apart: a field only reachable that way is one where the
        # value's words are scattered in a window rather than written as the value.
        strict = sum(
            v for k, v in entry["how"].items() if k in {"verbatim", "normalised", "number"}
        )
        found = strict + entry["how"].get("tokens_present", 0)
        cands = entry["cands"]
        unique = sum(1 for c in cands if c == 1)
        rows.append(
            {
                "field": field,
                "n": entry["n"],
                "recoverable": found / entry["n"],
                "recoverable_strict": strict / entry["n"],
                "unique": unique / entry["n"],
                "median_candidates": (sorted(cands)[len(cands) // 2] if cands else None),
                "how": dict(entry["how"]),
                "section": dict(entry["section"]),
                "kind": dict(entry["kind"]),
            }
        )
    rows.sort(key=lambda r: (-r["unique"], -r["n"]))
    print(
        f"{'field':52s} {'n':>4s} {'strict':>6s} {'recov':>6s} {'uniq':>6s} {'medC':>5s}  top section"
    )
    for r in rows:
        top = max(r["section"].items(), key=lambda x: x[1])[0] if r["section"] else "-"
        mc = r["median_candidates"]
        print(
            f"{r['field']:52s} {r['n']:4d} {r['recoverable_strict']:6.0%} "
            f"{r['recoverable']:6.0%} {r['unique']:6.0%} "
            f"{(str(mc) if mc is not None else '-'):>5s}  {top}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    print(f"\nfields: {len(rows)}   instances: {sum(r['n'] for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
