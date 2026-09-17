#!/usr/bin/env python3
"""How many proposed quotes does span resolution drop, and what does each pass buy?

    python scripts/measure_quote_drops.py [<run dir> ...]

Answers a question the built records cannot: a resolved span keeps the document's text, not
the quote that located it, so the model's quotes survive only in the payloads. Measured on
the 903-paper `pondie-907` run -- 104,957 quotes, 95.1% exact, 1.6% whitespace, 0.5%
case-only, 2.8% dropped.

Self-contained: it reimplements `spans.fold` / `_tolerant_pattern` / `resolve` rather than
importing pondie, so it measures the passes as they are written here and does not depend on
which pondie revision this host has checked out.

Input is the payloads, which hold the model's ORIGINAL quotes -- `build` replaces them with
document substrings on the way into the record, so a built corpus cannot be asked this.
"""
import json, re, sys
from collections import Counter
from pathlib import Path

_EQUIVALENT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "​": " ", "﻿": " ",
}
_TABLE = str.maketrans(_EQUIVALENT)

def fold(v): return v.translate(_TABLE)
def normalize(raw): return raw.replace("\r\n", "\n").replace("\r", "\n")

def classify(text, folded, quote):
    """Which pass places this quote, or None."""
    if not quote or not quote.strip():
        return "empty"
    if re.search(re.escape(quote), text):
        return "exact"
    tokens = [re.escape(t) for t in fold(quote).split()]
    if not tokens:
        return "empty"
    pattern = r"\s+".join(tokens)
    if re.search(pattern, folded):
        return "whitespace"
    if re.search(pattern, folded, re.IGNORECASE):
        return "case"
    return "dropped"

def quotes_of(node, out, path=""):
    if isinstance(node, dict):
        if isinstance(node.get("quotes"), list):
            for q in node["quotes"]:
                if isinstance(q, str):
                    out.append((path, q))
        for k, v in node.items():
            quotes_of(v, out, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            quotes_of(v, out, f"{path}[{i}]")

def main():
    runs = sys.argv[1:] or ["/data/james/pondie-vs-fulltext/pondie-data/runs/pondie-907"]
    corpus = Path("/data/james/pondie-vs-fulltext/corpus")
    verdicts = Counter(); per_slot = Counter(); per_slot_tot = Counter()
    papers = skipped = 0
    dropped_examples = []
    for run in runs:
        for study_dir in sorted(Path(run, "payloads").iterdir()):
            if not study_dir.is_dir():
                continue
            text = None
            for name in ("text.tables.txt", "text.txt"):
                for flavour in ("local", "pubget", "elsevier", "ace"):
                    candidate = corpus / study_dir.name / "processed" / flavour / name
                    if candidate.is_file():
                        text = normalize(candidate.read_text(encoding="utf-8", errors="replace"))
                        break
                if text is not None:
                    break
            if text is None:
                skipped += 1
                continue
            papers += 1
            folded = fold(text)
            found = []
            for payload in study_dir.glob("*.json"):
                try:
                    quotes_of(json.loads(payload.read_text(encoding="utf-8")), found)
                except Exception:
                    continue
            for path, quote in found:
                verdict = classify(text, folded, quote)
                verdicts[verdict] += 1
                slot = path.rsplit(".evidence", 1)[0].rsplit(".", 1)[-1].split("[")[0]
                per_slot_tot[slot] += 1
                if verdict in ("dropped", "case"):
                    per_slot[(slot, verdict)] += 1
                if verdict == "dropped" and len(dropped_examples) < 25:
                    dropped_examples.append(quote)
    total = sum(verdicts.values())
    print(f"{papers:,} papers ({skipped} with no text), {total:,} proposed quotes\n")
    for k in ("exact", "whitespace", "case", "dropped", "empty"):
        if verdicts[k]:
            print(f"  {verdicts[k]:8,d}  ({verdicts[k]/total:5.1%})  {k}")
    print(f"\nslots losing the most quotes:")
    rows = sorted({s for s, _v in per_slot}, key=lambda s: -per_slot[(s, "dropped")])
    print(f"  {'slot':34} {'quotes':>8} {'dropped':>9} {'case-only':>10}")
    for s in rows[:14]:
        print(f"  {s:34} {per_slot_tot[s]:8,d} {per_slot[(s,'dropped')]:9,d} "
              f"{per_slot[(s,'case')]:10,d}")
    print("\nexamples of quotes nothing could place:")
    for q in dropped_examples[:12]:
        print(f"    {q[:104]!r}")

main()
