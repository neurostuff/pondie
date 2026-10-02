#!/usr/bin/env python3
"""Put a paper's own tables back into its corpus text, where the build dropped them.

    python scripts/recover_page_tables.py --corpus <corpus> --articles <articles> [--write]

A paper built through the ACE route takes its tables from ACE's export, and the export is
empty for 350 of the corpus's ACE-route papers. Their saved journal pages are not: 220 of
them carry a `<table>`, and the first one is usually the demographics table. That is not a
cosmetic gap. `Group.age_minimum` is unanswerable on 21 of the 50 PTSD records, which
costs that project's query 7 of its 17 gold papers, and the model is not being coy --
`12853571`'s render contains no age at all because the table holding it never reached the
text.

APPENDS, NEVER RE-RENDERS. The obvious fix is to rebuild the text with
`build_corpus.build_ace`, which now falls back to the page's own tables -- and rebuilding
moves every character offset in every record already extracted against that text. This
writes the `## Tables` section onto the end of the existing render instead, so offsets
into the body are untouched and the tables land where `build_ace` would have put them.

The manifest is written the way the Tables stage expects, with `table_id` `page<n>`, so a
recovered table is distinguishable from one ACE parsed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_corpus import page_table_blocks  # noqa: E402

TABLES_HEADING = "\n\n## Tables\n\n"


def recover(corpus: Path, articles: Path, write: bool) -> dict[str, int]:
    tally = {"looked at": 0, "no saved page": 0, "page has no table": 0, "recovered": 0,
             "tables written": 0}
    for provenance_path in sorted(corpus.glob("*/provenance.json")):
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        study = provenance_path.parent
        flavour = study / "processed" / "local"
        manifest = flavour / "tables.jsonl"
        text = flavour / "text.tables.txt"
        if not text.is_file() or (manifest.is_file() and manifest.stat().st_size):
            continue
        tally["looked at"] += 1
        source = provenance.get("source_path") or ""
        page = articles.parent / source if source else None
        if page is None or not page.is_file() or page.suffix.lower() != ".html":
            tally["no saved page"] += 1
            continue
        blocks, rows = page_table_blocks(page.read_text(encoding="utf-8", errors="replace"))
        if not blocks:
            tally["page has no table"] += 1
            continue
        tally["recovered"] += 1
        tally["tables written"] += len(rows)
        if not write:
            continue
        body = text.read_text(encoding="utf-8")
        if TABLES_HEADING.strip() not in body:
            text.write_text(body.rstrip() + TABLES_HEADING + "\n\n".join(blocks) + "\n",
                            encoding="utf-8")
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        provenance["n_tables"] = len(rows)
        provenance["tables_recovered_from"] = "saved page"
        provenance_path.write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")
    return tally


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--articles", required=True, type=Path,
                    help="the directory `provenance.source_path` is relative to")
    ap.add_argument("--write", action="store_true", help="without it, nothing is written")
    args = ap.parse_args()
    tally = recover(args.corpus, args.articles, args.write)
    width = max(len(k) for k in tally)
    for key, value in tally.items():
        print(f"  {value:6d}  {key:{width}}")
    if not args.write:
        print("\n  nothing written; pass --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
