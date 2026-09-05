#!/usr/bin/env python3
"""Every `quote` in a ground-truth file must be verbatim in the article it names.

The quote is what makes a disagreement adjudicable, so a quote that is not in the paper
makes the file worse than useless -- it looks checkable and is not. Run this after any edit:

    python benchmarks/repair_truth/verify_quotes.py [CORPUS_ROOT]

CORPUS_ROOT defaults to the location the files name in `source`; pass a directory of
`<pmid>.txt` instead when working off a local copy.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def article(pmid: str, root: str | None) -> str:
    paths = [os.path.join(root, f"{pmid}.txt")] if root else []
    paths.append(f"/data/james/pondie-vs-fulltext/corpus/{pmid}/processed/local/text.tables.txt")
    for path in paths:
        if os.path.isfile(path):
            return norm(open(path, encoding="utf-8", errors="replace").read())
    raise SystemExit(f"no article text for {pmid}; looked in {paths}")


def main(root: str | None = None) -> int:
    bad = total = 0
    for path in sorted(glob.glob(os.path.join(HERE, "*.json"))):
        pmid = os.path.basename(path)[:-5]
        text = article(pmid, root)
        truth = json.load(open(path, encoding="utf-8"))
        for entity in truth["entities"]:
            for slot, spec in entity["fields"].items():
                total += 1
                quote = spec.get("quote")
                if not quote:
                    print(f"MISSING QUOTE  {pmid} {entity['key']}.{slot}")
                    bad += 1
                elif norm(quote) not in text:
                    print(f"NOT VERBATIM   {pmid} {entity['key']}.{slot}: {norm(quote)[:100]!r}")
                    bad += 1
    print(f"{total} quotes checked, {bad} not verbatim")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
