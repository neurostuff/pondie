"""Mine every available paper's abbreviation definitions into one referenceable file.

An abbreviation resolved one way in one record and another way in the next is a
normalization bug that no single record can show. Putting every expansion in one file
makes the collision visible -- `disagreements()` is the whole reason this is a corpus-level
artefact rather than a per-paper step.

    python build_abbreviations.py --texts data/texts --out data/vocab/abbreviations.json
    python build_abbreviations.py --records 'data/runs/*/records/*.extraction.json'
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import re
import sys
from pathlib import Path

from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

from pipeline.abbreviations import Abbreviations, detector  # noqa: E402
from pipeline.kinds import TEXT_FLAVOURS  # noqa: E402

#: Abbreviations papers use without ever defining, usually because they are assumed. Each
#: is a claim about the field rather than about a paper, which is why they are separated
#: from the mined ones and why the list is short: the mined store should be doing this
#: work, and a growing curated list is a sign it is not.
CURATED: dict[str, str] = {
    "SSRI": "selective serotonin reuptake inhibitor",
    "SSRIs": "selective serotonin reuptake inhibitors",
    "SNRI": "serotonin norepinephrine reuptake inhibitor",
    "TAU": "treatment as usual",
    "MDD": "major depressive disorder",
    "TRD": "treatment resistant depression",
    "HAMD": "Hamilton Depression Rating Scale",
    "HDRS": "Hamilton Depression Rating Scale",
    "HAMA": "Hamilton Anxiety Rating Scale",
    "BDI": "Beck Depression Inventory",
    "MADRS": "Montgomery Asberg Depression Rating Scale",
    "ADOS": "Autism Diagnostic Observation Schedule",
    "ADI-R": "Autism Diagnostic Interview Revised",
    "MMSE": "Mini Mental State Examination",
    "YBOCS": "Yale Brown Obsessive Compulsive Scale",
    "PANSS": "Positive and Negative Syndrome Scale",
    "CBT": "cognitive behavioural therapy",
    "ECT": "electroconvulsive therapy",
    "rTMS": "repetitive transcranial magnetic stimulation",
    "tDCS": "transcranial direct current stimulation",
    "iTBS": "intermittent theta burst stimulation",
    "dlPFC": "dorsolateral prefrontal cortex",
    "dmPFC": "dorsomedial prefrontal cortex",
    "vmPFC": "ventromedial prefrontal cortex",
    "ACC": "anterior cingulate cortex",
    "sgACC": "subgenual anterior cingulate cortex",
    "OFC": "orbitofrontal cortex",
    "HC": "healthy controls",
    "TD": "typically developing",
}


#: An expansion Schwartz & Hearst returns is not always a phrase. Run over a paper carrying
#: tables, the algorithm walks into cell content and returns things like
#: `ii -> "including seizures),"` and `hunger scan -> "Hunger Scan) a | 52.0 | 1.3 |"`. A
#: consumer that expands blindly then rewrites `bipolar II disorder` into nonsense, so the
#: store refuses to hold them rather than leaving every caller to notice.
MALFORMED = re.compile(r"[()\[\]|]|\d{3,}|\b(?:and|or|of|the|in|with|a)$")
#: Subtype and enumeration markers. `bipolar II` is not an abbreviation of anything.
NUMERAL = frozenset("i ii iii iv v vi vii viii ix x".split())


def usable(short: str, expansion: str) -> bool:
    return (len(short) >= 2 and short.lower() not in NUMERAL and bool(expansion)
            and not MALFORMED.search(expansion) and len(expansion.split()) <= 7
            and short.lower() != expansion.lower())


def strings_in(node, out: list) -> None:
    """Every string in a record long enough to hold a definition."""
    if isinstance(node, str):
        if len(node) > 12:
            out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            strings_in(value, out)
    elif isinstance(node, list):
        for value in node:
            strings_in(value, out)


def texts_under(root: Path):
    for study in sorted(p for p in root.iterdir() if p.is_dir()):
        for flavour, name in TEXT_FLAVOURS:
            candidate = study / "processed" / flavour / name
            if candidate.is_file():
                yield study.name, candidate
                break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, action="append", default=[],
                        help="a corpus root; may be repeated")
    parser.add_argument("--records", action="append", default=[],
                        help="a glob of extraction records, mined when the papers they were "
                             "extracted from are no longer on disk. A record carries the "
                             "paper's own `long form (SF)` phrasings in its string fields")
    parser.add_argument("--out", type=Path,
                        default=Path("data/vocab/abbreviations.json"))
    args = parser.parse_args()

    if detector() is None:
        print("scispacy is not installed; falling back to the built-in miner, which is "
              "known to miss definitions it should find", file=sys.stderr)

    store = Abbreviations.load(args.out)
    papers = 0
    for root in args.texts:
        if not root.is_dir():
            continue
        for study, path in texts_under(root):
            store.learn(path.read_text(encoding="utf-8", errors="replace"), study)
            papers += 1

    for pattern in args.records:
        for path in sorted(globlib.glob(pattern)):
            if path.endswith(".raw.json"):
                continue
            try:
                body = json.loads(Path(path).read_text())
            except Exception:  # noqa: BLE001 -- a truncated record is not a reason to stop
                continue
            found: list[str] = []
            strings_in(body, found)
            store.learn(". ".join(found), Path(path).name.split(".")[0])
            papers += 1

    for short, expansion in CURATED.items():
        store.add(short, expansion, "curated")

    dropped = {k: v for k, v in store.entries.items() if not usable(k, v["expansion"])}
    for key in dropped:
        del store.entries[key]
    store.save(args.out)
    mined = sum(1 for e in store.entries.values() if e["source"] == "mined")
    curated = len(store.entries) - mined
    print(f"{papers} paper(s) read")
    if dropped:
        print(f"{len(dropped)} mis-parsed expansion(s) refused, e.g. "
              + "; ".join(f"{k!r} -> {v['expansion'][:34]!r}" for k, v in list(dropped.items())[:3]))
    print(f"{len(store.entries)} abbreviations: {mined} mined, {curated} curated")
    clashes = store.disagreements()
    if clashes:
        print(f"\n{len(clashes)} short form(s) expanded more than one way -- these cannot "
              f"be pooled across papers without a decision:")
        for short, variants in clashes[:12]:
            print(f"   {short:10s} {variants}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
