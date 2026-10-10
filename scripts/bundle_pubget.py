"""Bundle every pubget-extracted paper with coordinates into a corpus `pondie extract` reads.

    python scripts/bundle_pubget.py --out /data/james/pubget-bundle --scan-only
    python scripts/bundle_pubget.py --out /data/james/pubget-bundle [--limit 5]

pubget-extracted means pubget, PMC or Europe PMC: ingestion fetches the last two for the
papers pubget's Open Access query misses and runs pubget's extraction over them, so each
writes pubget's layout under its own `processed/<source>/`. A folder has one of the three;
`SOURCES` is the order taken if it had more.

A paper is taken when its ns-pond data folder has that source's text, its table manifest and a
coordinate parse (`stage1/analyses.json`) with at least one point, all of whose tables are
in the manifest, and when it has a result coordinate: a parsed table the text carries (one
of its points is on a table row of `text.txt`, a tab-separated or `|` line), or a prose
entry upstream read as a `result`. A parsed table not found in the text is listed in the
paper's `tables_not_in_text`; most are parse errors (a table of lake depths read as
coordinates), so they do not exclude the paper. pubget inserts each table at its position
as tab-separated rows, so the text needs no rebuild.

The parse holds coordinates stated in prose as entries with `table_id: "prose"` and a
`role` (`result`, `roi`, `seed`, `target`). They have no table, so the manifest and
table-row tests skip them, and a paper whose only points are prose seeds, ROIs or targets
reports no result to extract.

The data folder is only read. Each bundled paper is a copy of what the stages read, in the
layout `pondie.paths` describes, extracted with `--flavour best`:

    <out>/corpus/<study>/identifiers.json
    <out>/corpus/<study>/processed/<source>/{text.txt,tables.jsonl,metadata.json}
    <out>/corpus/<study>/stage1/analyses.json       the parse; split rewrites it
    <out>/corpus/<study>/stage1/analyses.orig.json  as copied, to reset a re-run
    <out>/pubget.pmids          pmid<TAB>study<TAB>source, one per bundled paper
    <out>/bundle.jsonl          one line per scanned folder: taken, or why not
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pondie.formats.parse_keys import PROSE_TABLE_ID
from pondie.paths import Flavour
from pondie.formats.table_parse import normalize_number

DATA = Path("/data/alejandro/projects/ns-pond/data")

#: The sources pubget's extraction lays out, in the order `paths.Flavour` ranks them.
SOURCES = tuple(f.value for f in Flavour if f.pubget_layout)

#: What the stages read, relative to a study folder; `{source}` is the paper's.
COPIED = (
    "identifiers.json",
    "processed/{source}/text.txt",
    "processed/{source}/tables.jsonl",
    "processed/{source}/metadata.json",
    "stage1/analyses.json",
)
REQUIRED = ("identifiers.json", "processed/{source}/text.txt",
            "processed/{source}/tables.jsonl", "stage1/analyses.json")

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _table_rows(text: str) -> list[list[int]]:
    """The numbers on each table row of the text, rounded, in order."""
    return [
        [round(float(n)) for n in _NUMBER.findall(line)]
        for line in map(normalize_number, text.splitlines())
        if "\t" in line or line.startswith("|")
    ]


def tables_not_in_text(text: str, analyses: list[dict]) -> list[str]:
    """The parsed tables none of whose points is on a table row as three consecutive numbers."""
    triples = {tuple(row[i : i + 3]) for row in _table_rows(text) for i in range(len(row) - 2)}
    seen: dict[str, bool] = {}
    for a in analyses:
        if a.get("points"):
            table = str(a.get("table_id"))
            seen[table] = seen.get(table, False) or any(
                tuple(round(float(v)) for v in p.get("coordinates") or []) in triples
                for p in a["points"]
            )
    return sorted(t for t, hit in seen.items() if not hit)


def source_of(study_dir: Path) -> str | None:
    """The first of `SOURCES` this folder has processed output for."""
    return next((s for s in SOURCES if (study_dir / "processed" / s).is_dir()), None)


def scan(study_dir: Path) -> dict:
    """Whether a data folder can be bundled, and why not when it cannot."""
    row: dict = {"study": study_dir.name, "taken": False}
    source = source_of(study_dir)
    if source is None:
        return row | {"reason": "no pubget-extracted source"}
    row["source"] = source
    processed = study_dir / "processed" / source
    for need in REQUIRED:
        if not (study_dir / need.format(source=source)).is_file():
            return row | {"reason": f"no {need.format(source=source)}"}
    try:
        row["pmid"] = str(json.loads((study_dir / "identifiers.json").read_text())["pmid"] or "")
        parse = json.loads((study_dir / "stage1/analyses.json").read_text(encoding="utf-8"))
        manifest = {
            str(json.loads(line).get("table_id"))
            for line in (processed / "tables.jsonl").read_text().splitlines()
            if line.strip()
        }
        text = (processed / "text.txt").read_text(encoding="utf-8")
    except (ValueError, KeyError, UnicodeDecodeError) as error:
        return row | {"reason": f"unreadable: {error}"[:200]}
    if not row["pmid"]:
        return row | {"reason": "no pmid"}
    analyses = parse.get("analyses") or []
    points = sum(len(a.get("points") or []) for a in analyses)
    if not points:
        return row | {"reason": "no coordinates in the parse"}
    # Prose entries share the parse but have no table: they are held to neither the
    # manifest nor the table-row test, and count when upstream read them as a result.
    prose = [a for a in analyses if str(a.get("table_id")) == PROSE_TABLE_ID]
    tabled = [a for a in analyses if str(a.get("table_id")) != PROSE_TABLE_ID]
    results = sum(len(a.get("points") or []) for a in prose if a.get("role") == "result")
    parsed = {str(a.get("table_id")) for a in tabled if a.get("points")}
    if not parsed <= manifest:
        return row | {"reason": "parse names tables the manifest lacks: "
                      + ", ".join(sorted(parsed - manifest))}
    absent = tables_not_in_text(text, tabled)
    if len(absent) == len(parsed) and not results:
        detail = ", ".join(absent) if parsed else "only seed/roi/target points in prose"
        return row | {"reason": "no result coordinates: " + detail}
    return row | {"taken": True, "analyses": len(analyses), "points": points,
                  "tables": len(parsed), "tables_not_in_text": absent,
                  "prose_points": sum(len(a.get("points") or []) for a in prose),
                  "prose_result_points": results, "chars": len(text)}


def copy_one(src: Path, dest: Path, source: str = "pubget") -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for item in (c.format(source=source) for c in COPIED):
        if (src / item).is_file():
            (dest / item).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / item, dest / item)
    shutil.copy2(dest / "stage1/analyses.json", dest / "stage1/analyses.orig.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA, help="the ns-pond data folders (read only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scan-only", action="store_true", help="write only bundle.jsonl")
    ap.add_argument("--limit", type=int, help="bundle the first N eligible papers")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.out.resolve().is_relative_to(args.data.resolve()):
        sys.exit("--out must not be inside --data")
    args.out.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in args.data.iterdir() if source_of(p))
    with ProcessPoolExecutor(args.workers) as pool:
        scanned = list(pool.map(scan, folders, chunksize=64))
    # One folder per pmid: two neurostore ids for one article would be extracted twice.
    first: dict[str, str] = {}
    for r in scanned:
        if r["taken"]:
            if r["pmid"] in first:
                r.update(taken=False, reason=f"duplicate pmid of {first[r['pmid']]}")
            else:
                first[r["pmid"]] = r["study"]
    taken = [r for r in scanned if r["taken"]]
    print(f"{len(folders)} folders with {'/'.join(SOURCES)} output; {len(taken)} eligible",
          file=sys.stderr)

    if not args.scan_only:
        todo = taken[: args.limit] if args.limit else taken
        for r in todo:
            copy_one(args.data / r["study"], args.out / "corpus" / r["study"], r["source"])
        with open(args.out / "pubget.pmids", "w") as fh:
            fh.write("# pmid\tstudy\tsource\n")
            fh.writelines(f"{r['pmid']}\t{r['study']}\t{r['source']}\n" for r in todo)
        print(f"bundled {len(todo)} into {args.out / 'corpus'}", file=sys.stderr)

    with open(args.out / "bundle.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in scanned)
    reasons: dict[str, int] = {}
    for r in scanned:
        if not r["taken"]:
            reasons[r["reason"].split(":")[0]] = reasons.get(r["reason"].split(":")[0], 0) + 1
    for key, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6d}  {key}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
