"""Rebuild pubget's own extracted-data CSVs from the archived PMC articles, offline.

`pmc_articles_highest_v.zip` is byte-for-byte what `pubget.extract_articles` writes -- the
`table_NNN_info.json` files carry the identical key set to a live pubget run -- so the
downstream `extract_data` step can be replayed locally. That matters because the alternative
is letting the arm re-fetch from PMC months after the baseline did: any paper whose
availability changed would move `missing_fulltexts.csv`, and that file is subtracted from the
full-text recall denominator. Replaying offline holds the denominator still.

pubget wants `articles/<000-fff>/pmcid_<PMCID>/`, keyed on the first three hex digits of the
pmcid's md5; the archive is keyed by project instead. Hardlinks, so re-bucketing 287 MB costs
nothing.

`keep_tables` stays False here on purpose: this output feeds the *text* arm and the mirrors'
table lookup, and the baseline's pubget text did not carry inlined tables. The pondie corpus
gets its own table-inlined render from `build_corpus.py`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

EXP = Path("/data/james/pondie-vs-fulltext")
VENDORED_PUBGET = Path("/home/james/projects/fdcr/vendor/pubget/src")


def bucket(pmcid: str) -> str:
    return hashlib.md5(pmcid.encode()).hexdigest()[:3]


def rebucket(src_root: Path, dest: Path) -> int:
    articles = dest / "articles"
    count = 0
    for project_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for article in sorted(project_dir.glob("pmcid_*")):
            pmcid = article.name.split("_", 1)[1]
            target = articles / bucket(pmcid) / article.name
            target.mkdir(parents=True, exist_ok=True)
            for path in article.rglob("*"):
                if not path.is_file():
                    continue
                out = target / path.relative_to(article)
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    try:
                        os.link(path, out)
                    except OSError:
                        shutil.copy2(path, out)
            count += 1
    # pubget refuses to read a step directory that does not declare itself complete.
    (articles / "info.json").write_text(json.dumps(
        {"name": "extract_articles", "is_complete": True,
         "n_articles": count}, indent=1) + "\n")
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=EXP / "pmc_articles")
    ap.add_argument("--dest", type=Path, default=EXP / "pubget_pmc")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    count = rebucket(args.src, args.dest)
    print(f"re-bucketed {count} articles -> {args.dest / 'articles'}")

    sys.path.insert(0, str(VENDORED_PUBGET))
    from pubget import extract_data_to_csv  # type: ignore

    data_dir, code = extract_data_to_csv(
        args.dest / "articles", args.dest / "extracted",
        articles_with_coords_only=False, n_jobs=args.n_jobs)
    print(f"extract_data -> {data_dir} (exit {code})")

    # pmcid -> pmid, so the mirrors can be keyed the way autonima keys studies.
    meta = {}
    with (Path(data_dir) / "metadata.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("pmid"):
                meta[str(row["pmcid"])] = str(row["pmid"])
    print(f"metadata: {len(meta)} pmcid->pmid")

    processed = args.dest / "processed_by_pmid"
    processed.mkdir(parents=True, exist_ok=True)
    for name in ("coordinates.csv", "tables.csv"):
        src = Path(data_dir) / name
        if not src.is_file():
            print(f"  warn: {name} absent")
            continue
        with src.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        out = []
        for row in rows:
            pmid = meta.get(str(row.get("pmcid")))
            if not pmid:
                continue
            row = dict(row)
            row.pop("pmcid", None)
            out.append({"pmid": pmid, **row})
        if out:
            with (processed / name).open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(out[0]))
                writer.writeheader()
                writer.writerows(out)
        print(f"  {name}: {len(out)} rows keyed by pmid")
    # `table_data_file` in tables.csv is relative to the pubget root, so the lookup has to
    # be able to reach it from processed_data_path.
    link = processed / "articles"
    if not link.exists():
        link.symlink_to(args.dest / "articles")

    # The text arm's fallback source: the same body column the baseline screened.
    text_root = args.dest / "text_by_pmid"
    text_root.mkdir(parents=True, exist_ok=True)
    written = 0
    with (Path(data_dir) / "text.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            pmid = meta.get(str(row.get("pmcid")))
            if not pmid:
                continue
            parts = [str(row.get(k) or "").strip()
                     for k in ("title", "keywords", "abstract", "body")]
            text = "\n\n".join(p for p in parts if p)
            if not text:
                continue
            (text_root / pmid).mkdir(exist_ok=True)
            (text_root / pmid / "text.txt").write_text(text, encoding="utf-8")
            written += 1
    print(f"  text_by_pmid: {written} papers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
