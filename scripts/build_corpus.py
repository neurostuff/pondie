"""Build the pondie corpus for the record arm, one directory per paper.

Autonima flattens a paper before screening it: `_clean_html_with_readability` keeps the text
of every <p> and <h*> but throws the heading levels away, and pubget's XSL deletes tables
outright. Pondie needs both -- `paper_sections` is an index of markdown headings, and
`Flavour` is ordered `local > pubget > elsevier > ace` by how many tables survive the render.
So the corpus is built here from the richest source each paper has, rather than reusing the
string autonima screened. The text arm reading that same rich rendering is a separate arm
(A1r), which is how the rendering and the record stay separable.

Three routes, one output shape:

    <corpus>/<pmid>/processed/local/text.tables.txt   markdown, headings kept, tables inline
    <corpus>/<pmid>/processed/local/tables.jsonl      the manifest the Tables stage copies
    <corpus>/<pmid>/stage1/analyses.json              the coordinate parse -- an INPUT
    <corpus>/<pmid>/stage1/analyses.orig.json         immutable; SignSplit rewrites the above
    <corpus>/<pmid>/provenance.json                   route, source path, text sha256
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

EXP = Path("/data/james/pondie-vs-fulltext")
REPO = Path(__file__).resolve().parents[2]

#: pubget's `keep_tables` lives only in the vendored checkout; the wheel in the venv is
#: upstream 0.0.8 and predates it. Imported the way `pondie.extraction.corpus.rebuild`
#: does -- through the source tree, without running pubget's `__init__`, which drags in
#: neuroquery for two modules that need only lxml and pandas.
VENDORED_PUBGET = Path("/home/james/projects/fdcr/vendor/pubget/src")

PROJECT_RUNS = {
    "cue_reactivity": ("v5-gpt", "v5-annotation-only-gpt"),
    "vbm_of_substance_use": ("v2", "v2-annotation-only-gpt"),
    "vbm_of_ptsd": ("v1", "v1-annotation-only"),
}


def localname(elem) -> str:
    from lxml import etree
    return etree.QName(elem).localname


def squash(elem) -> str | None:
    return " ".join(" ".join(elem.itertext()).split()) if elem is not None else None


# --------------------------------------------------------------------------- table render

def csv_to_markdown(text: str, max_rows: int = 400) -> str:
    """A table as pipes and a delimiter row.

    Markdown rather than TSV for the same reason `rebuild.py` gives: the consumer renders
    plain text, so the grid has to *read* as a table without being rendered as one.
    """
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)][:max_rows]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    clean = lambda c: c.replace("|", "\\|").replace("\n", " ").strip()
    out = ["| " + " | ".join(clean(c) for c in rows[0]) + " |",
           "|" + "---|" * width]
    out += ["| " + " | ".join(clean(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def html_table_to_markdown(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return csv_to_markdown(buf.getvalue())


def table_block(label: str | None, caption: str | None, body: str,
                footer: str | None) -> str:
    head = label or "Table"
    parts = [f"### {head}"]
    if caption:
        parts.append(caption)
    if body:
        parts.append(body)
    if footer:
        parts.append(f"_{footer}_")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- routes

def build_elsevier(pmid: str, src: Path) -> tuple[str, list[dict]]:
    """Elsevier ships markdown with headings already; only the tables are missing."""
    from lxml import etree

    text = (src / "text.txt").read_text(encoding="utf-8", errors="replace")

    meta: dict[str, dict] = {}
    article = src / "article.xml"
    if article.is_file():
        try:
            root = etree.parse(str(article)).getroot()
            for elem in root.iter():
                if localname(elem) != "table" or not elem.get("id"):
                    continue
                kids = list(elem.iterchildren())
                pick = lambda *names: next(
                    (c for c in kids if localname(c) in names), None)
                # Elsevier writes the id upper-case (`TAB1`) while the exported CSV is
                # named from the lower-case form, and calls the footnote a `legend`.
                meta[elem.get("id").lower()] = {
                    "label": squash(pick("label")),
                    "caption": squash(pick("caption")),
                    "footer": squash(pick("legend", "table-footnote")),
                }
        except Exception as exc:  # a malformed float must not cost the paper its text
            print(f"    warn {pmid}: article.xml unreadable ({exc})", file=sys.stderr)

    manifest, blocks, ordinal = [], [], 1
    for path in sorted((src / "tables").glob("*.csv")) if (src / "tables").is_dir() else []:
        # `01_tbl1.csv` -> the element id `tbl1` that article.xml keys captions on
        table_id = path.stem.split("_", 1)[-1]
        info = meta.get(table_id.lower(), {})
        label = info.get("label")
        body = csv_to_markdown(path.read_text(encoding="utf-8", errors="replace"))
        blocks.append(table_block(label or f"Table {ordinal}", info.get("caption"),
                                  body, info.get("footer")))
        manifest.append({
            "table_id": table_id,
            "table_number": label,
            "caption": info.get("caption"),
            "footer": info.get("footer"),
            "contains_coordinates": None,
            "metadata": {"table_label": label,
                         "data_path": f"tables/{path.name}"},
        })
        ordinal += 1
    if blocks:
        text = text.rstrip() + "\n\n## Tables\n\n" + "\n\n".join(blocks) + "\n"
    return text, manifest


def build_ace(pmid: str, html_path: Path, ace_tables: dict[str, list[dict]],
              tables_root: Path) -> tuple[str, list[dict]]:
    """A journal page, so the article has to be found before it can be converted.

    Readability picks the same content autonima screens; the difference is that the heading
    levels are kept here instead of collapsed into paragraphs.
    """
    import readabilipy
    from bs4 import BeautifulSoup

    raw = html_path.read_text(encoding="utf-8", errors="replace")
    article = readabilipy.simple_json_from_html_string(raw, use_readability=True)
    content = article.get("content") or ""
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = (article.get("title") or "").strip()
    lines = [f"# {title}"] if title else []
    for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        body = elem.get_text(" ", strip=True)
        if not body:
            continue
        if elem.name.startswith("h"):
            lines.append(f"\n{'#' * (int(elem.name[1]) + 1)} {body}")
        else:
            lines.append(body)
    text = "\n\n".join(lines)

    manifest, blocks, ordinal = [], [], 1
    for row in ace_tables.get(pmid, []):
        raw_file = row.get("table_raw_file") or ""
        path = tables_root.parent / raw_file if raw_file else None
        body = ""
        if path is not None and path.is_file():
            body = html_table_to_markdown(
                path.read_text(encoding="utf-8", errors="replace"))
        # `table_id` is ACE's internal database key, not anything the paper printed, so a
        # missing label becomes a positional "Table N" in the prose and stays null in the
        # manifest -- pondie's Tables stage falls back to `tbl{index}` on its own.
        label = row.get("table_label") or None
        caption = row.get("table_caption") or None
        footer = row.get("table_foot") or None
        blocks.append(table_block(label or f"Table {ordinal}", caption, body, footer))
        manifest.append({
            "table_id": str(row["table_id"]),
            "table_number": label,
            "caption": caption,
            "footer": footer,
            "contains_coordinates": None,
            "metadata": {"table_label": label, "data_path": raw_file},
        })
        ordinal += 1
    if blocks:
        text = text.rstrip() + "\n\n## Tables\n\n" + "\n\n".join(blocks) + "\n"
    return text, manifest


def build_pmc(pmid: str, src: Path) -> tuple[str, list[dict]]:
    """pubget's own extractor with `keep_tables`, which is what `local` flavour means."""
    if str(VENDORED_PUBGET) not in sys.path:
        sys.path.insert(0, str(VENDORED_PUBGET))
    import types
    if "pubget" not in sys.modules:  # register a stub so __init__ never runs
        stub = types.ModuleType("pubget")
        stub.__path__ = [str(VENDORED_PUBGET / "pubget")]
        sys.modules["pubget"] = stub
    from pubget._text import TextExtractor  # type: ignore
    from lxml import etree

    extractor = TextExtractor(keep_tables=True)
    article = etree.parse(str(src / "article.xml"))
    result = extractor.extract(article, src, None)
    parts = [str(result.get(k) or "").strip()
             for k in ("title", "keywords", "abstract", "body")]
    text = "\n\n".join(p for p in parts if p)

    manifest = []
    for info_path in sorted((src / "tables").glob("*_info.json")):
        info = json.loads(info_path.read_text())
        manifest.append({
            "table_id": info.get("table_id") or info_path.stem,
            "table_number": info.get("table_label"),
            "caption": info.get("table_caption"),
            "footer": info.get("table_foot"),
            "contains_coordinates": None,
            "metadata": {"table_label": info.get("table_label"),
                         "data_path": f"tables/{info.get('table_data_file')}"},
        })
    return text, manifest


# --------------------------------------------------------------------------- stage 1

def load_baseline_analyses() -> dict[str, list[dict]]:
    """pmid -> the analyses autonima already parsed out of that paper's tables.

    Stage 1 is an input pondie never regenerates, and re-parsing would cost a model call per
    table *and* let the two arms disagree about how a table splits. Transporting the
    baseline's parse makes the table split identical by construction.
    """
    out: dict[str, list[dict]] = {}
    for project, runs in PROJECT_RUNS.items():
        for run in runs:
            path = REPO / "projects" / project / run / "outputs" / "coordinate_parsing_results.json"
            if not path.is_file():
                continue
            for study in json.loads(path.read_text()).get("studies", []):
                pmid = str(study.get("pmid") or "")
                if pmid and study.get("analyses"):
                    out.setdefault(pmid, study["analyses"])
    return out


def stage1_doc(pmid: str, analyses: list[dict], manifest: list[dict],
               tier: str) -> dict:
    by_id = {str(t["table_id"]): t for t in manifest}
    enriched = []
    for item in analyses:
        table = by_id.get(str(item.get("table_id") or ""), {})
        enriched.append({
            "name": item.get("name"),
            "description": item.get("description"),
            "points": item.get("points") or [],
            "table_id": item.get("table_id"),
            "table_number": table.get("table_number"),
            "table_caption": table.get("caption"),
            "table_footer": table.get("footer"),
        })
    return {
        "study": pmid,
        "source": f"autonima/{tier}",
        "analyses": enriched,
        # A fresh transport has not had the sign split applied and must say so rather than
        # imply it did; pondie's SignSplit stage reads this and rewrites the file.
        "sign_split_applied": False,
    }


# --------------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, default=EXP / "pmids" / "cohort.csv")
    ap.add_argument("--out", type=Path, default=EXP / "corpus")
    ap.add_argument("--pmids", type=Path, help="restrict to these pmids, one per line")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true",
                    help="rewrite stage1/analyses.json even if SignSplit changed it")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(args.cohort.open()) if r["in_cohort"] == "True"]
    if args.pmids:
        wanted = {l.strip() for l in args.pmids.read_text().splitlines() if l.strip()}
        rows = [r for r in rows if r["pmid"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    ace_tables: dict[str, list[dict]] = {}
    tables_csv = EXP / "articles" / "ace_outputs" / "processed" / "tables.csv"
    with tables_csv.open() as fh:
        for row in csv.DictReader(fh):
            ace_tables.setdefault(str(row["pmid"]), []).append(row)
    tables_root = EXP / "articles" / "ace_outputs" / "processed" / "tables"

    baseline = load_baseline_analyses()
    print(f"baseline coordinate parses: {len(baseline)} pmids")

    from collections import Counter
    tiers, failures = Counter(), []
    for index, row in enumerate(rows, 1):
        pmid, route = row["pmid"], row["build_source"]
        src = EXP / row["source_path"]
        study = args.out / pmid
        try:
            if route == "elsevier":
                text, manifest = build_elsevier(pmid, src)
            elif route == "ace":
                text, manifest = build_ace(pmid, src, ace_tables, tables_root)
            elif route == "pmc":
                text, manifest = build_pmc(pmid, src)
            else:
                raise ValueError(f"unknown route {route!r}")
            if not text.strip():
                raise ValueError("empty text")
        except Exception as exc:
            failures.append((pmid, route, f"{type(exc).__name__}: {exc}"))
            continue

        processed = study / "processed" / "local"
        processed.mkdir(parents=True, exist_ok=True)
        text_path = processed / "text.tables.txt"
        text_path.write_text(text, encoding="utf-8")
        with (processed / "tables.jsonl").open("w", encoding="utf-8") as fh:
            for entry in manifest:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        analyses = baseline.get(pmid) or []
        tier = "parsed" if analyses else "empty"
        tiers[tier] += 1
        doc = stage1_doc(pmid, analyses, manifest, tier)
        stage1 = study / "stage1"
        stage1.mkdir(parents=True, exist_ok=True)
        original = stage1 / "analyses.orig.json"
        current = stage1 / "analyses.json"
        payload = json.dumps(doc, indent=1) + "\n"
        # SignSplit rewrites analyses.json in place, so a rebuild would silently destroy a
        # split parse. Only the immutable copy is safe to overwrite unconditionally.
        if current.is_file() and original.is_file() and not args.force:
            if current.read_text() != original.read_text():
                print(f"    skip {pmid}: analyses.json diverged from orig (SignSplit ran)")
                original.write_text(payload)
                continue
        original.write_text(payload)
        current.write_text(payload)

        (study / "provenance.json").write_text(json.dumps({
            "pmid": pmid,
            "project": row["project"],
            "route": route,
            "mirror": row["mirror"],
            "source_path": row["source_path"],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "n_chars": len(text),
            "n_tables": len(manifest),
            "n_analyses": len(analyses),
            "stage1_tier": tier,
        }, indent=1) + "\n")

        if index % 50 == 0:
            print(f"  {index}/{len(rows)}")

    print(f"\nbuilt {sum(tiers.values())} / {len(rows)}   tiers={dict(tiers)}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for pmid, route, why in failures[:15]:
            print(f"  {pmid} [{route}] {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
