"""Build the ablation corpus from the ns-pond catalog, read-only.

    python build_corpus.py --out /data/james/pondie-ablation/corpus PMID [PMID ...]

Why not reuse /data/james/pondie-vs-fulltext/corpus: for 11 of the 19 vbm_of_ptsd gold papers
it holds there, the text carries no table and the stage-1 parse is empty, while the catalog
has the tables (and their coordinate parse) for several of them. An arm cannot be blamed for
missing a contrast whose table it was never shown.

One output shape per paper, the layout `pondie.paths` reads with flavour `local`:

    <out>/<pmid>/processed/local/text.tables.txt   article text + every table as markdown
    <out>/<pmid>/processed/local/tables.jsonl      the manifest `Tables` copies
    <out>/<pmid>/stage1/analyses.json              coordinate parse (split rewrites it)
    <out>/<pmid>/stage1/analyses.orig.json         immutable copy, to reset between arms
    <out>/<pmid>/provenance.json                   which source, and why

Source choice per paper: the catalog extract whose table ids the catalog's coordinate parse
is keyed by (so a parse entry's table exists in the text), then the one with most tables,
then the longest text. Falls back to the old corpus only when the catalog has no text.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sqlite3
from pathlib import Path

CAT = Path("/data/alejandro/projects/ns-pond/catalog")
OLD = Path("/data/james/pondie-vs-fulltext/corpus")


def blob(digest: str):
    for path in (CAT / "blobs" / digest[:2]).glob(digest + "*"):
        raw = path.read_bytes()
        return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    return None


def html_table_to_markdown(html: str, max_rows: int = 400) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    rows = rows[:max_rows]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    clean = lambda c: c.replace("|", "\\|").replace("\n", " ").strip()  # noqa: E731
    out = ["| " + " | ".join(clean(c) for c in rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(clean(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def label_of(table: dict) -> str | None:
    meta = table.get("metadata") or {}
    if meta.get("label"):
        return str(meta["label"])
    number = table.get("table_number")
    return f"Table {number}" if number not in (None, "") else None


def stage1_entries(parse: dict, tables: dict[str, dict]) -> list[dict]:
    """The catalog's coordinate parse, in the shape `TableParse` reads."""
    out = []
    for table_id, entry in parse.items():
        table = tables.get(str(table_id), {})
        for analysis in entry.get("analyses") or []:
            points = []
            for c in analysis.get("coordinates") or []:
                values = []
                if c.get("statistic_value") is not None:
                    values.append({"value": c["statistic_value"],
                                   "kind": (c.get("statistic_type") or "").lower() or None})
                points.append({"coordinates": [c["x"], c["y"], c["z"]],
                               "space": c.get("space"), "values": values})
            out.append({
                "name": analysis.get("name") or "",
                "description": analysis.get("description"),
                "points": points,
                "table_id": str(table_id),
                "table_number": label_of(table) or analysis.get("table_number"),
                "table_caption": table.get("caption") or analysis.get("table_caption"),
                "table_footer": table.get("footer") or analysis.get("table_footer"),
            })
    return out


def choose(extracts: list[tuple[str, dict]], parse_keys: set[str]):
    def rank(item):
        source, b = item
        ids = {str(t.get("table_id")) for t in b.get("tables") or []}
        text = b.get("full_text_path")
        size = Path(text).stat().st_size if text and Path(text).is_file() else 0
        return (bool(parse_keys) and parse_keys <= ids, len(ids & parse_keys),
                len(b.get("tables") or []), size)

    usable = [e for e in extracts if e[1].get("full_text_path")
              and Path(e[1]["full_text_path"]).is_file()]
    return max(usable, key=rank) if usable else None


def render_old(pmid: str) -> dict | None:
    """The old corpus entry, parse reset to its immutable original."""
    old = OLD / pmid
    if not (old / "processed/local/text.tables.txt").is_file():
        return None
    manifest = []
    if (old / "processed/local/tables.jsonl").is_file():
        manifest = [json.loads(line) for line in
                    (old / "processed/local/tables.jsonl").read_text().splitlines() if line.strip()]
    document = json.loads((old / "stage1/analyses.orig.json").read_text())
    document["sign_split_applied"] = False
    document.pop("prose_foci_applied", None)
    return {"text": (old / "processed/local/text.tables.txt").read_text(encoding="utf-8"),
            "manifest": manifest, "document": document,
            "prov": {"route": "old-corpus", "source": str(old)}}


def render_catalog(pmid: str, picked, parse: dict) -> dict:
    source, b = picked
    text = Path(b["full_text_path"]).read_text(encoding="utf-8", errors="replace").rstrip()
    tables = {str(t.get("table_id")): t for t in b.get("tables") or []}
    blocks, manifest = [], []
    for ordinal, (table_id, t) in enumerate(tables.items(), start=1):
        body = ""
        raw = t.get("raw_content_path")
        if raw and Path(raw).is_file():
            body = html_table_to_markdown(Path(raw).read_text(encoding="utf-8", errors="replace"))
        label = label_of(t)
        parts = [f"### {label or f'Table {ordinal}'}"]
        if t.get("caption"):
            parts.append(t["caption"])
        if body:
            parts.append(body)
        if t.get("footer"):
            parts.append(f"_{t['footer']}_")
        blocks.append("\n\n".join(parts))
        manifest.append({"table_id": table_id, "table_number": label,
                         "caption": t.get("caption"), "footer": t.get("footer"),
                         "contains_coordinates": t.get("contains_coordinates"),
                         "metadata": {"table_label": label, "data_path": raw or ""}})
    if blocks:
        text += "\n\n## Tables\n\n" + "\n\n".join(blocks) + "\n"
    document = {"study": pmid, "source": f"catalog/{source}",
                "analyses": stage1_entries(parse, tables), "sign_split_applied": False}
    return {"text": text + "\n", "manifest": manifest, "document": document,
            "prov": {"route": f"catalog/{source}", "source": b["full_text_path"],
                     "parse_tables": sorted(parse), "manifest_tables": sorted(tables)}}


def titles(pmids: list[str]) -> dict[str, str]:
    import urllib.request
    out = {}
    for i in range(0, len(pmids), 150):
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed"
               "&retmode=json&id=" + ",".join(pmids[i:i + 150]))
        res = json.load(urllib.request.urlopen(url))["result"]
        out |= {p: res.get(p, {}).get("title", "") for p in pmids[i:i + 150]}
    return out


def matches_title(text: str, title: str) -> float:
    """Share of the title's long words that appear in the first 6,000 characters.

    The catalog filed at least one paper under another pmid (16371250's ace text is
    15381021's), and a parse with more points is no reason to take a different article.
    """
    words = lambda s: set(re.findall(r"[a-z]{5,}", s.lower()))  # noqa: E731
    want = words(title)
    return len(want & words(text[:6000])) / max(1, len(want))


def build_one(pmid: str, out: Path, con: sqlite3.Connection, title: str = "") -> dict:
    row = con.execute("select article_id from aliases where kind='pmid' and value=?",
                      (pmid,)).fetchone()
    extracts, parse = [], {}
    if row:
        for stage, source, digest in con.execute(
                "select stage, source, blob from artifacts where article_id=? and status='ok' "
                "and stage in ('extract','analyses')", row):
            b = blob(digest) if digest else None
            if b is None:
                continue
            if stage == "extract":
                extracts.append((source, b))
            else:
                parse = b
    picked = choose(extracts, {str(k) for k in parse})
    dest = out / pmid
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "processed" / "local").mkdir(parents=True)
    (dest / "stage1").mkdir(parents=True)

    catalog = render_catalog(pmid, picked, parse) if picked else None
    legacy = render_old(pmid)
    candidates = [c for c in (catalog, legacy) if c is not None]
    for c in candidates:
        c["prov"]["title_match"] = round(matches_title(c["text"], title), 2) if title else None
    rejected = [c["prov"]["route"] for c in candidates
                if title and c["prov"]["title_match"] < 0.6]
    candidates = [c for c in candidates if c["prov"]["route"] not in rejected]
    if not candidates:
        shutil.rmtree(dest)
        return {"pmid": pmid, "route": "none"}

    def score(c):
        doc = c["document"]
        points = sum(len(a.get("points") or []) for a in doc.get("analyses") or [])
        return (points, len(c["manifest"]), len(c["text"]))

    best = max(candidates, key=score)
    (dest / "processed/local/text.tables.txt").write_text(best["text"], encoding="utf-8")
    with open(dest / "processed/local/tables.jsonl", "w", encoding="utf-8") as fh:
        for m in best["manifest"]:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    document, prov = best["document"], best["prov"]
    prov["alternatives"] = {c["prov"]["route"]: score(c) for c in candidates}
    prov["rejected_wrong_paper"] = rejected
    for name in ("analyses.json", "analyses.orig.json"):
        (dest / "stage1" / name).write_text(json.dumps(document, indent=1) + "\n")
    text = (dest / "processed/local/text.tables.txt").read_text()
    prov |= {"pmid": pmid, "n_chars": len(text),
             "n_tables": len(best["manifest"]),
             "n_parse_entries": len(document.get("analyses") or []),
             "n_points": sum(len(a.get("points") or []) for a in document.get("analyses") or [])}
    (dest / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    return prov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("pmids", nargs="+")
    args = ap.parse_args()
    con = sqlite3.connect(f"file:{CAT / 'catalog.sqlite'}?mode=ro", uri=True)
    pmids = [p for arg in args.pmids for p in
             (Path(arg).read_text().split() if Path(arg).is_file() else [arg])]
    known = titles(pmids)
    for pmid in pmids:
        prov = build_one(pmid, args.out, con, known.get(pmid, ""))
        print(pmid, prov.get("route"), prov.get("n_chars"), "tables", prov.get("n_tables"),
              "parse", prov.get("n_parse_entries"), "points", prov.get("n_points"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
