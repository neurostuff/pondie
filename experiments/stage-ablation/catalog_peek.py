"""Summarize what the ns-pond catalog holds for a list of pmids (read-only)."""
import gzip, json, sqlite3, sys
from pathlib import Path

CAT = Path("/data/alejandro/projects/ns-pond/catalog")


def blob(h):
    for p in (CAT / "blobs" / h[:2]).glob(h + "*"):
        raw = p.read_bytes()
        return json.loads(gzip.decompress(raw) if p.suffix == ".gz" else raw)


def main():
    c = sqlite3.connect(f"file:{CAT/'catalog.sqlite'}?mode=ro", uri=True)
    for pmid in sys.argv[1:]:
        row = c.execute("select article_id from aliases where kind='pmid' and value=?", (pmid,)).fetchone()
        if not row:
            print(pmid, "NOT IN CATALOG"); continue
        out = []
        for stage, source, status, h, summ in c.execute(
            "select stage,source,status,blob,summary from artifacts where article_id=? and stage in ('extract','analyses')", row):
            out.append(f"{stage}/{source}:{status}:{summ[:90]}")
        print(pmid, row[0], " | ".join(out))


if __name__ == "__main__":
    main()
