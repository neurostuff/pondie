"""Does each corpus text belong to the pmid it is filed under? Title words vs the text head."""
import json, re, sys, urllib.request
from pathlib import Path

corpus = Path(sys.argv[1]); pmids = Path(sys.argv[2]).read_text().split()
url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id="
       + ",".join(pmids))
res = json.load(urllib.request.urlopen(url))["result"]
words = lambda s: {w for w in re.findall(r"[a-z]{5,}", s.lower())}
for p in pmids:
    title = res.get(p, {}).get("title", "")
    f = corpus / p / "processed/local/text.tables.txt"
    head = f.read_text()[:6000] if f.is_file() else ""
    tw = words(title); overlap = len(tw & words(head)) / max(1, len(tw))
    flag = "OK " if overlap >= 0.6 else "BAD"
    print(flag, p, f"{overlap:.2f}", title[:90], "| head:", head[:80].replace("\n", " "))
