"""Per paper: is the new text the same document as the old one? Length ratio and shared 8-word shingles."""
import json, re, sys, pathlib
from concurrent.futures import ProcessPoolExecutor
O, N = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def sh(t, step=1):
    w = re.findall(r"[a-z0-9]+", t.lower())
    return {" ".join(w[i:i+8]) for i in range(0, max(0, len(w)-7), step)}
def one(sid):
    o, n = O/sid/"processed/pubget/text.txt", N/sid/"processed/pubget/text.txt"
    if not (o.exists() and n.exists()): return None
    a, b = o.read_text(), n.read_text()
    sa, sb = sh(a, 3), sh(b)
    return {"sid": sid, "old_len": len(a), "new_len": len(b), "old_in_new": len(sa & sb)/max(1, len(sa))}
if __name__ == "__main__":
    sids = [l.strip() for l in open(sys.argv[3])]
    with ProcessPoolExecutor(16) as ex, open(sys.argv[4], "w") as fo:
        for r in ex.map(one, sids, chunksize=50):
            if r: fo.write(json.dumps(r)+"\n")
