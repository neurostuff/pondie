"""Diff Astra's raw record against the migrated training record, entity by local_id, evidence stripped.

    python rawdiff.py RAW_DIR TRAIN_DIR OUT_JSON
Counts changed field paths (list indices as [] where items have no local_id) per collection, with example sids.
"""
import collections, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor

RAW, TRAIN, OUT = sys.argv[1:4]


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "evidence"}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def diff(a, b, path, out):
    if type(a) is not type(b):
        out.append((path, "type"))
        return
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a:
                out.append((f"{path}.{k}", "added"))
            elif k not in b:
                out.append((f"{path}.{k}", "removed"))
            else:
                diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list):
        if all(isinstance(v, dict) and "local_id" in v for v in a + b) and (a or b):
            A, B = {v["local_id"]: v for v in a}, {v["local_id"]: v for v in b}
            for k in set(A) - set(B):
                out.append((f"{path}[id]", "item_removed"))
            for k in set(B) - set(A):
                out.append((f"{path}[id]", "item_added"))
            for k in set(A) & set(B):
                diff(A[k], B[k], f"{path}[id]", out)
        elif len(a) != len(b):
            out.append((path, f"len"))
            if len(a) and len(b) and all(not isinstance(v, (dict, list)) for v in a + b):
                pass
        else:
            for x, y in zip(a, b):
                diff(x, y, f"{path}[]", out)
    elif a != b:
        out.append((path, "value"))


def one(sid):
    raw = strip(json.load(open(f"{RAW}/{sid}.extraction.json")))
    tr = strip(json.load(open(f"{TRAIN}/{sid}.extraction.json")))
    out = []
    for c in set(raw) | set(tr):
        if c in ("extraction_metadata",):
            continue
        a, b = raw.get(c), tr.get(c)
        if a is None or b is None:
            out.append((c, "collection_" + ("added" if a is None else "removed")))
            continue
        if isinstance(a, list) and all(isinstance(v, dict) and "local_id" in v for v in a + b):
            A, B = {v["local_id"]: v for v in a}, {v["local_id"]: v for v in b}
            for k in set(B) - set(A):
                out.append((f"{c}[id]", "item_added"))
            for k in set(A) & set(B):
                diff(A[k], B[k], f"{c}[id]", out)
        else:
            diff(a, b, c, out)
    return sid, out


if __name__ == "__main__":
    sids = sorted(f[:-16] for f in os.listdir(TRAIN))
    C, ex, papers = collections.Counter(), collections.defaultdict(list), collections.Counter()
    with ProcessPoolExecutor(16) as pool:
        for sid, out in pool.map(one, sids, chunksize=20):
            for k in set(out):
                papers[k] += 1
                if len(ex[k]) < 4:
                    ex[k].append(sid)
            C.update(out)
    rows = [{"path": p, "kind": k, "n": n, "papers": papers[(p, k)], "ex": ex[(p, k)]} for (p, k), n in C.most_common()]
    json.dump({"records": len(sids), "rows": rows}, open(OUT, "w"), indent=1)
    for r in rows[:80]:
        print(f'{r["n"]:>8} {r["papers"]:>6}  {r["kind"]:<14} {r["path"]}  {r["ex"][:2]}')
