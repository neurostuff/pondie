"""compare_effects.py REF_RAW NEW_RAW: compare two effects_nx raw.jsonl files (beast vs Lonestar6), keyed by sid|analysis."""
import json, sys

def load(p):
    return {r["key"]: r for r in map(json.loads, open(p)) if r}

ref, new = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(ref) & set(new))
print(f"ref {len(ref)} new {len(new)} common {len(keys)}")
ex = kind = cellset = 0
diffs = []
canon = lambda c: sorted(json.dumps(x, sort_keys=True) for x in c)
for k in keys:
    a, b = ref[k]["out"], new[k]["out"]
    e = a == b
    ex += e
    kind += a.get("kind") == b.get("kind")
    cs = canon(a.get("cells") or []) == canon(b.get("cells") or [])
    cellset += cs
    if not e:
        diffs.append((k, a, b))
n = max(len(keys), 1)
print(f"exact match (kind + ordered cells): {ex}/{len(keys)} = {ex/n:.3f}")
print(f"kind match: {kind}/{len(keys)}   cell set match (order ignored): {cellset}/{len(keys)}")
print(f"new failed: {sum(not r['ok'] for r in new.values())}  truncated: {sum(r.get('finish')=='length' for r in new.values())}")
for k, a, b in diffs:
    print("DIFF", k)
    print("  ref", json.dumps(a, ensure_ascii=False))
    print("  new", json.dumps(b, ensure_ascii=False))
