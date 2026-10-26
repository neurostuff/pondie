"""Match an old record's coordinate sets to the new coordinate parse by their points.

An old set's key is a positional stage-1 key (`t2#1`, `prose#3`), numbered over the old
`stage1/analyses.json` as `pondie.formats.parse_keys.positional_keys` numbers it. Its points come from that entry.
A new parse analysis has a content key `analysis#<h>` and its own points. Points match when every coordinate is
within TOL mm.

Per old set:
  exact    one new analysis holds exactly the same points
  split    the old points are covered by several new analyses (all of them old points)
  partial  the best new analysis shares some points (overlap reported)
  none     no new analysis shares a point
The old role is kept; `role_conflict` flags a matched new analysis whose role differs.
"""
import collections

TOL = 0.51


def positional_keys(entries):
    ordinals, keys = {}, []
    for e in entries:
        tid = str((e or {}).get("table_id") or "")
        ordinals[tid] = ordinals.get(tid, 0) + 1
        keys.append(f"{'text' if tid == 'prose' else tid}#{ordinals[tid]}")
    return keys


def canon_key(k):
    return "text#" + k[len("prose#"):] if isinstance(k, str) and k.startswith("prose#") else k


def pts(entry):
    out = []
    for p in entry.get("points") or []:
        c = p.get("coordinates") or []
        if len(c) == 3 and all(isinstance(v, (int, float)) for v in c):
            out.append(tuple(float(v) for v in c))
    return out


def _same(a, b):
    return all(abs(x - y) <= TOL for x, y in zip(a, b))


def _shared(a, b):
    """Points of `a` that have a partner in `b` (one-to-one, greedy)."""
    used, hit = set(), 0
    for p in a:
        for j, q in enumerate(b):
            if j not in used and _same(p, q):
                used.add(j)
                hit += 1
                break
    return hit


def match(old_entries, new_analyses):
    """{old key: {status, new_keys, overlap, old_n, new_n, new_roles, table_agrees}}."""
    keys = positional_keys(old_entries)
    new = [(a["key"], pts(a), a.get("role"), a.get("table_id")) for a in new_analyses]
    out = {}
    for k, e in zip(keys, old_entries):
        op = pts(e)
        if not op:
            out[k] = {"status": "no_old_points", "new_keys": [], "old_n": 0}
            continue
        scored = sorted(((_shared(op, np_), nk, np_, role, tid) for nk, np_, role, tid in new),
                        key=lambda x: -x[0])
        best = scored[0] if scored else (0, None, [], None, None)
        rec = {"old_n": len(op), "old_table_id": e.get("table_id")}
        if best[0] == 0:
            rec.update(status="none", new_keys=[])
        elif best[0] == len(op) == len(best[2]):
            rec.update(status="exact", new_keys=[best[1]], new_roles=[best[3]], new_table_ids=[best[4]])
        else:
            parts = [s for s in scored if s[0] and s[0] == len(s[2])]
            covered = sum(s[0] for s in parts)
            if len(parts) > 1 and covered >= len(op):
                rec.update(status="split", new_keys=[s[1] for s in parts], new_roles=[s[3] for s in parts],
                           new_table_ids=[s[4] for s in parts])
            else:
                rec.update(status="partial", new_keys=[best[1]], overlap=best[0], new_n=len(best[2]),
                           new_roles=[best[3]], new_table_ids=[best[4]])
        out[k] = rec
    return out


def census(sample):
    """sample: iterable of (old record, old stage1 entries, new parse analyses)."""
    C, R = collections.Counter(), collections.Counter()
    for rec, old_entries, new_analyses in sample:
        m = match(old_entries, new_analyses)
        for cs in rec.get("coordinate_sets") or []:
            k = canon_key(cs["local_id"])
            r = m.get(k)
            if r is None:
                C["key_not_in_old_parse"] += 1
                continue
            C[r["status"]] += 1
            old_role = ((cs.get("role") or {}).get("value"))
            for nr in r.get("new_roles") or []:
                R[(old_role, nr)] += 1
    return C, R
