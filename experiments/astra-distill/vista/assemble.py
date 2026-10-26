"""Put the outputs of call 1 and of every call 2 of a paper back into one compact record (and, through
compact.from_compact, one schema record).

Call 1 supplies the entities. Each call 2 supplies analyses, their model chain and tables, with ids it made up
itself; two calls can make up the same id for different things, so a created id that an earlier call already
defined with different content is renamed (`<id>~<n>`, everywhere it is used by that call). Models, tables and
analyses with identical content under the same id are one. Each analysis gets `source_table_analysis` = the key of
the parse entry that was asked, and the entry becomes a coordinate set pointing at the entry's first analysis.
"""
import copy

import compact as K


def _content_ids(obj):
    """ids a call 2 object defines, in a form the renamer can look up: models (with their terms), analyses, tables"""
    out = []
    for coll in ("analyses", "model_chain", "tables"):
        for e in obj.get(coll) or []:
            if isinstance(e, dict) and isinstance(e.get("local_id"), str):
                out.append((coll, e))
    return out


def assemble(call1, call2, slots):
    """call1: compact object or None; call2: {parse key: compact object or None} -> compact record"""
    rec = copy.deepcopy(call1) if isinstance(call1, dict) else {}
    defined = {"analyses": {}, "model_estimations": {}, "tables": {}}
    names = {"analyses": "analyses", "model_chain": "model_estimations", "tables": "tables"}
    sets, n_rename = [], 0
    for key, obj in call2.items():
        if not isinstance(obj, dict):
            continue
        obj = copy.deepcopy(obj)
        mapping = {}
        for coll, e in _content_ids(obj):
            have = defined[names[coll]].get(e["local_id"])
            if have is not None and have != e:
                n_rename += 1
                suffix = f"~{n_rename}"
                mapping[e["local_id"]] = e["local_id"] + suffix
                for t in e.get("terms") or []:
                    if isinstance(t, dict) and isinstance(t.get("local_id"), str):
                        mapping[t["local_id"]] = t["local_id"] + suffix
        if mapping:
            obj = K.rename_ids(obj, slots, mapping)
        first = None
        for a in obj.get("analyses") or []:
            if isinstance(a, dict):
                defined["analyses"][a.get("local_id")] = copy.deepcopy(a)
                a["source_table_analysis"] = key
                first = first or a.get("local_id")
                rec.setdefault("analyses", []).append(a)
        for m in obj.get("model_chain") or []:
            if isinstance(m, dict) and m.get("local_id") not in defined["model_estimations"]:
                rec.setdefault("model_estimations", []).append(m)
                defined["model_estimations"][m.get("local_id")] = m
        for t in obj.get("tables") or []:
            if isinstance(t, dict) and t.get("local_id") not in defined["tables"]:
                rec.setdefault("tables", []).append(t)
                defined["tables"][t.get("local_id")] = t
        if obj.get("analyses"):
            sets.append({"local_id": key, **({"analysis": first} if first else {})})
    if sets:
        rec["coordinate_sets"] = sets
    return rec


def reference_stats(rec, slots):
    """[resolved, total] over every reference value of a compact record"""
    defined = set(K.local_ids(rec))
    refs = K.refs_in(rec, slots)
    return [sum(r in defined for r in refs), len(refs)]
