"""Invariant checks over the training records: spans, keys, references, cell terms, and overlap with the 100 papers.

    python review_train.py TRAIN_DIR PARSE_DIR NEW_DIR SCHEMA [FIRST100_IDS]
"""
import collections
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prune as P  # noqa: E402

TRAIN, PARSE, NEW, SCHEMA = sys.argv[1:5]
SL = P.ref_slots(json.load(open(SCHEMA)))


def spans(x):
    if isinstance(x, dict):
        ev = x.get("evidence")
        if isinstance(ev, dict):
            for s in ev.get("sets") or []:
                yield from s.get("spans") or []
        for k, v in x.items():
            if k != "evidence":
                yield from spans(v)
    elif isinstance(x, list):
        for v in x:
            yield from spans(v)


def V(x):
    return x.get("value") if isinstance(x, dict) else None


def one(sid):
    r = json.load(open(f"{TRAIN}/records/{sid}.extraction.json"))
    t = open(f"{NEW}/{sid}/processed/pubget/text.txt").read()
    h = hashlib.sha256(t.encode()).hexdigest()
    C = collections.Counter()
    C["hash_mismatch"] += r["extraction_metadata"]["source_text_hash"] != h
    for sp in spans(r):
        C["spans"] += 1
        C["span_text_mismatch"] += t[sp["start_char"]:sp["end_char"]] != sp["text"]
    p = json.load(open(f"{PARSE}/{sid}/coordinate_parse.json"))
    C["parse_hash_mismatch"] += p["text_sha256"] != h
    pk = {a["key"] for a in p["analyses"]}
    pointed = {a["key"] for a in p["analyses"] if a.get("points")}
    C["implausible_points"] += sum(1 for a in p["analyses"] for pt in a.get("points") or []
                                   if any(abs(v) > 200 or (v != 0 and abs(v) < 0.01) for v in pt["coordinates"]))
    sk = {c["local_id"] for c in r.get("coordinate_sets") or []}
    C["sets_not_in_parse"] += len(sk - pk)
    own = P.owners(r)
    for c in P.COLLECTIONS:
        for e in r.get(c) or []:
            C["dangling_ref"] += sum(1 for ref in P.refs_in(e, SL) if ref not in own)
    models = {m["local_id"]: m for m in r.get("model_estimations") or []}
    for a in r.get("analyses") or []:
        C["analyses"] += 1
        C["analysis_without_set"] += V(a.get("source_table_analysis")) not in sk
        nopts = V(a.get("source_table_analysis")) not in pointed
        C["analysis_on_entry_without_points"] += nopts
        C["null_result_on_entry_without_points"] += nopts and V(a.get("outcome")) == "no_significant_effect"
        m = a.get("model_estimation")
        C["analysis_without_model"] += not m
        terms, stack, seen = set(), ([m] if m else []), set()
        while stack:
            k = stack.pop()
            if k in seen or k not in models:
                continue
            seen.add(k)
            terms |= {tm["local_id"] for tm in models[k].get("terms") or []}
            stack += models[k].get("inputs_from") or []
        eff = a.get("effect") or {}
        C["kind:" + str(V(eff.get("kind")))] += 1
        cells = eff.get("cells") or []
        C["analysis_without_cells"] += not cells
        C["cell_term_not_in_model_chain"] += sum(1 for cl in cells if cl.get("term") and cl["term"] not in terms)
    return sid, C


if __name__ == "__main__":
    sids = sorted(f[: -len(".extraction.json")] for f in os.listdir(f"{TRAIN}/records"))
    T, bad = collections.Counter(), collections.defaultdict(list)
    with ProcessPoolExecutor(16) as ex:
        for sid, c in ex.map(one, sids, chunksize=20):
            T.update(c)
            for k in ("span_text_mismatch", "sets_not_in_parse", "dangling_ref", "analysis_without_set",
                      "hash_mismatch", "parse_hash_mismatch", "cell_term_not_in_model_chain"):
                if c[k] and len(bad[k]) < 5:
                    bad[k].append(sid)
    if len(sys.argv) > 5:
        first = {line.strip() for line in open(sys.argv[5]) if line.strip()}
        T["overlap_with_first100"] = len(first & set(sids))
    print(json.dumps({"records": len(sids), **dict(sorted(T.items())), "examples": bad}, indent=1))
