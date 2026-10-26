"""Training records: keep only analyses attached to a coordinate-parser group, and the entities they need.

    python prune.py MIGRATED_RUN OUT_DIR --schema extraction-record.schema.json [--workers N]

An analysis is kept when its `source_table_analysis` names a coordinate set in the record whose parse entry holds
at least one point, or when it has no key and such a set's `analysis` names it (it then takes that set's key). Every
other analysis is
removed, and so is every entity reachable from a removed analysis but not from a kept one (or from a coordinate set):
models through `inputs_from`, their preprocessing, measures, inference settings, tasks and their acquisitions and
devices, groups, assessments, regions, tables, design arms and timepoints. References are followed through the slots
the extraction schema marks `x-reference`. An entity no analysis reaches is study-level and kept.
A nested entity (a task's condition, a model's term) stands for its container: reaching it reaches the container,
and it goes only with the container.

OUT_DIR/records/<sid>.extraction.json  the pruned record, validated
OUT_DIR/pruned/<sid>.json              what was removed, by collection and local_id
OUT_DIR/summary.json                   counts
"""
import argparse
import collections
import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

#: top-level collections whose entries have a local_id, as `Study` holds them
COLLECTIONS = ("groups", "tasks", "assessments", "regions", "devices", "acquisitions", "preprocessings",
               "model_estimations", "inference_settings", "measures", "analyses", "tables", "coordinate_sets",
               "external_datasets")
DESIGN = ("arms", "timepoints")
_REF = None


def ref_slots(schema):
    """Property names that hold references to another entity's local_id, from the schema's x-reference marks."""
    names = set()

    def walk(x, prop):
        if isinstance(x, dict):
            if "x-reference" in x and prop:
                names.add(prop)
            for k, v in x.items():
                walk(v, prop if k in ("items", "anyOf", "oneOf", "allOf") else (k if k != "properties" else prop))
        elif isinstance(x, list):
            for v in x:
                walk(v, prop)

    for d in schema["$defs"].values():
        for p, v in d.get("properties", {}).items():
            walk(v, p)
    return names


def refs_in(x, slots):
    """Every local_id string held under a reference slot anywhere inside `x`."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k in slots and isinstance(v, str):
                out.append(v)
            elif k in slots and isinstance(v, list) and all(isinstance(i, str) for i in v):
                out.extend(v)
            else:
                out.extend(refs_in(v, slots))
    elif isinstance(x, list):
        for v in x:
            out.extend(refs_in(v, slots))
    return out


def owners(rec):
    """local_id -> (collection, container local_id) for top-level and design entries, and for nested entities
    with a local_id (conditions, terms, levels), the top-level entry that contains them."""
    own = {}

    def nested(x, top):
        if isinstance(x, dict):
            if isinstance(x.get("local_id"), str) and x is not top[2]:
                own.setdefault(x["local_id"], (top[0], top[1]))
            for v in x.values():
                nested(v, top)
        elif isinstance(x, list):
            for v in x:
                nested(v, top)

    for c in COLLECTIONS:
        for e in rec.get(c) or []:
            if isinstance(e, dict) and isinstance(e.get("local_id"), str):
                own[e["local_id"]] = (c, e["local_id"])
                nested(e, (c, e["local_id"], e))
    for c in DESIGN:
        for e in (rec.get("design") or {}).get(c) or []:
            if isinstance(e, dict) and isinstance(e.get("local_id"), str):
                own[e["local_id"]] = ("design." + c, e["local_id"])
                nested(e, ("design." + c, e["local_id"], e))
    return own


def prune(rec, slots, pointed, null_entries=frozenset()):
    """`pointed`: the keys of the paper's parse entries that hold at least one point. `null_entries`: point-less
    table entries whose table has pointed entries (where a null result can sit)."""
    own = owners(rec)
    entries = {}
    for c in COLLECTIONS:
        for e in rec.get(c) or []:
            if isinstance(e, dict) and "local_id" in e:
                entries[(c, e["local_id"])] = e
    for c in DESIGN:
        for e in (rec.get("design") or {}).get(c) or []:
            if isinstance(e, dict) and "local_id" in e:
                entries[("design." + c, e["local_id"])] = e
    set_keys = {cs["local_id"] for cs in rec.get("coordinate_sets") or []}

    def sta(a):
        v = a.get("source_table_analysis")
        return v.get("value") if isinstance(v, dict) else None

    # coordinate-backed: the analysis's set is a parse entry holding at least one point (REVIEW-2 A)
    # ...or a null result: no point survived, the analysis says `no_significant_effect`, and its table reports
    # coordinates for other analyses (5vb5LXjqprf6 "Control>Somm": a header row with nothing under it). A point-less
    # entry in a table with no coordinates at all is a region-level table, not a coordinate analysis.
    def null_ok(a):
        v = a.get("outcome")
        return (sta(a) in null_entries and isinstance(v, dict) and v.get("value") == "no_significant_effect")

    kept = {a["local_id"] for a in rec.get("analyses") or []
            if sta(a) in set_keys and (sta(a) in pointed or null_ok(a))}
    # ...or linked from the other side: a pointed set's `analysis` names it and it has no key of its own. The schema
    # makes the join from either side ("here for sets read from the text as well"); reading only the key side dropped
    # 652 analyses in 431 training papers (6teXKZNVzJ62 ana_s224, 2026-10-26). The analysis takes that set's key, the
    # first in record order when several sets name it.
    backlink = {}
    for cs in rec.get("coordinate_sets") or []:
        if cs.get("analysis") and cs["local_id"] in pointed:
            backlink.setdefault(cs["analysis"], cs["local_id"])
    keyed = {}
    for a in rec.get("analyses") or []:
        if a["local_id"] not in kept and not sta(a) and backlink.get(a["local_id"]):
            kept.add(a["local_id"])
            keyed[a["local_id"]] = backlink[a["local_id"]]
    removed = {a["local_id"] for a in rec.get("analyses") or []} - kept

    # a coordinate set's `analysis` and an analysis's `mirror_of` point back at analyses, not at dependencies:
    # following them would keep a removed analysis's entities alive (REVIEW-2 issue 2)
    dep_slots = slots - {"analysis", "mirror_of"}

    def closure(seeds):
        seen, stack = set(), list(seeds)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for r in refs_in(entries.get(node, {}), dep_slots):
                if r in own:
                    stack.append(own[r])
        return seen

    removed_nodes = {("analyses", a) for a in removed}
    reach_removed = closure(list(removed_nodes))
    # candidates: what only removed analyses reach. Everything else stays, and whatever a staying entity
    # references stays with it (an unreferenced acquisition keeps its device), so the kept side is closed again
    # from every staying entity until nothing more is rescued
    drop = reach_removed - closure([("analyses", a) for a in kept] + [("coordinate_sets", k) for k in set_keys])
    while True:
        stay = [n for n in entries if n not in drop and n not in removed_nodes]
        rescued = drop & closure(stay)
        if not rescued - removed_nodes:
            break
        drop -= rescued
    drop |= removed_nodes
    out = dict(rec)
    gone = collections.defaultdict(list)
    for c in COLLECTIONS:
        if c in rec and isinstance(rec[c], list):
            out[c] = [e for e in rec[c] if not (isinstance(e, dict) and (c, e.get("local_id")) in drop)]
            gone[c] = [e.get("local_id") for e in rec[c] if isinstance(e, dict) and (c, e.get("local_id")) in drop]
    if isinstance(rec.get("design"), dict):
        out["design"] = dict(rec["design"])
        for c in DESIGN:
            if isinstance(rec["design"].get(c), list):
                out["design"][c] = [e for e in rec["design"][c] if ("design." + c, e.get("local_id")) not in drop]
                gone["design." + c] = [e.get("local_id") for e in rec["design"][c]
                                       if ("design." + c, e.get("local_id")) in drop]
    # a kept entity's reference to a dropped one cannot exist (the closure keeps it); a coordinate set's `analysis`
    # naming a removed analysis can: clear it
    cleared = []   # reported: a pointed set losing its analysis is how the one-sided keep rule hid 652 analyses
    for cs in out.get("coordinate_sets") or []:
        if cs.get("analysis") in removed:
            cleared.append((cs["local_id"], cs["analysis"], cs["local_id"] in pointed))
            cs["analysis"] = None
    gone["cleared_set_links"] = [c for c in cleared]
    out["analyses"] = [dict(an, source_table_analysis={"extraction_status": "extracted", "value": keyed[an["local_id"]],
                                                        "value_source": "generated", "evidence": {"status": "not_found"}})
                       if an.get("local_id") in keyed else an for an in out.get("analyses") or []]
    for an in out.get("analyses") or []:
        if an.get("mirror_of") in removed:
            an["mirror_of"] = None
    return out, {k: v for k, v in gone.items() if v}, len(kept)


def degenerate(rec, slots):
    """Reference faults in a pruned record: references to no entity, and effect cells naming a term outside the
    analysis's model chain (its model and the models it reads from), and analyses with no effect cells. Any of these
    makes the record unusable as a target."""
    own = owners(rec)
    dangling = [(c, e.get("local_id"), r) for c in COLLECTIONS for e in rec.get(c) or []
                for r in refs_in(e, slots) if r not in own]
    models = {m["local_id"]: m for m in rec.get("model_estimations") or []}
    off = []
    for a in rec.get("analyses") or []:
        terms, stack, seen = set(), [a.get("model_estimation")], set()
        while stack:
            k = stack.pop()
            if k in seen or k not in models:
                continue
            seen.add(k)
            terms |= {t["local_id"] for t in models[k].get("terms") or []}
            stack += models[k].get("inputs_from") or []
        off += [(a.get("local_id"), c.get("term")) for c in (a.get("effect") or {}).get("cells") or []
                if c.get("term") and c["term"] not in terms]
    empty = [a.get("local_id") for a in rec.get("analyses") or [] if not (a.get("effect") or {}).get("cells")]
    return dangling, off, empty


def one(args):
    p, out_dir, schema_path, parse_dir = args
    global _REF
    from jsonschema import Draft202012Validator
    if _REF is None:
        sch = json.load(open(schema_path))
        _REF = (ref_slots(sch), Draft202012Validator(sch))
    slots, val = _REF
    rec = json.load(open(p))
    sid = p.name[: -len(".extraction.json")]
    parse = json.load(open(pathlib.Path(parse_dir) / sid / "coordinate_parse.json"))
    pointed = {e["key"] for e in parse["analyses"] if e.get("points")}
    coord_tables = {e.get("table_id") for e in parse["analyses"] if e.get("points") and e.get("origin") == "table"}
    null_entries = {e["key"] for e in parse["analyses"]
                    if not e.get("points") and e.get("origin") == "table" and e.get("table_id") in coord_tables}
    out, gone, n_kept = prune(rec, slots, pointed, null_entries)
    errs = [f"/{'/'.join(map(str, e.absolute_path))}: {e.message[:160]}" for e in val.iter_errors(out)][:10]
    dangling, off, empty = degenerate(out, slots)
    status = ("invalid" if errs else "no_coordinate_analyses" if n_kept == 0
              else "excluded_dangling_reference" if dangling else "excluded_cell_term_off_chain" if off
              else "excluded_analysis_without_cells" if empty else "ok")
    if status == "ok":
        with open(out_dir / "records" / p.name, "w") as f:
            json.dump(out, f, ensure_ascii=False)
    with open(out_dir / "pruned" / f"{sid}.json", "w") as f:
        json.dump({"sid": sid, "status": status, "kept_analyses": n_kept, "removed": gone, "errors": errs,
                   "dangling": dangling[:20], "off_chain_cells": off[:20],
                   "without_cells": empty[:20]}, f)
    return status, n_kept, {k: len(v) for k, v in gone.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--parse-dir", required=True, help="RUN/parse: one coordinate_parse.json per paper")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "pruned").mkdir(exist_ok=True)
    files = sorted((pathlib.Path(a.run) / "records").glob("*.extraction.json"))
    S, K, G = collections.Counter(), 0, collections.Counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for st, n, g in ex.map(one, ((p, out, a.schema, a.parse_dir) for p in files), chunksize=8):
            S[st] += 1
            K += n
            G.update(g)
    summary = {"records": dict(S), "kept_analyses": K, "removed": dict(G)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
