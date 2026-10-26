"""Checks of the guided schema against the extraction schema.

  gold      the guided schema (record_shape, parse keys enumerated) must accept every gold target
  preds     for prediction files ({"record": {sid: compact record | null}}): valid under the old guided schema (the
            per-paper schema stored in an old bodies file), under the new one, and after from_compact against the
            extraction schema
  fuzz      mutate gold records (delete a key or a list element somewhere, or empty a value): whenever the mutated
            record is still guided-valid it must restore to an extraction-valid record (this is the property the
            guided schema exists for; a counterexample is a slot the guided schema should require)

    python verify_schema.py gold --split val.txt --split train.txt
    python verify_schema.py preds r0-qwen-preds.partial.json r0-nu-preds.json --old-bodies val.bodies.v1.jsonl
    python verify_schema.py fuzz --split val.txt --limit 300 --mutations 6
"""
import argparse
import collections
import copy
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compact as K  # noqa: E402
import tasks as T  # noqa: E402
from jsonschema import Draft202012Validator  # noqa: E402

P = T.Paths()
_C = {}


def ctx():
    if not _C:
        ix = T.index(str(P.schema))
        _C.update(ix=ix, ext=Draft202012Validator(ix.schema))
    return _C


def guided_validator(sid):
    keys = list(T.load_parse(sid, P, captions=False))
    return Draft202012Validator(T.record_shape(ctx()["ix"], "schema", keys))


def ext_valid(rec, sid):
    c = ctx()
    try:
        return c["ext"].is_valid(K.from_compact(rec, c["ix"], local_id=sid))
    except Exception:  # noqa: BLE001
        return False


def _gold(sid):
    rec = T.record_target(T.load_compact(sid, P))
    errs = list(guided_validator(sid).iter_errors(rec))
    return sid, (f"/{'/'.join(map(str, errs[0].absolute_path))}: {errs[0].message[:120]}" if errs else None)


def mutate(rec, rng):
    """delete one key (or list element), or empty one value, at a random place; None if nothing to do"""
    rec = copy.deepcopy(rec)
    places = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                places.append((path, k))
                walk(v, path + [k])
        elif isinstance(x, list):
            for i, v in enumerate(x):
                places.append((path, i))
                walk(v, path + [i])
    walk(rec, [])
    if not places:
        return None, None
    path, k = rng.choice(places)
    o = rec
    for p in path:
        o = o[p]
    how = rng.choice(("delete", "delete", "empty"))
    if how == "delete" or not isinstance(o[k], (dict, list, str)):
        del o[k]
    else:
        o[k] = type(o[k])()
    return rec, "/".join(map(str, path + [k]))


def _fuzz(args):
    sid, n, seed = args
    rng = random.Random(f"{seed}{sid}")
    gold = T.record_target(T.load_compact(sid, P))
    v = guided_validator(sid)
    out = {"mutated": 0, "guided_valid": 0, "ext_valid": 0, "bad": []}
    for _ in range(n):
        rec, where = mutate(gold, rng)
        if rec is None:
            continue
        out["mutated"] += 1
        if v.is_valid(rec):
            out["guided_valid"] += 1
            if ext_valid(rec, sid):
                out["ext_valid"] += 1
            elif len(out["bad"]) < 3:
                out["bad"].append((sid, where))
    return out


def _pred(args):
    sid, rec, old_schema = args
    if not isinstance(rec, dict):
        return sid, None
    new = guided_validator(sid).is_valid(rec)
    old = Draft202012Validator(old_schema).is_valid(rec) if old_schema else None
    return sid, {"old": old, "new": new, "ext": ext_valid(rec, sid)}


def sids_of(a):
    out = []
    for sp in a.split:
        f = Path(sp)
        out += (f if f.exists() else P.splits / sp).read_text().split()
    return out[: a.limit] if a.limit else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("gold", "preds", "fuzz"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--split", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mutations", type=int, default=6)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--old-bodies", default="")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    ex = ProcessPoolExecutor(a.workers)
    if a.cmd == "gold":
        sids = sids_of(a)
        res = list(ex.map(_gold, sids, chunksize=8))
        bad = [r for r in res if r[1]]
        print(json.dumps({"papers": len(sids), "rejected": len(bad), "examples": bad[:5]}, indent=1))
    elif a.cmd == "fuzz":
        sids = sids_of(a)
        tot = collections.Counter()
        bad = []
        for r in ex.map(_fuzz, [(s, a.mutations, a.seed) for s in sids], chunksize=4):
            for k in ("mutated", "guided_valid", "ext_valid"):
                tot[k] += r[k]
            bad += r["bad"]
        print(json.dumps({**tot, "guided_valid_but_not_extraction_valid": tot["guided_valid"] - tot["ext_valid"],
                          "examples": bad[:8]}, indent=1))
    else:
        old = {}
        if a.old_bodies:
            for line in open(a.old_bodies):
                d = json.loads(line)
                old[d["sid"]] = d["schema"]
        for f in a.files:
            recs = json.loads(Path(f).read_text())["record"]
            res = dict(ex.map(_pred, [(s, r, old.get(s)) for s, r in recs.items()], chunksize=2))
            ok = [v for v in res.values() if v]
            n = len(res)
            cnt = collections.Counter()
            for v in ok:
                cnt["old_guided_valid"] += bool(v["old"])
                cnt["new_guided_valid"] += bool(v["new"])
                cnt["extraction_valid"] += bool(v["ext"])
                cnt["new_guided_valid_and_extraction_valid"] += bool(v["new"] and v["ext"])
                cnt["old_guided_valid_but_extraction_invalid"] += bool(v["old"] and not v["ext"])
                cnt["new_guided_valid_but_extraction_invalid"] += bool(v["new"] and not v["ext"])
            print(json.dumps({"file": Path(f).name, "records": n, "null_or_not_object": n - len(ok), **cnt}, indent=1))


if __name__ == "__main__":
    main()
