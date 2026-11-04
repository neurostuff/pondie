"""Compact form of a study_schema extraction record, and its inverse (Vista phase).

`to_compact(record)` unwraps every ExtractedValue to its bare value and drops what a model is not asked to produce:
  * a `not_reported` value (and an `extracted` one whose value is null or empty) is omitted;
  * `evidence`, `value_source`, `unreported_reason`, `stable_id`, `extraction_metadata` are dropped;
  * the fields left out of extraction for now (TRAINING-PLAN): Task.paradigm, Task.variant_qualifiers,
    Table.title, Table.description;
  * None, [] and {} are omitted everywhere, so the form is canonical: `to_compact(from_compact(c)) == c`.
`local_id`s and references are kept.

`from_compact(compact, schema)` re-wraps by the extraction schema: a value in a wrapped slot becomes
`{extraction_status: extracted, value, value_source: generated, evidence: {status: not_found}}`; a wrapped slot
the compact record lacks becomes `{extraction_status: not_reported, evidence: {status: not_applicable}}`; required
plain arrays that were emptied come back as []. Keys the schema does not know are kept, so they show up as schema
errors (a model that invents a field is not silently repaired).

Also here, used by tasks.py and score.py: schema helpers (`SchemaIndex`) and `compact_schema` (the JSON schema of the
compact form of a definition, for guided decoding).

    python compact.py roundtrip SCHEMA RECORDS_DIR [SPLIT_LIST] [--workers N]   # the round-trip test
"""
import copy
import json
import sys
from pathlib import Path

DROP_KEYS = frozenset({"evidence", "value_source", "unreported_reason", "stable_id", "extraction_metadata"})
#: (def name) -> fields left out of extraction, in compact form and in the restored record
EXCLUDED = {"Task": frozenset({"paradigm", "variant_qualifiers"}), "Table": frozenset({"title", "description"})}
#: top-level collection -> def name, for the exclusion lookup in the schema-free to_compact
_COLLECTION_DEF = {"tasks": "Task", "tables": "Table"}
_MISSING = object()


def is_wrapped(x):
    return isinstance(x, dict) and "extraction_status" in x and "evidence" in x


def _clean(v):
    """None, empty containers -> _MISSING"""
    if v is None or v == [] or v == {}:
        return _MISSING
    return v


def _unwrap(x):
    if x.get("extraction_status") != "extracted":
        return _MISSING
    return _clean(_compact(x.get("value"), frozenset()))


def _compact(x, excluded):
    if is_wrapped(x):
        return _unwrap(x)
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in DROP_KEYS or k in excluded:
                continue
            c = _clean(_compact(v, frozenset()))
            if c is not _MISSING:
                out[k] = c
        return out
    if isinstance(x, list):
        out = []
        for v in x:
            c = _clean(_compact(v, frozenset()))
            if c is not _MISSING:
                out.append(c)
        return out
    return x


def to_compact(record):
    """Compact form of one extraction record (see the module docstring)."""
    out = {}
    for k, v in record.items():
        if k in DROP_KEYS:
            continue
        if k in _COLLECTION_DEF and isinstance(v, list):
            ex = EXCLUDED[_COLLECTION_DEF[k]]
            items = [_clean(_compact(e, ex)) for e in v]
            c = [i for i in items if i is not _MISSING]
        else:
            c = _clean(_compact(v, frozenset()))
        if c is not _MISSING and c != []:
            out[k] = c
    return out


# ----------------------------------------------------------------------------------------------- schema index

def _name(ref):
    return ref.rsplit("/", 1)[-1]


class SchemaIndex:
    """What the extraction schema says about each property: wrapped (an Extracted* slot), an object (one or
    several definitions: a union), an array, or plain; and which properties are references to which class."""

    def __init__(self, schema):
        self.schema = schema
        self.D = schema["$defs"]
        self._root = {"properties": schema["properties"], "required": schema.get("required", [])}

    def definition(self, name):
        return self._root if name == "<root>" else self.D[name]

    def wrapped(self, name):
        d = self.D.get(name)
        return bool(d) and name.startswith("Extracted") and "extraction_status" in d.get("properties", {})

    def options(self, ps):
        opts = ps["anyOf"] if "anyOf" in ps else [ps]
        return [o for o in opts if o != {"type": "null"}]

    def kind(self, ps):
        """('wrapped', def) | ('obj', [defs]) | ('arr', items schema) | ('plain', None)"""
        opts = self.options(ps)
        refs = [_name(o["$ref"]) for o in opts if "$ref" in o]
        if len(refs) == 1 and self.wrapped(refs[0]):
            return "wrapped", refs[0]
        if refs and all("$ref" in o for o in opts) and not any(self.D[r].get("enum") for r in refs):
            return "obj", refs
        if len(opts) == 1 and "items" in opts[0]:
            return "arr", opts[0]["items"]
        return "plain", None

    def discriminator(self, defs):
        """the property whose single-valued enum tells the members of a union apart"""
        for p, v in self.D[defs[0]]["properties"].items():
            if "enum" in v and len(v["enum"]) == 1 and all(p in self.D[d]["properties"] for d in defs):
                return p
        return None

    def pick(self, defs, value):
        if len(defs) == 1:
            return defs[0]
        p = self.discriminator(defs)
        for d in defs:
            e = self.D[d]["properties"][p].get("enum")
            if e and isinstance(value, dict) and value.get(p) == e[0]:
                return d
        return None

    def ref_slots(self):
        """{property name: class name} for every x-reference property"""
        out = {}

        def walk(x, prop):
            if isinstance(x, dict):
                if "x-reference" in x and prop:
                    out[prop] = x["x-reference"]
                for k, v in x.items():
                    walk(v, prop if k in ("items", "anyOf", "oneOf", "allOf") else (k if k != "properties" else prop))
            elif isinstance(x, list):
                for v in x:
                    walk(v, prop)
        for d in self.D.values():
            for p, v in d.get("properties", {}).items():
                walk(v, p)
        return out


# ----------------------------------------------------------------------------------------------- from_compact

WRAP_PRESENT = {"extraction_status": "extracted", "value_source": "generated", "evidence": {"status": "not_found"}}
WRAP_ABSENT = {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}


def _wrap(v):
    return {"extraction_status": "extracted", "value": v, "value_source": "generated",
            "evidence": {"status": "not_found"}}


def _absent():
    return {"extraction_status": "not_reported", "evidence": {"status": "not_applicable"}}


def _restore_value(ix, ps, v):
    """restore a value found under property schema ps"""
    kind, arg = ix.kind(ps)
    if kind == "wrapped":
        return _wrap(v)
    if kind == "obj":
        d = ix.pick(arg, v)
        if d is None or not isinstance(v, dict):
            return v
        return _restore_obj(ix, d, v)
    if kind == "arr" and isinstance(v, list):
        return [_restore_value(ix, arg, i) for i in v]
    return v


def _restore_obj(ix, defname, value):
    d = ix.definition(defname)
    props, req = d["properties"], d.get("required", [])
    excl = EXCLUDED.get(defname, frozenset())
    out = {}
    for k, v in value.items():
        out[k] = _restore_value(ix, props[k], v) if k in props else v
    for p, ps in props.items():
        if p in out or p in excl:
            continue
        kind, arg = ix.kind(ps)
        if kind == "wrapped":
            out[p] = _absent()
        elif p in req and kind == "arr":
            out[p] = []
        elif p in req and kind == "obj" and len(arg) == 1:
            out[p] = _restore_obj(ix, arg[0], {})
    return out


def from_compact(compact, schema, local_id=None):
    """Schema-valid record from a compact one (see the module docstring)."""
    ix = schema if isinstance(schema, SchemaIndex) else SchemaIndex(schema)
    rec = _restore_obj(ix, "<root>", copy.deepcopy(compact))
    rec.setdefault("local_id", local_id or "unknown")
    rec["extraction_metadata"] = {"extractor_model": "vista-compact", "extractor_version": "0"}
    return rec


# ----------------------------------------------------------------------------------------------- JSON schema / template of the compact form

MAX_TEMPLATE_ENUM = 60
EXTRA_REQUIRED = frozenset()  # (def, slot) pairs to require beyond the rule; none: gold has a mediation without a path (aht88ZqK8NhX)


def _val(ix, ps, mode):
    """schema (mode "schema") or NuExtract template (mode "template") of a bare value, from the `value` property of
    an Extracted* def. A closed enum is an enum; an open vocabulary (enum or string) is a string in the schema and
    its enum in the template (a hint). Template: string -> "string", enum -> [options], array -> [item]; an array
    of enum values is [[options]] (NuExtract's multi-label form)."""
    if "$ref" in ps:
        d = ix.D[_name(ps["$ref"])]
        if "enum" in d:
            return ({"type": "string", "enum": list(d["enum"])} if mode == "schema" else
                    (list(d["enum"]) if len(d["enum"]) <= MAX_TEMPLATE_ENUM else "string"))
        return _val(ix, d, mode)
    if "anyOf" in ps:
        opts = [o for o in ps["anyOf"] if o != {"type": "null"}]
        hint = open_values(ix, ps)
        if any(o.get("type") == "string" and "enum" not in o and "$ref" not in o for o in opts):
            if mode == "schema":
                return {"type": "string"}
            return hint if hint and len(hint) <= MAX_TEMPLATE_ENUM else "string"
        subs = [_val(ix, o, mode) for o in opts]
        return subs[0] if len(subs) == 1 else ({"anyOf": subs} if mode == "schema" else subs[0])
    t = ps.get("type")
    if isinstance(t, list):
        t = [x for x in t if x != "null"]
        t = t[0] if len(t) == 1 else t
    if mode == "schema":
        out = {}
        if "enum" in ps:
            out["enum"] = ps["enum"]
        if t is not None:
            out["type"] = t
        if "items" in ps:
            out["items"] = _val(ix, ps["items"], mode)
        return out
    if "enum" in ps:
        return list(ps["enum"])
    if t == "array":
        it = _val(ix, ps.get("items", {}), mode)
        return [it]
    return {"string": "string", "number": "number", "integer": "integer", "boolean": "boolean"}.get(t, "string")


def open_values(ix, ps):
    """enum values that an open-vocabulary slot suggests (for templates), else None"""
    for o in ps.get("anyOf", []):
        if "$ref" in o and "enum" in ix.D[_name(o["$ref"])]:
            return list(ix.D[_name(o["$ref"])]["enum"])
    return None


def shape(schema, defs, mode="schema", ref_enums=None, skip=(), only=None, override=None):
    """JSON schema (mode "schema": no $defs, no nulls) or NuExtract template (mode "template") of the compact form
    of `defs` (a def name, or a list: a union, merged in the template).
    Every property is optional but the plain required strings (local_id, a cell's term, ...): compact omits what
    Astra left unreported. ref_enums: {class name: [local_ids]} turns reference slots into enums of those ids
    (schema) -- in a template a reference is the placeholder "local_id". skip: property names left out; only: if
    given, the properties of the top-level `<root>` definition kept; override: {(def name, property): schema or
    template value} replacing what the schema says for that property (used for the parse-key enums)."""
    ix = schema if isinstance(schema, SchemaIndex) else SchemaIndex(schema)
    ref_enums = ref_enums or {}

    def ref(cls):
        if mode == "template":
            return "local_id"
        ids = ref_enums.get(cls)
        if ids is None:
            return {"type": "string"}
        return {"type": "string", "enum": list(ids) or ["__none__"]}

    def prop(ps):
        kind, arg = ix.kind(ps)
        if kind == "wrapped":
            return _val(ix, ix.D[arg]["properties"]["value"], mode)
        if kind == "obj":
            return union(arg)
        opts = ix.options(ps)
        o = opts[0] if opts else ps
        if kind == "arr":
            if "x-reference" in o:
                r = ref(o["x-reference"])
                return {"type": "array", "items": r} if mode == "schema" else [r]
            it = prop(o["items"])
            return {"type": "array", "items": it} if mode == "schema" else [it]
        if "x-reference" in o or "x-reference" in ps:
            return ref(o.get("x-reference") or ps["x-reference"])
        return _val(ix, ps, mode)

    def obj(name):
        d = ix.definition(name)
        excl = EXCLUDED.get(name, frozenset())
        props, req = {}, []
        for p, ps in d["properties"].items():
            if p in DROP_KEYS or p in excl or p in skip or (name == "<root>" and only is not None and p not in only):
                continue
            props[p] = override[(name, p)] if override and (name, p) in override else prop(ps)
            kind, arg = ix.kind(ps)
            # required in the guided schema: what the extraction schema requires and from_compact cannot make up. A
            # wrapped slot or an emptied array is restored (not_reported / []), a single-def object is restored from
            # {}; a plain slot (a string or a reference) and a union object (`details`) cannot be, so the model must
            # write them. EXTRA_REQUIRED: none at present.
            if (p in d.get("required", []) and (kind == "plain" or (kind == "obj" and len(arg) > 1))) \
                    or (name, p) in EXTRA_REQUIRED:
                req.append(p)
        if mode == "template":
            return props
        out = {"type": "object", "additionalProperties": False, "properties": props}
        if req:
            out["required"] = req
        return out

    def union(names):
        subs = [obj(n) for n in names]
        if len(subs) == 1:
            return subs[0]
        if mode == "schema":
            return {"anyOf": subs}
        merged = {}
        for sub in subs:
            for k, v in sub.items():
                if k in merged and merged[k] != v:
                    if isinstance(merged[k], list) and isinstance(v, list) and all(isinstance(x, str) for x in merged[k] + v):
                        merged[k] = merged[k] + [x for x in v if x not in merged[k]]
                else:
                    merged.setdefault(k, v)
        return merged

    return union([defs] if isinstance(defs, str) else defs)


# ----------------------------------------------------------------------------------------------- references

def ref_names(ix):
    """property names that hold references (x-reference), from the extraction schema"""
    return set(ix.ref_slots())


def refs_in(x, slots):
    """every local_id string held under a reference slot anywhere inside x (as migrate/prune.py's refs_in)"""
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


def local_ids(x):
    """every local_id defined anywhere inside x"""
    out = []
    if isinstance(x, dict):
        if isinstance(x.get("local_id"), str):
            out.append(x["local_id"])
        for v in x.values():
            out.extend(local_ids(v))
    elif isinstance(x, list):
        for v in x:
            out.extend(local_ids(v))
    return out


def rename_ids(x, slots, mapping):
    """copy of x with local_ids and reference values renamed through mapping"""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "local_id" and isinstance(v, str):
                out[k] = mapping.get(v, v)
            elif k in slots and isinstance(v, str):
                out[k] = mapping.get(v, v)
            elif k in slots and isinstance(v, list) and all(isinstance(i, str) for i in v):
                out[k] = [mapping.get(i, i) for i in v]
            else:
                out[k] = rename_ids(v, slots, mapping)
        return out
    if isinstance(x, list):
        return [rename_ids(v, slots, mapping) for v in x]
    return x


# ----------------------------------------------------------------------------------------------- round-trip test

def _one(args):
    path, schema_path = args
    from jsonschema import Draft202012Validator
    global _V
    try:
        _V
    except NameError:
        sch = json.load(open(schema_path))
        _V = (sch, SchemaIndex(sch), Draft202012Validator(sch))
    sch, ix, val = _V
    rec = json.load(open(path))
    c1 = to_compact(rec)
    r2 = from_compact(c1, ix)
    errs = [f"/{'/'.join(map(str, e.absolute_path))}: {e.message[:120]}" for e in val.iter_errors(r2)][:3]
    same = to_compact(r2) == c1
    # idempotent and JSON-stable
    c3 = json.loads(json.dumps(c1))
    return Path(path).name.split(".")[0], same, not errs, errs, c3 == c1


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("roundtrip",))
    ap.add_argument("schema")
    ap.add_argument("records")
    ap.add_argument("split", nargs="?")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    files = sorted(Path(a.records).glob("*.extraction.json"))
    if a.split:
        keep = set(Path(a.split).read_text().split())
        files = [f for f in files if f.name.split(".")[0] in keep]
    n = same = valid = both = 0
    bad = []
    with ProcessPoolExecutor(a.workers) as ex:
        for sid, s, v, errs, st in ex.map(_one, ((f, a.schema) for f in files), chunksize=8):
            n += 1
            same += s
            valid += v
            both += s and v
            if not (s and v) and len(bad) < 10:
                bad.append((sid, s, errs))
    print(json.dumps({"records": n, "roundtrip_equal": same, "schema_valid": valid, "both": both,
                      "failures": bad}, indent=1))


if __name__ == "__main__":
    main()

