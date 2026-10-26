"""Training examples for the Vista phase: ONE call per paper that extracts everything.

Input (a single user turn): the instruction, the paper text, then the parse listing: every parse entry in document
order with its key (`analysis#...`), origin, table id, name, description, coordinate space, table caption (300
characters) and footer (700) from the old stage-1 parse, and all its points.
Target (the assistant turn): the whole compact pruned record, `to_compact(record)` without the root local_id: entity
collections, design, `coordinate_sets` (local_id = a parse key), analyses (`source_table_analysis` = a parse key),
model_estimations, tables. References are kept; ids the output creates are free strings.
Loss on the assistant turn only (token-sum, in the trainer).

Rows (JSONL): {"sid", "call": 1, "key": "paper", "prompt", "template", "target" (object), "target_json",
"parse_keys", "n_tokens", "n_input_tokens", "n_listing_tokens", "n_target_tokens"}. The prompt is the user message; the
template goes through NuExtract3's chat template (`template=`, `enable_thinking=False`); the trainer adds EOS after
`target_json`. `--max-tokens N` leaves out papers whose total exceeds N (default: none are left out).

    python tasks.py build --split val.txt --out out/val [--max-tokens N]   # -> out/val.jsonl
    python tasks.py stats --split train.txt --split val.txt                 # sizes only
    python tasks.py schemas --out-dir schemas                               # generic schema and template
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compact as K  # noqa: E402

ROOT = Path(os.environ.get("ASTRA_ROOT", "/data/james/agents/astra-distill"))
MIG = ROOT / "migrate"
DEFAULTS = {  # data v5 (frozen); v4 was train-v4/records, run-old-v4/parse, splits-v4
    "records": MIG / "train/records",
    "parse": MIG / "run-old/parse",
    "text": Path("/data/alejandro/projects/ns-pond/data"),
    "schema": MIG / "in/extraction-record.schema.json",
    "splits": MIG / "splits",
    "stage1": MIG / "in/old/pubget-bundle-20261007/corpus",
}
MAX_ANALYSES = 0           # no entry is dropped for its number of analyses

MAX_POINTS = None          # all points of an entry go into the listing

INSTR = ("Extract the full record of this neuroimaging paper: participant groups, tasks with their conditions, "
         "assessments, brain regions, devices, acquisitions, preprocessing pipelines, inference settings, measures, "
         "the study design, the statistical models with their terms and levels, the tables, and every statistical "
         "analysis the paper reports. The coordinate sets listed after the text are the paper's reported coordinates: "
         "give each its own entry in coordinate_sets (local_id = its key) and attach to it the analyses it reports "
         "(source_table_analysis = its key). Give each entity a short local_id and use those ids wherever one entity "
         "refers to another. Leave out anything the paper does not state.")


# ----------------------------------------------------------------------------------------------- paper access

class Paths:
    def __init__(self, **kw):
        for k, v in DEFAULTS.items():
            setattr(self, k, Path(kw.get(k) or v))


_IX = {}


def index(schema_path):
    if schema_path not in _IX:
        _IX[schema_path] = K.SchemaIndex(json.load(open(schema_path)))
    return _IX[schema_path]


def load_compact(sid, P):
    return K.to_compact(json.load(open(P.records / f"{sid}.extraction.json")))


def load_parse(sid, P, captions=True):
    """{key: entry} in document order, each entry with `caption` / `footer` from the old stage-1 parse when it has them"""
    import stage1
    p = json.load(open(P.parse / sid / "coordinate_parse.json"))
    caps = stage1.captions(sid, P.stage1) if captions else {}
    return {e["key"]: dict(e, **caps.get(e["key"], {})) for e in p["analyses"]}


def load_text(sid, P):
    return (P.text / sid / "processed/pubget/text.txt").read_text()


# ----------------------------------------------------------------------------------------------- entities

def entity_lists(c):
    """{collection name: [entity]} of the compact record's entity classes, conditions and design entries split out"""
    ok = lambda xs: [e for e in xs or [] if isinstance(e, dict)]  # noqa: E731  (a model's output may be malformed)
    out = {k: ok(c.get(k)) for k in ("groups", "tasks", "assessments", "regions", "devices", "acquisitions",
                                     "preprocessings", "inference_settings", "measures")}
    out["conditions"] = [dict(cd, _task=t.get("local_id")) for t in out["tasks"] for cd in ok(t.get("conditions"))]
    d = c.get("design") if isinstance(c.get("design"), dict) else {}
    out["arms"], out["timepoints"] = ok(d.get("arms")), ok(d.get("timepoints"))
    return out


def candidates(c):
    """{schema class name: [local_ids]} for the guided schema's reference enums"""
    el = entity_lists(c)
    cls = {"Group": "groups", "Task": "tasks", "Assessment": "assessments", "Region": "regions", "Device": "devices",
           "Acquisition": "acquisitions", "Preprocessing": "preprocessings", "InferenceSettings": "inference_settings",
           "Measure": "measures", "Condition": "conditions", "Arm": "arms", "Timepoint": "timepoints"}
    return {k: [e["local_id"] for e in el[v] if isinstance(e.get("local_id"), str)] for k, v in cls.items()}


def _fmt(v, n=60):
    if isinstance(v, list):
        v = ", ".join(_fmt(x, n) for x in v[:3]) + (", ..." if len(v) > 3 else "")
    elif isinstance(v, bool):
        v = "yes" if v else "no"
    elif isinstance(v, float) and v == v and abs(v) != float("inf") and v == int(v):
        v = int(v)
    v = str(v).replace("\n", " ")
    return v if len(v) <= n else v[: n - 3] + "..."


def parse_block(e, max_points=MAX_POINTS):
    """the parse entry as the model sees it"""
    pts = e.get("points") or []
    out = [f"Coordinate set: {e['key']}", f"Origin: {e['origin']}" + (f" (table {e['table_id']})" if e.get("table_id") else ""),
           f"Name: {e.get('name') or ''}"]
    if e.get("description"):
        out.append(f"Description: {e['description']}")
    if e.get("coordinate_space"):
        out.append(f"Coordinate space: {e['coordinate_space']}")
    if e.get("caption"):
        out.append(f"Caption: {e['caption']}")
    if e.get("footer"):
        out.append(f"Footer: {e['footer']}")
    out.append(f"Points: {len(pts)}")
    for p in pts[:max_points]:
        row = " ".join(_fmt(x, 8) for x in p["coordinates"])
        extra = []
        if p.get("values"):
            vs = p["values"] if isinstance(p["values"], list) else [p["values"]]
            extra.append(" ".join(f"{v.get('kind')}={_fmt(v.get('value'), 8)}" if isinstance(v, dict) else _fmt(v, 8) for v in vs[:2]))
        if p.get("cluster_size") is not None:
            extra.append(f"k={_fmt(p['cluster_size'])}")
        out.append(row + (f"  [{'; '.join(extra)}]" if extra else ""))
    if max_points and len(pts) > max_points:
        out.append(f"... ({len(pts) - max_points} more points)")
    return "\n".join(out)



# ----------------------------------------------------------------------------------------------- target, schema, template

ROOT_DROP = ("local_id", "extraction_metadata", "stable_id")


def record_target(c):
    """the compact record without its root local_id (the paper's id is not something to generate)"""
    return {k: v for k, v in c.items() if k != "local_id"}



def analyses_by_key(c):
    by = collections.defaultdict(list)
    for a in (c.get("analyses") or []) if isinstance(c, dict) else []:
        if isinstance(a, dict):
            by[a.get("source_table_analysis")].append(a)
    return by


def model_chain(c, analyses):
    """models of the analyses and their inputs_from closure, the analyses' models first (breadth-first)"""
    models = {m.get("local_id"): m for m in c.get("model_estimations") or [] if isinstance(m, dict)}
    order, todo = [], [a.get("model_estimation") for a in analyses if a.get("model_estimation")]
    while todo:
        m = todo.pop(0)
        if m in order or m not in models:
            continue
        order.append(m)
        todo += models[m].get("inputs_from") or []
    return [models[m] for m in order]


def call2_target(c, analyses):
    """the part of a record that belongs to one parse entry: its analyses (source_table_analysis left out), their
    model chain and the tables they cite. Used by the scorer to compare a record per parse key."""
    an = [{k: v for k, v in a.items() if k != "source_table_analysis"} for a in analyses]
    tabs = {t.get("local_id"): t for t in c.get("tables") or [] if isinstance(t, dict)}
    tids = []
    for a in analyses:
        for t in a.get("tables") or []:
            if t in tabs and t not in tids:
                tids.append(t)
    out = {"analyses": an}
    chain = model_chain(c, analyses)
    if chain:
        out["model_chain"] = chain
    if tids:
        out["tables"] = [tabs[t] for t in tids]
    return out


def select_keys(c, parse, empty_keys="include", max_an=0):
    """[(key, entry, analyses)] for the entries with points, in document order (the scorer's units)"""
    by = analyses_by_key(c)
    return [(k, e, by.get(k, [])) for k, e in parse.items() if e.get("points") and (by.get(k) or empty_keys == "include")
            and not (max_an and len(by.get(k, [])) > max_an)], []


def record_shape(ix, mode, keys=None):
    """JSON schema (mode "schema") or template of the whole compact record. keys: the parse keys of the paper's
    listing: `coordinate_sets[].local_id` and `analyses[].source_table_analysis` must be one of them."""
    only = [p for p in ix.schema["properties"] if p not in ROOT_DROP]
    ov = None
    if mode == "schema" and keys is not None:
        e = {"type": "string", "enum": list(keys) or ["__none__"]}
        ov = {("CoordinateSet", "local_id"): e, ("Analysis", "source_table_analysis"): e}
    elif mode == "template":
        ov = {("CoordinateSet", "local_id"): "parse_key", ("Analysis", "source_table_analysis"): "parse_key"}
    return K.shape(ix, "<root>", mode, only=only, override=ov)


def template_str(t):
    return json.dumps(t, ensure_ascii=False)


# ----------------------------------------------------------------------------------------------- examples

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("numind/NuExtract3")
    return _TOK


def ntok(s):
    return len(tokenizer()(s, add_special_tokens=False)["input_ids"])


def dumps(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


IM_END = "<|im_end|>"


def user_turn(doc, template):
    """NuExtract3's single-turn wire format (equal to apply_chat_template(..., template=, enable_thinking=False))"""
    return (f"<|im_start|>user\n【task】structured\n【template_start】{template}【template_end】\n【document_start】\n"
            f"{doc}\n【document_end】{IM_END}\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def listing(parse):
    """the parse listing: every entry, in document order"""
    return "\n\n".join(parse_block(e) for e in parse.values())


def prompt_of(text, parse):
    return INSTR + "\n\nText:\n" + text + "\n\nCoordinate sets:\n" + listing(parse)


def build_paper(args):
    """-> (row | None, notes) for one paper"""
    sid, P, opts = args
    P = Paths(**P)
    ix = index(str(P.schema))
    try:
        c = load_compact(sid, P)
        parse = load_parse(sid, P)
        text = load_text(sid, P)
    except FileNotFoundError as e:
        return None, [f"{sid}: missing {Path(str(e.filename)).name}"]
    tmpl = template_str(record_shape(ix, "template"))
    prompt = prompt_of(text, parse)
    tgt = record_target(c)
    tj = dumps(tgt)
    head = user_turn(prompt, tmpl)
    n_in, n_tg = ntok(head), ntok(tj) + 1
    n_list = ntok("\n\nCoordinate sets:\n" + listing(parse))
    row = {"sid": sid, "call": 1, "key": "paper", "prompt": prompt, "template": tmpl, "target": tgt, "target_json": tj,
           "parse_keys": list(parse), "n_tokens": n_in + n_tg, "n_input_tokens": n_in, "n_listing_tokens": n_list,
           "n_target_tokens": n_tg, "n_text_tokens": ntok(text)}
    notes = []
    if opts.get("max_tokens") and row["n_tokens"] > opts["max_tokens"]:
        notes.append(f"{sid}: {row['n_tokens']} tokens, over --max-tokens")
        row["over_max"] = True
    return row, notes


def check_row(row, ix_path):
    """gold target against its guided schema and its own references -> (schema error | None, unresolved refs)"""
    from jsonschema import Draft202012Validator
    ix = index(ix_path)
    v = Draft202012Validator(record_shape(ix, "schema", row["parse_keys"]))
    errs = list(v.iter_errors(row["target"]))
    err = f"/{'/'.join(map(str, errs[0].absolute_path))}: {errs[0].message[:150]}" if errs else None
    slots = K.ref_names(ix)
    defined = set(K.local_ids(row["target"])) | set(row["parse_keys"])
    return err, [x for x in K.refs_in(row["target"], slots) if x not in defined]


def _worker(args):
    row, notes = build_paper(args)
    if row is None:
        return None, notes, None, []
    err, unres = check_row(row, str(Paths(**args[1]).schema))
    return row, notes, err, unres


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def dist(xs):
    return {"n": len(xs), "median": pct(xs, .5), "p90": pct(xs, .9), "p99": pct(xs, .99), "max": max(xs) if xs else None,
            "total": sum(xs)}


def cmd_build(a, stats_only):
    P = {k: getattr(a, k) for k in DEFAULTS}
    sids = []
    for sp in a.split:
        f = Path(sp)
        f = f if f.exists() else Path(P["splits"]) / sp
        if f.name == "test.txt":
            raise SystemExit("test split: not built here (frozen until the end)")
        sids += f.read_text().split()
    if a.limit:
        sids = sids[: a.limit]
    opts = {"max_tokens": a.max_tokens}
    out = open(f"{a.out}.jsonl", "w") if (a.out and not stats_only) else None
    rows, notes, bad, unres = [], [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for row, n, err, un in ex.map(_worker, [(s, P, opts) for s in sids], chunksize=2):
            notes += n
            if row is None:
                continue
            rows.append({k: v for k, v in row.items() if k not in ("prompt", "template", "target", "target_json")})
            if err:
                bad.append((row["sid"], err))
            if un:
                unres.append((row["sid"], un[:3]))
            if out and not row.get("over_max"):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if out:
        out.close()
    f = lambda k: [r[k] for r in rows]  # noqa: E731
    tot = f("n_tokens")
    S = {"papers": len(rows), "max_tokens": a.max_tokens,
         "input_tokens": dist(f("n_input_tokens")), "listing_tokens": dist(f("n_listing_tokens")),
         "text_tokens": dist(f("n_text_tokens")), "target_tokens": dist(f("n_target_tokens")),
         "total_tokens": dist(tot),
         "over_16k": sum(x > 16384 for x in tot), "over_32k": sum(x > 32768 for x in tot),
         "over_48k": sum(x > 49152 for x in tot), "over_64k": sum(x > 65536 for x in tot),
         "over_max_tokens_left_out": sum(bool(r.get("over_max")) for r in rows) if a.max_tokens else 0,
         "parse_entries_per_paper": dist([len(r["parse_keys"]) for r in rows]),
         "gold_target_schema_failures": len(bad), "schema_failure_examples": bad[:5],
         "gold_unresolved_reference_papers": len(unres), "unresolved_examples": unres[:5],
         "n_notes": len(notes), "notes": notes[:10]}
    if a.manifest:
        with open(a.manifest, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps(S, indent=1))
    if a.stats_out:
        Path(a.stats_out).write_text(json.dumps(S, indent=1))


def cmd_schemas(a):
    ix = index(str(a.schema))
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, obj in (("schema-record.json", record_shape(ix, "schema")), ("template-record.json", record_shape(ix, "template"))):
        (d / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))
        print(name, len(json.dumps(obj)), "bytes;", "tokens", ntok(json.dumps(obj)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "stats", "schemas"))
    ap.add_argument("--split", action="append", default=[], help="a split list (path, or a name under --splits); repeatable")
    ap.add_argument("--out", default="", help="prefix: writes PREFIX.jsonl")
    ap.add_argument("--out-dir", default="schemas")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--stats-out", default="")
    ap.add_argument("--max-tokens", type=int, default=0, help="leave out papers whose input + target exceeds this (0: none)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=str(v))
    a = ap.parse_args()
    if a.cmd == "schemas":
        cmd_schemas(a)
    else:
        cmd_build(a, a.cmd == "stats")


if __name__ == "__main__":
    main()
