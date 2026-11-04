"""The paper prompt the astra-distill models were trained on and are run with.

One user turn per paper: the instruction, the paper text, then the parse listing (every parse entry in document order
with its key, origin, table id, name, description, coordinate space, table caption and footer, and all its points).
The template is the compact record's shape (`template()`); NuExtract3 takes it through its chat template, the 27B as
a suffix of the prompt (`user_content`). Moved from experiments/astra-distill/vista/tasks.py and bodies_ns.py
unchanged, so a prompt built here is the one training and the ns extraction saw.
"""
import json
import os
from functools import lru_cache
from importlib import resources

from . import compact as K

MAX_POINTS = None          # all points of an entry go into the listing

INSTR = ("Extract the full record of this neuroimaging paper: participant groups, tasks with their conditions, "
         "assessments, brain regions, devices, acquisitions, preprocessing pipelines, inference settings, measures, "
         "the study design, the statistical models with their terms and levels, the tables, and every statistical "
         "analysis the paper reports. The coordinate sets listed after the text are the paper's reported coordinates: "
         "give each its own entry in coordinate_sets (local_id = its key) and attach to it the analyses it reports "
         "(source_table_analysis = its key). Give each entity a short local_id and use those ids wherever one entity "
         "refers to another. Leave out anything the paper does not state.")


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


ROOT_DROP = ("local_id", "extraction_metadata", "stable_id")


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


#: The trim training's captions had (stage1.py).
CAPTION_CHARS, FOOTER_CHARS = 300, 700


def captions(tables, entries):
    """{parse key: {"caption", "footer"}} for table entries, from a paper parse's `tables[]` (study_schema), trimmed
    as training's captions were"""
    tables = {t.get("table_id"): t for t in tables or [] if isinstance(t, dict)}
    out = {}
    for e in entries:
        t = tables.get(e.get("table_id"))
        if not t:
            continue
        cap, foot = (t.get("caption") or "").strip(), (t.get("footer") or "").strip()
        if cap or foot:
            out[e["key"]] = {"caption": cap[:CAPTION_CHARS], "footer": foot[:FOOTER_CHARS]}
    return out


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


#: What the 27B's user turn adds after the prompt (tacc/eval_record.py, as train_record.render does).
TEMPLATE_SUFFIX = "\n\nReturn the record as JSON that follows this template (values show the type or the allowed choices):\n"


def user_content(prompt, template, family):
    """(user message, chat_template_kwargs) for a model family: "nu" (NuExtract3) or "q27" (Qwen3.8-27B)"""
    if family == "nu":
        return prompt, {"enable_thinking": False, "template": template}
    if family == "q27":
        return prompt + TEMPLATE_SUFFIX + template, {"enable_thinking": False}
    raise ValueError(f"unknown model family {family!r}")


@lru_cache(maxsize=None)
def schema():
    """The extraction-record JSON Schema the models were trained against (migrate/in, sha256 27f8cf61...)."""
    return json.loads(resources.files(__package__).joinpath("extraction-record.schema.json").read_text())


@lru_cache(maxsize=None)
def schema_index():
    return K.SchemaIndex(schema())


@lru_cache(maxsize=None)
def template():
    """The compact record's template, as every paper's request carries it."""
    return template_str(record_shape(schema_index(), "template"))
