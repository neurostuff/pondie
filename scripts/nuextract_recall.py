"""A second opinion per entity class, aimed at recall rather than fidelity.

Pondie builds entities demand-driven: an analysis names what it needs, and `satisfy` builds
exactly that and nothing else. That is what stops it inventing a task the study never ran,
and it is also why it cannot recover an entity no analysis thought to ask for -- the missing
whole-brain contrast in PMID 19794316, or the 450 `roi` analyses whose `regions` were never
named. Nothing in a demand-driven pass is looking for what the demands left out.

So this asks a different model, one class at a time, the open question pondie never asks:
*list every entity of this class in the paper.* One call per class means the whole prompt is
that class's slots, which is the attention argument behind axis B1 in
`docs/extraction-workflow-experiments.md`; asking for the full list rather than a named set
is what makes it a recall pass. The output is not merged -- it is diffed against pondie's
record, and the diff is the finding.

Templates are projected from the LinkML schema so the two extractors are answering the same
question about the same slots, and a schema change reaches both.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXP = Path("/data/james/pondie-vs-fulltext")
MODEL = "numind/NuExtract3"

#: Storage classes worth a recall sweep, and the record key each lands under. Analyses and
#: regions first: those are the two with measured recall failures.
CLASSES = {
    "analyses": "Analysis",
    "regions": "Region",
    "groups": "Group",
    "inference_settings": "InferenceSettings",
    "tasks": "Task",
    "measures": "Measure",
    "model_estimations": "ModelEstimation",
    "acquisitions": "Acquisition",
}

#: LinkML range -> NuExtract type. Anything unmapped becomes a plain string, which is the
#: safe default: NuExtract validates its own output against the template, so a wrong type is
#: a dropped field rather than a wrong one.
RANGE_TO_TYPE = {
    "string": "string", "integer": "integer", "float": "number", "double": "number",
    "boolean": "boolean", "date": "date", "datetime": "date-time", "uriorcurie": "string",
}

#: Slots that carry cross-references or pipeline bookkeeping rather than anything a reader
#: could find in the paper. Asking for them invites invented identifiers.
SKIP_SLOTS = {"id", "mirror_of", "source_table_analysis", "inputs_from",
              "defines_regions", "tables", "model_representation_notes"}

#: `local_id` is offered on every class, not just Analysis. Without it the model could name
#: an entity but never address one, so every correction to a region, group or measure had to
#: be matched by label -- the path that minted a second copy of the CAPS the record already
#: held. It is placed first so the reply reads as an edit list.
ID_SLOT = "local_id"

#: `Analysis` does not survive a flat projection. Its slots are name/definition/scope, and
#: asked for those alone the model answered "Voxel-based morphometry" on two papers -- the
#: method, not a contrast. What makes something an analysis here is the comparison it tests,
#: and that lives in `effect.cells`, which the projection drops because it is a reference.
#: So this class gets a hand-authored template that puts the comparison in the template
#: itself, and an instruction that says what to enumerate.
ANALYSIS_TEMPLATE = {
    "analyses": [{
        # `local_id` makes the returned list an edit of the existing set rather than a
        # separate proposal: reuse an id to correct that analysis, or say NEW to add one.
        "local_id": "string",
        "name": "verbatim-string",
        "definition": "string",
        "groups_compared": ["verbatim-string"],
        "conditions_compared": ["verbatim-string"],
        "direction": ["increase", "decrease", "difference", "correlation_positive",
                      "correlation_negative", "not_stated"],
        "measure": "string",
        "spatial_scope": ["whole_brain", "roi", "searchlight"],
        "correction_scope": ["whole_brain", "roi", "not_stated"],
        "correction_method": "string",
        "cluster_threshold": "string",
        "reported_in": "verbatim-string",
    }]
}

ANALYSIS_INSTRUCTION = """\
Return the COMPLETE list of statistical ANALYSES this paper reports on brain data --
correcting the ones already extracted, and adding any that are missing.

An analysis is one tested comparison or association that produces a statistical map or a set
of regional results -- for example "PTSD < controls in grey matter volume", "drug cues >
neutral cues", or "correlation of symptom severity with hippocampal volume". It is NOT a
method or a software package: "voxel-based morphometry", "SPM8" and "FreeSurfer" are how the
analyses were run, not analyses themselves.

For each analysis, name in `regions_searched` the brain regions its search space was
restricted to, and leave that list empty when the analysis was run over the whole brain --
an empty list is a claim that inference was not restricted, so do not leave it empty for a
region-of-interest analysis. Name in `groups_used` the participant groups it compared.

Enumerate them exhaustively, including analyses reported only in tables, figures or
supplementary material. If the same comparison is reported twice under different statistical
thresholds -- once corrected within a region of interest and once corrected across the whole
brain -- list it TWICE, once per threshold, because each reports its own coordinates.

"""


def nu_type(schema, slot) -> str | list | None:
    """LinkML range -> NuExtract type. `slot` is a linkml SlotDefinition, not a dict."""
    rng = getattr(slot, "range", None)
    if rng is None:
        # `any_of: [SomeEnum, string]` is how the schema keeps a vocabulary open. Taking the
        # enum branch keeps the closed values in the template; without this the slot
        # degrades to a free string and the two extractors stop being comparable.
        for alt in (getattr(slot, "any_of", None) or []):
            alt_range = getattr(alt, "range", None)
            if alt_range in schema.enums:
                rng = alt_range
                break
    if rng in schema.enums:
        values = list((getattr(schema.enums[rng], "permissible_values", None) or {}).keys())
        return values or "string"
    if rng in schema.classes:
        return None  # a reference to another entity; the recall sweep handles it separately
    return RANGE_TO_TYPE.get(str(rng or "string").lower(), "string")


def template_for(schema, class_name: str, key: str) -> dict:
    cls = schema.classes.get(class_name)
    if cls is None:
        raise SystemExit(f"{class_name} not in schema")
    fields: dict = {ID_SLOT: "string"}
    for name, slot in (getattr(cls, "attributes", None) or {}).items():
        if name in SKIP_SLOTS:
            continue
        kind = nu_type(schema, slot)
        if kind is None:
            continue
        multi = bool(getattr(slot, "multivalued", False))
        fields[name] = [kind] if multi and isinstance(kind, str) else kind
    return {key: [fields]}


def pondie_entities(record: dict, key: str) -> list[str]:
    """The labels pondie already has for this class, for the diff."""
    out = []
    for item in (record.get(key) or []):
        if not isinstance(item, dict):
            continue
        for field in ("name", "local_id", "definition"):
            node = item.get(field)
            value = node.get("value") if isinstance(node, dict) else node
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
                break
    return out


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmids", nargs="+", required=True)
    ap.add_argument("--classes", nargs="*", default=["analyses", "regions"])
    ap.add_argument("--records", type=Path,
                    default=EXP / "pondie-data/runs/pondie-907/records")
    ap.add_argument("--corpus", type=Path, default=EXP / "corpus")
    ap.add_argument("--out", type=Path, default=EXP / "reports" / "nuextract_recall.jsonl")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--dump-templates", action="store_true")
    ap.add_argument("--load-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    args = ap.parse_args()

    from pondie import schema as pondie_schema
    from pondie.schema import reader
    sch = reader.load(pondie_schema.STORAGE)

    templates = {k: (ANALYSIS_TEMPLATE if k == "analyses" else template_for(sch, CLASSES[k], k))
                 for k in args.classes}
    instructions = {"analyses": ANALYSIS_INSTRUCTION}
    if args.dump_templates:
        for key, tpl in templates.items():
            print(f"--- {key}\n{json.dumps(tpl, indent=1)[:1400]}")
        return 0

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    # `device_map="auto"` silently placed 9.3 GB of bf16 weights on the CPU rather than
    # split them over two 8 GB cards, and a 4B model decoding a 20k-token prompt on CPU is
    # not a pipeline. 4-bit puts the whole model on one card with room for the KV cache.
    kwargs: dict = {"trust_remote_code": True}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, **kwargs).eval()
    print(f"loaded {MODEL} (4bit={args.load_4bit}): "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB on GPU")

    def run(text: str, template: dict, instruction: str = "") -> str:
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": instruction + text}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", template=json.dumps(template, indent=2),
            enable_thinking=args.thinking).to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        return processor.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0].strip()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        for pmid in args.pmids:
            text = (args.corpus / pmid / "processed/local/text.tables.txt").read_text(
                errors="replace")
            record = json.loads((args.records / f"{pmid}.extraction.json").read_text())
            for key in args.classes:
                raw = run(text, templates[key], instructions.get(key, ""))
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"_unparsed": raw[:2000]}
                found = parsed.get(key) or [] if isinstance(parsed, dict) else []
                have = pondie_entities(record, key)
                have_norm = {normalize(h) for h in have}
                labels = []
                for item in found:
                    if not isinstance(item, dict):
                        continue
                    lab = item.get("name") or item.get("definition") or ""
                    labels.append(str(lab))
                extra = [l for l in labels if normalize(l) not in have_norm and l.strip()]
                print(f"\n{pmid} [{key}]  pondie={len(have)}  nuextract={len(labels)}  "
                      f"not-in-pondie={len(extra)}")
                for h in have:
                    print(f"    pondie   : {h[:95]}")
                for l in labels:
                    mark = "NEW " if normalize(l) not in have_norm else "    "
                    print(f"    nuextract{mark}: {l[:95]}")
                fh.write(json.dumps({"pmid": pmid, "class": key, "pondie": have,
                                     "nuextract": labels, "extra": extra,
                                     "raw": found}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
