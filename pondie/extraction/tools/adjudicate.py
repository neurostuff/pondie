"""Read a record the way a curator has to, and apply corrections the way a record needs.

An extraction record is unreadable as JSON: every value is an `ExtractedValue` wrapper
carrying evidence spans, so a 300-field record is 300 KB and the structure is buried in
provenance. `show` prints the assertions and hides the wrappers. `apply` takes a
corrections file, edits the record, re-resolves any quote it is given into verified
spans, and re-stamps the extraction metadata so the corrected record is honestly a
different extractor's output rather than a doctored copy of the model's.

    python adjudicate.py show  <id> [--section analyses]
    python adjudicate.py apply <id>          # reads corrections/<id>.corrections.json
    python adjudicate.py apply --all

The corrections file is a list of operations, each carrying its reason:

    [{"op": "set", "path": "groups[0].age_mean", "value": 42.4,
      "quote": "mean age 42.4 years", "why": "table 1, not the text's 42"},
     {"op": "status", "path": "design.blinding", "value": "not_reported",
      "why": "the paper never says who was blinded"},
     {"op": "delete", "path": "analyses[3]", "why": "duplicate of analyses[2]"},
     {"op": "raw", "path": "groups[0].sex_distribution", "value": [...],
      "why": "whole nested structure replaced"}]

`set` writes a value and keeps the wrapper; `status` sets extraction_status (and clears
the value for not_reported); `raw` replaces a subtree wholesale for nested records;
`delete` removes a list element or key. `quote` is optional on any of them and is
resolved against the paper text, so a correction can carry its own evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from pondie import paths
from pondie.extraction.record import builder
from pondie.extraction.record import spans as span_tools
from pondie.formats import text_index, values

TEXTS = paths.CORPUS
CORRECTIONS = paths.DATA / "corrections"

#: There is no single records directory: records belong to the run that produced them, so
#: which run to read is the caller's to say rather than this module's to assume.

#: Stamped onto every corrected record. The corrections are themselves a model's output
#: -- read by Claude against the paper, not typed by a curator -- so they are labelled as
#: such and stay reviewable rather than being promoted to ground truth.
CORRECTED_MODEL = "claude-opus-5"
CORRECTED_VERSION = "adjudicated-0.1.0"


# ------------------------------------------------------------------ reading


def plain(node: Any) -> Any:
    """The assertion a wrapper makes, with the provenance dropped."""

    if values.is_field(node):
        if node.get("extraction_status") != "extracted":
            return "<not_reported>"
        return node.get("value")
    return node


def _fmt(value: Any, width: int = 150) -> str:
    if value is None:
        return "<null>"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= width else text[: width - 1] + "…"


def _evidence_mark(node: Any) -> str:
    if not values.is_field(node):
        return ""
    status = (node.get("evidence") or {}).get("status")
    return {"present": "", "not_found": " [no-quote]", "not_applicable": ""}.get(status, "")


def render(node: Any, path: str = "", depth: int = 0, out: list | None = None) -> list[str]:
    out = [] if out is None else out
    pad = "  " * depth
    if values.is_field(node):
        out.append(f"{pad}{path}: {_fmt(plain(node))}{_evidence_mark(node)}")
        return out
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("extraction_metadata", "local_id"):
                continue
            if values.is_field(value) or not isinstance(value, (dict, list)):
                out.append(f"{pad}{key}: {_fmt(plain(value))}{_evidence_mark(value)}")
            else:
                label = key
                if isinstance(value, dict) and value.get("local_id"):
                    label = f"{key} ({value['local_id']})"
                out.append(f"{pad}{label}:")
                render(value, "", depth + 1, out)
        return out
    if isinstance(node, list):
        for index, item in enumerate(node):
            tag = f"[{index}]"
            if isinstance(item, dict) and item.get("local_id"):
                tag = f"[{index}] {item['local_id']}"
            if values.is_field(item) or not isinstance(item, (dict, list)):
                out.append(f"{pad}{tag} {_fmt(plain(item))}")
            else:
                out.append(f"{pad}{tag}")
                render(item, "", depth + 1, out)
    return out


def command_show(args: argparse.Namespace) -> int:
    record = json.loads(
        (args.records_dir / f"{args.paper}.extraction.json").read_text("utf-8")
    )
    meta = record.get("extraction_metadata", {})
    print(
        f"# {args.paper}   extractor={meta.get('extractor_model')} "
        f"{meta.get('extractor_version')}"
    )
    body = {k: v for k, v in record.items() if k not in ("extraction_metadata",)}
    if args.section:
        body = {k: v for k, v in body.items() if k in args.section}
    print("\n".join(render(body)))
    return 0


# ------------------------------------------------------------------ writing

_STEP = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")


def walk_path(root: Any, path: str, create: bool = False) -> tuple[Any, Any]:
    """Return `(container, key_or_index)` for a dotted/indexed path.

    With `create`, a missing intermediate *object* is made on the way down. Whole
    containers go missing, not just leaves -- an analysis with nothing to say about
    thresholding carries no `inference_settings` key at all -- so filling one in means
    building the object first. List indices are never created: an out-of-range index is
    a mistake in the path, not an omission in the record.
    """

    steps: list[Any] = []
    for name, index in _STEP.findall(path):
        steps.append(name if name else int(index))
    node = root
    for step in steps[:-1]:
        if create and isinstance(node, dict) and step not in node:
            node[step] = {}
        node = node[step]
    return node, steps[-1]


def attach_quote(field: dict, quote: str, normalized: str, folded: str, path: str) -> None:
    found = span_tools.resolve(normalized, quote, folded_text=folded)
    field["evidence"] = {"status": "present", "sets": [{"spans": [found.as_record()]}]}


def apply_one(
    paper: str,
    *,
    dry_run: bool,
    records: Path,
    texts: Path = TEXTS,
    model: str = CORRECTED_MODEL,
    version: str = CORRECTED_VERSION,
) -> int:
    record_path = records / f"{paper}.extraction.json"
    corrections_path = CORRECTIONS / f"{paper}.corrections.json"
    if not corrections_path.is_file():
        print(f"{paper}: no corrections file, left as the model wrote it")
        return 0

    record = json.loads(record_path.read_text("utf-8"))
    operations = json.loads(corrections_path.read_text("utf-8"))

    text_file = paths.text(paper, paths.Flavour.local, texts)
    normalized = text_index.normalize(text_file.read_text("utf-8"))
    folded = span_tools.fold(normalized) if hasattr(span_tools, "fold") else normalized

    report = builder.BuildReport()
    applied = failed = 0
    for operation in operations:
        path, kind = operation["path"], operation.get("op", "set")
        try:
            container, key = walk_path(record, path, create=(kind in ("set", "status")))
            if kind == "delete":
                del container[key]
            elif kind in ("raw", "append"):
                value = operation["value"]
                # A subtree written by hand carries quotes, not offsets. Resolving them
                # through the builder's own walker is what keeps a corrected span
                # indistinguishable from an extracted one.
                builder._walk(value, normalized, folded, path, report)
                if kind == "raw":
                    container[key] = value
                else:
                    container[key].append(value)
            else:
                # A slot the model never emitted is absent, not `not_reported`, so a
                # correction that fills one has to create the wrapper. Only for dict
                # containers: an out-of-range list index is a mistake in the path.
                if isinstance(container, dict) and key not in container:
                    container[key] = {
                        "extraction_status": "not_reported",
                        "evidence": {"status": "not_applicable"},
                    }
                field = container[key]
                # A cross-reference is a bare local_id string, not a wrapper -- there is
                # no `not_reported` form of a reference -- so `set` writes it directly.
                if not values.is_field(field):
                    if isinstance(field, (dict, list)):
                        raise TypeError(f"{path} is a nested structure; use op 'raw'")
                    container[key] = operation["value"]
                    applied += 1
                    continue
                if kind == "status":
                    field["extraction_status"] = operation["value"]
                    if operation["value"] == "not_reported":
                        field.pop("value", None)
                        field.pop("value_source", None)
                        field["evidence"] = {"status": "not_applicable"}
                else:
                    field["extraction_status"] = "extracted"
                    field["value"] = operation["value"]
                    field.setdefault("value_source", operation.get("value_source", "reported"))
                if operation.get("quote"):
                    attach_quote(field, operation["quote"], normalized, folded, path)
                elif kind != "status":
                    # A corrected value whose old quote supported the old value would be
                    # a span asserting something the field no longer says.
                    field["evidence"] = {"status": "not_found"}
            applied += 1
        except Exception as error:
            print(f"  FAILED {path}: {type(error).__name__}: {error}", file=sys.stderr)
            failed += 1

    record.setdefault("extraction_metadata", {}).update(
        {
            "extractor_model": model,
            "extractor_version": version,
        }
    )

    print(
        f"{paper}: {applied} applied, {failed} failed"
        + (f", {len(report.failures)} quote(s) unresolved" if report.failures else "")
    )
    for failure in report.failures:
        print(f"  unresolved: {failure}", file=sys.stderr)
    if failed:
        return 1
    if not dry_run:
        record_path.write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


def command_apply(args: argparse.Namespace) -> int:
    papers = (
        [p.stem.split(".")[0] for p in sorted(CORRECTIONS.glob("*.corrections.json"))]
        if args.all
        else [args.paper]
    )
    return (
        max(
            apply_one(
                p,
                dry_run=args.dry_run,
                records=args.records_dir,
                texts=args.texts_dir,
                model=args.extractor_model,
                version=args.extractor_version,
            )
            for p in papers
        )
        if papers
        else 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="print a record's assertions without the wrappers")
    show.add_argument("paper")
    show.add_argument("--records-dir", type=Path, required=True)
    show.add_argument("--section", nargs="*")
    show.set_defaults(func=command_show)

    apply_cmd = sub.add_parser("apply", help="apply corrections/<id>.corrections.json")
    apply_cmd.add_argument("paper", nargs="?")
    apply_cmd.add_argument("--all", action="store_true")
    apply_cmd.add_argument("--dry-run", action="store_true")
    # The §5 referents are a second corpus of records, in the schema submodule, and a
    # correction to one has to resolve its quotes against that copy's text.
    apply_cmd.add_argument("--records-dir", type=Path, required=True)
    apply_cmd.add_argument("--texts-dir", type=Path, default=TEXTS)
    # A record whose corrections a human reviewed field by field is a different kind of
    # artifact from one a model corrected, and the stamp is the only thing that says so.
    apply_cmd.add_argument("--extractor-model", default=CORRECTED_MODEL)
    apply_cmd.add_argument("--extractor-version", default=CORRECTED_VERSION)
    apply_cmd.set_defaults(func=command_apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
