"""Render a pondie extraction record as the text autonima screens in place of the paper.

Markdown, not the record JSON. Three reasons, and the first decides it: the record on disk
is ~500 leaves each wrapped in `{"extraction_status":…,"value_source":…,"evidence":{…}}`, and
that envelope is most of the bytes. Flattening it to prose keeps the facts and drops the
scaffolding. Second, the consumer is a screening model prompted for prose --
`screening/prompts.py` splices this into `Full Text Content: {content}` with no fence and a
prose template on either side, so a JSON object would sit adjacent to `Authors:` with nothing
marking where it ends. Third, the completeness instruction the screener is given needs an
answerable question, which the section banner provides.

Three variants, from the same record, so the arms differ only in this file's `--evidence`:

    full   values + the supporting quote for each      the proposal under test
    none   values only                                 what the quotes buy
    only   the quotes, no schema scaffolding           is it structure, or just retrieval?

Determinism is a requirement, not a nicety: autonima hashes the file content into the
fulltext and annotation cache signatures, so a renderer that emitted a timestamp or iterated
a set would silently force a full re-screen on every resume. Lists keep record order (it is
meaningful); nothing is sorted by hash; there are no timestamps.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: Stamped by the builder, not extracted, and meaningless to a screener.
SKIP_KEYS = {"local_id", "extraction_metadata"}

#: Strings a publisher leaves on a page the scraper could not get past. A record built from
#: one of these looks complete -- the model will happily extract an analysis out of an
#: abstract -- so the render has to say what it was built from rather than assert that the
#: article was whole. Asserting it cost a false include on PMID 16508348, where the source
#: was a Karger pay-per-view stub and the text arm correctly returned fulltext_incomplete.
PAYWALL_MARKERS = (
    "do not currently have access",
    "pay-per-view",
    "purchase this article",
    "get access to the full version",
    "sign in to access",
)

#: A body that reports original results has these. Their absence is the cheap structural
#: test for "this is a landing page, not a paper".
BODY_SECTIONS = ("method", "material", "result", "discussion", "conclusion")


def provenance(pmid: str, corpus: Path | None) -> tuple[list[str], bool]:
    """What the record was built from, and whether that source looks whole.

    Returns the lines to print and whether the source is suspect. Nothing here is inferred
    from the record: a record is exactly as complete as its source, and the record cannot
    know what its source was missing.
    """
    if corpus is None:
        return [], False
    study = corpus / pmid
    text_path = study / "processed" / "local" / "text.tables.txt"
    prov_path = study / "provenance.json"
    if not text_path.is_file():
        return [], False
    text = text_path.read_text(encoding="utf-8", errors="replace")
    meta = json.loads(prov_path.read_text()) if prov_path.is_file() else {}
    headings = [line.lstrip("#").strip().lower()
                for line in text.splitlines() if line.startswith("#")]
    found = [s for s in BODY_SECTIONS if any(s in h for h in headings)]
    lowered = text.lower()
    paywalled = [m for m in PAYWALL_MARKERS if m in lowered]
    suspect = bool(paywalled) or len(found) < 2

    lines = [
        f"Source: {meta.get('route', 'unknown')} render, {len(text):,} characters, "
        f"{len(headings)} headings, {meta.get('n_tables', 0)} tables."
    ]
    if paywalled:
        lines.append(
            f"WARNING — the source page carried a publisher access notice "
            f"({paywalled[0]!r}). The full article body was probably NOT available, so this "
            f"record may have been extracted from the abstract alone.")
    elif len(found) < 2:
        # A fact, not a verdict. This fires when the heading detector found no `#` lines --
        # true of 10 of 40 ACE renders, whose text is complete and whose headings simply are
        # not marked up. Stated as "may be built from a partial source", it cost three
        # papers: the screener confirmed every inclusion criterion, then abstained anyway,
        # writing "the source record includes a WARNING ... so fulltext_incomplete is set
        # true". The renderer had told it to distrust a record it had just read.
        lines.append(
            f"Section headings detected: {', '.join(found) or 'none'} — this source is not "
            f"marked up with headings, which says nothing about whether the article body "
            f"was present.")
    else:
        lines.append(f"Body sections present: {', '.join(found)}.")
    lines.append(
        "A field this record does not carry may be one the paper did not report or one the "
        "extraction missed. Either way it is not evidence against the study: judge it on "
        "what is stated here, and if what you need to decide is genuinely absent, say so "
        "rather than inferring the paper lacks it.")
    return lines, suspect

#: Top-level record keys in the order a reader wants them, which is roughly the order a
#: methods section introduces them. Anything not named here is appended in record order.
SECTION_ORDER = [
    "description", "hypothesis", "design", "groups", "tasks", "acquisitions", "devices",
    "preprocessings", "measures", "regions", "model_estimations", "inference_settings",
    "analyses", "tables", "assessments", "external_datasets",
]

TITLES = {
    "description": "Study description", "hypothesis": "Hypotheses", "design": "Design",
    "groups": "Participant groups", "tasks": "Tasks", "acquisitions": "Acquisitions",
    "devices": "Devices", "preprocessings": "Preprocessing", "measures": "Measures",
    "regions": "Regions", "model_estimations": "Model estimations",
    "inference_settings": "Inference settings", "analyses": "Analyses",
    "tables": "Tables", "assessments": "Assessments", "external_datasets": "External datasets",
}


def is_field(node) -> bool:
    return isinstance(node, dict) and "extraction_status" in node


def humanize(key: str) -> str:
    return key.replace("_", " ")


def flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float, str)):
        return re.sub(r"\s+", " ", str(value)).strip()
    if isinstance(value, list):
        return "; ".join(x for x in (flatten(v) for v in value) if x)
    if isinstance(value, dict):
        return "; ".join(f"{humanize(k)}={flatten(v)}" for k, v in value.items()
                         if flatten(v))
    return str(value)


MARK = "\x00"


class QuoteIndex:
    """Each supporting sentence written once, referenced everywhere else by a short id.

    Measured over 902 records: 101,658 span instances but only 64,692 distinct sentences,
    and in one paper a single sentence supports 34 fields. Repeating the text under every
    field it warrants spends 41% of the quote budget restating the same string.

    Ids are sequential in document order, so `[s1]` is the earliest sentence in the paper and
    the numbering is stable for a given record without needing a hash. The offset travels
    with the sentence in the appendix, which is what makes a reference checkable.
    """

    def __init__(self) -> None:
        self.by_text: dict[str, str] = {}
        self.spans: dict[str, dict] = {}
        self.used_by: dict[str, list[str]] = {}

    def add(self, span: dict, field_path: str) -> str:
        """Register a span and return the *placeholder* to write into the body.

        A placeholder rather than the id, because ids are only final after `renumber`, and a
        body line written during the walk cannot be rewritten by remapping the appendix. The
        first version did exactly that and every citation in all 903 files pointed at the
        wrong sentence -- self-consistent, and wrong.
        """
        text = re.sub(r"\s+", " ", span.get("text", "")).strip()
        key = self.by_text.get(text)
        if key is None:
            key = f"k{len(self.by_text) + 1}"
            self.by_text[text] = key
            self.spans[key] = {"text": text, "start": span.get("start_char"),
                               "sources": []}
        # A sentence both locators found is warranted twice over, which is worth saying.
        source = span.get("_source")
        if source and source not in self.spans[key]["sources"]:
            self.spans[key]["sources"].append(source)
        if field_path not in self.used_by.setdefault(key, []):
            self.used_by[key].append(field_path)
        return f"{MARK}{key}{MARK}"

    def renumber(self, text: str) -> str:
        """Number by position in the paper and resolve the placeholders in `text`."""
        order = sorted(self.spans, key=lambda k: (self.spans[k]["start"] is None,
                                                  self.spans[k]["start"] or 0))
        mapping = {old: f"s{i}" for i, old in enumerate(order, 1)}
        self.spans = {mapping[k]: v for k, v in self.spans.items()}
        self.by_text = {t: mapping[k] for t, k in self.by_text.items()}
        self.used_by = {mapping[k]: v for k, v in self.used_by.items()}
        return re.sub(f"{MARK}(k[0-9]+){MARK}", lambda m: mapping[m.group(1)], text)

    def appendix(self, show_users: bool) -> list[str]:
        if not self.spans:
            return []
        tagged = any(v.get("sources") for v in self.spans.values())
        out = ["", "### Supporting passages", "",
               "Each sentence is stated once here and referenced above by its id."]
        if tagged:
            out.append("A bracketed label names how the sentence was located: model_quote "
                       "(the extracting model cited it), retriever (a cross-encoder ranked "
                       "it), repair_pass (a later verification pass replaced a weaker "
                       "citation). An unlabelled passage comes from the original extraction.")
        for key in sorted(self.spans, key=lambda k: int(k[1:])):
            entry = self.spans[key]
            tag = (" [" + ", ".join(entry["sources"]) + "]") if entry.get("sources") else ""
            out.append(f'[{key}]{tag} "{entry["text"]}"')
            if show_users and len(self.used_by.get(key, [])) > 1:
                out.append(f"      supports: {', '.join(self.used_by[key])}")
        return out


def quotes_of(node: dict) -> list[str]:
    out, seen = [], set()
    for group in ((node.get("evidence") or {}).get("sets") or []):
        for span in group.get("spans") or []:
            text = re.sub(r"\s+", " ", span.get("text", "")).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def spans_with_source(node: dict) -> list[dict]:
    """Every span on a field, each carrying the locator that produced its set."""
    out = []
    for group in ((node.get("evidence") or {}).get("sets") or []):
        source = group.get("source")
        for span in group.get("spans") or []:
            if span.get("text"):
                out.append({**span, "_source": source})
    return out


class Renderer:
    def __init__(self, mode: str, index: "QuoteIndex | None" = None) -> None:
        self.mode = mode
        self.index = index
        self.lines: list[str] = []
        self.unreported: list[str] = []
        self.unsupported: list[str] = []
        self.all_quotes: list[str] = []
        self.seen_quotes: set[str] = set()
        self.malformed: list[str] = []

    def field(self, label: str, node: dict, indent: str, path: str) -> None:
        status = node.get("extraction_status")
        evidence = (node.get("evidence") or {}).get("status")
        if status == "not_reported":
            self.unreported.append(path)
            return
        text = flatten(node.get("value"))
        if not text:
            return

        quotes = quotes_of(node)
        for quote in quotes:
            if quote not in self.seen_quotes:
                self.seen_quotes.add(quote)
                self.all_quotes.append(quote)
        if evidence == "not_found":
            self.unsupported.append(path)

        ids = []
        if self.index is not None:
            for span in spans_with_source(node):
                # One sentence both locators returned is one citation, not two.
                mark = self.index.add(span, path)
                if mark not in ids:
                    ids.append(mark)
        if self.mode == "only":
            return

        marks = ""
        if evidence == "not_found":
            marks += " [unsupported]"
        if node.get("value_source") == "generated":
            marks += " [inferred]"
        cite = f"  [{', '.join(ids)}]" if ids else ""
        self.lines.append(f"{indent}- {label}: {text}{marks}{cite}")
        if self.mode == "full" and self.index is None:
            for quote in quotes:
                self.lines.append(f'{indent}  > "{quote}"')

    def node(self, value, indent: str, path: str) -> None:
        if is_field(value):
            self.field(humanize(path.rsplit(".", 1)[-1]), value, indent, path)
        elif isinstance(value, dict):
            for key, sub in value.items():
                if key in SKIP_KEYS:
                    continue
                if is_field(sub) or not isinstance(sub, (dict, list)):
                    if is_field(sub):
                        self.field(humanize(key), sub, indent, f"{path}.{key}")
                elif isinstance(sub, list) and not sub:
                    continue
                else:
                    before = len(self.lines)
                    self.lines.append(f"{indent}- {humanize(key)}:")
                    self.node(sub, indent + "  ", f"{path}.{key}")
                    if len(self.lines) == before + 1:
                        self.lines.pop()
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.node(item, indent, f"{path}[{index}]")

    def entity(self, item, index: int, path: str) -> None:
        """One entity in a top-level list, headed by whatever identifies it.

        A handful of records carry a bare string where an entity belongs -- a dangling
        local_id, or a fragment of the model's own JSON that the repairs did not catch. It
        is counted rather than rendered or crashed on, so the deficiency analysis can see
        how often the extractor emits one.
        """
        if not isinstance(item, dict):
            self.malformed.append(path)
            return
        local = item.get("local_id")
        name = item.get("name")
        name_text = flatten(name.get("value")) if is_field(name) else flatten(name)
        heading = " — ".join(x for x in [local, name_text] if x) or f"item {index + 1}"
        before = len(self.lines)
        self.lines.append(f"\n#### {heading}")
        self.node({k: v for k, v in item.items() if k != "name"}, "", path)
        if len(self.lines) == before + 1:
            self.lines.pop()


def render(record: dict, mode: str, pmid: str, corpus: Path | None = None,
           cite_ids: bool = True, show_users: bool = False) -> str:
    meta = record.get("extraction_metadata") or {}
    sections = [s.get("title") for s in (meta.get("paper_sections") or []) if s.get("title")]
    keys = [k for k in SECTION_ORDER if k in record]
    keys += [k for k in record if k not in keys and k not in SKIP_KEYS]

    # A3 is the arm with no evidence at all, so it gets no appendix either: leaving the index
    # on emitted the full passage list under `--evidence none` and made A3 measure 0.96x of
    # A2 instead of stripping ~40% of the document.
    index = QuoteIndex() if cite_ids and mode != "none" else None
    renderer = Renderer(mode, index)
    malformed: list[str] = []
    body: list[str] = []
    for key in keys:
        value = record[key]
        if isinstance(value, list) and not value:
            continue
        renderer.lines = []
        title = TITLES.get(key, humanize(key).capitalize())
        if isinstance(value, list) and value and any(isinstance(v, dict) for v in value):
            for pos, item in enumerate(value):
                renderer.entity(item, pos, f"{key}[{pos}]")
        else:
            renderer.node(value, "", key)
        if renderer.lines:
            body.append(f"### {title}\n" + "\n".join(renderer.lines))
        malformed.extend(renderer.malformed)
        renderer.malformed = []

    counts = {k: len(record[k]) for k in ("analyses", "groups", "tasks", "tables")
              if isinstance(record.get(k), list) and record[k]}
    prov_lines, suspect = provenance(pmid, corpus)
    header = [
        f"# Structured extraction record — PMID {pmid}",
        "",
        "This document is a machine-extracted structured record of the study, not the "
        "article prose. It states what the extraction schema captured from the source named "
        "below. A field absent from the record means either that the paper did not report "
        "it or that it was not captured — judge which from the source description here.",
    ]
    header.extend(prov_lines)
    if sections:
        header.append(f"Source article sections ({len(sections)}): " + ", ".join(sections) + ".")
    if counts:
        header.append("Record contents: "
                      + ", ".join(f"{v} {humanize(k)}" for k, v in counts.items()) + ".")

    if mode == "only" and index is not None:
        parts = list(header)          # the index appendix is the whole body
        show_users = True
    elif mode == "only":
        parts = header + ["", "## Supporting passages quoted from the article", ""]
        parts += [f'- "{q}"' for q in renderer.all_quotes]
    else:
        parts = header + [""] + body

    appendix = []
    if renderer.unreported:
        appendix.append(f"Not reported by the paper ({len(renderer.unreported)}): "
                        + "; ".join(renderer.unreported) + ".")
    if malformed:
        appendix.append(f"Malformed entries skipped ({len(malformed)}): "
                        + "; ".join(malformed) + ".")
    if renderer.unsupported and mode != "none":
        appendix.append(
            f"Extracted but with no supporting sentence located ({len(renderer.unsupported)}): "
            + "; ".join(renderer.unsupported) + ".")
    if appendix:
        parts += ["", "### Fields this record could not fill", ""] + appendix

    if index is None:
        return "\n".join(parts).rstrip() + "\n"
    body_text = index.renumber("\n".join(parts))
    return "\n".join([body_text] + index.appendix(show_users)).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--evidence", default="full", choices=["full", "none", "only"])
    ap.add_argument("--inline-quotes", action="store_true",
                    help="repeat each quote under its field instead of citing an id")
    ap.add_argument("--show-users", action="store_true",
                    help="list, per passage, the fields it supports")
    ap.add_argument("--corpus", type=Path,
                    default=Path("/data/james/pondie-vs-fulltext/corpus"),
                    help="corpus root, read for source provenance; omit to skip it")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    for path in sorted(args.records.glob("*.extraction.json")):
        pmid = path.name.split(".")[0]
        text = render(json.loads(path.read_text()), args.evidence, pmid, args.corpus,
                      cite_ids=not args.inline_quotes, show_users=args.show_users)
        (args.out / f"{pmid}.txt").write_text(text, encoding="utf-8")
        total += 1
    print(f"rendered {total} records ({args.evidence}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
