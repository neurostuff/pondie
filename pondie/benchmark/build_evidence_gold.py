"""Assemble a human evidence gold set from the review layer and the correction files.

Three signals, and they are not equally strong:

  added/adjusted   a reviewer drew or moved a highlight. Positive, and unambiguous.
  removed          a reviewer deleted a span the LLM evidence pass had produced.
                   Negative, and the only direct read anywhere on whether that pass
                   is right.
  kept             a reviewer left the LLM's span in place. Recorded but NOT scored as
                   positive: the same pre-fill problem that made the first direction
                   benchmark read 100% applies here, because a reviewer who never
                   touched the evidence layer leaves every span kept.

The set is deliberately treated as incomplete. A reviewer highlights a sentence that
supports the value, not every sentence that would; a pick that matches nothing human is
recorded `unknown` rather than wrong. Scoring anything against this must report the
unknown share, or it is reporting a denominator it made up.

Span offsets are validated against the paper text before use. Some review tasks
highlight a rendered table rather than the article, so their offsets do not index the
text at all, and those are dropped rather than silently mis-resolved.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def paper_text(paper: str) -> str | None:
    found = sorted(glob.glob(str(ROOT / f"data/texts/{paper}/processed/*/text.txt")))
    return Path(found[0]).read_text(encoding="utf-8") if found else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def resolves(text: str, span: dict) -> bool:
    """Does this span's stored text actually sit at its stored offsets?"""
    start, end, quoted = span.get("start"), span.get("end"), span.get("text") or ""
    if start is None or end is None or not quoted:
        return False
    actual = text[start:end]
    return actual == quoted or _norm(actual) == _norm(quoted)


def locate(text: str, quote: str, near: int | None = None) -> tuple[int, int] | None:
    """Where a passage sits in the paper, or None if it genuinely is not there.

    Used for two things. Correction files carry the quote a reviewer wrote and never
    offsets, so there is nothing to trust. And review-layer spans carry offsets that
    sometimes do not resolve -- the review UI stores escaped newlines, and the text a
    task was built against is not always byte-identical to the copy on disk. Dropping
    those loses real reviewer work, so the text is re-located instead.

    Exact match first, then with runs of whitespace made elastic, then a windowed fuzzy
    match. `near` biases the choice toward the original offset when a passage occurs
    more than once, which is what keeps a repeated sentence from being re-anchored to
    the wrong occurrence.
    """

    quote = quote.replace("\\n", " ").replace("\\t", " ").strip()
    if not quote or len(quote) < 8:
        return None

    def pick(spots: list[tuple[int, int]]) -> tuple[int, int] | None:
        if not spots:
            return None
        if near is None:
            return spots[0]
        return min(spots, key=lambda s: abs(s[0] - near))

    exact = [(m.start(), m.end()) for m in re.finditer(re.escape(quote), text)]
    if exact:
        return pick(exact)

    elastic = r"\s+".join(re.escape(word) for word in quote.split())
    loose = [(m.start(), m.end()) for m in re.finditer(elastic, text)]
    if loose:
        return pick(loose)

    # Last resort: anchor on the longest distinctive run of words the passage has, then
    # take a window of the original length. Below 0.75 similarity the passage is not
    # this text and must not be forced onto it.
    words = quote.split()
    for size in (8, 6, 4):
        if len(words) < size:
            continue
        for start in range(0, len(words) - size + 1):
            probe = r"\s+".join(re.escape(w) for w in words[start : start + size])
            found = [m.start() for m in re.finditer(probe, text)]
            if not found:
                continue
            at = min(found, key=lambda s: abs(s - near)) if near is not None else found[0]
            begin = max(0, at - sum(len(w) + 1 for w in words[:start]))
            window = (begin, min(len(text), begin + len(quote) + 40))
            ratio = SequenceMatcher(
                None, _norm(quote), _norm(text[window[0] : window[1]])
            ).ratio()
            if ratio >= 0.75:
                return window
    return None


def entity_target(review_key: str) -> tuple[str, str] | None:
    """(paper, layer) for a key whose spans are labelled with an entity's local_id.

    The entity, model and contrast layers put one task per class and label each
    highlight with the local_id it supports, so the span's target is in the label
    rather than the key. These carry most of the reviewer's evidence work -- and all
    but two of the deletions.
    """

    parts = review_key.split("|")
    if len(parts) < 2 or parts[1] not in ("entities", "model", "contrast"):
        return None
    return parts[0], parts[1]


def field_path(review_key: str) -> tuple[str, str] | None:
    """(paper, address) for a value-layer key, or None for a non-field task.

    Keys look like `xevP8UDRAVh9|value|Arm||arms[0].name`: class, local_id and the
    entity-relative path `record.Record` gave the field. All three are kept, because
    the path alone is not unique -- every Region has a `definition_method` -- and
    matching on it alone silently credits one entity's evidence to another.

    The `#` prefix marks this address as review-derived and entity-relative, to keep
    it apart from the absolute record paths the correction files use.
    """

    parts = review_key.split("|")
    if len(parts) < 5 or not parts[4]:
        return None
    return parts[0], f"#{parts[2]}|{parts[3]}|{parts[4]}"


def build(decoded: list[dict], corrections_dir: Path) -> dict:
    gold: dict[str, dict] = {}
    texts: dict[str, str | None] = {}
    dropped = 0
    realigned = [0]

    def slot(paper: str, path: str) -> dict:
        return gold.setdefault(
            f"{paper}|{path}",
            {
                "paper": paper,
                "path": path,
                "positive": [],
                "negative": [],
                "kept": [],
                "retracted": [],
            },
        )

    for entry in decoded:
        by_path = field_path(entry["review_key"])
        by_entity = entity_target(entry["review_key"])
        if by_path is None and by_entity is None:
            continue
        paper = (by_path or by_entity)[0]
        text = texts.setdefault(paper, paper_text(paper))
        if text is None:
            continue
        evidence = entry.get("evidence") or {}
        buckets = [
            ("positive", evidence.get("added", [])),
            ("positive", [a["to"] for a in evidence.get("adjusted", [])]),
            ("negative", evidence.get("removed", [])),
            ("kept", evidence.get("kept", [])),
        ]
        for bucket, spans in buckets:
            for span in spans:
                start, end, source = span.get("start"), span.get("end"), "review"
                if not resolves(text, span):
                    at = locate(text, span.get("text") or "", near=start)
                    if at is None:
                        dropped += 1
                        continue
                    start, end, source = at[0], at[1], "review-realigned"
                    realigned[0] += 1
                record = {
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "source": source,
                }
                if by_path is not None:
                    slot(paper, by_path[1])[bucket].append(dict(record))
                    continue
                # An entity-layer span names its target in the label. A span with no
                # label supports the task's whole class and cannot be attributed, so
                # it is dropped rather than credited to an arbitrary entity.
                labels = [lab for lab in (span.get("labels") or []) if lab]
                if not labels:
                    dropped += 1
                    continue
                # The third key part means a different thing in each layer: a class
                # for `entities` (the label is then the instance), an instance for
                # `model` and `contrast` (the label is then a row within it). Both are
                # kept so the resolver does not have to guess which it is looking at.
                parts = entry["review_key"].split("|")
                scope = parts[2] if len(parts) > 2 else ""
                for label in labels:
                    entry_slot = slot(paper, f"@{by_entity[1]}|{scope}|{label}")
                    entry_slot[bucket].append(dict(record))
                    entry_slot["layer"] = by_entity[1]
                    entry_slot["scope"] = scope
                    entry_slot["label"] = label

    quotes = 0
    for file in sorted(corrections_dir.glob("*.corrections.json")):
        paper = file.name.split(".")[0]
        text = texts.setdefault(paper, paper_text(paper))
        if text is None:
            continue
        for op in json.loads(file.read_text(encoding="utf-8")):
            path, quote = op.get("path"), op.get("quote")
            if not path or not isinstance(quote, str):
                continue
            at = locate(text, quote.strip())
            if at is None:
                dropped += 1
                continue
            slot(paper, path)["positive"].append(
                {
                    "start": at[0],
                    "end": at[1],
                    "text": text[at[0] : at[1]],
                    "source": "correction",
                }
            )
            quotes += 1

    # A deletion only counts against a locator when the reviewer put something else in
    # its place. Reading every deletion as "this sentence does not support the value"
    # was wrong: of 68 slots carrying one, 53 were bare, 7 deleted a passage the same
    # reviewer highlighted again, and hand-reading five of them found five picks that
    # were correct. Reviewers delete to re-scope a label and to trim an over-broad span
    # at least as often as to reject. See docs/evidence-unknown-judgements.md.
    retracted = 0
    for slot_data in gold.values():
        kept_negative = []
        for span in slot_data["negative"]:
            replaced = any(
                not (other["start"] < span["end"] and span["start"] < other["end"])
                for other in slot_data["positive"]
            )
            if replaced:
                kept_negative.append(span)
            else:
                slot_data["retracted"].append(span)
                retracted += 1
        slot_data["negative"] = kept_negative

    return {
        "gold": gold,
        "dropped": dropped,
        "quotes": quotes,
        "realigned": realigned[0],
        "retracted": retracted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded", required=True, type=Path)
    parser.add_argument("--corrections", type=Path, default=ROOT / "corrections")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    decoded = json.loads(args.decoded.read_text(encoding="utf-8"))
    built = build(decoded, args.corrections)
    gold = built["gold"]

    positive = sum(len(g["positive"]) for g in gold.values())
    negative = sum(len(g["negative"]) for g in gold.values())
    kept = sum(len(g["kept"]) for g in gold.values())
    scorable = sum(1 for g in gold.values() if g["positive"] or g["negative"])
    papers = len({g["paper"] for g in gold.values()})

    args.out.write_text(
        json.dumps(gold, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(gold)} field slots over {papers} papers, {scorable} scorable")
    print(
        f"  {positive} positive ({built['quotes']} from correction quotes), "
        f"{negative} negative, {kept} kept and {built['retracted']} retracted "
        f"(neither scored)"
    )
    print(
        f"  {built['realigned']} span(s) re-located by text after their offsets "
        f"failed to resolve"
    )
    print(f"  {built['dropped']} span(s) dropped: their text is not in the paper at all")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
