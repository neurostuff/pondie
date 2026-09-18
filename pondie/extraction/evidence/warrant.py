"""Warranting a value: turning the quotes a pass offered into spans on the document.

The step the pipeline's third arrow crosses. Nothing between the parse and here has needed
the paper text; this resolves each quote against it, and it is why a record's spans can be
asserted to satisfy `normalized[start:end] == span.text`.

`quote` asks the model for a supporting sentence and `record.spans` locates one string. This
walks the record applying the second to the output of the first, rewrites each evidence block
in place, and counts what happened. It lived in `builder`, where it owned thirteen of
`BuildReport`'s sixteen fields while the build itself owned three.

WHAT THE COUNTS ARE FOR. `not_found` collapsed two opposite faults until `unlocated` split
them: a value nobody quoted is a recall failure in the pass that should have, and a value
whose quote no locator could place is a fidelity failure in the quote or the matcher. Over
1,817 records 32,500 fields carried the first status with no way to tell which had happened.
`case_insensitive` and `elided` exist for the same reason -- each is the measurement for the
matcher pass that produced it, and a built corpus cannot be asked what either bought, because
a resolved span keeps the document's text rather than the quote that located it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pondie.extraction.record import spans as span_tools
from pondie.formats import values

@dataclass
class Warrant:
    """What warranting did to one record: how spans were placed, and what was not."""

    fields: int = 0
    extracted: int = 0
    not_reported: int = 0

    #: Which pass placed each span. Separate counters because each is the measurement for
    #: the pass that produced it.
    exact: int = 0
    whitespace_tolerant: int = 0
    case_insensitive: int = 0
    elided: int = 0
    unresolved: list[str] = field(default_factory=list)

    present: int = 0
    not_found: int = 0
    #: Fields left `not_found` after a quote WAS offered and nothing could place it, against
    #: `not_found`, which counts that and the fields nobody quoted together.
    unlocated: int = 0
    #: Fields that kept some evidence and lost some. Invisible in every field-level count:
    #: the field reads as fully evidenced and only the quote tally shows the loss.
    partly_unlocated: int = 0

    #: Fields whose evidence claimed something the record could not support, rewritten.
    downgraded: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"fields={self.fields} "
            f"(extracted={self.extracted}, not_reported={self.not_reported})\n"
            f"evidence: present={self.present}, not_found={self.not_found}\n"
            f"spans: exact={self.exact}, whitespace-tolerant={self.whitespace_tolerant}, "
            f"case-insensitive={self.case_insensitive}, elided={self.elided}, "
            f"unresolved={len(self.unresolved)}\n"
            f"fields unevidenced despite a quote={self.unlocated}, "
            f"partly={self.partly_unlocated}\n"
            f"downgraded fields={len(self.downgraded)}"
        )


def warrant(record: dict[str, Any], normalized: str) -> Warrant:
    """Resolve every quote in the record against the document, in place."""
    report = Warrant()
    _walk_record(record, normalized, span_tools.fold(normalized), "", report)
    return report


def verify(record: dict[str, Any], normalized: str) -> None:
    """Assert every span addresses the document it claims to. Raises if one does not."""
    for evidence_set in _iter_sets(record):
        for span in evidence_set.get("spans", []):
            span_tools.verify(normalized, span)


def _resolve_field(
    node: dict[str, Any], normalized: str, folded: str, path: str, report: Warrant
) -> None:
    """Rewrite one FIELD object's evidence quotes into verified spans, in place."""

    report.fields += 1
    status = node.get("extraction_status")
    if status == "extracted":
        report.extracted += 1
    elif status == "not_reported":
        report.not_reported += 1

    evidence = node.get("evidence")
    if not isinstance(evidence, dict):
        # A field with no `evidence` object at all is the same defect as one claiming the
        # wrong status: `evidence` is REQUIRED on every ExtractedValue, so returning here
        # left the record to fail validation at the end of the run rather than be fixed and
        # counted. `repair_wrappers` produces exactly this shape.
        #
        # Branching on the status, because the two answers are different claims and an
        # earlier version of this wrote `not_found` for both. A `not_reported` field is one
        # the paper never mentioned; saying its support could not be located is the exact
        # conflation `values.py` exists to prevent, and the schema rejects the pair outright
        # ("not_reported fields must have evidence.status not_applicable"). That shape is
        # what `--no-evidence` instructs the model to emit, so every such run produced it.
        located = status == "extracted"
        node["evidence"] = evidence = {"status": "not_found" if located else "not_applicable"}
        report.downgraded.append(f"{path}: {status} with no evidence block")

    raw_sets = evidence.get("sets")
    if not isinstance(raw_sets, list):
        # An extracted field may not claim its evidence is not_applicable: the value is
        # asserted, so support for it was either found or not. The branch below enforces
        # this once a set has been tried; a field that arrived with no `sets` at all has
        # never been through it, and used to keep the contradiction all the way into the
        # record.
        if status == "extracted" and evidence.get("status") == "not_applicable":
            evidence["status"] = "not_found"
            report.downgraded.append(path)
        if evidence.get("status") == "not_found":
            report.not_found += 1
        return

    rebuilt: list[dict[str, Any]] = []
    unlocated = 0
    for index, evidence_set in enumerate(raw_sets):
        quotes = evidence_set.get("quotes") if isinstance(evidence_set, dict) else None
        if not isinstance(quotes, list):
            continue
        resolved: list[dict[str, object]] = []
        for quote in quotes:
            try:
                placed = [span_tools.resolve(normalized, quote, folded_text=folded)]
            except span_tools.SpanResolutionError as error:
                # A quote joining two non-adjacent fragments with "..." is not in the
                # document and never can be, but each fragment is, and an EvidenceSet
                # already holds several spans. Tried only after the whole quote fails.
                try:
                    placed = span_tools.resolve_elided(normalized, quote, folded_text=folded)
                    report.elided += len(placed)
                except span_tools.SpanResolutionError:
                    report.unresolved.append(f"{path} set[{index}]: {error}")
                    unlocated += 1
                    continue
            else:
                found = placed[0]
                if found.exact:
                    report.exact += 1
                elif found.cased:
                    report.case_insensitive += 1
                else:
                    report.whitespace_tolerant += 1
            resolved.extend(span.as_record() for span in placed)

        # An EvidenceSet requires at least one span (minimum_cardinality: 1), so a
        # set whose every quote failed to resolve cannot be emitted at all.
        if resolved:
            # `source` says which locator found this set. Rebuilding the set without it
            # would drop the distinction the evidence pass just recorded, leaving the two
            # locators told apart only by position again.
            rebuilt.append(
                {"spans": resolved}
                if not evidence_set.get("source")
                else {"source": evidence_set["source"], "spans": resolved}
            )

    # Written whenever a quote was dropped, not only when every one was. A field offering
    # two quotes and keeping one is `present` and fully evidenced to any reader, and the
    # dropped half is exactly the thing this slot exists to count -- recording it only on
    # total failure would measure unevidenced FIELDS while claiming to measure dropped
    # QUOTES, and the two differ by every partial loss.
    if unlocated:
        evidence["unlocated_quotes"] = unlocated

    if rebuilt:
        evidence["sets"] = rebuilt
        evidence["status"] = "present"
        report.present += 1
        if unlocated:
            report.partly_unlocated += 1
    else:
        # No usable evidence survived. The value may still be right, so keep it
        # and record that support was not located rather than deleting the field.
        evidence.pop("sets", None)
        if status == "extracted":
            evidence["status"] = "not_found"
            report.not_found += 1
            report.downgraded.append(path)
            # Which of the two faults this was. A field nobody quoted keeps the slot
            # absent, so the slot's presence is the claim that support WAS proposed and
            # nothing could place it -- a fidelity failure rather than a recall one.
            if unlocated:
                report.unlocated += 1
        else:
            evidence["status"] = "not_applicable"


def _walk_record(node: Any, normalized: str, folded: str, path: str, report: Warrant) -> None:
    if values.is_field(node):
        _resolve_field(node, normalized, folded, path, report)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _walk_record(value, normalized, folded, f"{path}.{key}" if path else str(key), report)
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            _walk_record(value, normalized, folded, f"{path}[{index}]", report)


def _iter_sets(node: Any):
    if isinstance(node, dict):
        if "spans" in node and isinstance(node["spans"], list):
            yield node
        for value in node.values():
            yield from _iter_sets(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_sets(value)

