#!/usr/bin/env python3
"""Fill extraction fields from code where code can be trusted, and abstain otherwise.

Every deriver here answers one field and is allowed to return `None`. Abstention is the
point: a deriver that guesses is worse than no deriver, because a model value at least
carries evidence and a reviewer can see it was read off the paper.

    python derive_fields.py --audit 'data/runs/<run>/records/*.extraction.json'
    python derive_fields.py --fill  'data/runs/<run>/records/*.extraction.json' --apply

Measured precision against the 16-paper corpus, and why each is in or out:
docs/deterministic-fields.md.

Two modes, and the distinction matters. `--audit` compares a derivation against what the
model wrote and reports agreement, conflict and abstention; it changes nothing. `--fill`
writes a derived value only where the field is **empty**. Neither overwrites a model value:
a conflict is a question for a reviewer, not something a regex settles.
"""

from __future__ import annotations

import argparse
import collections
import glob as globlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from pondie import paths
from pondie.extraction.record.direction import same_level
from pondie.formats import values

ROOT = Path(__file__).resolve().parents[3]
TEXTS = paths.CORPUS






# -- sources ----------------------------------------------------------------


def paper_text(paper: str) -> str:
    path = paths.text(paper, paths.Flavour.local, TEXTS)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def stage1_analyses(paper: str) -> list[dict]:
    path = paths.stage1(paper, TEXTS)
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("analyses") or []


# -- derivers ---------------------------------------------------------------

#: `1.5 T`, `3T`, `3.0 Tesla`. The lookbehind keeps it off the `3` of `p < 0.003 T-value`,
#: and the modal hit wins because a Methods section names its scanner's strength repeatedly
#: while a stray match appears once.
_TESLA = re.compile(
    r"(?<![\d.])(1\.5|3\.0|3|4|4\.7|7|9\.4|11\.7)\s*-?\s*(?:T\b|Tesla\b)", re.I
)

#: Counted, not merely found. One mention of "rat" is a citation or an analogy; a rodent
#: study says it on every other line. Three is well clear of both observed populations --
#: the human papers here peak at 1 and the rodent one is in the dozens.
_ANIMAL = re.compile(
    r"\b(rats?|mice|mouse|murine|rodents?|macaques?|marmosets?|rhesus|zebrafish"
    r"|porcine|canine|felines?)\b",
    re.I,
)
_ANIMAL_FLOOR = 3

#: Stage 1's value kinds, which are what a table prints, mapped onto the statistic family.
#: `f` and `chi_square` are deliberately absent: the parser has no kind for either, so an
#: F table arrives labelled `t-statistic` or `other`. Deriving `t` from that would overwrite
#: a correct `f` with a wrong value, which is why a conflict here is reported and never
#: applied.
_KIND_TO_FAMILY = {
    "t-statistic": "t",
    "z-statistic": "z",
    "correlation": "correlation",
    "beta": "beta",
}

#: A contrast name is often a formal expression -- `FESZ>NC`, `Baseline > week 6`. 51% of
#: parsed names carry one of these operators, and the side a level sits on then gives its
#: sign outright.
_COMPARISON = re.compile(
    r"(>=|<=|>|<|\bversus\b|\bvs\.?\b|\bgreater than\b|\bless than\b)", re.I
)
_GREATER = {">", ">=", "greater than"}

#: A direction word in the name of a slope analysis -- `Left dlPFC parcel — Negative FC`.
_POSITIVE_WORD = re.compile(r"\b(positive|increases?d?|greater|higher|stronger)\b", re.I)
_NEGATIVE_WORD = re.compile(r"\b(negative|decreases?d?|reduces?d?|lower|weaker)\b", re.I)


def derive_field_strength(paper: str, **_: Any) -> float | None:
    hits = collections.Counter(m.group(1) for m in _TESLA.finditer(paper_text(paper)))
    return float(hits.most_common(1)[0][0]) if hits else None


def derive_species(paper: str, **_: Any) -> str | None:
    text = paper_text(paper)
    if not text:
        return None
    return "human" if len(_ANIMAL.findall(text)) < _ANIMAL_FLOOR else None


def derive_age_unit(paper: str, **_: Any) -> str | None:
    """Only for a human sample. A rodent age is in weeks and a human one is not in weeks,
    but "not weeks" is not a value -- an animal study is abstained on rather than guessed.
    """
    return "years" if derive_species(paper) == "human" else None


def derive_statistic_family(
    paper: str, analysis: Mapping | None = None, **_: Any
) -> str | None:
    if not analysis:
        return None
    name = values.read(analysis.get("name")) or ""
    for parse in stage1_analyses(paper):
        if not same_level(name, parse.get("name") or ""):
            continue
        kinds = collections.Counter(
            value.get("kind")
            for point in (parse.get("points") or [])
            for value in (point.get("values") or [])
            if value.get("kind") in _KIND_TO_FAMILY
        )
        if kinds:
            return _KIND_TO_FAMILY[kinds.most_common(1)[0][0]]
    return None


def derive_cell_direction(
    paper: str, analysis: Mapping | None = None, level: str | None = None, **_: Any
) -> str | None:
    """A cell's sign from the contrast's name, or from the statistic's sign for a slope.

    Recovers 52 of 101 reviewer-gold signed cells with no model judgement and one error --
    which turned out to be an inverted gold answer, so 52/52 against corrected gold. It
    abstains on the other 49: the level matches neither side of the name (24), the name
    carries no operator (20), or it is a slope whose statistics are unsigned (15).
    """
    if not analysis:
        return None
    name = values.read(analysis.get("name")) or ""

    if level:
        parts = _COMPARISON.split(name)
        if len(parts) < 3:
            return None
        left, operator, right = parts[0], parts[1].strip().lower(), parts[2]
        greater = operator in _GREATER
        if same_level(level, left):
            return "positive" if greater else "negative"
        if same_level(level, right):
            return "negative" if greater else "positive"
        return None

    # No level: a slope or a product column, whose cell carries its direction alone.
    positive, negative = _POSITIVE_WORD.search(name), _NEGATIVE_WORD.search(name)
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    for parse in stage1_analyses(paper):
        if not same_level(name, parse.get("name") or ""):
            continue
        statistics = [
            v["value"]
            for p in (parse.get("points") or [])
            for v in (p.get("values") or [])
            if v.get("kind") != "p-value"
            and isinstance(v.get("value"), (int, float))
            and v["value"] != 0
        ]
        if statistics and all(v > 0 for v in statistics):
            return "positive"
        if statistics and all(v < 0 for v in statistics):
            return "negative"
    return None


#: Closed-vocabulary fields whose value is invariant across instances within a paper, so a
#: paper-wide match cannot be assigned to the wrong instance. Ordered by specificity: `DSM-5`
#: is tried before `DSM-IV` because `DSM-IV` would otherwise match `DSM-IV-TR` first, and
#: `double-blind` before `single-blind` for the same reason.
KEYWORD_RULES: dict[str, list[tuple[str, str]]] = {
    "diagnostic_system": [
        ("DSM-5", r"\bDSM[- ]?5\b|\bDSM[- ]?V\b"),
        ("DSM-IV", r"\bDSM[- ]?IV\b|\bDSM[- ]?4\b"),
        ("ICD-11", r"\bICD[- ]?11\b"),
        ("ICD-10", r"\bICD[- ]?10\b"),
    ],
    "mr_acquisition_type": [("3D", r"\b3D\b"), ("2D", r"\b2D\b")],
    "blinding": [
        ("double_blind", r"\bdouble[- ]blind"),
        ("single_blind", r"\bsingle[- ]blind"),
        ("open_label", r"\bopen[- ]label"),
        ("none", r"\bunblinded\b"),
    ],
    "assignment_structure": [
        ("crossover", r"\bcross[- ]?over\b"),
        ("parallel", r"\bparallel[- ]group|\bparallel arms?\b"),
        ("within_subject", r"\bwithin[- ]subjects?\b"),
        ("between_subject", r"\bbetween[- ]subjects?\b"),
    ],
    "hrf_model": [
        ("canonical", r"\bcanonical (?:HRF|h[ae]modynamic)"),
        ("gamma", r"\bgamma (?:function|basis|HRF)"),
    ],
    "handedness_category": [
        ("right", r"\bright[- ]handed\b"),
        ("left", r"\bleft[- ]handed\b"),
        ("mixed", r"\bambidextrous\b"),
    ],
}


def _keyword(paper: str, rule: str) -> str | None:
    text = paper_text(paper)
    if not text:
        return None
    return next(
        (value for value, pattern in KEYWORD_RULES[rule] if re.search(pattern, text, re.I)),
        None,
    )


def derive_diagnostic_system(paper: str, **_: Any) -> str | None:
    return _keyword(paper, "diagnostic_system")


def derive_mr_acquisition_type(paper: str, **_: Any) -> str | None:
    return _keyword(paper, "mr_acquisition_type")


def derive_blinding(paper: str, **_: Any) -> str | None:
    return _keyword(paper, "blinding")


def derive_assignment_structure(paper: str, **_: Any) -> str | None:
    return _keyword(paper, "assignment_structure")


def derive_hrf_model(paper: str, **_: Any) -> str | None:
    return _keyword(paper, "hrf_model")


def derive_handedness(paper: str, **_: Any) -> str | None:
    return _keyword(paper, "handedness_category")


#: (label, where the field lives, deriver). `scope` names the traversal, not a path: a
#: field on every Group is reached differently from one on a Cell.
DERIVERS: list[tuple[str, str, str, Callable[..., Any]]] = [
    (
        "Acquisition.magnetic_field_strength_tesla",
        "acquisitions",
        "magnetic_field_strength_tesla",
        derive_field_strength,
    ),
    ("Group.species", "groups", "species", derive_species),
    ("Group.age_unit", "groups", "age_unit", derive_age_unit),
    ("Statistic.family", "analysis.statistic", "family", derive_statistic_family),
    ("Cell.direction", "analysis.cells", "direction", derive_cell_direction),
    # Group.diagnostic_system and ModelEstimation.hrf_model are deliberately absent. Both
    # are `range: string`, and the model's value carries more than a code does:
    # `DSM-IV-TR; NINCDS-ADRDA` against a derived `DSM-IV` loses an edition and a whole
    # second system. `diagnostic_system` also has a documented false positive a keyword
    # cannot avoid -- the schema warns that an edition inside an instrument's title
    # ("SCID for DSM-IV Axis II Disorders") does not establish it, and the text offers no
    # way to tell those mentions apart.
    (
        "Acquisition.mr_acquisition_type",
        "acquisitions",
        "mr_acquisition_type",
        derive_mr_acquisition_type,
    ),
    ("StudyDesign.blinding", "design", "blinding", derive_blinding),
    (
        "StudyDesign.assignment_structure",
        "design",
        "assignment_structure",
        derive_assignment_structure,
    ),
]


def _targets(record: Mapping, paper: str):
    """(label, container, key, derived, current) for every field a deriver covers."""
    for label, scope, key, deriver in DERIVERS:
        if scope == "design":
            node = record.get("design")
            if isinstance(node, Mapping):
                yield label, node, key, deriver(paper), values.read(node.get(key))
        elif scope in ("acquisitions", "groups", "model_estimations"):
            for entity in record.get(scope) or []:
                yield label, entity, key, deriver(paper), values.read(entity.get(key))
        elif scope == "analysis.statistic":
            for analysis in record.get("analyses") or []:
                node = (analysis.get("effect") or {}).get("statistic")
                if isinstance(node, Mapping):
                    yield (
                        label,
                        node,
                        key,
                        deriver(paper, analysis=analysis),
                        values.read(node.get(key)),
                    )
        elif scope == "analysis.cells":
            for analysis in record.get("analyses") or []:
                for cell in (analysis.get("effect") or {}).get("cells") or []:
                    if not isinstance(cell, Mapping):
                        continue
                    yield (
                        label,
                        cell,
                        key,
                        deriver(paper, analysis=analysis, level=values.read(cell.get("level"))),
                        values.read(cell.get(key)),
                    )


def agrees(derived: Any, current: Any) -> bool:
    """Whether a derivation and a model value say the same thing.

    Numbers are compared as numbers: `3` and `3.0` are one field strength, and a string
    comparison reported 18 conflicts on this corpus where there were none.
    """
    try:
        return abs(float(derived) - float(current)) < 1e-9
    except (TypeError, ValueError):
        return str(derived).strip().casefold() == str(current).strip().casefold()


def audit(records: list[Path]) -> dict[str, collections.Counter]:
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    conflicts: list[str] = []
    for path in records:
        paper = path.name.split(".")[0]
        record = json.loads(path.read_text(encoding="utf-8"))
        for label, _holder, _key, derived, current in _targets(record, paper):
            bucket = stats[label]
            if derived is None:
                bucket["abstain"] += 1
            elif current in (None, ""):
                bucket["fillable"] += 1
            elif agrees(derived, current):
                bucket["agree"] += 1
            else:
                bucket["conflict"] += 1
                conflicts.append(
                    f"    {paper} {label}: model {current!r} vs derived {derived!r}"
                )
    stats["_conflicts"] = conflicts  # type: ignore[assignment]
    return stats


def fill(records: list[Path], apply: bool) -> int:
    filled = 0
    for path in records:
        paper = path.name.split(".")[0]
        record = json.loads(path.read_text(encoding="utf-8"))
        touched = 0
        for label, holder, key, derived, current in _targets(record, paper):
            if derived is None or current not in (None, ""):
                continue
            node = holder.get(key)
            # `generated`, because that is what the schema's ValueSource offers for a value
            # the pipeline minted rather than read off the page -- there is no `derived`.
            # The whole wrapper is built through `values.wrap` rather than by hand, which is
            # what makes the required `evidence` block structurally impossible to omit; the
            # branch below used to write three keys and leave that one out.
            # `not_found` and not `not_applicable`: the value could have been read off the
            # page -- code just got there first -- so the schema treats the pair
            # `extracted` + `not_applicable` as an error.
            filled_field = values.wrap(derived, source="generated", evidence="not_found")
            if isinstance(node, Mapping):
                # Key by key, and evidence only if the node has none. A field can reach here
                # with an empty `value` and a real span set behind it; replacing the block
                # wholesale threw that span away.
                node["value"] = filled_field["value"]
                node["extraction_status"] = filled_field["extraction_status"]
                node["value_source"] = filled_field["value_source"]
                node.setdefault("evidence", filled_field["evidence"])
            else:
                holder[key] = filled_field
            touched += 1
        if touched:
            filled += touched
            print(f"  {paper}: filled {touched} field(s)")
            if apply:
                path.write_text(
                    json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
                )
    print(f"\n{filled} field(s) {'written' if apply else 'fillable (dry run)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("records", nargs="+")
    ap.add_argument(
        "--fill", action="store_true", help="write derived values into empty fields"
    )
    ap.add_argument("--apply", action="store_true", help="with --fill, actually write")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    records = sorted({Path(p) for spec in args.records for p in globlib.glob(spec)})
    if not records:
        print("no records matched", file=sys.stderr)
        return 1

    if args.fill:
        return fill(records, args.apply)

    stats = audit(records)
    conflicts = stats.pop("_conflicts", [])
    print(
        f"{'field':46s} {'agree':>6s} {'conflict':>9s} {'fillable':>9s} {'abstain':>8s}  precision"
    )
    for label, bucket in stats.items():
        answered = bucket["agree"] + bucket["conflict"]
        precision = f"{bucket['agree'] / answered:.1%}" if answered else "   --"
        print(
            f"{label:46s} {bucket['agree']:6d} {bucket['conflict']:9d} "
            f"{bucket['fillable']:9d} {bucket['abstain']:8d}  {precision:>9s}"
        )
    if conflicts and args.verbose:
        print("\nconflicts (never applied; a reviewer decides):")
        for line in conflicts:
            print(line)
    elif conflicts:
        print(f"\n{len(conflicts)} conflict(s); -v to list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
