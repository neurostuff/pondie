"""Score an extraction, and say what it got right field by field.

One call, one result:

    from pondie.benchmark import run
    print(run().report())

Two golds, because they answer different questions and neither substitutes for the other:

  gold records      `benchmarks/gold/*.extraction.json` -- a whole record, hand-built. Gives
                    per-field precision, recall and F1, and per-entity-type object scores.
  direction tables  `benchmarks/gold/direction/*.direction.json` -- a reviewer's answer for
                    one cell. Gives polarity accuracy on the terms that carry weight.

The direction half needs a third set. `reference` is the record the reviewer was **shown**:
it supplies identity -- which term a row is a row of -- and nothing else, and its own
direction values are never read. Passing `reference` as `candidate` is a real measurement
rather than a tautology, because the gold is a third party; that configuration is where the
96.6% quoted elsewhere comes from, and the set shipped as `candidate` gets 94.5%.

Both sets ship, so a fresh clone gets a real number with no corpus and no credentials.

**Read the headline against the right thing.** Two reviewers scoring the same 239 cells agree
78.2% read naively; the 95.8% sometimes quoted is that figure weighed by provenance tier, and
the narrowest defensible number is 44 cells at 95.5% where both chose a sign. None of those
shares a denominator with a polarity score over this gold, so none is a ceiling for it. What
the doubly-reviewed set does show is a shape: of 52 disputed cells only 2 are `positive` vs
`negative`. Humans agree about polarity and argue about membership.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pondie import paths
from pondie.benchmark import scoring
from pondie.benchmark.scoring import Semantics, compare, load_gold, score
from pondie.schema import reader

BENCHMARKS = paths.REPO / "benchmarks"
GOLD = BENCHMARKS / "gold"
DIRECTION_GOLD = GOLD / "direction"
REFERENCE = BENCHMARKS / "reference"
CANDIDATE = BENCHMARKS / "candidate"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FieldScore(Strict):
    """How one field did, across every paper that had it.

    `precision` and `recall` are about *presence*: did the extractor fill the field the gold
    fills, and only that field. `accuracy` is about the value, over the pairs where both
    sides filled it -- so a field can have perfect presence and no accuracy at all.

    `agreed_absent` is neither. Both sides saying the paper is silent is a correct answer
    that no P/R/F1 can represent -- with tp=fp=fn=0 the formula yields 0.0, which reads as
    total failure -- so those are counted here and kept out of the scored set.
    """

    field: str
    precision: float
    recall: float
    f1: float
    accuracy: float | None
    compared: int
    both_filled: int
    missed: int
    spurious: int
    agreed_absent: int

    @property
    def scored(self) -> bool:
        """Was there anything to score? A field both sides left empty everywhere was not."""
        return bool(self.both_filled or self.missed or self.spurious)


class DirectionScore(Strict):
    """Polarity over the cells that carry weight, with the coverage behind it.

    A cell carries weight when it is on one side of the contrast or the other. A `held`
    level is held from both sides and has no sign to get right, so it is not scored -- and
    a term the extractor never mentioned is reported as coverage rather than penalised,
    which is why the headline cannot be a lie by omission.
    """

    papers: int = 0
    scored_cells: int = 0
    correct: int = 0
    gold_cells: int = 0
    skipped: tuple[str, ...] = ()

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.scored_cells if self.scored_cells else None

    @property
    def coverage(self) -> float | None:
        """Share of reviewed cells that could be scored at all."""
        return self.scored_cells / self.gold_cells if self.gold_cells else None


class Result(Strict):
    """Everything one benchmark run measured."""

    direction: DirectionScore
    fields: tuple[FieldScore, ...] = ()
    entities: tuple[FieldScore, ...] = ()
    records_scored: int = 0

    def summary(self) -> str:
        """One line, for a test to assert on and a log to carry."""
        direction = self.direction
        accuracy = f"{direction.accuracy:.1%}" if direction.accuracy is not None else "n/a"
        coverage = f"{direction.coverage:.0%}" if direction.coverage is not None else "n/a"
        line = (
            f"{direction.papers} paper(s) · polarity {accuracy} on "
            f"{direction.scored_cells} weighted cell(s) · covering {coverage} of "
            f"{direction.gold_cells} reviewed"
        )
        if direction.skipped:
            line += f" · {len(direction.skipped)} skipped"
        scored = [f for f in self.fields if f.scored]
        if scored:
            mean = sum(f.f1 for f in scored) / len(scored)
            line += f"\n{self.records_scored} record(s) · {len(scored)} field(s) · macro-F1 {mean:.1%}"
        return line

    def report(self, limit: int = 0) -> str:
        """The full table: every field, worst F1 first, so the fixable thing is at the top."""
        out = [self.summary(), ""]
        if self.entities:
            out += ["ENTITIES", _table(self.entities), ""]
        scored = sorted(
            (f for f in self.fields if f.scored), key=lambda f: (f.f1, f.field)
        )
        if scored:
            out += [f"FIELDS ({len(scored)} scored, worst first)"]
            out += [_table(scored[:limit] if limit else scored)]
        silent = [f.field for f in self.fields if not f.scored]
        if silent:
            out += ["", f"{len(silent)} field(s) both sides left empty everywhere, not scored"]
        return "\n".join(out)


def _table(rows: tuple[FieldScore, ...] | list[FieldScore]) -> str:
    head = f"  {'':44} {'P':>6} {'R':>6} {'F1':>6} {'acc':>7} {'n':>5}"
    lines = [head]
    for row in rows:
        accuracy = f"{row.accuracy:.1%}" if row.accuracy is not None else "--"
        lines.append(
            f"  {row.field:44} {row.precision:>6.1%} {row.recall:>6.1%} "
            f"{row.f1:>6.1%} {accuracy:>7} {row.compared:>5}"
        )
    return "\n".join(lines)


def _accumulate(into: dict[str, dict], summaries: dict[str, dict]) -> None:
    for name, metrics in summaries.items():
        presence = metrics["presence"]
        bucket = into.setdefault(
            name,
            {"tp": 0, "fp": 0, "fn": 0, "compared": 0, "correct": 0.0, "both": 0, "absent": 0},
        )
        bucket["tp"] += presence["tp"]
        bucket["fp"] += presence["fp"]
        bucket["fn"] += presence["fn"]
        bucket["compared"] += metrics["fields_compared"]
        bucket["both"] += metrics["both_extracted"]
        bucket["absent"] += metrics["agree_not_reported"]
        accuracy = metrics["value_accuracy"]
        if accuracy == accuracy:  # not NaN
            bucket["correct"] += accuracy * metrics["both_extracted"]


def _scores(buckets: dict[str, dict]) -> tuple[FieldScore, ...]:
    from pondie.benchmark.scoring import prf

    out = []
    for name, b in sorted(buckets.items()):
        metrics = prf(b["tp"], b["fp"], b["fn"])
        out.append(
            FieldScore(
                field=name,
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                accuracy=(b["correct"] / b["both"]) if b["both"] else None,
                compared=b["compared"],
                both_filled=b["both"],
                missed=b["fn"],
                spurious=b["fp"],
                agreed_absent=b["absent"],
            )
        )
    return tuple(out)


def run(
    candidate: Path = CANDIDATE,
    reference: Path = REFERENCE,
    gold: Path = GOLD,
    semantic: bool = False,
) -> Result:
    """Score every paper the gold covers, and return the numbers.

    Returns a `Result` rather than printing one: a caller that wants text calls `.report()`,
    a test asserts on `.direction.accuracy` or on one `FieldScore`.
    """
    schema, semantics = reader.load(scoring.SCHEMA), Semantics(semantic)

    papers = scored = correct = reviewed = 0
    skipped: list[str] = []
    for table in sorted((gold / "direction").glob("*.direction.json")):
        paper = table.name.split(".")[0]
        cand = candidate / f"{paper}.extraction.json"
        ref = reference / f"{paper}.extraction.json"
        answers = load_gold(table)
        if not (cand.is_file() and ref.is_file() and answers):
            skipped.append(paper)
            continue
        result = score(
            json.loads(ref.read_text()),
            json.loads(cand.read_text()),
            answers,
            schema,
            semantics,
            paper,
        )
        tier1 = result.get("tier1") or {}
        papers += 1
        scored += int(tier1.get("n", 0))
        correct += int(tier1.get("correct", 0))
        reviewed += int(result.get("gold_signed", 0))

    field_buckets: dict[str, dict] = {}
    entity_buckets: dict[str, dict] = {}
    records = 0
    for gold_record in sorted(gold.glob("*.extraction.json")):
        paper = gold_record.name.split(".")[0]
        cand = candidate / f"{paper}.extraction.json"
        if not cand.is_file():
            continue
        records += 1
        measured = compare(
            json.loads(gold_record.read_text()),
            json.loads(cand.read_text()),
            schema,
            semantics,
            paper,
        )
        _accumulate(field_buckets, measured["fields"]["per_field"])
        _accumulate(entity_buckets, measured["fields"]["per_type"])

    return Result(
        direction=DirectionScore(
            papers=papers,
            scored_cells=scored,
            correct=correct,
            gold_cells=reviewed,
            skipped=tuple(skipped),
        ),
        fields=_scores(field_buckets),
        entities=_scores(entity_buckets),
        records_scored=records,
    )


__all__ = [
    "run",
    "Result",
    "FieldScore",
    "DirectionScore",
    "CANDIDATE",
    "REFERENCE",
    "GOLD",
    "DIRECTION_GOLD",
]
