"""Run the direction benchmark and report one number a test can assert on.

Three sets, and confusing two of them makes the benchmark meaningless:

  gold        the reviewer's answer for a cell -- the only thing scored against
  reference   the records the reviewer was SHOWN. Supplies identity, which term a row is a
              row of, and nothing else; its own direction values are never read
  candidate   the extraction being evaluated

Passing `reference` as `candidate` is a real measurement, not a tautology: the gold is a
third party, so what is scored is the deployed extraction's own polarity against the
reviewers. That configuration is where the 96.6% quoted elsewhere comes from. A different
run scores differently -- the set shipped here as `candidate` gets 94.5% -- which is what
makes this a benchmark rather than a smoke test.

Both sets ship, so a fresh clone gets a real number with no corpus and no credentials.

Only cells both sides signed are scored. Missing terms are reported as coverage rather than
penalised, so the headline cannot be a lie by omission.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pondie.benchmark.compare import Schema, Semantics
from pondie.benchmark.direction import load_gold, score

ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
GOLD = ROOT / "gold" / "direction"
REFERENCE = ROOT / "reference"
CANDIDATE = ROOT / "candidate"


class DirectionScore(BaseModel):
    """Polarity over cells both sides signed, with the coverage behind it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

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

    def summary(self) -> str:
        acc = f"{self.accuracy:.1%}" if self.accuracy is not None else "n/a"
        cov = f"{self.coverage:.0%}" if self.coverage is not None else "n/a"
        return (
            f"{self.papers} paper(s) · polarity {acc} on {self.scored_cells} signed "
            f"cell(s) · covering {cov} of {self.gold_cells} reviewed"
            + (f" · {len(self.skipped)} skipped" if self.skipped else "")
        )


def run(
    candidate: Path = CANDIDATE,
    reference: Path = REFERENCE,
    gold: Path = GOLD,
    semantic: bool = False,
) -> DirectionScore:
    schema, sem = Schema(), Semantics(semantic)
    papers = scored = correct = reviewed = 0
    skipped: list[str] = []
    for table in sorted(gold.glob("*.direction.json")):
        paper = table.name.split(".")[0]
        cand, ref = (
            candidate / f"{paper}.extraction.json",
            reference / f"{paper}.extraction.json",
        )
        answers = load_gold(table)
        if not (cand.is_file() and ref.is_file() and answers):
            skipped.append(paper)
            continue
        result = score(
            json.loads(ref.read_text()),
            json.loads(cand.read_text()),
            answers,
            schema,
            sem,
            paper,
        )
        tier1 = result.get("tier1") or {}
        papers += 1
        scored += int(tier1.get("n", 0))
        correct += int(tier1.get("correct", 0))
        reviewed += int(result.get("gold_signed", 0))
    return DirectionScore(
        papers=papers,
        scored_cells=scored,
        correct=correct,
        gold_cells=reviewed,
        skipped=tuple(skipped),
    )


if __name__ == "__main__":
    print(run().summary())
