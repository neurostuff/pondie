"""Score `derive_direction` against the reviewed direction gold.

Coverage and accuracy are reported separately and neither alone is the answer. A rule
that fires on 5% of cells at 100% is not worth wiring in; one that fires on 60% at 80%
is worse than the model it would replace, which scores 96.6%.
"""
import json, sys, re
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes
from derive_direction import direction_of, polarity

def unwrap(n): return n.get("value") if isinstance(n, dict) and "value" in n else n

stats = Counter(); wrong = []
for file in sorted((ROOT / "benchmarks/gold/direction").glob("*.direction.json")):
    gold = json.loads(file.read_text(encoding="utf-8"))
    paper = gold["paper_id"]
    record_file = ROOT / f"data/records/{paper}.extraction.json"
    if not record_file.is_file():
        continue
    record = json.loads(record_file.read_text(encoding="utf-8"))
    names = {}
    for analysis in record.get("analyses", []):
        local_id = unwrap(analysis.get("local_id"))
        names[local_id] = " . ".join(filter(None, [
            str(unwrap(analysis.get("name")) or ""),
            str(unwrap(analysis.get("definition")) or "")]))
    for cell in gold["cells"]:
        truth = cell.get("direction")
        if not truth or cell.get("tier") == "silent":
            continue
        stats["cells"] += 1
        contrast = names.get(cell["analysis"], "")
        guess = direction_of(cell.get("level") or "", contrast)
        if guess is None:
            stats["no answer"] += 1
            continue
        stats["answered"] += 1
        if guess == truth:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1
            if len(wrong) < 8:
                wrong.append((paper, cell.get("level"), truth, guess, contrast[:90]))

n = stats["cells"]; answered = stats["answered"]
print(f"{n} reviewed cells\n")
print(f"  fires on          {answered:4d}  ({answered*100/max(n,1):.0f}% coverage)")
print(f"  correct           {stats['correct']:4d}  "
      f"({stats['correct']*100/max(answered,1):.0f}% of those it answers)")
print(f"  wrong             {stats['wrong']:4d}")
print(f"  declines          {stats['no answer']:4d}")
print("\nwrong answers:")
for paper, level, truth, guess, contrast in wrong:
    print(f"  {paper} level={level!r} gold={truth} got={guess}\n      {contrast}")
