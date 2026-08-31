#!/usr/bin/env python3
"""Score extraction runs against each other when there is no gold record to score against.

One verified record exists. Fifteen more papers have text, a table parse and no answer, so
the only signal available on them is whether independent runs of the pipeline say the same
thing. This module turns that into numbers, and then -- on the one paper where both are
available -- asks the question those numbers are worth nothing without: **does agreement
track correctness?**

Three quantities, and the distinction between the first two is the whole design:

    self        pairwise agreement between replicates of ONE configuration.
                Reproducibility. A configuration can be perfectly self-consistent and
                perfectly wrong.
    consensus   agreement between one configuration and the facts a majority of ALL runs
                assert. Centrality. High means "says what everyone says".
    gold        agreement with the verified record, where one exists. Correctness.

A fact is a tuple, not an object, so no entity matching is needed and nothing is scored
through a bipartite assignment that could itself be wrong. Four families, in the order
they matter for this record: the signed cells, the analyses, the entities, the field
values. `compare_extractions.py` remains the instrument for gold scoring -- it reads the
schema and matches entities properly. This is the cheaper instrument that works without an
answer key, and the two are reported side by side rather than one standing in for the other.

    python compare_agreement.py --configs 'pre_*' --papers xevP8UDRAVh9 --replicates 3
"""

from __future__ import annotations

import argparse
from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

import fnmatch
import json
import math
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "benchmarks" / "gold"
RUNS = ROOT / "data" / "sweep"

from . import compare_extractions as ce  # noqa: E402

#: The families a fact can belong to, most consequential first. `cell` carries the sign,
#: which is the fact the record exists to hold; `field` is the long tail of descriptive
#: slots. Reported separately and never averaged into one number, because a pipeline that
#: fills two hundred fields and inverts one sign is not 99.5% right.
FAMILIES = ("cell", "analysis", "entity", "field")

#: Fields that name an entity, in the order tried. The name is the key a fact is filed
#: under, so two records must derive it the same way or nothing lines up.
NAME_FIELDS = ("name", "label", "term", "definition", "description")


#: `>` and `<` are the only punctuation in this corpus that carries meaning. The scorer's
#: normalisation strips punctuation, which turns "GM heroin > GM placebo" and
#: "GM heroin < GM placebo" into one string -- collapsing gold's two VBM analyses into one
#: fact and understating both the gold count and every run's recall against it.
_OPERATORS = ((">", " gt "), ("<", " lt "), ("≥", " ge "), ("≤", " le "))


def normalise(value: Any) -> str:
    text = str(value)
    for symbol, word in _OPERATORS:
        text = text.replace(symbol, word)
    return ce.normalize(text)


def entity_key(entity: ce.Entity) -> str:
    """(type, name) as one string, the identity a fact is filed under.

    Local ids are not comparable across records -- one run's `dev_verio` is another's
    `device_magnetom` -- so the name is the only thing two runs can agree on without a
    matcher. It is the weak point of this instrument and the reason `compare_extractions`
    is still what scores against gold: a run that renamed an entity correctly is counted
    here as having invented one.
    """

    for candidate in NAME_FIELDS:
        field = entity.fields.get(candidate)
        if field and field.status == "extracted" and field.value:
            return f"{entity.etype}:{normalise(field.value)[:60]}"
    return f"{entity.etype}:{normalise(entity.local_id)}"


def analysis_key(analysis: dict) -> str:
    """Name plus the tables it reports, because the name alone is not unique.

    A paper reporting one contrast in an ROI table and again whole-brain has two analyses
    called "Positive correlation", and gold for the verified paper has exactly that: four
    analyses under two names. Keyed on the name alone they collapse to two facts, and both
    the consensus and the gold count silently halve. The table local_ids are safe to key
    on where entity ids are not -- they come from the pubget manifest positionally, so
    every run of every configuration numbers them the same way.
    """

    label = normalise((analysis.get("name") or {}).get("value") or "")[:40]
    tables = analysis.get("tables")
    tables = tables.get("value") if isinstance(tables, dict) else tables
    if isinstance(tables, str):
        tables = [tables]
    named = ",".join(sorted(str(t) for t in (tables or []))) or "no-table"
    return f"{label}@{named}"


def cell_facts(document: dict) -> set[str]:
    """One fact per signed cell: which analysis, which term, which level, which way.

    The term and level are read through the model, not off the cell, because a cell holds
    a `local_id` and two records do not share one. An unresolvable reference is kept as
    the raw id with a marker, so it can never agree with anything -- which is the correct
    verdict for a cell pointing at a term the record does not declare.
    """

    terms: dict[str, str] = {}
    for model in document.get("model_estimations") or []:
        for term in model.get("terms") or []:
            name = term.get("name") or {}
            terms[str(term.get("local_id"))] = normalise(name.get("value") or "")[:40]

    facts = set()
    for analysis in document.get("analyses") or []:
        label = analysis_key(analysis)
        effect = analysis.get("effect") or {}
        for cell in effect.get("cells") or []:
            term = cell.get("term")
            term = term.get("local_id") if isinstance(term, dict) else term
            named = terms.get(str(term), f"?{term}")
            level = normalise((cell.get("level") or {}).get("value")
                              if isinstance(cell.get("level"), dict) else cell.get("level") or "")
            facts.add(f"{label}|{named}|{level}|{ce._cell_direction(cell)}")
    return facts


def record_facts(document: dict, schema: ce.Schema) -> dict[str, set[str]]:
    """The four fact families for one record."""

    flat = ce.flatten(document, schema, "run")
    entities = {key: set() for key in FAMILIES}
    entities["cell"] = cell_facts(document)
    # local_id -> the table-qualified key, so a field fact on an analysis is filed under
    # the same identity the analysis fact is.
    analysis_keys = {str(a.get("local_id")): analysis_key(a)
                     for a in document.get("analyses") or []}
    for entity in flat.entities.values():
        key = analysis_keys.get(entity.local_id) or entity_key(entity)
        if entity.etype == "Analysis":
            entities["analysis"].add(key)
        elif entity.etype != "Study":
            entities["entity"].add(key)
        for path, field in entity.fields.items():
            # `not_reported` is a claim too -- "the paper does not say" -- and two runs
            # that both decline to fill a field agree about it. Recording only extracted
            # values would let a run raise its agreement by emitting less.
            value = normalise(field.value) if field.status == "extracted" else "#absent"
            entities["field"].add(f"{key}|{path}|{value[:60]}")
    return entities


def f1(left: set[str], right: set[str]) -> float:
    """Symmetric F1 over two fact sets. Undefined for two empty sets, which is `nan`."""

    if not left and not right:
        return float("nan")
    shared = len(left & right)
    return 2 * shared / (len(left) + len(right)) if (left or right) else float("nan")


def mean(values: Iterable[float]) -> float:
    usable = [v for v in values if isinstance(v, float) and not math.isnan(v)]
    return sum(usable) / len(usable) if usable else float("nan")


def consensus(facts: Sequence[dict[str, set[str]]], threshold: float = 0.5
              ) -> dict[str, set[str]]:
    """The facts more than `threshold` of the runs assert.

    A majority vote over runs, which is the only answer key available on a paper with no
    gold. It is not ground truth and this module's headline result is about exactly how
    far it is from being one.
    """

    out = {}
    for family in FAMILIES:
        counts = Counter(fact for run in facts for fact in run[family])
        needed = threshold * len(facts)
        out[family] = {fact for fact, count in counts.items() if count > needed}
    return out


def load(config: str, paper: str, replicates: int) -> list[tuple[str, dict]]:
    """(run label, document) for each replicate of one configuration that produced one."""

    found = []
    for replicate in range(replicates):
        directory = RUNS / (config if replicate == 0 else f"{config}#{replicate}")
        path = directory / "records" / f"{paper}.extraction.json"
        if path.is_file():
            found.append((f"{config}#{replicate}", json.loads(path.read_text(encoding="utf-8"))))
    return found


def score_paper(paper: str, configs: Sequence[str], replicates: int, schema: ce.Schema
                ) -> dict[str, Any]:
    facts: dict[str, list[dict[str, set[str]]]] = {}
    for config in configs:
        runs = load(config, paper, replicates)
        if runs:
            facts[config] = [record_facts(document, schema) for _, document in runs]
    if not facts:
        return {}

    every = [run for runs in facts.values() for run in runs]
    agreed = consensus(every)

    gold_path = GOLD / f"{paper}.extraction.json"
    gold = (record_facts(json.loads(gold_path.read_text(encoding="utf-8")), schema)
            if gold_path.is_file() else None)

    rows = {}
    for config, runs in facts.items():
        others = [run for name, group in facts.items() if name != config for run in group]
        row: dict[str, Any] = {"runs": len(runs)}
        for family in FAMILIES:
            row[f"self_{family}"] = mean(
                f1(a[family], b[family]) for a, b in combinations(runs, 2))
            row[f"consensus_{family}"] = mean(f1(run[family], agreed[family]) for run in runs)
            row[f"cross_{family}"] = mean(
                f1(run[family], other[family]) for run in runs for other in others)
            if gold is not None:
                row[f"gold_{family}"] = mean(f1(run[family], gold[family]) for run in runs)
        rows[config] = row

    out: dict[str, Any] = {"paper": paper, "configs": rows, "runs": len(every),
                           "consensus_size": {f: len(agreed[f]) for f in FAMILIES}}
    if gold is not None:
        # The result this module exists for. Consensus precision is the fraction of what
        # the runs agree on that is actually right; recall is the fraction of the truth
        # they agree on. A high precision means agreement is worth acting on; a low recall
        # means agreement cannot find the facts every run misses, however many runs there
        # are, because they all miss them the same way.
        out["consensus_vs_gold"] = {
            family: {
                "precision": (len(agreed[family] & gold[family]) / len(agreed[family])
                              if agreed[family] else float("nan")),
                "recall": (len(agreed[family] & gold[family]) / len(gold[family])
                           if gold[family] else float("nan")),
                "consensus": len(agreed[family]), "gold": len(gold[family]),
            } for family in FAMILIES}
        out["gold_facts"] = {family: len(gold[family]) for family in FAMILIES}
    return out


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Rank correlation, which is what a question about ordering wants.

    Pearson on eleven configurations would be dominated by whichever arm happened to fall
    over, and the question here is whether *ranking* by agreement ranks by correctness.
    """

    pairs = [(a, b) for a, b in zip(left, right)
             if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return float("nan")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            shared = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                out[order[position]] = shared
            index = stop + 1
        return out

    a_ranks, b_ranks = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs)
    mean_a, mean_b = sum(a_ranks) / n, sum(b_ranks) / n
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a_ranks, b_ranks))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a_ranks)
                            * sum((y - mean_b) ** 2 for y in b_ranks))
    return numerator / denominator if denominator else float("nan")


def render(results: Sequence[dict[str, Any]], configs: Sequence[str]) -> str:
    lines: list[str] = []
    add = lines.append

    for result in results:
        if not result:
            continue
        add(f"\n=== {result['paper']}  ({result['runs']} runs)"
            + ("  [gold available]" if "gold_facts" in result else "  [no gold]"))
        gold = "gold_facts" in result
        for family in FAMILIES:
            counts = result["consensus_size"][family]
            add(f"\n  -- {family} facts (consensus holds {counts}"
                + (f", gold holds {result['gold_facts'][family]}" if gold else "") + ")")
            add("  " + "config".ljust(24) + f"{'self':>9}{'consensus':>11}{'cross':>9}"
                + (f"{'GOLD':>9}" if gold else ""))
            for config in configs:
                row = result["configs"].get(config)
                if not row:
                    continue
                line = ("  " + config.ljust(24)
                        + ce.pct(row[f"self_{family}"]).rjust(9)
                        + ce.pct(row[f"consensus_{family}"]).rjust(11)
                        + ce.pct(row[f"cross_{family}"]).rjust(9))
                if gold:
                    line += ce.pct(row[f"gold_{family}"]).rjust(9)
                add(line)
        if gold:
            add("\n  -- is the consensus right? (majority-vote facts vs the gold record)")
            add("  " + "family".ljust(24) + f"{'precision':>11}{'recall':>9}"
                f"{'consensus n':>13}{'gold n':>8}")
            for family in FAMILIES:
                verdict = result["consensus_vs_gold"][family]
                add("  " + family.ljust(24) + ce.pct(verdict["precision"]).rjust(11)
                    + ce.pct(verdict["recall"]).rjust(9)
                    + f"{verdict['consensus']:>13}{verdict['gold']:>8}")

            add("\n  -- does agreement predict correctness? "
                "(Spearman over configurations, n="
                f"{len(result['configs'])})")
            add("  " + "family".ljust(24) + f"{'self vs gold':>14}{'consensus vs gold':>20}")
            for family in FAMILIES:
                rows = [result["configs"][c] for c in configs if c in result["configs"]]
                gold_values = [r[f"gold_{family}"] for r in rows]
                add("  " + family.ljust(24)
                    + f"{spearman([r[f'self_{family}'] for r in rows], gold_values):>+14.2f}"
                    + f"{spearman([r[f'consensus_{family}'] for r in rows], gold_values):>+20.2f}")
    return "\n".join(lines)


def known_configs() -> list[str]:
    """Every configuration with a run directory, `#n` replicate suffixes collapsed."""

    return sorted({re.sub(r"#\d+$", "", path.name) for path in RUNS.iterdir()
                   if path.is_dir() and (path / "records").is_dir()})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=["pre_*"],
                        help="configuration names or fnmatch patterns")
    parser.add_argument("--papers", nargs="+", required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    available = known_configs()
    configs = [name for name in available
               if any(fnmatch.fnmatch(name, pattern) for pattern in args.configs)]
    if not configs:
        print(f"no run directories match {args.configs}; have: {', '.join(available)}",
              file=sys.stderr)
        return 1

    schema = ce.Schema()
    results = [score_paper(paper, configs, args.replicates, schema) for paper in args.papers]
    print(render(results, configs))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
