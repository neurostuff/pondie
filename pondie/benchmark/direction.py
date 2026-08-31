#!/usr/bin/env python3
"""Score an extraction's contrast polarity against the reviewer direction table.

One question: for a term both sides agree is in the contrast, did the extractor put it on
the right side? Missing terms and mislabelled other terms are deliberately not penalised --
they are reported as coverage so the headline cannot be a lie by omission, and nothing more.

    python score_direction.py data/records/84rGLhCbUJTh.extraction.json
    python score_direction.py 'runs/*/*.extraction.json' --json out.json

Why these tiers and not others: study_schema/contrast-direction-rubric.md.

The gold is a reviewer artefact produced elsewhere; the benchmark that reads it lives here, so
a change to extraction is scored by the repository that made it rather than by whichever
checkout happens to hold the tables.

Term same-ness is established by compare_extractions' entity map -- optimal bipartite
assignment over attributes and both reference directions, iterated to a fixed point -- and
not by a second, weaker string comparison invented here.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .compare import (
    CELL_THRESHOLD,
    Aligner,
    Schema,
    Semantics,
    _all_text,
    _cell_direction,
    _cell_term,
    bootstrap,
    cohen_kappa,
    flatten,
    inline_similarity,
    match,
)

ROOT = Path(__file__).resolve().parent
SIGNED = {"positive", "negative"}
#: Everything the vocabulary offers that is not a side. Gold in this set with a signed
#: candidate is Tier 3 invention; gold signed with candidate here is Tier 2 loss.
UNSIGNED = {"absent", "held", "undirected", "not_reported"}


def _level_of(cell: Mapping) -> str:
    level = cell.get("level")
    if isinstance(level, Mapping):
        return str(level.get("value") or "")
    return str(level or "")


def _same_level(a: str, b: str) -> bool:
    """Whether two spellings of a level name the same level.

    A reviewer row is labelled with the level the *ModelTerm declares*; a `Cell` carries its
    own `level`, which the schema allows to differ in wording (`Cell.label` exists for that
    case). So "schizophrenia or schizoaffective disorder" and "Patients with schizophrenia or
    schizoaffective disorder" are one level and an exact key matches neither.

    A graded similarity is the wrong repair. These vocabularies pair levels that differ by an
    affix and mean opposite things -- `men`/`women`, `synchronous`/`asynchronous`, the latter
    at 0.96 on an edit ratio -- so a threshold pairs a level with its negation and scores the
    flip as correct. Whole words only: equal, or one word set contained in the other.

    Kept in step with `tasks._same_level`, which decides the same question when the row grid
    is built. The two drifting apart would mean the scorer pairs cells the exporter did not.
    """

    def words(text: str) -> list[str]:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).split()

    x, y = words(a), words(b)
    if not x or not y:
        return not x and not y
    return x == y or set(x) <= set(y) or set(y) <= set(x)


def _pair_gold_to_cells(ref_cells: Sequence[Mapping], analysis_id: str,
                        gold: Mapping[str, dict]) -> tuple[dict[int, dict], list[dict]]:
    """Pair each reviewed cell with the reference cell it was a row for.

    Returns the pairing and the gold entries nothing could be paired to. The second half is
    not a detail: a gold cell that finds no reference cell is a reviewer asserting a
    direction for something the extraction never proposed, and dropping it unreported is
    how a coverage failure comes to look like a perfect score.
    """

    wanted = {k: v for k, v in gold.items() if v["analysis"] == analysis_id}
    by_term: dict[str, list[int]] = defaultdict(list)
    for i, cell in enumerate(ref_cells):
        term = _cell_term(cell)
        if term:
            by_term[term].append(i)

    paired: dict[int, dict] = {}
    unresolved: list[dict] = []
    for entry in wanted.values():
        candidates = [i for i in by_term.get(entry["term"], []) if i not in paired]
        if not candidates:
            unresolved.append(entry)
            continue
        target = str(entry.get("level") or "")
        if not target:
            # A slope or product column declares no level, so its cell names none and the
            # term alone identifies it.
            paired[candidates[0]] = entry
            continue
        hit = next((i for i in candidates if _same_level(target, _level_of(ref_cells[i]))), None)
        if hit is None:
            unresolved.append(entry)
        else:
            paired[hit] = entry
    return paired, unresolved


def load_gold(path: Path) -> dict[str, dict]:
    """The reviewer table, keyed exactly as build_direction_gold wrote it."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    table = {}
    for cell in doc["cells"]:
        if cell.get("disputed") or not cell.get("direction"):
            continue
        # `silent` answers are the extractor's own prediction, left untouched by a reviewer
        # who may or may not have read the row. Scoring against them measures agreement
        # with the prediction, not with a human.
        if cell.get("tier") == "silent":
            continue
        table[f"{cell['analysis']}|{cell['term']}|{cell.get('level') or ''}"] = cell
    return table


def score(reference_doc: Mapping, cand_doc: Mapping, gold: Mapping[str, dict],
          schema: Schema, sem: Semantics, label: str) -> dict[str, Any]:
    """Align candidate to the record the reviewer was shown, then read only directions.

    The reference record supplies identity -- which term a row is a row of -- and nothing
    else. Its own direction values are the extractor's guesses and are never read; the
    reviewer's answer for that cell is.
    """
    reference = flatten(reference_doc, schema, "reference")
    cand = flatten(cand_doc, schema, "candidate")
    sem.prepare(_all_text(reference) + _all_text(cand))
    aligner = Aligner(reference, cand, schema, sem)

    cand_by_id = {e.local_id: e for e in cand.by_type.get("Analysis", [])}
    coverage = Counter()
    seen_analyses: set[str] = set()
    pairs: list[tuple[str, str]] = []          # (gold, candidate) over signed-vs-signed
    by_tier: dict[str, list[float]] = defaultdict(list)
    tier2 = Counter()
    tier3 = Counter()
    detail: list[dict[str, Any]] = []
    per_paper_hits: list[float] = []

    for ref_ent in reference.by_type.get("Analysis", []):
        ref_cells = ref_ent.inline.get("effect.cells", ("Cell", []))[1]
        # Only cells a reviewer actually judged are in play; the rest of the record was
        # never shown as a question and has no gold.
        paired, unresolved = _pair_gold_to_cells(ref_cells, ref_ent.local_id, gold)
        seen_analyses.add(ref_ent.local_id)
        for entry in unresolved:
            coverage["gold_cell_not_in_reference"] += 1
            detail.append({"analysis": entry["analysis"], "term": entry["term"],
                           "level": entry.get("level"), "gold": entry["direction"],
                           "tier": entry.get("tier", "accepted"),
                           "outcome": "gold_cell_not_in_reference"})
        judged = sorted(paired.items())
        if not judged:
            continue

        cand_id = aligner.inverse.get(ref_ent.local_id)
        if cand_id is None:
            coverage["analysis_unaligned"] += len(judged)
            for _, c in judged:
                detail.append({"analysis": ref_ent.local_id, "term": _cell_term(c),
                               "level": _level_of(c), "outcome": "analysis_unaligned"})
            continue

        cand_ent = cand_by_id[cand_id]
        cand_cells = cand_ent.inline.get("effect.cells", ("Cell", []))[1]
        aligned = match(ref_cells, cand_cells,
                        lambda a, b: inline_similarity(a, b, "Cell", schema, sem),
                        CELL_THRESHOLD)
        by_ref = {i: j for i, j, _ in aligned}

        for i, entry in judged:
            ref_cell = ref_cells[i]
            g_dir = entry["direction"]
            row = {"analysis": ref_ent.local_id, "term": _cell_term(ref_cell),
                   "level": _level_of(ref_cell), "gold": g_dir,
                   "tier": entry.get("tier", "accepted")}

            j = by_ref.get(i)
            if j is None:
                coverage["cell_unaligned"] += 1
                if g_dir in SIGNED:
                    tier2["sign_missing"] += 1
                detail.append({**row, "outcome": "cell_unaligned"})
                continue

            cand_cell = cand_cells[j]
            c_term = _cell_term(cand_cell)
            grounded = bool(c_term) and aligner.map.get(c_term) == _cell_term(ref_cell)
            c_dir = _cell_direction(cand_cell)
            row["candidate"] = c_dir

            if not grounded:
                # A right sign on a term that is not this term names a different
                # comparison. Not credited, not counted against Tier 1.
                coverage["term_unaligned"] += 1
                detail.append({**row, "outcome": "term_unaligned"})
                continue

            coverage["scorable"] += 1
            if g_dir in SIGNED and c_dir in SIGNED:
                pairs.append((g_dir, c_dir))
                hit = float(g_dir == c_dir)
                per_paper_hits.append(hit)
                by_tier[row["tier"]].append(hit)
                detail.append({**row, "outcome": "correct" if hit else "sign_flip"})
            elif g_dir in SIGNED:
                tier2["sign_loss"] += 1
                detail.append({**row, "outcome": "sign_loss"})
            elif c_dir in SIGNED:
                tier3["sign_invention"] += 1
                detail.append({**row, "outcome": "sign_invention"})
            else:
                tier3["unsigned_agreement" if g_dir == c_dir
                      else "unsigned_substitution"] += 1
                detail.append({**row, "outcome": "unsigned"})

    for entry in gold.values():
        if entry["analysis"] not in seen_analyses:
            coverage["gold_analysis_not_in_reference"] += 1
            detail.append({"analysis": entry["analysis"], "term": entry["term"],
                           "level": entry.get("level"), "gold": entry["direction"],
                           "outcome": "gold_analysis_not_in_reference"})

    correct = sum(1 for g, c in pairs if g == c)
    return {
        "record": label,
        "gold_cells": len(gold),
        "gold_signed": sum(1 for c in gold.values() if c["direction"] in SIGNED),
        "coverage": dict(coverage),
        "tier1": {
            "n": len(pairs),
            "correct": correct,
            "accuracy": (correct / len(pairs)) if pairs else None,
            "sign_flip_rate": (1 - correct / len(pairs)) if pairs else None,
            "kappa": cohen_kappa(pairs) if pairs else None,
        },
        "tier2": dict(tier2),
        "tier3": dict(tier3),
        "hits": per_paper_hits,
        "by_tier": {k: v for k, v in by_tier.items()},
        "detail": detail,
    }


def render(results: Sequence[Mapping[str, Any]], verbose: bool) -> str:
    out: list[str] = []
    tot_n = sum(r["tier1"]["n"] for r in results)
    tot_ok = sum(r["tier1"]["correct"] for r in results)
    cov = Counter()
    t2, t3 = Counter(), Counter()
    for r in results:
        cov.update(r["coverage"]); t2.update(r["tier2"]); t3.update(r["tier3"])

    out.append("Tier 0 -- coverage (reported, never scored)")
    gold_total = sum(r["gold_cells"] for r in results)
    gold_signed = sum(r["gold_signed"] for r in results)
    out.append(f"  reviewer cells {gold_total} ({gold_signed} signed) across "
               f"{len(results)} record(s)")
    for k in ("scorable", "cell_unaligned", "term_unaligned", "analysis_unaligned",
              "gold_cell_not_in_reference", "gold_analysis_not_in_reference"):
        if cov.get(k):
            out.append(f"  {k:20s} {cov[k]:5d}")

    out.append("")
    out.append("Tier 1 -- polarity accuracy  [THE HEADLINE]")
    if tot_n:
        acc = tot_ok / tot_n
        lo, hi = bootstrap([h for r in results for h in r["hits"]])
        out.append(f"  n = {tot_n} signed-vs-signed cell(s)")
        out.append(f"  accuracy       {acc:.1%}  ({tot_ok}/{tot_n})   95% CI "
                   f"[{lo:.1%}, {hi:.1%}]")
        out.append(f"  sign_flip_rate {1 - acc:.1%}")
        out.append("  baselines:  coin flip 50.0%   human ceiling 95.8%")
        tiers: dict[str, list[float]] = defaultdict(list)
        for r in results:
            for k, v in r["by_tier"].items():
                tiers[k].extend(v)
        out.append("")
        out.append("  split by how the reviewer produced the gold:")
        for k, label in (("changed", "reviewer moved the radio  [prediction-independent]"),
                         ("accepted", "left, analysis ticked accept"),
                         ("unflagged", "left, analysis flagged but not for direction")):
            v = tiers.get(k) or []
            if v:
                out.append(f"    {k:9s} n={len(v):4d}  accuracy {sum(v) / len(v):6.1%}   {label}")
            else:
                out.append(f"    {k:9s} n=   0                        {label}")
    else:
        out.append("  n = 0 -- no cell was both signed in gold and signed by the candidate")
        out.append("  NO HEADLINE IS AVAILABLE. This is a coverage failure, not a score.")

    out.append("")
    out.append("Tier 2 -- polarity retention (gold signed, candidate did not sign)")
    out.append(f"  sign_loss    {t2.get('sign_loss', 0):5d}  candidate produced the cell unsigned")
    out.append(f"  sign_missing {t2.get('sign_missing', 0):5d}  candidate produced no such cell "
               f"(out of scope, not penalised)")

    out.append("")
    out.append("Tier 3 -- sign invention (excluded from the headline)")
    out.append(f"  sign_invention {t3.get('sign_invention', 0):5d}  gold unsigned, candidate signed")

    if verbose:
        out.append("")
        out.append("per record")
        for r in sorted(results, key=lambda x: x["record"]):
            t = r["tier1"]
            acc = f"{t['accuracy']:.0%}" if t["accuracy"] is not None else "  --"
            out.append(f"  {r['record']:24s} tier1 {acc} ({t['correct']}/{t['n']})"
                       f"   scorable {r['coverage'].get('scorable', 0)}")
        flips = [d for r in results for d in r["detail"] if d["outcome"] == "sign_flip"]
        if flips:
            out.append("")
            out.append(f"sign flips ({len(flips)})")
            for d in flips[:40]:
                out.append(f"  {d['analysis']}  {d['term']} : {d['level']}"
                           f"   gold {d['gold']} -> candidate {d['candidate']}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", help="candidate extraction records (globs ok)")
    ap.add_argument("--gold-dir", type=Path, default=ROOT / "benchmarks/gold/direction")
    ap.add_argument("--reference-dir", type=Path, default=ROOT / "data/records",
                    help="the records the reviewers were shown; supplies term identity only")
    ap.add_argument("--semantic", action="store_true", help="embeddings for term same-ness")
    ap.add_argument("--json", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    paths: list[Path] = []
    for spec in args.records:
        hits = [Path(p) for p in globlib.glob(spec)]
        paths.extend(hits or [Path(spec)])

    schema = Schema()
    sem = Semantics(args.semantic)
    results = []
    for path in sorted(paths):
        paper = path.name.split(".")[0]
        gold_path = args.gold_dir / f"{paper}.direction.json"
        if not gold_path.exists():
            print(f"skip {path.name}: no reviewer gold for {paper}", file=sys.stderr)
            continue
        ref_path = args.reference_dir / f"{paper}.extraction.json"
        if not ref_path.exists():
            print(f"skip {path.name}: no reference record at {ref_path}", file=sys.stderr)
            continue
        gold = load_gold(gold_path)
        if not gold:
            print(f"skip {path.name}: every reviewed cell disputed", file=sys.stderr)
            continue
        results.append(score(
            json.loads(ref_path.read_text(encoding="utf-8")),
            json.loads(path.read_text(encoding="utf-8")),
            gold, schema, sem, path.name.split(".")[0]))

    if not results:
        print("nothing scored", file=sys.stderr)
        return 1
    print(render(results, args.verbose))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
