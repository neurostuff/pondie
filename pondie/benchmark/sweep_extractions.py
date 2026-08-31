#!/usr/bin/env python3
"""Run extraction configurations over the gold papers and score each one.

One row per (configuration, paper): run the pipeline, build the record, score it with
`compare_extractions.py`, and report the deltas against a named baseline configuration.

Comparisons are **within paper**. Paper difficulty spans 44% to 100% on the current gold
set, so a mean over papers is mostly a statement about which papers were in it; the delta
matrix is the finding and the mean of deltas is the summary.

    python sweep_extractions.py --list
    python sweep_extractions.py --configs baseline effort_high --jobs 3
    python sweep_extractions.py --configs baseline effort_high --report-only

Design, cost model and the caveats that bound what this can conclude:
docs/extraction-workflow-experiments.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "benchmarks" / "gold"
TEXTS = ROOT / "data" / "texts"
RUNS = ROOT / "data" / "sweep"

from pondie.benchmark import compare_extractions as ce  # noqa: E402

DEFAULT_MODEL = "@psyc-aid338-ope-333f18/gpt-5.6-luna"

#: Stage 4 adds 4-7 calls per paper and `compare_extractions.py` scores no evidence spans,
#: so it is off for every sweep run. Leaving it on would triple the cost of the sweep to
#: move no number in the report.
STAGES = ["tables", "entities", "analyses", "build"]


@dataclass
class Config:
    """One point in the workflow space.

    `flags` are arguments `run_extraction.py` already accepts; `needs` names the pipeline
    work a configuration requires that does not exist yet, so `--list` can say what is
    runnable today rather than failing halfway through a sweep.
    """

    name: str
    axes: str
    describe: str
    flags: dict[str, str] = field(default_factory=dict)
    needs: str = ""

    @property
    def runnable(self) -> bool:
        return not self.needs


#: A high-effort call spends reasoning out of the same budget as the answer, so the ceiling
#: has to rise with the effort or the pass returns an empty payload having thought itself
#: out of room. `extract_record.py` reports that case as DEGENERATE rather than hiding it.
HIGH_EFFORT_MAX_OUT = "120000"

#: The flags every preprocessing arm shares, so the only difference between two of those
#: rows is the `--preprocess` value. Anything else varying would make the comparison a
#: comparison of two things.
PREPROCESS_SUBSTRATE = {
    "--effort": "low",
    "--zero-foci-rule": "",
    "--max-attempts": "3",
    "--stages": "tables demands satisfy build",
}

CONFIGS: list[Config] = [
    Config(
        "baseline",
        "A0 B0 C0 D0",
        "entity-first, monolithic, no reconciliation, low effort",
        {"--effort": "low"},
    ),
    Config(
        "effort_medium",
        "A0 B0 C0 D-",
        "as baseline, medium reasoning effort on both passes",
        {"--effort": "medium", "--max-out": HIGH_EFFORT_MAX_OUT},
    ),
    Config(
        "effort_high",
        "A0 B0 C0 D1",
        "as baseline, high reasoning effort on both passes",
        {"--effort": "high", "--max-out": HIGH_EFFORT_MAX_OUT},
    ),
    Config(
        "effort_graded",
        "A0 B0 C0 D2",
        "low effort filling entity slots, high effort deciding analyses and cells",
        {
            "--entities-effort": "low",
            "--analyses-effort": "high",
            "--max-out": HIGH_EFFORT_MAX_OUT,
        },
    ),
    Config(
        "no_stage1",
        "A- B0 C0 D0",
        "analyses pass without the stage-1 table listing: what the anchor is worth",
        {"--effort": "low", "--no-stage1": ""},
    ),
    Config(
        "table_rows",
        "A2 B0 C0 D0",
        "stage-1 foci and statistic values in the prompt, not just a count",
        {"--effort": "low", "--table-rows": ""},
    ),
    Config(
        "recheck_cells",
        "A0 B0 C3 D0",
        "baseline, then one targeted re-ask per analysis about term and direction",
        {"--effort": "low", "--stages": "tables entities analyses recheck build"},
    ),
    Config(
        "recheck_high",
        "A0 B0 C3 D2",
        "targeted cell re-ask at high effort, everything else low",
        {
            "--entities-effort": "low",
            "--analyses-effort": "high",
            "--max-out": HIGH_EFFORT_MAX_OUT,
            "--stages": "tables entities analyses recheck build",
        },
    ),
    # The zero-foci rule and the demand-driven ordering, alone and together, so that a
    # combined win can be attributed to one of them rather than to the pair.
    Config(
        "zero_foci",
        "A0 B0 C0 D0 +Z",
        "baseline plus: a stage-1 entry with no coordinates is still an analysis",
        {"--effort": "low", "--zero-foci-rule": ""},
    ),
    Config(
        "analysis_first",
        "A1 B0 C0 D0",
        "analyses declare the entities they need; the entity pass is held to that list",
        {"--effort": "low", "--stages": "tables demands satisfy build"},
    ),
    Config(
        "analysis_first_zf",
        "A1 B0 C0 D0 +Z",
        "demand-driven ordering plus the zero-foci rule",
        {
            "--effort": "low",
            "--zero-foci-rule": "",
            "--stages": "tables demands satisfy build",
        },
    ),
    Config(
        "analysis_first_zf_retry",
        "A1 B0 C0 D0 +Z +R",
        "demand-driven ordering, zero-foci rule, and a post-condition retry per pass",
        {
            "--effort": "low",
            "--zero-foci-rule": "",
            "--max-attempts": "3",
            "--stages": "tables demands satisfy build",
        },
    ),
    Config(
        "analysis_first_zf_recheck",
        "A1 B0 C3 D0 +Z",
        "demand-driven ordering, zero-foci rule, and the targeted cell re-ask",
        {
            "--effort": "low",
            "--zero-foci-rule": "",
            "--stages": "tables demands satisfy recheck build",
        },
    ),
    # --- text preprocessing, all on the same substrate ---------------------------------
    # One arm per `review/preprocess.py` strategy, plus `pre_control` which is that
    # substrate with no preprocessing at all. The substrate is the best measured
    # configuration (demand-driven, zero-foci rule, post-condition retry) and not
    # `baseline`, so a win here is a win on top of what the pipeline already does. They
    # are listed explicitly rather than generated because a sweep argument that does not
    # appear in this file is a sweep nobody can reproduce from it.
    Config(
        "pre_control",
        "A1 +Z +R P0",
        "the preprocessing substrate with no preprocessing: the control arm",
        dict(PREPROCESS_SUBSTRATE),
    ),
    Config(
        "pre_sections",
        "A1 +Z +R P1",
        "drop the Introduction, Discussion and back matter from the prompt",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "sections"}),
    ),
    Config(
        "pre_reorder",
        "A1 +Z +R P2",
        "same content, Methods and Results first and the tables last",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "reorder"}),
    ),
    Config(
        "pre_retrieval",
        "A1 +Z +R P3",
        "BM25 sentence selection against a schema-derived query, 45% of prose kept",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "retrieval"}),
    ),
    Config(
        "pre_abbrev",
        "A1 +Z +R P4",
        "Schwartz-Hearst abbreviation glossary ahead of the paper",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "abbrev"}),
    ),
    Config(
        "pre_stats",
        "A1 +Z +R P5",
        "inventory of sentences reporting a statistic, to the analyses pass",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "stats"}),
    ),
    Config(
        "pre_contrasts",
        "A1 +Z +R P6",
        "cue-phrase sweep for tested comparisons, to the analyses pass",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "contrasts"}),
    ),
    Config(
        "pre_methods",
        "A1 +Z +R P7",
        "Methods parameters labelled by extraction field, to the entity pass",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "methods"}),
    ),
    Config(
        "pre_cohort",
        "A1 +Z +R P8",
        "sample, sex, age, arm and timepoint phrases, to the entity pass",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "cohort"}),
    ),
    Config(
        "pre_regions",
        "A1 +Z +R P9",
        "anatomy mentions split into ROI-context and result-table labels",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "regions"}),
    ),
    Config(
        "pre_combo",
        "A1 +Z +R P10",
        "section-scoped text plus the digests each pass is served by",
        dict(PREPROCESS_SUBSTRATE, **{"--preprocess": "combo"}),
    ),
    # --- everything below needs pipeline work; listed so the space is visible ----------
    Config(
        "back_pressure",
        "A0 B0 C5 D0",
        "baseline, then re-ask about entities no analysis references",
        needs="a back-pressure pass (see docs, this replaced the prune idea)",
    ),
    Config(
        "per_class",
        "A0 B1 C0 D0",
        "one call per entity class over a cached prefix",
        needs="prompt reordered for caching + a per-class driver",
    ),
    Config(
        "per_analysis",
        "A0 B3 C0 D0",
        "one call per analysis on the analysis side",
        needs="per-analysis mode in extract_record.py",
    ),
    Config(
        "self_consistency3",
        "A0 B0 C4 D0",
        "three analyses passes, majority vote per cell",
        needs="k-sampling and a cell-level vote",
    ),
]

BY_NAME = {c.name: c for c in CONFIGS}
# A duplicate name silently shadows the earlier entry, and if the survivor is an unbuilt
# stub the whole sweep refuses to start with a message about work that is already done.
assert len(BY_NAME) == len(CONFIGS), "duplicate configuration name(s): " + ", ".join(
    sorted({c.name for c in CONFIGS if sum(o.name == c.name for o in CONFIGS) > 1})
)


def gold_papers() -> list[str]:
    return sorted(p.name.split(".")[0] for p in GOLD.glob("*.extraction.json"))


def write_pmids(papers: Sequence[str], path: Path) -> None:
    """run_extraction reads `pmid<TAB>study<TAB>axis`; only the study id is used here."""

    lines = ["# generated by sweep_extractions.py"]
    for study in papers:
        identifiers = TEXTS / study / "identifiers.json"
        pmid = study
        if identifiers.is_file():
            try:
                pmid = str(
                    json.loads(identifiers.read_text(encoding="utf-8")).get("pmid", study)
                )
            except (OSError, ValueError):
                pass
        lines.append(f"{pmid}\t{study}\tsweep")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dir(config_name: str, replicate: int) -> Path:
    return RUNS / (config_name if replicate == 0 else f"{config_name}#{replicate}")


def run_one(
    config: Config, paper: str, model: str, redo: bool, replicate: int = 0
) -> dict[str, Any]:
    """Run the pipeline for one configuration on one paper, into that run's own tree."""

    out = run_dir(config.name, replicate)
    record = out / "records" / f"{paper}.extraction.json"
    if record.is_file() and not redo:
        return {
            "paper": paper,
            "config": config.name,
            "replicate": replicate,
            "status": "cached",
        }

    out.mkdir(parents=True, exist_ok=True)
    (out / "records").mkdir(exist_ok=True)
    pmids = out / f"{paper}.pmids"
    write_pmids([paper], pmids)

    command = [
        sys.executable,
        "-m",
        "pondie.extraction.passes.run_extraction",
        "--pmids",
        str(pmids),
        "--texts",
        str(TEXTS),
        # Payloads and examples are per configuration. Pointing --examples at
        # review/examples would overwrite the suite's own fixture corpus.
        "--payloads",
        str(out / "payloads"),
        "--examples",
        str(out / "records"),
        "--key-file",
        str(ROOT / ".env"),
        "--model",
        model,
    ]
    flags = dict(config.flags)
    command += ["--stages", *flags.pop("--stages", " ".join(STAGES)).split()]
    for flag, value in flags.items():
        command += [flag] if value == "" else [flag, value]
    if redo:
        command.append("--redo")

    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - started
    log = out / "logs"
    log.mkdir(exist_ok=True)
    (log / f"{paper}.log").write_text(
        completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8"
    )
    return {
        "paper": paper,
        "config": config.name,
        "replicate": replicate,
        "seconds": round(elapsed, 1),
        "status": "ok" if record.is_file() else "FAILED",
        "returncode": completed.returncode,
    }


def degenerate(config_name: str, paper: str, replicate: int) -> bool:
    """Whether any stage of this run emitted an empty payload.

    Tracked and not repaired: the sweep measures configurations as they are, and a
    configuration that silently loses a pass should show that in its numbers rather than
    have it papered over by a retry the pipeline does not actually have.
    """

    log = run_dir(config_name, replicate) / "logs" / f"{paper}.log"
    return log.is_file() and "DEGENERATE" in log.read_text(encoding="utf-8")


def score(
    config_name: str,
    papers: Sequence[str],
    schema: ce.Schema,
    replicate: int = 0,
    scope: str = "tables",
) -> dict[str, dict]:
    """Score whatever this configuration produced. A missing record is a zero, not a gap.

    Dropping a paper a configuration failed on would let a pipeline improve its mean by
    crashing on the papers it finds hard. A paper with no *gold* record is the opposite
    case and is dropped: there is nothing to score it against, and calling that zero
    would punish every configuration equally for the corpus's missing answer keys.
    `compare_agreement.py` is what those papers are measured with.
    """

    out = run_dir(config_name, replicate) / "records"
    results: dict[str, dict] = {}
    for paper in papers:
        record = out / f"{paper}.extraction.json"
        gold_path = GOLD / f"{paper}.extraction.json"
        if not gold_path.is_file():
            results[paper] = {
                "failed": False,
                "ungraded": True,
                "degenerate": False,
                "direction_cells": {"tp": 0, "fp": 0, "fn": 0},
            }
            continue
        if not record.is_file():
            results[paper] = {
                "failed": True,
                "ungraded": False,
                "composite": 0.0,
                "direction_f1": 0.0,
                "direction_cells": {"tp": 0, "fp": 0, "fn": 0},
                "degenerate": True,
            }
            continue
        result = ce.compare(
            json.loads(gold_path.read_text(encoding="utf-8")),
            json.loads(record.read_text(encoding="utf-8")),
            schema,
            ce.Semantics(False),
            f"{config_name}/{paper}",
            scope=scope,
        )
        primary = result["direction"]["primary"]
        results[paper] = {
            "failed": False,
            "ungraded": False,
            "degenerate": degenerate(config_name, paper, replicate),
            "composite": result["composite"]["score"],
            "direction_f1": primary["cell_prf"]["f1"],
            "direction_accuracy": primary["accuracy_term_grounded"],
            "direction_cells": primary["cell_prf"],
            "grounding_rate": primary["cells"]["grounding_rate"],
            "sign_flip_rate": primary["sign_flip_rate"],
            "analyses_matched": result["entities"]["per_type"]
            .get("Analysis", {})
            .get("recall"),
            "entity_f1": result["entities"]["micro"]["f1"],
            "relationship_f1": result["relationships"]["micro"]["f1"],
            "field_accuracy": result["fields"]["overall"]["value_accuracy"],
        }
    return results


def cell_pool(scores: dict[str, dict]) -> dict[str, float]:
    """Direction F1 pooled over cells rather than averaged over papers.

    Six papers is too few to resolve anything; the 65 gold cells across them are the unit
    the headline can actually be estimated at.
    """

    usable = [s for s in scores.values() if not s["failed"] and not s.get("ungraded")]
    return ce.prf(*(sum(s["direction_cells"][k] for s in usable) for k in ("tp", "fp", "fn")))


#: Direction on a matched term first, and everything else after it. That is the fact the
#: record exists to carry: a wrong sign inverts the finding, where a wrong scanner model
#: does not.
METRICS = [
    ("direction_f1", "DIRECTION F1 (term-grounded)"),
    ("direction_accuracy", "direction accuracy, matched cells only"),
    ("analyses_matched", "analysis recall"),
    ("composite", "composite"),
    ("entity_f1", "entity F1"),
    ("relationship_f1", "relationship F1"),
    ("field_accuracy", "field acc"),
]


def spread(values: Sequence[float]) -> tuple[float, float]:
    usable = [v for v in values if isinstance(v, float) and not math.isnan(v)]
    if not usable:
        return float("nan"), float("nan")
    mean = sum(usable) / len(usable)
    if len(usable) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in usable) / (len(usable) - 1)
    return mean, math.sqrt(variance)


def render_replicates(
    runs: dict[str, list[dict[str, dict]]], papers: Sequence[str], baseline: str
) -> str:
    """Config means with the run-to-run spread beside them.

    With one paper in the gold set, a difference between two configurations is only a
    finding if it is larger than the difference between two runs of the *same*
    configuration. That comparison is the whole point of this table, so the standard
    deviation is printed next to every mean rather than in a footnote.
    """

    lines: list[str] = []
    add = lines.append
    base_runs = runs.get(baseline, [])

    for key, label in METRICS:
        add(f"\n-- {label} " + "-" * (60 - len(label)))
        add(
            "  "
            + "config".ljust(20)
            + "  n     mean       sd      min      max"
            + ("        Δ vs base" if base_runs else "")
        )
        for name, replicates in runs.items():
            values = [
                r[p][key]
                for r in replicates
                for p in papers
                if not r[p].get("ungraded")
                and isinstance(r[p].get(key), float)
                and not math.isnan(r[p][key])
            ]
            mean, sd = spread(values)
            row = (
                "  "
                + name.ljust(20)
                + f" {len(values):2d}  "
                + ce.pct(mean)
                + f"  {sd * 100:6.1f}  "
                + ce.pct(min(values) if values else float("nan"))
                + "  "
                + ce.pct(max(values) if values else float("nan"))
            )
            if base_runs and name != baseline:
                base_values = [
                    r[p][key]
                    for r in base_runs
                    for p in papers
                    if not r[p].get("ungraded")
                    and isinstance(r[p].get(key), float)
                    and not math.isnan(r[p][key])
                ]
                base_mean, base_sd = spread(base_values)
                delta = mean - base_mean
                # Pooled sd of the two configurations, as the yardstick the delta has to
                # clear. Not a test -- with this many runs a test would be theatre -- just
                # the scale the difference should be read against.
                pooled = math.sqrt((sd**2 + base_sd**2) / 2)
                row += (
                    f"   {delta * 100:+6.1f}  ({delta / pooled:+.1f} sd)"
                    if pooled > 1e-9
                    else f"   {delta * 100:+6.1f}"
                )
            add(row)

    add("\n== DIRECTION, pooled over every cell of every replicate " + "=" * 6)
    add("  " + "config".ljust(20) + "     P       R      F1     tp   fp   fn   degen")
    ranked = sorted(
        runs.items(),
        key=lambda kv: -ce.prf(
            *[sum(cell_pool(r)[k] for r in kv[1]) for k in ("tp", "fp", "fn")]
        )["f1"],
    )
    for name, replicates in ranked:
        tp = fp = fn = 0
        for replicate in replicates:
            pool = cell_pool(replicate)
            tp, fp, fn = tp + pool["tp"], fp + pool["fp"], fn + pool["fn"]
        pool = ce.prf(tp, fp, fn)
        degen = sum(1 for r in replicates for p in papers if r[p].get("degenerate"))
        add(
            "  "
            + name.ljust(20)
            + f"{ce.pct(pool['precision'])} {ce.pct(pool['recall'])} {ce.pct(pool['f1'])}"
            + f"  {tp:4d} {fp:4d} {fn:4d}   {degen:3d}"
        )

    graded = [p for p in papers if (GOLD / f"{p}.extraction.json").is_file()]
    ungraded = [p for p in papers if p not in graded]
    add(
        f"\nScored against gold: {', '.join(graded) or 'none'}."
        + (
            f"  Run but not scored (no gold record): {', '.join(ungraded)};"
            " use compare_agreement.py for those."
            if ungraded
            else ""
        )
    )
    add("'degen' counts runs where a pass emitted no entities at all -- a silent failure,")
    add("left unrepaired here so it shows in the numbers. A delta smaller than the")
    add("same-config sd is not a finding, whatever its sign.")
    return "\n".join(lines)


def render(
    all_scores: dict[str, dict[str, dict]], papers: Sequence[str], baseline: str
) -> str:
    lines: list[str] = []
    add = lines.append
    metrics = [
        ("composite", "composite"),
        ("direction_f1", "direction F1"),
        ("entity_f1", "entity F1"),
        ("relationship_f1", "relationship F1"),
        ("field_accuracy", "field acc"),
    ]

    base = all_scores.get(baseline)
    for key, label in metrics:
        add(f"\n-- {label} " + "-" * (68 - len(label)))
        add(
            "  "
            + "config".ljust(20)
            + "".join(p[:10].rjust(11) for p in papers)
            + "     mean"
            + ("      Δ" if base else "")
        )
        for name, scores in all_scores.items():
            row = "  " + name.ljust(20)
            values = []
            for paper in papers:
                value = None if scores[paper].get("ungraded") else scores[paper].get(key)
                values.append(value)
                row += (ce.pct(value) if value is not None else "   n/a ").rjust(11)
            usable = [v for v in values if isinstance(v, float) and not math.isnan(v)]
            mean = sum(usable) / len(usable) if usable else float("nan")
            row += ce.pct(mean).rjust(9)
            if base and name != baseline:
                # Paired on paper: the mean of per-paper deltas, not a difference of means.
                deltas = [
                    scores[p][key] - base[p][key]
                    for p in papers
                    if not scores[p].get("ungraded")
                    and isinstance(scores[p].get(key), float)
                    and isinstance(base[p].get(key), float)
                    and not math.isnan(scores[p][key])
                    and not math.isnan(base[p][key])
                ]
                row += f"  {sum(deltas) / len(deltas) * 100:+6.1f}" if deltas else "       "
            lines.append(row)

    add("\n-- direction, pooled over cells (the unit with resolution) " + "-" * 12)
    add("  " + "config".ljust(20) + "     P       R      F1     tp   fp   fn")
    for name, scores in all_scores.items():
        pool = cell_pool(scores)
        add(
            "  "
            + name.ljust(20)
            + f"{ce.pct(pool['precision'])} {ce.pct(pool['recall'])} {ce.pct(pool['f1'])}"
            + f"  {pool['tp']:4d} {pool['fp']:4d} {pool['fn']:4d}"
        )

    add("\nEvery number above is scored against gold that, for five of these six papers,")
    add("was derived by correcting the baseline pipeline's own output. Rank order is the")
    add("finding; absolute values are not. See docs/extraction-workflow-experiments.md §0.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline"],
        help="configuration names, or 'runnable' for every buildable one",
    )
    parser.add_argument("--papers", nargs="+", help="default: every paper with a gold record")
    parser.add_argument(
        "--baseline", default="baseline", help="configuration deltas are against"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--jobs", type=int, default=3, help="papers extracted concurrently")
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="independent runs per configuration. With one gold paper this "
        "is the measurement that matters: a config delta means nothing "
        "until the same-config spread is known.",
    )
    parser.add_argument("--redo", action="store_true", help="re-run stages already written")
    parser.add_argument("--report-only", action="store_true", help="score what is on disk")
    parser.add_argument("--list", action="store_true", help="print the configuration space")
    parser.add_argument(
        "--scope",
        default="tables",
        choices=["all", "tables"],
        help="'tables' scores only analyses a publication table reported",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    if args.list:
        print(f"{'name':<20} {'axes':<12} runnable  description")
        for config in CONFIGS:
            mark = "yes" if config.runnable else "NO "
            print(f"{config.name:<20} {config.axes:<12} {mark:<9} {config.describe}")
            if config.needs:
                print(f"{'':<20} {'':<12} {'':<9} needs: {config.needs}")
        return 0

    names = (
        [c.name for c in CONFIGS if c.runnable]
        if args.configs == ["runnable"]
        else args.configs
    )
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        print(f"unknown configuration(s): {', '.join(unknown)}", file=sys.stderr)
        return 1
    blocked = [n for n in names if not BY_NAME[n].runnable]
    if blocked and not args.report_only:
        for name in blocked:
            print(f"{name}: not runnable yet -- needs {BY_NAME[name].needs}", file=sys.stderr)
        return 1

    papers = args.papers or gold_papers()
    missing = [p for p in papers if not (TEXTS / p / "stage1" / "analyses.json").is_file()]
    if missing:
        print(
            f"no stage-1 parse for: {', '.join(missing)}. "
            "Run review/parse_tables.py for them first (it costs money).",
            file=sys.stderr,
        )
        papers = [p for p in papers if p not in missing]
    if not papers:
        return 1

    replicates = range(max(1, args.replicates))
    if not args.report_only:
        jobs = [
            (BY_NAME[name], paper, r) for name in names for paper in papers for r in replicates
        ]
        print(
            f"{len(jobs)} runs: {len(names)} configuration(s) x {len(papers)} paper(s) "
            f"x {len(replicates)} replicate(s), {args.jobs} at a time\n"
        )
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for outcome in pool.map(
                lambda job: run_one(job[0], job[1], args.model, args.redo, job[2]), jobs
            ):
                print(
                    f"  {outcome['config']:<18} #{outcome['replicate']} "
                    f"{outcome['paper']:<14} {outcome['status']}"
                    + (f"  {outcome['seconds']}s" if "seconds" in outcome else "")
                )

    schema = ce.Schema()
    runs = {
        name: [score(name, papers, schema, r, args.scope) for r in replicates]
        for name in names
    }
    print(
        render_replicates(runs, papers, args.baseline)
        if len(replicates) > 1 or len(papers) == 1
        else render({n: r[0] for n, r in runs.items()}, papers, args.baseline)
    )

    if args.json:
        args.json.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
