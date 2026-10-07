"""Score runs against the benchmark: selection, answerability, foci, cost.

    python score.py RUN [RUN ...] [--raw] [--negatives negatives.pmids] [--detail]

Selection is the meta-analysis query in `queries.py`. Recall is over the gold papers a run
holds; precision needs `--negatives` (papers screened for this meta-analysis and NOT in its
included set) to have been run through the same arm.

Foci: for each gold paper, the coordinates of the analyses the strict query selected,
resolved through `source_table_analysis` into the run's own stage-1 parse, against the gold
foci (within 1 mm). Arms with no parse cannot be scored on foci and report n/a.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import gold
import queries
from pondie import paths
from pondie.extraction import pubmed
from pondie.formats.parse_keys import parse_keys

PUBMED_CACHE = Path(__file__).with_name("pubmed.json")
#: Set by --gold-coords: an included paper's gold foci answer "reported coordinates".
GOLD_COORDS = False
#: Set by --overlap: exclude a selected paper whose cohorts re-report an earlier one's.
OVERLAP = False
#: Set by --labels: "benchmark" (as published) or "adjudicated" (gold.ADJUDICATED applied).
LABELS = "benchmark"


def pubmed_facts(pmids: list[str]) -> dict:
    cache = json.loads(PUBMED_CACHE.read_text()) if PUBMED_CACHE.is_file() else {}
    missing = [p for p in pmids if p not in cache]
    if missing:
        cache |= pubmed.summaries(missing)
        PUBMED_CACHE.write_text(json.dumps(cache, indent=1))
    return cache


def authorship(pmids: list[str]) -> dict:
    path = PUBMED_CACHE.with_name("authorship.json")
    cache = json.loads(path.read_text()) if path.is_file() else {}
    missing = [p for p in pmids if p not in cache]
    if missing:
        cache |= pubmed.authorship(missing)
        path.write_text(json.dumps(cache, indent=1))
    return cache


def pubyears(pmids: list[str]) -> dict[str, int]:
    import urllib.request
    path = PUBMED_CACHE.with_name("pubyear.json")
    cache = json.loads(path.read_text()) if path.is_file() else {}
    missing = [p for p in pmids if p not in cache and p.isdigit()]
    for i in range(0, len(missing), 150):
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed"
               "&retmode=json&id=" + ",".join(missing[i:i + 150]))
        res = json.load(urllib.request.urlopen(url))["result"]
        for p in missing[i:i + 150]:
            date = res.get(p, {}).get("pubdate", "")
            cache[p] = int(date[:4]) if date[:4].isdigit() else None
    path.write_text(json.dumps(cache, indent=1))
    return cache


def parse_index(run_dir: Path, pmid: str) -> dict[str, dict]:
    parse = run_dir / "corpus" / pmid / "stage1" / "analyses.json"
    if not parse.is_file():
        return {}
    entries = json.loads(parse.read_text()).get("analyses") or []
    return dict(zip(parse_keys(entries), entries))


def foci_of(run_dir: Path, pmid: str, keys: list[str]) -> list[tuple]:
    by_key = parse_index(run_dir, pmid)
    return [tuple(p["coordinates"]) for k in keys for p in (by_key.get(k) or {}).get("points") or []]


def input_coordinates(pmid: str) -> int:
    """Coordinates the paper's INPUTS carry: the points of its coordinate parse.

    Deterministic and arm-independent, read from the base corpus, so every arm answers
    "reported coordinate-based results" the same way. The upstream parse holds prose
    coordinates as well as table ones; a corpus built from an older parse undercounts a
    paper that states its coordinates only in prose.
    """
    base = paths.DATA / "corpus" / pmid
    if not base.is_dir():
        return 0
    parse = json.loads((base / "stage1/analyses.orig.json").read_text())
    return sum(len(a.get("points") or []) for a in parse.get("analyses") or [])


def annotate_foci(record: dict, run_dir: Path, pmid: str) -> None:
    """`_n_foci` on each analysis linked to the parse; absent where it is not linked."""
    by_key = parse_index(run_dir, pmid)
    for a in record.get("analyses") or []:
        if not isinstance(a, dict):
            continue
        keys = queries.refs(a.get("source_table_analysis"))
        known = [k for k in keys if k in by_key]
        if known:
            a["_n_foci"] = sum(len(by_key[k].get("points") or []) for k in known)


def load(run_dir: Path, raw: bool) -> dict[str, dict]:
    folder = run_dir / ("records_raw" if raw else "records")
    return {p.name.split(".")[0]: json.loads(p.read_text()) for p in folder.glob("*.extraction.json")}


def cost(run_dir: Path) -> dict:
    total = Counter()
    for line in (run_dir / "outcomes.jsonl").read_text().splitlines() if (run_dir / "outcomes.jsonl").is_file() else []:
        for o in json.loads(line).get("outcomes") or []:
            for k, v in (o.get("cost") or {}).items():
                total[k] += v
    return dict(total)


def score(run: str, raw: bool, meta: str, negatives: set[str], detail: bool,
          quiet: bool = False) -> dict:
    run_dir = paths.run(run)
    records = load(run_dir, raw)
    facts = pubmed_facts(sorted(records))
    years = pubyears(sorted(records))
    authors = authorship(sorted(records))
    positives = gold.labels(meta, LABELS)
    golds = gold.gold_studyset(meta)
    rows, tally = [], {}
    foci_hit = foci_total = foci_sel = foci_sel_gold = 0
    for pmid, record in sorted(records.items()):
        record["local_id"] = pmid
        pubmed.fill(record, facts)
        record["_pubyear"] = years.get(pmid)
        from pondie.query.overlap import date_key
        when = authors.get(pmid, {}).get("pubdate")
        record["_pubdate"] = date_key(when, pmid)[:2] if when else None
        annotate_foci(record, run_dir, pmid)
        record["_input_coordinates"] = input_coordinates(pmid)
        # The pooled foci; for a benchmark whose studyset merges papers into blocks
        # (dementia), being in the included set is what says its coordinates were pooled.
        record["_gold_coordinates"] = (
            len([p for a in golds.get(pmid, []) for p in a["points"]])
            or int(pmid in gold.included(meta))) if GOLD_COORDS else 0
        r = queries.evaluate(record, meta)
        pool = negatives | gold.included(meta) | positives
        label = "gold" if pmid in positives else ("neg" if pmid in pool else "other")
        if pmid in gold.unscored(meta, LABELS):
            label = "unscored"
        rows.append((pmid, label, r))
        for name, v in {**r["study"], **r["analysis_best"]}.items():
            tally.setdefault(name, Counter())[(label, v)] += 1
        if label == "gold":
            keys = [k for a in r["analyses"] if a["local_id"] in r["pooled_hits"] for k in a["source"]]
            selected = foci_of(run_dir, pmid, keys)
            gpts = [p for a in golds.get(pmid, []) for p in a["points"]]
            foci_total += len(gpts)
            foci_hit += sum(any(gold.near(g, q) for q in selected) for g in gpts)
            foci_sel += len(selected)
            foci_sel_gold += sum(any(gold.near(q, g) for g in gpts) for q in selected)
            if detail and gpts:
                hit = sum(any(gold.near(g, q) for q in selected) for g in gpts)
                print(f"  foci {pmid}: {hit}/{len(gpts)} gold found, {len(selected)} selected")

    def rate(label, key):
        group = [r for _, lab, r in rows if lab == label]
        return sum(r[key] for r in group), len(group)

    if OVERLAP and queries.SPECS[meta].excludes_overlap:
        from pondie.query.overlap import overlapping
        meta_pubmed = authorship(sorted(records))
        chosen = {pmid for pmid, _, r in rows if r["veto"]}
        excluded = overlapping(records, chosen, meta_pubmed, status=queries.SPECS[meta].status)
        for pmid, lab, r in rows:
            if pmid in excluded:
                r["veto"] = r["strict"] = False
                r["overlaps"] = excluded[pmid]
        print("  overlap exclusions:", ", ".join(
            f"{p} ({'gold' if p in positives else 'neg'}) ~ {e}" for p, e in excluded.items()) or "none")

    out = {"run": run + (":raw" if raw else ""), "n": len(rows)}
    for key in ("strict", "veto", "permissive"):
        tp, ng = rate("gold", key)
        fp, nn = rate("neg", key)
        out[key] = {"recall": f"{tp}/{ng}", "fp": f"{fp}/{nn}",
                    "precision": f"{tp / (tp + fp):.2f}" if tp + fp else "n/a"}
    out["tally"] = {name: {f"{lab}:{v}": n for (lab, v), n in c.items()} for name, c in tally.items()}
    out["rows"] = {pmid: {"label": lab, "veto": r["veto"], "strict": r["strict"],
                          "analyses": len(r["analyses"]),
                          "best": {**r["study"], **r["analysis_best"]}}
                   for pmid, lab, r in rows}
    out["foci_recall"] = f"{foci_hit}/{foci_total}"
    out["foci_precision"] = f"{foci_sel_gold}/{foci_sel}"
    out["cost"] = cost(run_dir)
    if quiet:
        return out
    print(f"\n=== {out['run']}  ({len(rows)} records, labels={LABELS})")
    for key in ("strict", "veto", "permissive"):
        print(f"  {key:10s} recall {out[key]['recall']:6s} false-pos {out[key]['fp']:6s} "
              f"precision {out[key]['precision']}")
    print(f"  foci of pooled veto hits on gold: recall {out['foci_recall']}  precision {out['foci_precision']}")
    c = out["cost"]
    if c:
        print(f"  cost: {c.get('calls', 0)} calls, {c.get('input_tokens', 0):,} in "
              f"({c.get('cached_tokens', 0):,} cached), {c.get('output_tokens', 0):,} out")
    print("  criterion             gold T/None/F      neg T/None/F")
    for name, counts in tally.items():
        g = "/".join(str(counts[("gold", v)]) for v in (True, None, False))
        n = "/".join(str(counts[("neg", v)]) for v in (True, None, False))
        print(f"  {name:22s} {g:18s} {n}")
    if detail:
        for pmid, label, r in rows:
            fails = [k for k, v in {**r["study"], **r["analysis_best"]}.items() if v is not True]
            mark = "SEL " if r["strict"] else ("veto" if r["veto"] else ("perm" if r["permissive"] else "    "))
            print(f"  {mark} {label:4s} {pmid}  " + ", ".join(
                f"{k}={'None' if r['study'].get(k, r['analysis_best'].get(k)) is None else 'F'}"
                for k in fails))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--raw", action="store_true", help="also score records_raw/ where present")
    ap.add_argument("--meta", default="36100907")
    ap.add_argument("--negatives", type=Path)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--overlap", action="store_true", help="judge sample overlap across papers")
    ap.add_argument("--labels", default="benchmark", choices=["benchmark", "adjudicated"])
    ap.add_argument("--gold-coords", action="store_true",
                    help="let gold foci answer 'reported coordinates' for included papers")
    args = ap.parse_args()
    negatives = set(args.negatives.read_text().split()) if args.negatives else set()
    global GOLD_COORDS
    GOLD_COORDS = args.gold_coords
    global OVERLAP, LABELS
    OVERLAP, LABELS = args.overlap, args.labels
    results = []
    for run in args.runs:
        if args.raw and (paths.run(run) / "records_raw").is_dir():
            results.append(score(run, True, args.meta, negatives, args.detail))
        results.append(score(run, False, args.meta, negatives, args.detail))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
