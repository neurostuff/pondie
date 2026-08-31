#!/usr/bin/env python3
"""Map `Group.medical_condition` onto MONDO, and through MONDO onto UMLS.

Why this shape rather than clustering, and the measurements behind each stage:
docs/normalization-pipelines.md.

Conditions are the opposite problem to tasks. There is no prose -- the median value is 30
characters -- so nothing can be inferred from context, and 70% of the distinct forms occur in
exactly one paper. Two consequences shape the pipeline:

  * the encoder has to work on short entity strings, which is a different competence from
    reading a paragraph. Measured on ONVOC synonym retrieval, SapBERT reaches R@1 66.3%
    where a general sentence encoder reaches 50.6% -- the reverse of their order on task
    descriptions, where SapBERT is last.
  * most of the tail is not a rare disease. Of the 400 forms used once, 24% are *negations*
    ("no neurological or psychiatric disorder"), 17% are compounds ("lifetime marijuana abuse
    or dependence"), and 8% carry a qualifier over a common term ("remitted anorexia
    nervosa"). Matching those against a disease ontology is the wrong operation, so triage
    runs before any lookup.

  0. triage      -> negation to a sentinel; split compounds; lift qualifiers off the head term
  1. expand      -> the corpus abbreviation store, mined with scispacy's Schwartz-Hearst
                    detector, over acronyms the extraction left short
  2. lexical     -> fold-exact against MONDO labels and exact synonyms
  3. embedding   -> SapBERT retrieval over what is left, accepted above a threshold
  4. rollup      -> walk MONDO's is_a edges to the nearest ancestor the corpus itself uses,
                    so a one-paper subtype becomes queryable alongside its parent
  5. report      -> what could not be placed, with support, as vocabulary evidence

    python normalize_conditions.py --build-index      # encode MONDO labels once
    python normalize_conditions.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from schema_utils import value_of  # noqa: E402

CORPORA = (
    "data/runs/mid/records/*.extraction.json",
    "data/runs/schiz/final2/*.extraction.json",
    "data/runs/depression/records/*.extraction.json",
)

#: A condition field recording the ABSENCE of a condition. Left in, these dominate the tail
#: and every one of them retrieves some disease at moderate similarity.
NEGATION = re.compile(
    r"^\s*(none|no\b|not\b|nil\b|without\b|free of\b|absence of\b|"
    r"unaffected|healthy|n/?a\b)",
    re.I,
)
#: Qualifiers that modify a course or state rather than naming a different disease. Lifted off
#: and kept, because "first-episode schizophrenia" and "chronic schizophrenia" are the same
#: MONDO term and a query that cannot tell them apart is a different problem from one that
#: cannot find them at all.
QUALIFIER = re.compile(
    r"\b(first[- ]episode|chronic|acute|early[- ]onset|late[- ]onset|"
    r"remitted|in remission|treatment[- ]resistant|refractory|drug[- ]na[iï]ve|"
    r"medicated|unmedicated|antipsychotic[- ]na[iï]ve|recent[- ]onset|stable|"
    r"current|lifetime|past|subclinical|mild|moderate|severe|recurrent|"
    r"childhood|adolescent|adult|p(?:a)?ediatric|comorbid|probable|possible)\b",
    re.I,
)
#: A mined expansion is only usable if it looks like a phrase. Schwartz-Hearst run over a
#: paper containing tables mis-parses cell content, and the store holds the results:
#: `ii -> "including seizures),"` and `control -> "control of wishes and urges"`. Applied
#: blind, those rewrite `bipolar II disorder` and `Impulse control disorder` into nonsense.
MALFORMED = re.compile(r"[()\[\]|]|\d{3,}|\b(and|or|of|the|in|with|a)$")
#: Subtype markers, never abbreviations, however the store has them keyed.
NUMERAL = frozenset("i ii iii iv v vi vii viii ix x".split())

SPLIT = re.compile(r"\s+(?:or|and/or|and)\s+|\s*[;/]\s*|\s*,\s*(?![^()]*\))")
TRAILING = re.compile(
    r"\s*\b(in some participants|in a subset|patients?|subjects?|"
    r"participants?|group|cohort|disorder patients)\b\s*$",
    re.I,
)


def fold(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def load_mondo(path: Path):
    """(labels, ids, umls, surface->index, parent edges) from the MONDO json release."""
    graph = json.loads(path.read_text())["graphs"][0]
    nodes, surface, umls = [], {}, {}
    index_of = {}
    for n in graph["nodes"]:
        meta = n.get("meta") or {}
        if (
            n.get("type") != "CLASS"
            or not n.get("lbl")
            or meta.get("deprecated")
            or "MONDO_" not in n["id"]
        ):
            continue
        i = len(nodes)
        index_of[n["id"]] = i
        nodes.append(n["lbl"])
        surface.setdefault(fold(n["lbl"]), i)
        for s in meta.get("synonyms") or []:
            if s.get("pred") == "hasExactSynonym" and s.get("val"):
                surface.setdefault(fold(s["val"]), i)
        cui = [
            x["val"]
            for x in (meta.get("xrefs") or [])
            if str(x.get("val", "")).startswith("UMLS:")
        ]
        if cui:
            umls[i] = cui[0].split(":", 1)[1]
    parents = defaultdict(list)
    for e in graph.get("edges") or []:
        if e.get("pred") == "is_a" and e["sub"] in index_of and e["obj"] in index_of:
            parents[index_of[e["sub"]]].append(index_of[e["obj"]])
    ids = {i: k.rsplit("_", 1)[-1] for k, i in index_of.items()}
    return nodes, ids, umls, surface, parents


def triage(raw: str):
    """(head terms, qualifiers, kind). `kind` is what the value turned out to be."""
    text = TRAILING.sub("", str(raw or "").strip())
    if not text:
        return [], [], "empty"
    if NEGATION.match(text):
        return [], [], "no condition reported"
    parts = [p.strip() for p in SPLIT.split(text) if p and p.strip()]
    heads, quals = [], []
    for part in parts:
        found = QUALIFIER.findall(part)
        quals += [q.lower() for q in found]
        head = TRAILING.sub("", QUALIFIER.sub("", part)).strip(" -,")
        head = re.sub(r"\s{2,}", " ", head)
        if head:
            heads.append(head)
    return heads, sorted(set(quals)), ("compound" if len(heads) > 1 else "single")


def load_abbreviations(path: Path) -> dict[str, str]:
    """Usable expansions only, keyed by lowercase short form."""
    if not path.is_file():
        return {}
    store = json.loads(path.read_text()).get("entries") or {}
    return {
        k: v["expansion"]
        for k, v in store.items()
        if len(k) >= 2
        and k not in NUMERAL
        and v.get("expansion")
        and not MALFORMED.search(v["expansion"])
        and len(v["expansion"].split()) <= 7
        and k.lower() != v["expansion"].lower()
    }


def expand(text: str, table: dict[str, str]) -> str:
    """Expand acronyms only where the source wrote them as acronyms.

    Case is the guard. `control` is a table key and an ordinary word, and only the paper's
    own capitalisation tells them apart.
    """
    out = text
    for token in sorted(
        set(re.findall(r"\b[A-Za-z][A-Za-z0-9-]{1,9}\b", text)), key=len, reverse=True
    ):
        if not (token.isupper() and len(token) >= 2):
            continue
        expansion = table.get(token.lower())
        if expansion:
            out = re.sub(rf"\b{re.escape(token)}\b", expansion, out)
    return out


def load_conditions(patterns=CORPORA):
    import glob

    seen = Counter()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path.endswith(".raw.json"):
                continue
            try:
                body = json.loads(Path(path).read_text())
            except Exception:
                continue
            body = body.get("study") or body
            if not isinstance(body, dict):
                continue
            for group in body.get("groups") or []:
                if not isinstance(group, dict):
                    continue
                for item in value_of(group.get("medical_condition"), True):
                    item = value_of(item)
                    if isinstance(item, str) and item.strip():
                        seen[item.strip()] += 1
    return seen


def ancestors(node: int, parents, limit: int = 12):
    """MONDO ancestors, nearest first."""
    seen, frontier, out = {node}, [node], []
    while frontier and len(out) < limit:
        nxt = []
        for n in frontier:
            for p in parents.get(n, ()):
                if p not in seen:
                    seen.add(p)
                    out.append(p)
                    nxt.append(p)
        frontier = nxt
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mondo", type=Path, default=Path("data/vocab/mondo.json"))
    ap.add_argument("--index", type=Path, default=Path("data/eval/mondo-sapbert.npy"))
    ap.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument(
        "--accept",
        type=float,
        default=0.90,
        help="auto-accept a retrieved MONDO term at or above this cosine. Measured "
        "on 1500 held-out MONDO synonym queries: 86%% precision at 0.90, 79%% "
        "at 0.85, 72%% at 0.80",
    )
    ap.add_argument(
        "--review",
        type=float,
        default=0.80,
        help="below --accept and at or above this, the match goes to a review queue "
        "with its top candidates rather than being taken or dropped. The two "
        "score distributions genuinely overlap -- a wrong top-1 has a median "
        "cosine of 0.820 while the 10th percentile of correct is 0.807 -- so "
        "no single cut separates them and the band is where curation earns its "
        "keep",
    )
    ap.add_argument(
        "--abbreviations", type=Path, default=Path("data/vocab/abbreviations.json")
    )
    ap.add_argument(
        "--min-support",
        type=int,
        default=3,
        help="papers a MONDO term needs before the rollup will stop at it",
    )
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/eval/condition-vocabulary.json"))
    args = ap.parse_args()

    import numpy as np

    labels, ids, umls, surface, parents = load_mondo(args.mondo)
    print(
        f"MONDO: {len(labels)} classes, {len(surface)} surface forms, "
        f"{len(umls)} with a UMLS xref, {sum(len(v) for v in parents.values())} is_a edges"
    )

    if args.build_index or not args.index.exists():
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model, device="cpu")
        emb = model.encode(
            labels, normalize_embeddings=True, batch_size=256, show_progress_bar=False
        )
        args.index.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.index, emb)
        print(f"wrote {args.index}")
        if args.build_index:
            return 0
    emb = np.load(args.index)

    counts = load_conditions()
    forms = sorted(counts)
    print(
        f"corpus: {len(forms)} distinct medical_condition forms over {sum(counts.values())} values"
    )

    kinds = Counter()
    heads_of = {}
    for f in forms:
        heads, quals, kind = triage(f)
        kinds[kind] += counts[f]
        heads_of[f] = (heads, quals, kind)
    print("  triage: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))

    table = load_abbreviations(args.abbreviations)
    todo = sorted({h for hs, _q, _k in heads_of.values() for h in hs})
    expanded = {h: expand(h, table) for h in todo}
    n_changed = sum(1 for h, e in expanded.items() if e != h)
    print(
        f"  stage 1 expand: {len(table)} usable expansions, {n_changed} head terms rewritten"
    )

    hit = {}
    for h in todo:
        for form in (h, expanded[h]):
            if fold(form) in surface:
                hit[h] = surface[fold(form)]
                break
    print(f"  stage 2 lexical: {len(hit)}/{len(todo)} head terms matched exactly")

    rest = [h for h in todo if h not in hit]
    if rest:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model, device="cpu")
        q = model.encode(
            [expanded[h] for h in rest],
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=False,
        )
        sim = q @ emb.T
        best = sim.argmax(1)
        score = sim.max(1)
        top5 = np.argsort(-sim, axis=1)[:, :5]
        taken, review, unmapped = 0, [], []
        for h, b, sc, five in zip(rest, best, score, top5):
            if sc >= args.accept:
                hit[h] = int(b)
                taken += 1
            elif sc >= args.review:
                review.append(
                    {
                        "text": h,
                        "cosine": float(sc),
                        "candidates": [labels[int(k)] for k in five],
                    }
                )
            else:
                unmapped.append((h, float(sc), labels[int(b)]))
        print(
            f"  stage 3 SapBERT: {taken} auto-accepted (>= {args.accept}), "
            f"{len(review)} to review ({args.review}-{args.accept}), "
            f"{len(unmapped)} rejected  (median best {np.median(score):.3f})"
        )
    else:
        review, unmapped = [], []

    support = Counter()
    for f, (heads, _q, _k) in heads_of.items():
        for h in heads:
            if h in hit:
                support[hit[h]] += counts[f]

    rolled = {}
    for node in list(support):
        if support[node] >= args.min_support:
            rolled[node] = node
            continue
        rolled[node] = next(
            (a for a in ancestors(node, parents) if support.get(a, 0) >= args.min_support),
            node,
        )

    entries = defaultdict(lambda: {"forms": [], "papers": 0, "qualifiers": Counter()})
    for f, (heads, quals, kind) in heads_of.items():
        for h in heads:
            if h not in hit:
                continue
            target = rolled.get(hit[h], hit[h])
            e = entries[target]
            e["forms"].append(f)
            e["papers"] += counts[f]
            e["qualifiers"].update(quals)
    out = [
        {
            "mondo": f"MONDO:{ids[i]}",
            "umls": umls.get(i),
            "label": labels[i],
            "papers": e["papers"],
            "n_forms": len(set(e["forms"])),
            "forms": sorted(set(e["forms"]))[:12],
            "qualifiers": [q for q, _ in e["qualifiers"].most_common(6)],
        }
        for i, e in entries.items()
    ]
    out.sort(key=lambda e: -e["papers"])
    with_cui = sum(1 for e in out if e["umls"])
    print(
        f"\n  {len(out)} MONDO terms cover the corpus; {with_cui} carry a UMLS CUI "
        f"({with_cui/max(1,len(out)):.0%})"
    )
    print(f"  {'papers':>6s} {'forms':>5s}  {'MONDO':14s} {'UMLS':10s} label")
    for e in out[:14]:
        print(
            f"  {e['papers']:6d} {e['n_forms']:5d}  {e['mondo']:14s} {e['umls'] or '-':10s} "
            f"{e['label'][:44]}"
        )
    unmapped.sort(key=lambda t: -counts.get(t[0], 0))
    print(f"\n  review queue ({len(review)}), highest first:")
    for r in sorted(review, key=lambda r: -r["cosine"])[:10]:
        print(f"     {r['cosine']:.2f}  {r['text'][:40]:40s} -> {r['candidates'][0][:40]}")
    print(f"\n  {len(unmapped)} head terms rejected below {args.review}:")
    for h, s, near in unmapped[:12]:
        n = sum(v for f, v in counts.items() if h in f)
        if n > 1:
            print(f"     {n:3d} paper(s)  {h[:44]:44s} nearest {near[:28]} ({s:.2f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "terms": out,
                "review": sorted(review, key=lambda r: -r["cosine"]),
                "unplaced": [
                    {"text": h, "nearest": near, "cosine": s} for h, s, near in unmapped
                ],
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
