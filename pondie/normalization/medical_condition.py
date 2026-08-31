"""`Group.medical_condition` -> MONDO, and through MONDO to UMLS.

The link shape. Short strings -- median 30 characters -- so nothing can be inferred from
context, and 70% of the distinct forms occur in exactly one paper.

Most of the value is in triage, before any lookup. Of 1565 values, **315 (20%) are negations**
-- "no neurological or psychiatric disorder" -- recording the *absence* of a condition, and
matched against a disease ontology every one of them retrieves something at plausible
similarity. `Group.is_healthy` agrees with the negation reading on 1070/1106 groups and is
consulted first where it is set.

The accept threshold cannot be a single cut. On 1500 held-out MONDO synonym queries the score
distributions overlap -- correct matches have p10 0.807 while *wrong* top-1s have a median of
0.820 -- so a retrieval routes three ways: taken, queued for review with its candidates, or
rejected with its nearest miss recorded.

The field is multivalued by design: a cohort has comorbidities, and 237 groups carry two or
more. Each entry is normalized separately.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pondie.normalization import UNKNOWN
from pondie.normalization._folding import fold, variants
from pondie.normalization._negation import available as parser_available
from pondie.normalization._records import DEFAULT, iter_records, strings_at, value_of
from pondie.normalization._vocabulary import Vocabulary, load_mondo

#: A condition field recording the ABSENCE of a condition. Left in, these dominate the tail.
NEGATION = re.compile(
    r"^\s*(none|no\b|not\b|nil\b|without\b|free of\b|absence of\b|"
    r"unaffected|healthy|n/?a\b)",
    re.I,
)
#: Course and state, not a different disease. Lifted off and kept: "first-episode
#: schizophrenia" and "chronic schizophrenia" are one MONDO term and a query that cannot tell
#: them apart is a different problem from one that cannot find them.
QUALIFIER = re.compile(
    r"\b(first[- ]episode|chronic|acute|early[- ]onset|late[- ]onset|"
    r"remitted|in remission|treatment[- ]resistant|refractory|"
    r"drug[- ]na[iï]ve|medicated|unmedicated|recent[- ]onset|stable|"
    r"current|lifetime|past|subclinical|mild|moderate|severe|recurrent|"
    r"childhood|adolescent|adult|p(?:a)?ediatric|comorbid)\b",
    re.I,
)
#: Split on `or` only. The slot is multivalued, so several conditions crammed into one string
#: is an extraction defect; splitting on `and` also breaks single names -- "attention deficit
#: and hyperactivity disorder" is one disorder, not two.
SPLIT = re.compile(r"\s+or\s+|\s*/\s*")
TRAILING = re.compile(
    r"\s*\b(in some participants|in a subset|patients?|subjects?|"
    r"participants?|group|cohort)\b\s*$",
    re.I,
)

NO_CONDITION = "NO_CONDITION"


@dataclass(frozen=True)
class Mapped:
    heads: tuple[str, ...]
    qualifiers: tuple[str, ...]
    kind: str


def triage(raw: object) -> Mapped:
    """Head terms, the qualifiers lifted off them, and what the value turned out to be."""
    text = TRAILING.sub("", str(raw or "").strip())
    if not text:
        return Mapped((), (), "empty")
    if NEGATION.match(text):
        return Mapped((), (), NO_CONDITION)
    heads, quals = [], []
    for part in (p.strip() for p in SPLIT.split(text) if p and p.strip()):
        quals += [q.lower() for q in QUALIFIER.findall(part)]
        head = TRAILING.sub("", QUALIFIER.sub("", part)).strip(" -,")
        head = re.sub(r"\s{2,}", " ", head)
        if head:
            heads.append(head)
    return Mapped(
        tuple(heads), tuple(sorted(set(quals))), "compound" if len(heads) > 1 else "single"
    )


def link(
    heads: list[str], vocab: Vocabulary, accept: float = 0.90, review: float = 0.80
) -> tuple[dict, list, list]:
    """(head -> node, review queue, rejected). Lexical first, then retrieval."""
    import numpy as np

    from pondie.normalization._embedding import for_phrases

    hit = {h: vocab.surface[v] for h in heads for v in variants(h) if v in vocab.surface}
    rest = [h for h in heads if h not in hit]
    if not rest:
        return hit, [], []
    corpus = for_phrases(vocab.labels)
    query = for_phrases(rest, cache=False)
    sim = query @ corpus.T
    best, score, top5 = sim.argmax(1), sim.max(1), np.argsort(-sim, axis=1)[:, :5]
    queued, rejected = [], []
    for head, b, s, five in zip(rest, best, score, top5):
        if s >= accept:
            hit[head] = int(b)
        elif s >= review:
            queued.append(
                {
                    "text": head,
                    "cosine": float(s),
                    "candidates": [vocab.labels[int(k)] for k in five],
                }
            )
        else:
            rejected.append(
                {"text": head, "cosine": float(s), "nearest": vocab.labels[int(b)]}
            )
    return hit, queued, rejected


def normalize(
    patterns: tuple[str, ...] = DEFAULT,
    min_support: int = 3,
    accept: float = 0.90,
    review: float = 0.80,
) -> dict:
    """Every condition in the corpus, mapped, rolled up, with the residual reported."""
    vocab = load_mondo()
    counts: Counter = Counter()
    healthy_only: Counter = Counter()
    for _study, body in iter_records(patterns):
        for group in body.get("groups") or []:
            if not isinstance(group, dict):
                continue
            flag = value_of(group.get("is_healthy"))
            for item in strings_at(group, "medical_condition"):
                counts[item] += 1
                if flag is True:
                    healthy_only[item] += 1

    triaged = {form: triage(form) for form in counts}
    heads = sorted({h for m in triaged.values() for h in m.heads})
    hit, queued, rejected = link(heads, vocab, accept, review)

    support: Counter = Counter()
    for form, mapped in triaged.items():
        for head in mapped.heads:
            if head in hit:
                support[hit[head]] += counts[form]
    rolled = {node: vocab.rollup(node, support, min_support) for node in support}

    terms: dict[int, dict] = {}
    for form, mapped in triaged.items():
        for head in mapped.heads:
            if head not in hit:
                continue
            target = rolled.get(hit[head], hit[head])
            entry = terms.setdefault(
                target,
                {
                    "mondo": f"MONDO:{vocab.ids[target]}",
                    "umls": vocab.umls.get(target),
                    "label": vocab.labels[target],
                    "values": 0,
                    "forms": set(),
                    "qualifiers": Counter(),
                },
            )
            entry["values"] += counts[form]
            entry["forms"].add(form)
            entry["qualifiers"].update(mapped.qualifiers)
    for entry in terms.values():
        entry["forms"] = sorted(entry["forms"])
        entry["qualifiers"] = [q for q, _ in entry["qualifiers"].most_common(6)]
    return {
        "terms": sorted(terms.values(), key=lambda e: -e["values"]),
        "review": sorted(queued, key=lambda r: -r["cosine"]),
        "rejected": rejected,
        "triage": Counter(m.kind for m in triaged.values()),
        "parser": parser_available(),
    }


def report(patterns: tuple[str, ...] = DEFAULT) -> str:
    out = normalize(patterns)
    lines = [
        f"{len(out['terms'])} MONDO terms; "
        f"{sum(1 for t in out['terms'] if t['umls'])} carry a UMLS CUI",
        f"triage: {dict(out['triage'])}",
        f"{'values':>7s} {'forms':>5s}  {'MONDO':14s} {'UMLS':10s} label",
    ]
    for term in out["terms"][:12]:
        lines.append(
            f"{term['values']:7d} {len(term['forms']):5d}  {term['mondo']:14s} "
            f"{term['umls'] or '-':10s} {term['label'][:44]}"
        )
    lines.append(f"review queue {len(out['review'])}, rejected {len(out['rejected'])}")
    for r in out["review"][:6]:
        lines.append(
            f"   {r['cosine']:.2f}  {r['text'][:40]:40s} -> {r['candidates'][0][:36]}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
