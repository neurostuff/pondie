"""`Task` -> a task identity, and a family of identities, clustered from the corpus itself.

The cluster shape, because no target vocabulary is usable: ONVOC has no task branch, and
Cognitive Atlas retrieval from a description alone has no threshold separating covered from
new. So tasks are grouped against each other and the group's most frequent name becomes its
label.

Six channels, kept separate. Folding `performance_measures` into one concatenated signature
*shrank* the stop-signal / go-no-go margin from +0.040 to +0.031: a sentence embedding is a
mean over its passage and the one discriminating token gets averaged away by the shared
vocabulary around it. As its own channel the same field contributes without diluting.

`prose` and `prose_lex` are dense and sparse views of the SAME text and both carry weight,
because IDF preserves a rare exact phrase that mean pooling discards.

Conditions are a SET, not a paragraph: soft overlap keeps `win` beside `gain` without letting
a long condition list drown a short one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pondie.normalization._clustering import (
    cluster,
    components,
    distances,
    families,
    name_links,
    rescue,
    sample_pairs,
)
from pondie.normalization._embedding import for_prose
from pondie.normalization._records import DEFAULT, iter_records, strings_at, value_of

CHANNELS = ("name", "prose", "setting", "measures", "conditions", "prose_lex")


@dataclass
class Task:
    study: str
    name: str
    description: str
    instructions: str
    stimuli: str
    design_type: str
    response_mode: str
    performance_measures: str
    conditions: tuple[str, ...]

    @property
    def prose(self) -> str:
        return ". ".join(x for x in (self.description, self.instructions) if x) or "none"

    @property
    def setting(self) -> str:
        return (
            ". ".join(x for x in (self.stimuli, self.design_type, self.response_mode) if x)
            or "none"
        )


def load(patterns: tuple[str, ...] = DEFAULT, minimum: int = 60) -> list[Task]:
    """Every task with enough description to compare. `minimum` keeps a bare name out."""

    def one(node, slot):
        return " ".join(strings_at({"x": node}, f"x.{slot}"))

    out = []
    for study, body in iter_records(patterns):
        for task in body.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            name = str(value_of(task.get("name")) or "").strip()
            if not name:
                continue
            item = Task(
                study=study,
                name=name,
                description=one(task, "description"),
                instructions=one(task, "instructions"),
                stimuli=one(task, "stimuli"),
                design_type=one(task, "design_type"),
                response_mode=one(task, "response_mode"),
                performance_measures=one(task, "performance_measures"),
                conditions=tuple(strings_at(task, "conditions.name")),
            )
            if len(item.description) + len(item.instructions) >= minimum:
                out.append(item)
    return out


def _channels(tasks: list[Task]):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as l2

    dense = {
        "name": for_prose([t.name for t in tasks], cache=False),
        "prose": for_prose([t.prose for t in tasks], cache=False),
        "setting": for_prose([t.setting for t in tasks], cache=False),
        "measures": for_prose([t.performance_measures or "none" for t in tasks], cache=False),
    }
    lexical = l2(
        TfidfVectorizer(
            stop_words="english", sublinear_tf=True, ngram_range=(1, 3), min_df=2
        ).fit_transform([t.prose for t in tasks])
    )
    vocab = sorted({c for t in tasks for c in t.conditions})
    index = {c: i for i, c in enumerate(vocab)}
    cvec = for_prose(vocab, cache=False) if vocab else np.zeros((0, 384))
    ids = [[index[c] for c in t.conditions] for t in tasks]

    def condition_overlap(i: int, j: int) -> float:
        a, b = ids[i], ids[j]
        if not a or not b:
            return 0.0
        m = cvec[a] @ cvec[b].T
        return float((m.max(1).mean() + m.max(0).mean()) / 2)

    return dense, lexical, condition_overlap


def _features(pairs, dense, lexical, overlap):
    import numpy as np

    rows = []
    for i, j in pairs:
        rows.append(
            [
                float(dense[k][i] @ dense[k][j])
                for k in ("name", "prose", "setting", "measures")
            ]
            + [overlap(i, j), float(lexical[i].multiply(lexical[j]).sum())]
        )
    return np.asarray(rows)


def normalize(
    patterns: tuple[str, ...] = DEFAULT,
    identity: float = 0.5,
    family: float = 0.35,
    rescue_at: float = 0.70,
    seed: int = 0,
) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    tasks = load(patterns)
    links = name_links([t.name for t in tasks])
    comp = components(len(tasks), links)
    dense, lexical, overlap = _channels(tasks)

    rng = np.random.default_rng(seed)
    pos, neg = sample_pairs(comp, rng)
    y = np.array([1] * len(pos) + [0] * len(neg))
    model = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
        _features(pos + neg, dense, lexical, overlap), y
    )

    n = len(tasks)
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    p = model.predict_proba(_features(idx, dense, lexical, overlap))[:, 1]
    d = distances(p, idx, n, links)

    ident = cluster(d, identity)
    ident, moved = rescue(ident, d, rescue_at)
    fam = families(ident, dense["prose"], family)

    def describe(labels):
        groups: dict[int, list[int]] = {}
        for i, c in enumerate(labels):
            groups.setdefault(int(c), []).append(i)
        out = []
        for cid, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            names = Counter(tasks[i].name for i in members)
            out.append(
                {
                    "label": names.most_common(1)[0][0],
                    "n_tasks": len(members),
                    "n_studies": len({tasks[i].study for i in members}),
                    "corpora": Counter(tasks[i].study[:0] or "" for i in members) and None,
                    "variants": [v for v, _ in names.most_common(8)],
                    "members": [f"{tasks[i].study}|{tasks[i].name}" for i in members],
                }
            )
        return out

    return {
        "tasks": tasks,
        "identity": describe(ident),
        "family": describe(fam),
        "rescued": moved,
        "weights": dict(zip(CHANNELS[1:], model.coef_[0])),
    }


def report(patterns: tuple[str, ...] = DEFAULT) -> str:
    out = normalize(patterns)
    lines = [
        f"{len(out['tasks'])} tasks -> {len(out['identity'])} identities, "
        f"{len(out['family'])} families ({out['rescued']} singletons rescued)",
        f"{'tasks':>6s} {'studies':>7s}  identity",
    ]
    for e in out["identity"][:10]:
        lines.append(f"{e['n_tasks']:6d} {e['n_studies']:7d}  {e['label'][:52]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
