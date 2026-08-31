#!/usr/bin/env python3
"""Group the task entities across corpora into task identities, data-driven.

Why this shape rather than a vocabulary lookup, and the measurements behind each stage:
docs/normalization-pipelines.md.

One concatenated signature per task does not work. Measured on this corpus, folding
`performance_measures` into the blob *shrank* the stop-signal/go-no-go margin from +0.040 to
+0.031, because a sentence embedding is a mean over the passage and the one discriminating
token ("stop-signal reaction time") is averaged away by the shared vocabulary around it. So
each field is its own similarity channel, and the channels are combined by a model rather
than by concatenation.

Three stages, precision first. Nothing here is specific to a paradigm or a corpus: the
channels are schema fields and the rules are string operations, so the same pipeline runs on
any set of extracted Task entities.

  1. name ladder      -> must-link. Folded equality and bidirectional containment. Also the
                         weak labels stage 2 trains on: learn where the cheap rule succeeded,
                         apply where it fails.
  2. pair model       -> logistic regression over per-channel similarities. Giving the name
                         its own weighted channel is what separates paradigms whose prose is
                         near-identical -- stop-signal against go/no-go, for instance.
  3. constrained      -> average-linkage agglomerative clustering on 1 - P(same task) gives
     clustering          task IDENTITY. Families are then built over the identities using the
                         prose cosine, not the model: a logistic probability saturates near 0
                         for non-matches, so it decides well and measures badly, and at a
                         coarse cut almost every distance piles up at 1.0.

    python normalize_tasks.py --fit          # train and report held-out pair metrics
    python normalize_tasks.py --cluster      # emit the vocabulary
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

from pondie import _schema  # noqa: F401 -- puts the schema submodule on the path
from pondie.extraction import passes  # noqa: F401 -- and the extraction passes

from schema_utils import NOT_REPORTED, value_of  # noqa: E402



CORPORA = {"mid": "data/runs/mid/records/*.extraction.json",
           "schiz": "data/runs/schiz/final2/*.extraction.json",
           "dep": "data/runs/depression/records/*.extraction.json"}

def text(x) -> str:
    """The words in a value. Empty for an absent slot and for `not_reported`, which carries
    no words: serialising the wrapper instead puts `{'extraction_status': ...}` into the text
    a matcher then reads."""
    if x is None or x is NOT_REPORTED:
        return ""
    return x if isinstance(x, str) else json.dumps(x)


def fold(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def load_tasks(corpora=CORPORA) -> list[dict]:
    import glob
    out = []
    for corpus, pattern in corpora.items():
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
            for task in (body.get("tasks") or []):
                if not isinstance(task, dict):
                    continue
                name = text(value_of(task.get("name"))).strip()
                if not name:
                    continue
                conditions = [text(value_of(c.get("name"))).strip()
                              for c in (task.get("conditions") or []) if isinstance(c, dict)]
                out.append({
                    "corpus": corpus,
                    "study": Path(path).name.split(".")[0],
                    "name": name,
                    "description": text(value_of(task.get("description"))),
                    "instructions": text(value_of(task.get("instructions"))),
                    "stimuli": text(value_of(task.get("stimuli"))),
                    "design_type": text(value_of(task.get("design_type"))),
                    "response_mode": text(value_of(task.get("response_mode"))),
                    "performance_measures": text(value_of(task.get("performance_measures"))),
                    "conditions": [c for c in conditions if c],
                })
    return out


def name_links(tasks: list[dict]) -> list[tuple[int, int]]:
    """Pairs the name alone is enough to join: folded equality, or one name inside the other."""
    folded = [fold(t["name"]) for t in tasks]
    by_exact = defaultdict(list)
    for i, f in enumerate(folded):
        by_exact[f].append(i)
    links = []
    for group in by_exact.values():
        links += [(group[0], j) for j in group[1:]]
    long = [(i, f) for i, f in enumerate(folded) if len(f) > 8]
    for a, (i, fi) in enumerate(long):
        for j, fj in long[a + 1:]:
            if fi != fj and (fi in fj or fj in fi):
                links.append((i, j))
    return links


def channels(tasks: list[dict], model_name: str):
    """Per-channel similarity matrices. Separate, never concatenated."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    model = SentenceTransformer(model_name, device="cpu")

    def enc(texts):
        return model.encode(texts, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False)

    name = enc([t["name"] for t in tasks])
    prose = enc([". ".join(x for x in (t["description"], t["instructions"]) if x) or "none"
                 for t in tasks])
    setting = enc([". ".join(x for x in (t["stimuli"], t["design_type"], t["response_mode"]) if x)
                   or "none" for t in tasks])
    measures = enc([t["performance_measures"] or "none" for t in tasks])

    # Conditions are a SET, not a paragraph. Soft overlap keeps `win` next to `gain` without
    # letting a long condition list drown a short one, which concatenation would do.
    vocab = sorted({c for t in tasks for c in t["conditions"]})
    index = {c: i for i, c in enumerate(vocab)}
    cvec = enc(vocab) if vocab else np.zeros((0, name.shape[1]))
    cond_ids = [[index[c] for c in t["conditions"]] for t in tasks]

    def cond_sim(i, j):
        a, b = cond_ids[i], cond_ids[j]
        if not a or not b:
            return 0.0
        m = cvec[a] @ cvec[b].T
        return float((m.max(1).mean() + m.max(0).mean()) / 2)

    # A sparse view of the same prose. Mean pooling discards a rare exact phrase; IDF is what
    # keeps it, which is why this channel and the dense one disagree usefully. Scoped to the
    # description: over `performance_measures` the vocabulary is a short shared list of
    # "reaction time" and "accuracy", so the cosine tracks common terms and measures nothing.
    lexical = normalize(TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                        ngram_range=(1, 3), min_df=2).fit_transform(
        [f"{t['description']} {t['instructions']}".strip() or "none" for t in tasks]))

    return ({"name": name, "prose": prose, "setting": setting, "measures": measures},
            cond_sim, lexical)


def features(pairs, mats, cond_sim, lexical):
    import numpy as np
    rows = []
    for i, j in pairs:
        f = [float(mats[k][i] @ mats[k][j]) for k in ("name", "prose", "setting", "measures")]
        f.append(cond_sim(i, j))
        f.append(float(lexical[i].multiply(lexical[j]).sum()))
        rows.append(f)
    return np.asarray(rows)


FEATURE_NAMES = ("name", "prose", "setting", "measures", "conditions", "prose_lex")


def components(n: int, links) -> list[int]:
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def sample_pairs(comp, rng, per_positive=3):
    """Positives inside a name-ladder component, negatives across. The model is trained on
    where the name ladder SUCCEEDED so it can be applied where the name ladder fails."""
    by = defaultdict(list)
    for i, c in enumerate(comp):
        by[c].append(i)
    usable = [g for g in by.values() if len(g) >= 3]
    pos = [(a, b) for g in usable for x, a in enumerate(g) for b in g[x + 1:]]
    members = [i for g in usable for i in g]
    neg = []
    while len(neg) < per_positive * len(pos):
        a, b = rng.choice(members), rng.choice(members)
        if comp[a] != comp[b]:
            neg.append((int(a), int(b)))
    return pos, neg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--identity", type=float, default=0.5, help="cut for one task")
    ap.add_argument("--rescue", type=float, default=0.70,
                    help="attach a singleton to its nearest cluster above this P(same task). "
                         "Average linkage asks a joiner to be close to a cluster's whole "
                         "membership, so a task adjacent to one member of a large cluster is "
                         "voted down by the rest -- measured here on `one-back visual task`, "
                         "held out of the n-back cluster at P=0.90. Set to 1.0 to disable")
    ap.add_argument("--family", type=float, default=0.35,
                    help="cosine cut over identity centroids, for a family of tasks. The "
                         "name-derived gold scores identity, not family, so it cannot pick "
                         "this -- every increase costs a little V by construction. It is a "
                         "granularity choice, not a fitted parameter")
    ap.add_argument("--out", type=Path, default=Path("data/eval/task-vocabulary.json"))
    ap.add_argument("--cluster", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

    tasks = load_tasks()
    tasks = [t for t in tasks if len(t["description"]) + len(t["instructions"]) >= 60]
    print(f"{len(tasks)} task entities  {dict(Counter(t['corpus'] for t in tasks))}")

    links = name_links(tasks)
    comp = components(len(tasks), links)
    print(f"stage 1  name ladder: {len(links)} must-link pairs -> {len(set(comp))} components")

    mats, cond_sim, lexical = channels(tasks, args.model)
    rng = np.random.default_rng(0)
    pos, neg = sample_pairs(comp, rng)
    pairs = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    X = features(pairs, mats, cond_sim, lexical)

    groups = np.array([comp[a] for a, _b in pairs])
    held = set(rng.choice(sorted(set(groups)), size=max(2, len(set(groups)) // 3), replace=False))
    test = np.array([g in held for g in groups])
    print(f"stage 2  {len(pos)} positive / {len(neg)} negative pairs, "
          f"grouped split -> {test.sum()} test")

    keep_no_name = [i for i, f in enumerate(FEATURE_NAMES) if f != "name"]
    for label, cols in (("without the name channel", keep_no_name),
                        ("with the name channel", list(range(len(FEATURE_NAMES))))):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[~test][:, cols],
                                                                            y[~test])
        p = clf.predict_proba(X[test][:, cols])[:, 1]
        print(f"   {label:26s} AUC {roc_auc_score(y[test], p):.3f}  "
              f"AP {average_precision_score(y[test], p):.3f}")
        if label.startswith("without"):
            print("      weights: " + ", ".join(
                f"{FEATURE_NAMES[c]} {w:+.2f}" for c, w in zip(cols, clf.coef_[0])))

    if not args.cluster:
        return 0

    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
    n = len(tasks)
    D = np.ones((n, n), dtype=np.float32)
    idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    P = clf.predict_proba(features(idx, mats, cond_sim, lexical))[:, 1]
    for (i, j), p in zip(idx, P):
        D[i, j] = D[j, i] = 1.0 - p
    np.fill_diagonal(D, 0.0)
    for a, b in links:                      # must-link survives the model
        D[a, b] = D[b, a] = 0.0

    gold_counts = Counter(fold(t["name"]) for t in tasks)
    ev = [i for i, t in enumerate(tasks) if gold_counts[fold(t["name"])] >= 3]
    uniq = {l: k for k, l in enumerate(sorted({fold(tasks[i]["name"]) for i in ev}))}
    ytrue = np.array([uniq[fold(tasks[i]["name"])] for i in ev])

    out = {}
    identity = AgglomerativeClustering(n_clusters=None, distance_threshold=args.identity,
                                       metric="precomputed",
                                       linkage="average").fit_predict(D)
    # Families are groups of identities, measured on prose geometry rather than on the model.
    sizes = Counter(identity)
    rescued = 0
    for i in range(n):
        if sizes[identity[i]] != 1:
            continue
        order = np.argsort(D[i])
        near = next((j for j in order if j != i and sizes[identity[j]] > 1), None)
        if near is not None and (1.0 - D[i][near]) >= args.rescue:
            identity[i] = identity[near]; rescued += 1
    if rescued:
        print(f"         rescue: {rescued} singleton(s) attached at P >= {args.rescue}")

    ids = sorted(set(identity))
    centroid = np.vstack([mats["prose"][identity == c].mean(0) for c in ids])
    centroid /= np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-9
    fam_of = AgglomerativeClustering(n_clusters=None, distance_threshold=args.family,
                                     metric="cosine", linkage="average").fit_predict(centroid)
    family = np.array([fam_of[ids.index(c)] for c in identity])

    for level, th, lab in (("identity", args.identity, identity),
                           ("family", args.family, family)):
        h, c, v = homogeneity_completeness_v_measure(ytrue, lab[ev])
        print(f"\nstage 3  {level:8s} th {th}: {len(set(lab)):4d} clusters  "
              f"ARI {adjusted_rand_score(ytrue, lab[ev]):.3f}  V {v:.3f}")
        groups_ = defaultdict(list)
        for i, cid in enumerate(lab):
            groups_[cid].append(i)
        entries = []
        for cid, members in sorted(groups_.items(), key=lambda kv: -len(kv[1])):
            names = Counter(tasks[i]["name"] for i in members)
            entries.append({"label": names.most_common(1)[0][0], "n_tasks": len(members),
                            "n_studies": len({tasks[i]["study"] for i in members}),
                            "corpora": dict(Counter(tasks[i]["corpus"] for i in members)),
                            "variants": [n for n, _ in names.most_common(8)],
                            "studies": sorted({tasks[i]["study"] for i in members})})
        out[level] = entries
        for e in entries[:8]:
            print(f"   {e['n_tasks']:4d} tasks {e['n_studies']:4d} studies  {e['label'][:44]:44s}"
                  f" {e['corpora']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
