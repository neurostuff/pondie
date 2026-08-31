"""The cluster shape: no usable target vocabulary, so the corpus is clustered against itself.

Used where linking is not available -- ONVOC has no task branch at all, and retrieval against
Cognitive Atlas from a description alone has no threshold separating covered from new (81% of
unmatched signatures score above the 10th percentile of the known-covered set).

Three stages, and the ordering is the design:

  name ladder    cheap, near-certain pairs -> MUST-LINK, and the weak labels stage 2 trains
                 on. Learn where the cheap rule succeeded; apply where it fails.
  pair model     a logistic regression over per-channel similarities. Channels are kept
                 SEPARATE rather than concatenated: a sentence embedding is a mean over its
                 passage, so folding a weak field into one signature averages away the token
                 that discriminates.
  clustering     average linkage on 1 - P(same), must-link enforced, then a rescue pass --
                 average linkage asks a joiner to be close to a cluster's whole membership,
                 so an item adjacent to one member of a large cluster is voted down by the
                 rest. Families are built OVER the identities from plain geometry, because a
                 logistic probability saturates near 0 and decides well while measuring badly.
"""
from __future__ import annotations

from collections import defaultdict

from ._folding import fold, squash


def name_links(names: list[str]) -> list[tuple[int, int]]:
    """Pairs a name alone settles: folded equality, or one name's tokens inside the other's.

    Containment is over TOKEN sequences and not raw substrings. `saccade task` is a substring
    of `reward cue antisaccade task`, and joining an antisaccade study to a prosaccade one is
    a labelling error that then trains the pair model.
    """
    tokens = [tuple(fold(n).split()) for n in names]
    by_exact: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(names):
        by_exact[squash(name)].append(i)
    links = [(g[0], j) for g in by_exact.values() for j in g[1:]]
    long = [(i, t) for i, t in enumerate(tokens) if len(t) >= 2]
    for a, (i, ti) in enumerate(long):
        for j, tj in long[a + 1:]:
            if ti != tj and (_subsequence(ti, tj) or _subsequence(tj, ti)):
                links.append((i, j))
    return links


def _subsequence(short: tuple, long: tuple) -> bool:
    return any(long[k:k + len(short)] == short for k in range(len(long) - len(short) + 1))


def components(n: int, links) -> list[int]:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def sample_pairs(comp: list[int], rng, per_positive: int = 3):
    """Positives inside a component, negatives across. Distant supervision from the ladder.

    The negatives are assumed rather than verified: two items in different components may be
    the same thing under different names, which is exactly the population this model exists
    to find. It biases the model conservative, and that is the safe direction.
    """
    by: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(comp):
        by[c].append(i)
    usable = [g for g in by.values() if len(g) >= 3]
    pos = [(a, b) for g in usable for x, a in enumerate(g) for b in g[x + 1:]]
    members = [i for g in usable for i in g]
    neg = []
    while len(neg) < per_positive * len(pos) and members:
        a, b = rng.choice(members), rng.choice(members)
        if comp[a] != comp[b]:
            neg.append((int(a), int(b)))
    return pos, neg


def distances(probabilities, pairs, n: int, must_link, cannot_link=()):
    """1 - P(same), with the ladder's certainties written in and any exclusions written out."""
    import numpy as np
    d = np.ones((n, n), dtype="float32")
    for (i, j), p in zip(pairs, probabilities):
        d[i, j] = d[j, i] = 1.0 - p
    np.fill_diagonal(d, 0.0)
    for a, b in must_link:
        d[a, b] = d[b, a] = 0.0
    for a, b in cannot_link:
        d[a, b] = d[b, a] = 1.0
    return d


def cluster(d, threshold: float):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                   metric="precomputed", linkage="average").fit_predict(d)


def rescue(labels, d, threshold: float):
    """Attach a singleton to its nearest non-singleton when the model is confident.

    Average linkage rejects a joiner that is far from most of a large cluster even when it is
    adjacent to one member -- measured on `one-back visual task`, held out of the n-back
    cluster at P=0.90.
    """
    import numpy as np
    from collections import Counter
    sizes = Counter(labels)
    labels = np.array(labels)
    moved = 0
    for i in range(len(labels)):
        if sizes[labels[i]] != 1:
            continue
        near = next((j for j in np.argsort(d[i])
                     if j != i and sizes[labels[j]] > 1), None)
        if near is not None and (1.0 - d[i][near]) >= threshold:
            labels[i] = labels[near]
            moved += 1
    return labels, moved


def families(labels, prose, threshold: float):
    """Groups of identities, from prose geometry rather than from the model."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    ids = sorted(set(labels))
    centroid = np.vstack([prose[labels == c].mean(0) for c in ids])
    centroid /= np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-9
    fam = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                  metric="cosine", linkage="average").fit_predict(centroid)
    return np.array([fam[ids.index(c)] for c in labels])
