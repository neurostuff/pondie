"""`Task` -> a paradigm category, and the stimulus each task used.

One module, three steps, and the order is the design:

    seed     match each task name against the normalised Cognitive Atlas (`atlas`)
    cluster  everything unmatched, on a paradigm distance over six channels
    label    the stimulus, read off `Condition.stimulus_content` as its own column

**Seeding comes first because the Atlas is a target and clustering is not.** There used to
be a second module here that skipped the seeds and clustered the whole corpus against
itself with a fitted pair model. Both were reachable -- this one only through a script, so
`pondie normalize task` ran the other one -- and they disagreed. On the 100-paper defect
set the unseeded route merged `novelty oddball task`, `Go/No-go tasks` and `sustained
attention task` into one identity called `stop signal task`; the Atlas names those as three
separate paradigms and keeps them apart. The unseeded route also had no way not to: its
pair model trained by distant supervision on name components of three or more members, and
over 90 tasks only two such components exist, so 20 of 90 tasks appeared in any training
pair and the model learned one axis -- resting-state or not.

So the seeds are the vocabulary and the clustering is the residual. Only folded name
EQUALITY is used as a hard constraint: equality closes transitively, containment does not.
Why it is shaped this way, with the measurements, is docs/task-clustering-method.md.

    python -m pondie.normalization.task --out data/task-facets
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pondie.normalization._embedding import for_phrases, for_prose
from pondie.normalization._records import DEFAULT, iter_records, strings_at, value_of
from pondie.normalization.atlas import (
    ATLAS,
    GENERIC_TOK,
    build as build_seeds,
    core,
    fold,
    sq,
)
from pondie.vocabularies.folding import squash

@dataclass
class Task:
    study: str
    name: str
    description: str
    instructions: str
    design_type: str
    response_modality: str
    performance_measures: str
    conditions: tuple[str, ...]
    #: `Condition.stimulus_content`, pooled over the task's conditions -- the stimulus
    #: facet as the paper stated it. Empty where no condition carried one, which the
    #: schema says is the right answer when the stimulus does not vary across conditions.
    stimulus_content: tuple[str, ...]

    @property
    def prose(self) -> str:
        return ". ".join(x for x in (self.description, self.instructions) if x) or "none"

    @property
    def stimulus(self) -> str:
        """The stimulus facet, stated rather than inferred.

        Its own property and never folded into `apparatus` or `prose`: those measure how
        the task RAN and what it was, and the whole point of the facet split is that the
        stimulus must not separate two tasks running one paradigm.
        """
        return ". ".join(self.stimulus_content) or "none"

    @property
    def apparatus(self) -> str:
        """How the task ran: the design and what the subject answered with.

        `Task.stimuli` used to be folded in here and the channel deliberately excluded it,
        so the slot was read off every record and never used. It is not loaded any more.
        """
        return ". ".join(x for x in (self.design_type, self.response_modality) if x) or "none"


def load(patterns: tuple[str, ...] = DEFAULT) -> list[Task]:
    """Every task in the records, named or not usefully described.

    No description threshold. One existed and dropped 24 of 1,758 tasks -- 1% -- for the
    sake of a comparison the channels already handle: a task with no prose scores `none`
    against `none` on the prose channels and is separated by the rest. Withholding it does
    not improve the partition, it removes a task from the corpus the partition describes.
    """

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
                design_type=one(task, "design_type"),
                response_modality=one(task, "response_modality"),
                performance_measures=one(task, "performance_measures"),
                conditions=tuple(strings_at(task, "conditions.name")),
                stimulus_content=tuple(strings_at(task, "conditions.stimulus_content")),
            )
            out.append(item)
    return out


#: `2-back`, `two-back`, `0-back` all instantiate `n-back`: the n is a variable and the
#: corpus fills it in.
_NBACK = re.compile(r"\b(\d+|one|two|three|four|zero)([- ]?back)\b", re.I)

#: British and American spellings of the same paradigm. `Balloon Analog Risk Task` against
#: the Atlas's `balloon analogue risk task` is the whole list so far.
_SPELLING = ((re.compile(r"\banalog\b", re.I), "analogue"),
             (re.compile(r"\bcolor\b", re.I), "colour"))

#: Words too common to identify a paradigm on their own. A one-token seed core may only match
#: if it is distinctive: a bare common word matching alone is how `art emotion test` became
#: the Angling Risk Task under an earlier matcher.
_COMMON = {
    "memory", "recall", "attention", "control", "learning", "reward", "emotion", "faces",
    "decision", "judgment", "perception", "imagery", "viewing", "naming", "reading",
    "counting", "listening", "speech", "motor", "working", "risk", "choice", "search",
    "span", "fluency", "discrimination", "detection", "matching", "rating", "induction",
    "inhibition", "switching", "recognition", "identification", "generation", "processing",
}

#: A bare acronym alias -- `ART`, `BART`, `DMT`, `PVT` -- only counts when the name writes it
#: in capitals. The Atlas aliases `Angling Risk Task` as `ART`, and case-insensitively that
#: matches `art emotion test`, which is how the Angling Risk Task acquired an art study. Same
#: principle as `vocabularies.onvoc.corroborated`: an acronym needs evidence, not similarity.
_ACRONYM = re.compile(r"^[A-Z][A-Za-z]{1,5}$")


_STORE = None


def abbreviations():
    """The corpus's own abbreviation store, loaded once. Empty if it was never built.

    Schwartz-Hearst over each paper's text, mined by `pondie.extraction.corpus`. The papers
    define their own short forms -- `MID`, `ERT`, `CR` -- and expanding them is what turns a
    name nothing can match into one that matches exactly. It is the paper's own definition,
    so it is a fact about that paper rather than a guess.
    """
    global _STORE
    if _STORE is None:
        from pondie.vocabularies.abbreviations import Abbreviations
        _STORE = Abbreviations.load()
    return _STORE


def normalise(text: str, paper: str = "") -> str:
    """Case, separators, spelling, the n-back variable, and the paper's own acronyms.

    Everything here is orthography or a variable, never a judgement about what two
    paradigms have in common -- which is what lets the result be used as a hard constraint.
    """
    out = str(text or "")
    # No paper, no expansion: an expansion is a fact about the article that wrote it, and an
    # Atlas label comes from no article. Only the acronym stage is skipped.
    from pondie.vocabularies.abbreviations import expansions_in
    for short, expansion in (expansions_in(out, abbreviations(), paper) if paper else ()):
        # A method word is dropped by `core` anyway, so expanding it can only add noise --
        # and the store's `fMRI` entry is a mining defect, two copies of "Functional
        # magnetic resonance imaging" run together with no separator.
        if fold(short) in GENERIC_TOK or len(expansion) <= len(short):
            continue
        out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(short)}(?![A-Za-z0-9])", expansion, out)
    out = _NBACK.sub(r"n\2", out)
    for pattern, canonical in _SPELLING:
        out = pattern.sub(canonical, out)
    return re.sub(r"\s{2,}", " ", out).strip()


def stimulus_of(task) -> str:
    """The stimulus facet, as the paper stated it on the task's conditions.

    Stated or nothing; never inferred from a name. `(unspecified)` is the schema's own
    empty case -- the stimulus did not vary across conditions -- not a missing value.
    """
    return task.stimulus if task.stimulus_content else "(unspecified)"


class Seeds:
    """The normalised Cognitive Atlas, and matching against it."""

    def __init__(self, path: Path = ATLAS):
        built = build_seeds(path)
        self.source = built
        self.labels = built["seeds"]
        self.core = {n: core(n) for n in self.labels}
        self.by_fold: dict[str, str] = {}
        self.by_core: dict[tuple, str] = {}
        self.acronyms: dict[str, str] = {}
        for n in self.labels:
            # Every string this seed can be recognised by: its label and the Atlas's own
            # aliases, both as written.
            for surface in [n, *built["aliases"].get(n, ())]:
                if _ACRONYM.match(surface.strip()):
                    self.acronyms.setdefault(surface.strip(), n)
                    continue
                key = fold(normalise(surface))
                if key:
                    self.by_fold.setdefault(key, n)
                c = core(normalise(surface))
                if c:
                    self.by_core.setdefault(c, n)

    def match(self, name: str, paper: str = "") -> tuple[str | None, str]:
        """(seed, how) for one task name, or (None, "").

        Ranked by where the seed sits in the name, then by how much of it matched, then by
        the shortest label. Position first because the paradigm is named first and the
        qualifier follows it -- ranking on length alone sent `Stop signal task with dot
        motion discrimination` to `dot motion task`.

        `paper` expands that article's own abbreviations before matching, which is the
        whole reason `normalise` takes one. This call did not pass it, so a task the paper
        named only by its short form -- `MID`, `ERT`, `SST` -- reached a list of expanded
        Atlas labels as an acronym and matched nothing, then went to the clustering as an
        unmatched residual. The expansion is the paper's own definition, mined by
        `extraction.corpus`, so it is a fact about that article and not a guess.
        """
        name = normalise(name, paper)
        for token in re.findall(r"\b[A-Za-z]{2,6}\b", name):
            if token in self.acronyms and token.isupper():
                return self.acronyms[token], "acronym"
        if fold(name) in self.by_fold:
            return self.by_fold[fold(name)], "exact"
        plain = core(name)
        # Exact core equality, before the containment loop and NOT subject to its
        # common-word guard. `switching task` and `task-switching` have the same core and
        # were refused only because `switching` is on the common list -- a guard meant for
        # a seed matching on one bare word inside a longer name, not for an exact equality.
        if plain in self.by_core:
            return self.by_core[plain], "core-exact"
        hits = []
        for seed, c in self.core.items():
            if not c or not (len(c) >= 2 or (len(c[0]) >= 6 and c[0] not in _COMMON)):
                continue
            at, how = _run_at(c, plain), "core"
            if at is None:
                # Squashed, to bridge hyphenation: `go/no-go task` against `GoNoGo`. Guarded
                # on a word boundary, or `Motion processing` matches e-MOTIONPROCESSING-task.
                if len(sq(c)) >= 6 and sq(c) in sq(plain) and any(
                    w.startswith(c[0]) for w in plain
                ):
                    at, how = len(plain), "squash"
                else:
                    continue
            hits.append(((at, -len(c), len(fold(seed))), seed, how))
        if not hits:
            return None, ""
        best = min(hits)
        return best[1], best[2]


def _run_at(short: tuple, long: tuple) -> int | None:
    return next(
        (k for k in range(len(long) - len(short) + 1) if long[k : k + len(short)] == short),
        None,
    )


def _similarities(pairs, dense, lexical, overlap):
    """One row per pair, one column per channel: name, prose, apparatus, measures,
    conditions, prose_lex.

    Six channels kept separate rather than concatenated: a sentence embedding is a mean
    over its passage, so folding a weak field into one signature averages away the token
    that discriminates. `prose` and `prose_lex` are dense and sparse views of the same
    text; conditions are a set, compared by soft overlap.

    The columns are returned rather than combined so the caller decides. `paradigm_distances`
    means them; a fitted weighting is what the deleted unseeded route did instead, and the
    module docstring says why that is not here. There was a `CHANNELS` tuple naming this
    order and nothing read it -- the order is here, in the one place that builds it.
    """
    import numpy as np

    return np.asarray([
        [float(dense[k][i] @ dense[k][j])
         for k in ("name", "prose", "apparatus", "measures")]
        + [overlap(i, j), float(lexical[i].multiply(lexical[j]).sum())]
        for i, j in pairs
    ])


def paradigm_distances(tasks, encoder: str = "minilm"):
    """1 - the mean of six channel similarities, with identical folded names forced to 0.

    `Condition.stimulus_content` is deliberately not a channel -- the stimulus must not
    separate two tasks running one paradigm. `stimulus_of` reads it instead, as its own
    output column. Measurements in docs/task-clustering-method.md.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as l2

    n = len(tasks)
    entity = for_phrases if encoder == "sapbert" else for_prose
    dense = {
        "name": entity([t.name for t in tasks], cache=False),
        "prose": for_prose([t.prose for t in tasks], cache=False),
        "apparatus": for_prose([t.apparatus for t in tasks], cache=False),
        "measures": for_prose([t.performance_measures or "none" for t in tasks], cache=False),
    }
    lexical = l2(
        TfidfVectorizer(
            stop_words="english", sublinear_tf=True, ngram_range=(1, 3), min_df=2
        ).fit_transform([t.prose for t in tasks])
    )
    conditions = [list(t.conditions) for t in tasks]
    vocab = sorted({c for cs in conditions for c in cs})
    at = {c: i for i, c in enumerate(vocab)}
    cvec = entity(vocab, cache=False) if vocab else np.zeros((0, 384))
    ids = [[at[c] for c in cs] for cs in conditions]

    def overlap(i: int, j: int) -> float:
        a, b = ids[i], ids[j]
        if not a or not b:
            return 0.0
        m = cvec[a] @ cvec[b].T
        return float((m.max(1).mean() + m.max(0).mean()) / 2)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    similarity = _similarities(pairs, dense, lexical, overlap).mean(axis=1)
    d = np.ones((n, n), dtype="float32")
    for (i, j), value in zip(pairs, similarity):
        d[i, j] = d[j, i] = 1.0 - value
    np.fill_diagonal(d, 0.0)

    # The one constraint kept from the old name ladder. Equality is safe to close
    # transitively; its containment rule is not, and chained a 288-task component.
    for group in same_name(tasks).values():
        # EVERY pair, not a star from the first member. The star was anchored on group[0],
        # and the clustering runs on the unseeded submatrix -- so whenever group[0] was
        # seeded, every other member lost its only zero and the group scattered. 49 tasks
        # keyed `cuereactivitytask` ended up in seven categories that way.
        for x, a in enumerate(group):
            for b in group[x + 1:]:
                d[a, b] = d[b, a] = 0.0
    return d


def same_name(tasks) -> dict[str, list[int]]:
    """Tasks whose names are the same once normalised, grouped. The one hard constraint.

    Equality after normalisation closes transitively without pathology, which is what makes
    it usable as a constraint where a containment rule is not.
    """
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, task in enumerate(tasks):
        groups[name_key(task)].append(i)
    return groups


def name_key(task) -> str:
    """The string two tasks must share to be forced together.

    The CORE, not the folded name: method words are not a distinction, so `cue reactivity`
    and `cue reactivity task` are one key. Orthographic throughout -- no rule here claims
    two differently-named paradigms are one.
    """
    c = core(normalise(task.name, task.study))
    return squash(" ".join(c))


def rescue(labels, sizes, d, threshold: float):
    """Attach a singleton to its nearest non-singleton when the distance is small enough.

    Average linkage votes down a task adjacent to one member of a large cluster. This is
    the second chance for the tail that leaves.
    """
    import numpy as np

    moved = {}
    for i, c in list(labels.items()):
        if sizes[c] != 1:
            continue
        near = [j for j in np.argsort(d[i]) if j != i and labels.get(j) is not None
                and sizes[labels[j]] > 1]
        if near and (1.0 - d[i][near[0]]) >= threshold:
            moved[i] = labels[near[0]]
    for i, c in moved.items():
        labels[i] = c
    return labels, moved


def merge_on_name(groups: dict, origin: dict, tasks) -> tuple[dict, dict]:
    """Join two categories when a task in each has the same normalised name.

    Applied after clustering, because clustering sees only the unseeded half and a shared
    name spans both. It overrides the clustering by design. The key must be substantive:
    merging on `task` alone would join `emotional faces task` to `taste task`.
    """
    key_of: dict[int, str] = {}
    for key, members in same_name(tasks).items():
        c = core(normalise(tasks[members[0]].name, tasks[members[0]].study))
        if len(c) >= 2 or (len(c) == 1 and len(c[0]) >= 6 and c[0] not in _COMMON):
            for i in members:
                key_of[i] = key

    parent = {label: label for label in groups}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    seen: dict[str, str] = {}
    for label, members in groups.items():
        for i in members:
            key = key_of.get(i)
            if key is None:
                continue
            if key in seen:
                ra, rb = find(seen[key]), find(label)
                if ra != rb:
                    parent[ra] = rb
            else:
                seen[key] = label

    merged: dict[str, list[int]] = collections.defaultdict(list)
    for label, members in groups.items():
        merged[find(label)] += members
    # an Atlas name beats a borrowed one when a merged group has both
    out, out_origin = {}, {}
    for root, members in merged.items():
        component = [l for l in groups if find(l) == root]
        named = [l for l in component if origin[l] == "cognitive atlas"]
        label = min(named, key=len) if named else max(
            component, key=lambda l: (len(groups[l]), -len(l)))
        out[label] = members
        out_origin[label] = "cognitive atlas" if named else "clustered"
    return out, out_origin


def categorise(tasks, seeds: Seeds, cut: float, encoder: str, rescue_at: float = 0.0) -> dict:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    matched = {}
    # Keyed by (name, study) and not by the name alone: an expansion is article-scoped, so
    # the same short form in two papers may expand to two different things and a
    # name-only cache would serve the first paper's answer to the second.
    cache: dict[tuple[str, str], tuple] = {}
    for i, t in enumerate(tasks):
        key = (t.name.lower(), t.study)
        if key not in cache:
            cache[key] = seeds.match(t.name, t.study)
        if cache[key][0]:
            matched[i] = cache[key]
    rest = [i for i in range(len(tasks)) if i not in matched]

    d = paradigm_distances(tasks, encoder)
    labels = {}
    rescued: list[dict] = []
    if rest:
        sub = d[np.ix_(rest, rest)]
        found = AgglomerativeClustering(
            n_clusters=None, distance_threshold=cut, metric="precomputed", linkage="average"
        ).fit_predict(sub)
        labels = {rest[x]: int(c) for x, c in enumerate(found)}
        if rescue_at:
            sizes0 = collections.Counter(labels.values())
            labels, moved = rescue(labels, sizes0, d, rescue_at)
            # Reported through the result, not printed. `categorise` is a library call --
            # `report` and `main` are the only things here that write to a terminal.
            for i, c in moved.items():
                home = [k for k, v in labels.items() if v == c and k != i]
                rescued.append({
                    "name": tasks[i].name,
                    "joined": collections.Counter(
                        tasks[k].name for k in home).most_common(1)[0][0],
                })

    groups: dict[str, list[int]] = collections.defaultdict(list)
    origin: dict[str, str] = {}
    for i, (seed, _how) in matched.items():
        groups[seed].append(i)
        origin[seed] = "cognitive atlas"
    sizes = collections.Counter(labels.values())
    for i, c in labels.items():
        if sizes[c] == 1:
            continue
        members = [k for k, v in labels.items() if v == c]
        # Labelled from the UNSTRIPPED names: stripping is for the distance, not the name.
        name = collections.Counter(tasks[k].name for k in members).most_common(1)[0][0]
        groups[name].append(i)
        origin[name] = "clustered"
    alone = [i for i, c in labels.items() if sizes[c] == 1]

    def describe(label: str, members: list[int]) -> dict:
        return {
            "category": label,
            "origin": origin[label],
            "n_tasks": len(members),
            "n_studies": len({tasks[i].study for i in members}),
            "n_names": len({tasks[i].name for i in members}),
            "stimuli": collections.Counter(
                stimulus_of(tasks[i]) for i in members
            ).most_common(),
            "members": [
                {
                    "study": tasks[i].study,
                    "name": tasks[i].name,
                    "stimulus": stimulus_of(tasks[i]),
                    "conditions": list(tasks[i].conditions),
                }
                for i in sorted(members, key=lambda k: (tasks[k].name.lower(), tasks[k].study))
            ],
        }

    groups, origin = merge_on_name(groups, origin, tasks)
    return {
        "cut": cut,
        "encoder": encoder,
        "n_tasks": len(tasks),
        "n_seeds": len(seeds.labels),
        "seeded_tasks": len(matched),
        "rescued": rescued,
        "categories": sorted(
            (describe(k, v) for k, v in groups.items()), key=lambda e: -e["n_studies"]
        ),
        "uncategorised": [
            {
                "study": tasks[i].study,
                "name": tasks[i].name,
                "stimulus": stimulus_of(tasks[i]),
                "conditions": list(tasks[i].conditions),
            }
            for i in sorted(alone, key=lambda k: tasks[k].name.lower())
        ],
    }

# ---------------------------------------------------------------- the field contract
#
# `normalization.fields()` lists a module that exposes `normalize`, and the CLI verb calls
# `report`. The seeded route had neither -- it exposed `categorise` and was reached only
# from `scripts/task_categories.py`, which is why `pondie normalize task` ran the other
# implementation. These two are the whole reason this module is the one that survives.


def normalize(
    patterns: tuple[str, ...] = DEFAULT,
    cut: float = 0.60,
    encoder: str = "minilm",
    rescue_at: float = 0.0,
    atlas: Path = ATLAS,
) -> dict:
    """Every task in the records, seeded against the Atlas and then clustered."""

    return categorise(load(patterns), Seeds(atlas), cut, encoder, rescue_at)


def summarise(out: dict) -> str:
    """The run as text. Takes the result so a caller that already has one does not re-run."""

    seeded = sum(1 for c in out["categories"] if c["origin"] == "cognitive atlas")
    lines = [
        f"{out['n_tasks']} tasks, {out['n_seeds']} Atlas seeds -> "
        f"{len(out['categories'])} categories "
        f"({seeded} named by the Atlas, {len(out['categories']) - seeded} clustered), "
        f"{out['seeded_tasks']} tasks seeded, "
        f"{len(out['uncategorised'])} in no category",
        f"{'studies':>7s}  {'origin':<16s} category",
    ]
    for entry in out["categories"]:
        stim = ", ".join(f"{s} ({n})" for s, n in entry["stimuli"][:3])
        lines.append(f"{entry['n_studies']:7d}  {entry['origin']:<16s} "
                     f"{entry['category'][:40]:42s}{stim[:44]}")
    if out.get("rescued"):
        lines += ["", f"{len(out['rescued'])} singleton(s) attached to a nearest category:"]
        lines += [f"         {r['name'][:46]:48s} -> {r['joined'][:42]}"
                  for r in out["rescued"]]
    if out["uncategorised"]:
        lines += ["", f"{len(out['uncategorised'])} task(s) in no category:"]
        lines += [f"         {s['name'][:60]}" for s in out["uncategorised"]]
    return "\n".join(lines)


def report(patterns: tuple[str, ...] = DEFAULT, **kw) -> str:
    return summarise(normalize(patterns, **kw))


def write(out: dict, directory: Path) -> None:
    """The JSON and the TSV, one row per task. A reviewer reads the TSV."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "task-categories.json").write_text(json.dumps(out, indent=1))
    with (directory / "task-categories.tsv").open("w", newline="") as handle:
        w = csv.writer(handle, delimiter="\t")
        w.writerow(["category", "origin", "category_studies", "study", "task_name",
                    "stimulus", "conditions"])
        for c in out["categories"]:
            for m in c["members"]:
                w.writerow([c["category"], c["origin"], c["n_studies"], m["study"],
                            m["name"], m["stimulus"], "; ".join(m["conditions"])])
        for s in out["uncategorised"]:
            w.writerow(["(none)", "", 0, s["study"], s["name"], s["stimulus"],
                        "; ".join(s["conditions"])])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="*", default=list(DEFAULT))
    parser.add_argument("--cut", type=float, default=0.60)
    parser.add_argument("--encoder", choices=("minilm", "sapbert"), default="minilm")
    parser.add_argument("--rescue", type=float, default=0.0,
                        help="attach a singleton to its nearest category at this similarity")
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--out", type=Path, help="directory for the JSON and TSV")
    args = parser.parse_args(argv)

    seeds = Seeds(args.atlas)
    tasks = load(tuple(args.records))
    print(f"{len(seeds.source['all'])} Atlas labels -> {len(seeds.labels)} seeds")
    print(f"{len(tasks)} tasks")
    out = categorise(tasks, seeds, args.cut, args.encoder, args.rescue)
    print(summarise(out))
    if args.out:
        write(out, args.out)
        print(f"wrote {args.out}/task-categories.json and .tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
