"""Fixed train / validation / test splits by paper, with linked papers kept together.

    python splits.py TRAIN_DIR NS_POND_DATA FLAGGED_CSV OUT_DIR [--test 500 --val 500 --seed 20261023]

- Papers are linked when they share any identifier (pmid, pmcid, doi, or a preprint id in either direction), a
  normalized title, or at least half their text (5-word shingles, 1 in 16 sampled by hash, overlap over the smaller
  paper's sample: the same paper published twice under unrelated ids, REVIEW-2), so a preprint and its published version, or a duplicate record, can never straddle a split.
  Linked papers form one group, and groups are assigned whole.
- Only unflagged training records are split. Flagged ones are listed separately and never used.
- Test and validation are drawn by stratified sampling over the number of coordinate analyses (1, 2-3, 4-7, 8+),
  with a fixed seed. Groups larger than 5 papers go to train, so the evaluation sets don't hinge on one cluster.
- OUT_DIR gets `train.txt`, `val.txt`, `test.txt` (sorted study ids), `groups.json`, and `SPLITS.sha256`, which holds
  each file's hash. The test set is frozen: training code reads only train and val.
"""
import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random
import re
import unicodedata


def norm_title(t):
    t = unicodedata.normalize("NFKC", t or "").casefold()
    return re.sub(r"[^0-9a-z]+", " ", t).strip()


def bucket(n):
    return "1" if n <= 1 else "2-3" if n <= 3 else "4-7" if n <= 7 else "8+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("data")
    ap.add_argument("flagged")
    ap.add_argument("out")
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--extend", default="", help="a previous OUT_DIR: keep its assignments; a new paper linked to a "
                    "val or test paper is held out (held_out.txt), any other new paper goes to train")
    a = ap.parse_args()
    flagged = {r[0] for r in csv.reader(open(a.flagged))}
    sids = sorted(p.name[: -len(".extraction.json")] for p in pathlib.Path(a.train, "records").glob("*.extraction.json"))
    sids = [s for s in sids if s not in flagged]
    parent = {s: s for s in sids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    owner, n_an = {}, {}
    for s in sids:
        rec = json.load(open(pathlib.Path(a.train, "records", f"{s}.extraction.json")))
        n_an[s] = len(rec.get("analyses") or [])
        keys = []
        ip = pathlib.Path(a.data, s, "identifiers.json")
        if ip.exists():
            d = json.load(open(ip))
            for k in ("pmid", "pmcid", "doi"):
                if d.get(k):
                    keys.append(f"id:{str(d[k]).lower()}")
            for v in (d.get("other_ids") or {}).values():
                for x in (v if isinstance(v, list) else [v]):
                    if x:
                        keys.append(f"id:{str(x).lower()}")
        tp = pathlib.Path(a.data, s, "processed/pubget/text.txt")
        if tp.exists():
            title = norm_title(tp.read_text().split("\n", 1)[0])
            if len(title) > 20:
                keys.append(f"title:{title}")
        for k in keys:
            if k in owner:
                union(s, owner[k])
            else:
                owner[k] = s
    shingles, index = {}, collections.defaultdict(list)
    for s in sids:
        tp = pathlib.Path(a.data, s, "processed/pubget/text.txt")
        w = re.findall(r"[0-9a-z]+", tp.read_text().casefold()) if tp.exists() else []
        sh = {h for h in (int.from_bytes(hashlib.blake2b(" ".join(w[i:i + 5]).encode(), digest_size=8).digest(), "big")
                          for i in range(len(w) - 4)) if h % 16 == 0}
        shingles[s] = sh
        for h in sh:
            index[h].append(s)
    near = 0
    for s in sids:
        shared = collections.Counter()
        for h in shingles[s]:
            if len(index[h]) <= 50:          # boilerplate shared by many papers says nothing
                shared.update(index[h])
        for o, c in shared.items():
            if o > s and c >= 0.5 * max(1, min(len(shingles[s]), len(shingles[o]))) and c >= 20:
                union(s, o)
                near += 1
    groups = collections.defaultdict(list)
    for s in sids:
        groups[find(s)].append(s)
    if a.extend:
        return extend(a, sids, groups, n_an, near)
    rng = random.Random(a.seed)
    strata = collections.defaultdict(list)
    big = []
    for g in groups.values():
        (big if len(g) > 5 else strata[bucket(max(n_an[s] for s in g))]).append(sorted(g))
    for v in strata.values():
        v.sort()
        rng.shuffle(v)
    total = sum(len(g) for v in strata.values() for g in v)
    split = {}

    def draw(name, want):
        got = 0
        for b in sorted(strata):
            quota = round(want * sum(len(g) for g in strata[b]) / total)
            taken = 0
            while strata[b] and taken < quota:
                g = strata[b].pop()
                for s in g:
                    split[s] = name
                taken += len(g)
                got += len(g)
        return got

    draw("test", a.test)
    draw("val", a.val)
    for v in list(strata.values()) + [big]:
        for g in v:
            for s in g:
                split.setdefault(s, "train")
    write(a, split, groups, n_an, near)


def extend(a, sids, groups, n_an, near):
    prev = {}
    for name in ("train", "val", "test"):
        for s in pathlib.Path(a.extend, f"{name}.txt").read_text().split():
            prev[s] = name
    split, held = {s: v for s, v in prev.items() if s in n_an}, []
    for g in groups.values():
        names = {prev[s] for s in g if s in prev}
        for s in g:
            if s not in prev:
                if names & {"val", "test"}:
                    held.append(s)
                else:
                    split[s] = "train"
    write(a, split, groups, n_an, near)
    pathlib.Path(a.out, "held_out.txt").write_text("".join(f"{s}\n" for s in sorted(held)))
    lost = sorted(s for s in prev if s not in n_an)
    print(json.dumps({"new_to_train": sum(1 for s in split if s not in prev), "held_out_linked": len(held),
                      "previous_missing": lost}, indent=1))


def write(a, split, groups, n_an, near):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("train", "val", "test"):
        ids = sorted(s for s, v in split.items() if v == name)
        body = "\n".join(ids) + "\n"
        (out / f"{name}.txt").write_text(body)
        hashes[name] = hashlib.sha256(body.encode()).hexdigest()
    linked = {k: v for k, v in groups.items() if len(v) > 1}
    json.dump({"seed": a.seed, "linked_groups": linked}, open(out / "groups.json", "w"), indent=1)
    (out / "SPLITS.sha256").write_text("".join(f"{h}  {n}.txt\n" for n, h in hashes.items()))
    summary = {n: sum(1 for v in split.values() if v == n) for n in ("train", "val", "test")}
    summary["analyses"] = {n: sum(n_an[s] for s, v in split.items() if v == n) for n in ("train", "val", "test")}
    summary["linked_groups"] = len(linked)
    summary["near_duplicate_links"] = near
    summary["papers_in_linked_groups"] = sum(len(v) for v in linked.values())
    summary["cross_split_groups"] = sum(1 for v in groups.values() if len({split.get(s, "held_out") for s in v}) > 1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
