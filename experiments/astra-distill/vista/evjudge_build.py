"""Pilot set for judging evidence units: does a unit state a record slot's value?

    python evjudge_build.py VAL_PRED.jsonl OUT.jsonl [--far ADJUDICATION-ev1-far.tsv] [--pairs 300]   (run from vista/ on beast; v6 records and splits)

Rows: {"set", "id", "sid", "slot", "label", "claim", "bare", "docs": [unit text, ...], "units": [unit number, ...]}.
  far    the 60 adjudicated far-away model units (ADJUDICATION-ev1-far.md and .tsv): label 1 states the
         value, 0.5 partly, 0 does not; docs are the model's far units
  pos    Astra's own units for a random val slot (label 1; Astra is occasionally wrong)
  neg    the same slot with as many random units of the same paper, each more than 10 units from Astra's (label 0)
A table row's doc carries its table's first row and the line before the table. The claim is the slot's field and
value in a sentence, named by the entity it belongs to; "bare" is the value alone in a sentence (`PHRASE`), naming no
entity.
"""
import argparse
import collections
import json
import random
import re
import sys

sys.path.insert(0, ".")
import evidence as E  # noqa: E402
import tasks as T  # noqa: E402

M = "/data/james/agents/astra-distill/migrate/"
P = T.Paths(records=M + "train-v6/records", splits=M + "splits-v6")
VERDICT = {"states": 1, "partly": 0.5, "not": 0}
SINGULAR = {"analyses": "analysis", "groups": "group", "regions": "region", "tasks": "task", "measures": "measure",
            "model_estimations": "model", "coordinate_sets": "coordinate set", "inference_settings": "inference setting",
            "acquisitions": "acquisition", "preprocessings": "preprocessing", "assessments": "assessment",
            "devices": "device", "external_datasets": "external dataset", "tables": "table"}


def steps(path):
    return [(n, int(i) if i else None) for n, i in re.findall(r"\.([^.\[]+)|\[(\d+)\]", path)]


def value_at(comp, path):
    x = comp
    for name, idx in steps(path):
        x = x[name] if name else x[idx]
    return x


def words(s):
    return str(s).replace("_", " ")


def say(v):
    if isinstance(v, list):
        return "; ".join(say(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{words(k)} {say(x)}" for k, x in v.items() if k != "local_id")
    return words(v) if isinstance(v, str) else json.dumps(v)


#: (parent field, field) -> the claim for its value, for the commonest slots; others get "The <parent> <field> is v."
PHRASE = {
    ("cells", "level"): "{v} is one of the conditions or groups compared.",
    ("levels", "level"): "{v} is one level of a factor in the analysis.",
    ("cells", "direction"): "The effect has a {v} direction.",
    ("statistic", "family"): "The test statistic is a {v} statistic.",
    ("analyses", "definition"): "An analysis tests: {v}",
    ("effect", "kind"): "The effect tested is a {v}.",
    ("analyses", "interpretations"): "{v}",
    ("terms", "type"): "A model term is {v}.",
    ("groups", "n"): "The group has {v} participants.",
    ("conditions", "description"): "A task condition: {v}",
    ("analyses", "outcome"): "The analysis found a {v}.",
    ("analyses", "name"): "An analysis is reported: {v}.",
    ("analyses", "spatial_scope"): "The analysis covers the {v}.",
    ("regions", "name"): "A brain region studied is the {v}.",
    ("groups", "name"): "A participant group is: {v}.",
    ("groups", "acquired_count"): "{v} participants of the group were enrolled.",
    ("conditions", "name"): "A task condition is {v}.",
    ("acquisitions", "magnetic_field_strength_tesla"): "Images were acquired at {v} tesla.",
    ("acquisitions", "modality"): "The imaging modality is {v}.",
    ("inference_settings", "alpha_level"): "The significance threshold is p < {v}.",
    ("analyses", "coordinate_space"): "Coordinates are reported in {v} space.",
    ("groups", "inclusion_criteria"): "Participants were included if: {v}.",
    ("groups", "exclusion_criteria"): "Participants were excluded if: {v}.",
}


def bare_claim(comp, path):
    """the value in a sentence of its own, naming no entity: 'The test statistic is a t statistic.'"""
    f = [n for n, i in steps(path) if n]
    parent, field = (f[-2] if len(f) > 1 else "study"), f[-1]
    v = say(value_at(comp, path))
    if (parent, field) in PHRASE:
        return PHRASE[parent, field].format(v=v)
    return f"The {SINGULAR.get(parent, words(parent))} {words(field)} is {v}."


def claim(comp, path):
    """'For the analysis "X": statistic family is t.'"""
    st = steps(path)
    owner, field = "the study", [n for n, i in st if n]
    if len(st) > 1 and st[1][1] is not None and st[0][0] in SINGULAR:
        item = comp[st[0][0]][st[1][1]]
        name = item.get("name") or item.get("label") or item.get("local_id")
        owner = f'the {SINGULAR[st[0][0]]} "{name}"' if name else f"a {SINGULAR[st[0][0]]}"
        field = field[1:]
    return f"For {owner}: {' '.join(words(f) for f in field[-2:])} is {say(value_at(comp, path))}."


def doc(text, U, i):
    """unit i's text; a table row gets its table's first row and the line before the table"""
    def t(j):
        a, b = U[j - 1]
        return text[a:b].strip()
    u = t(i)
    if "\t" not in u:
        return u
    j = i
    while j > 1 and "\t" in t(j - 1):
        j -= 1
    head = [t(j - 1)] if j > 1 else []
    if j != i:
        head.append(t(j))
    return "\n".join(head + [u])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("out")
    ap.add_argument("--far", default="ADJUDICATION-ev1-far.tsv")
    ap.add_argument("--pairs", type=int, default=300)
    a = ap.parse_args()
    preds = {}
    for line in open(a.preds):
        d = json.loads(line)
        preds[d["sid"]] = d.get("prediction")
    rows = []

    # far: the adjudicated items
    for line in open(a.far).read().splitlines()[1:]:
        n, s, k, verdict = line.split("\t")
        rec, text, U, comp, ev = E.paper(s, P)
        g, p = set(), set()
        for kk, i in E.pairs(ev):
            if kk == k:
                g.add(i)
        for kk, i in E.pairs(preds[s]):
            if kk == k:
                p.add(i)
        far = sorted(i for i in p - g if min(abs(i - j) for j in g) > 10)
        rows.append({"set": "far", "id": int(n), "sid": s, "slot": k, "label": VERDICT[verdict],
                     "claim": claim(comp, k), "bare": bare_claim(comp, k), "docs": [doc(text, U, i) for i in far],
                     "units": far})

    # pos / neg: random val slots
    rng = random.Random(23)
    val = open(M + "splits-v6/val.txt").read().split()
    n = 0
    while n < a.pairs:
        s = rng.choice(val)
        rec, text, U, comp, ev = E.paper(s, P)
        g = collections.defaultdict(set)
        for k, i in E.pairs(ev):
            g[k].add(i)
        k = rng.choice(sorted(g))
        gold = sorted(g[k])
        far = [i for i in range(1, len(U) + 1) if min(abs(i - j) for j in gold) > 10]
        if len(far) < len(gold):
            continue
        n += 1
        c, b = claim(comp, k), bare_claim(comp, k)
        rows.append({"set": "pos", "id": n, "sid": s, "slot": k, "label": 1, "claim": c, "bare": b,
                     "docs": [doc(text, U, i) for i in gold], "units": gold})
        neg = sorted(rng.sample(far, len(gold)))
        rows.append({"set": "neg", "id": n, "sid": s, "slot": k, "label": 0, "claim": c, "bare": b,
                     "docs": [doc(text, U, i) for i in neg], "units": neg})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(collections.Counter(r["set"] for r in rows))


if __name__ == "__main__":
    main()
