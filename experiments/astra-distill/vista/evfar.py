"""    python evfar.py VAL_PRED.jsonl N   (run from vista/ on beast; v6 records and splits)

Random sample of shared slots where the model cites a unit more than 10 units from every Astra unit: the slot's
value and both sides' unit texts, for adjudication"""
import collections, json, random, re, sys
sys.path.insert(0, ".")
import evidence as E, tasks as T
M = "/data/james/agents/astra-distill/migrate/"
P = T.Paths(records=M + "train-v6/records", splits=M + "splits-v6")
preds = {json.loads(l)["sid"]: json.loads(l).get("prediction") for l in open(sys.argv[1])}
N = int(sys.argv[2])

def get(comp, path):
    x = comp
    for name, idx in re.findall(r"\.([^.\[]+)|\[(\d+)\]", path):
        x = x[name] if name else x[int(idx)]
    return x

sids = [s for s in open(M + "splits-v6/val.txt").read().split() if isinstance(preds.get(s), (dict, list))]
rng = random.Random(11)
rng.shuffle(sids)
pool = []
for s in sids:
    rec, text, U, comp, ev = E.paper(s, P)
    g, p = collections.defaultdict(set), collections.defaultdict(set)
    for k, i in E.pairs(ev): g[k].add(i)
    for k, i in E.pairs(preds[s]): p[k].add(i)
    for k in sorted(g.keys() & p.keys()):
        far = [i for i in p[k] - g[k] if min(abs(i - j) for j in g[k]) > 10]
        if far: pool.append((s, k, sorted(far), sorted(g[k]), sorted(p[k] & g[k]), text, U, comp))
    if len(pool) > 40 * N:  # enough candidates; papers come in shuffled order
        break
rng.shuffle(pool)
def ut(text, U, i):
    a, b = U[i - 1] if 0 < i <= len(U) else (0, 0)
    return text[a:b][:350].replace("\n", " ")
for n, (s, k, far, gold, both, text, U, comp) in enumerate(pool[:N], 1):
    try: v = get(comp, k)
    except Exception: v = "?"
    print(f"### {n}. {s} {k}\nVALUE: {json.dumps(v, ensure_ascii=False)[:300]}")
    for i in far: print(f"  MODEL S{i}: {ut(text, U, i)}")
    for i in gold: print(f"  ASTRA S{i}{' (both)' if i in both else ''}: {ut(text, U, i)}")
