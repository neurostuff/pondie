"""Judge the pilot set (`evjudge_build.py`) with a decision model, each on its own runtime (Vista gh-dev).

    python evjudge_decision.py elephant PILOT.jsonl OUT.jsonl   # baselabs/elephant-1: OneJev qev + serve.patch
    python evjudge_decision.py nimble   PILOT.jsonl OUT.jsonl   # bespokelabs/Bespoke-Nimble-9B-v3: its own engine
    python evjudge_decision.py d3       PILOT.jsonl OUT.jsonl   # vllm-sr/d3: its remote code's system_one()

Models are read from the local snapshots under $DECISION_HUB (offline). Each row is asked two yes/no questions over
the same state (its docs joined), one per claim form, and gains "scores": {"<model>": P(yes) for the entity-named
claim, "<model>_bare": for the bare claim}, as `evjudge_run.py` names them; `evjudge_run.py score` scores them.
Nimble-9B-v3 runs on the engine in its Decision Index repo (bespokelabs/bespoke-nimble-9b-v3-decision-index,
code/), its LoRA adapter on the pinned Qwen3.5-9B: the nimble GitHub repo's scorer checks a newer contract than v3's.
"""
import json
import os
import sys
import time
from pathlib import Path

HUB = Path(os.environ["DECISION_HUB"])
FORMS = {"claim": "", "bare": "_bare"}


def snapshot(repo):
    snaps = list((HUB / ("models--" + repo.replace("/", "--")) / "snapshots").iterdir())
    assert len(snaps) == 1, snaps
    return snaps[0]


def question(claim):
    return f"Does the text state the following claim, all of it? Claim: {claim}"


def system_one_judge(decide):
    """decide(state, questions) -> {question id: P(yes)}"""
    def judge(r):
        qs = {f: {"type": "noul", "instructions": question(r[f])} for f in FORMS}
        return decide("\n".join(r["docs"]), qs)
    return judge


def elephant():
    from qev.engine import DecisionEngine
    from qev.schema import SystemOneRequest
    engine = DecisionEngine(str(snapshot("baselabs/elephant-1")), device="cuda:0")

    def decide(state, qs):
        resp, _ = engine.decide(SystemOneRequest(state=state, questions=qs))
        return {k: a.noul for k, a in resp.answers.items()}
    return system_one_judge(decide)


def d3():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(snapshot("vllm-sr/d3")), trust_remote_code=True)

    def decide(state, qs):
        answers = model.system_one(state=state, questions=qs)["answers"]
        return {k: a["noul"] for k, a in answers.items()}
    return system_one_judge(decide)


def nimble():
    # v3's own engine (decision-index repo code/, which pins its prompt files by hash) imports only these two names
    # from the Decision Index kit, which is not installed here
    import types
    kit = types.ModuleType("decision_index.engines")

    class Engine:
        def __init__(self, **options):
            self.options = options

    class Unsupported(Exception):
        pass
    kit.Engine, kit.Unsupported = Engine, Unsupported
    sys.modules["decision_index"] = types.ModuleType("decision_index")
    sys.modules["decision_index.engines"] = kit
    from nimble.evaluation.decision_index_engine import NimbleEngine
    adapter = snapshot("bespokelabs/Bespoke-Nimble-9B-v3")
    contract = json.loads((adapter / "schema_config.json").read_text())
    base = HUB / ("models--" + contract["model"].replace("/", "--")) / "snapshots" / contract["revision"]
    engine = NimbleEngine(adapter=str(adapter), model=str(base), revision=contract["revision"])

    def decide(state, qs):
        return {k: a["noul"] for k, a in engine(state, qs)[0]["answers"].items()}
    return system_one_judge(decide)


def main():
    name, pilot, out = sys.argv[1:4]
    judge = {"elephant": elephant, "nimble": nimble, "d3": d3}[name]()
    rows = [json.loads(line) for line in open(pilot)]
    t = time.time()
    with open(out, "w") as f:
        for n, r in enumerate(rows, 1):
            p = judge(r)
            r["scores"] = {name + suffix: p[form] for form, suffix in FORMS.items()}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                print(f"{n}/{len(rows)} in {time.time() - t:.0f} s", flush=True)
    print(f"done {len(rows)} in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
