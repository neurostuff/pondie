"""Judge the pilot set (`evjudge_build.py`) with a decision model, each on its own runtime (Vista gh-dev).

    python evjudge_decision.py elephant PILOT.jsonl OUT.jsonl   # baselabs/elephant-1: OneJev qev + serve.patch
    python evjudge_decision.py nimble   PILOT.jsonl OUT.jsonl   # bespokelabs/Bespoke-Nimble-9B-v3: nimble's CUDA scorer
    python evjudge_decision.py d3       PILOT.jsonl OUT.jsonl   # vllm-sr/d3: its remote code's system_one()

Models are read from the local snapshots under $DECISION_HUB (offline). Each row is asked two yes/no questions over
the same state (its docs joined), one per claim form, and gains "scores": {"<model>": P(yes) for the entity-named
claim, "<model>_bare": for the bare claim}, as `evjudge_run.py` names them; `evjudge_run.py score` scores them.
Nimble-9B-v3 is a LoRA adapter: it is merged into its pinned base once (as nimble's README does; the v3 contract's
keys do not match nimble's merge_local_adapter.py) and saved under $DECISION_HUB/merged.
"""
import hashlib
import json
import os
import shutil
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
    import torch
    adapter = snapshot("bespokelabs/Bespoke-Nimble-9B-v3")
    contract = json.loads((adapter / "schema_config.json").read_text())
    base = HUB / ("models--" + contract["model"].replace("/", "--")) / "snapshots" / contract["revision"]
    out = HUB / "merged" / ("nimble-9b-v3-" + adapter.name)
    if not (out / "READY.json").is_file():
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        t = time.time()
        model = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
        model.save_pretrained(out)
        AutoTokenizer.from_pretrained(base).save_pretrained(out)
        shutil.copy(adapter / "schema_config.json", out / "schema_config.json")
        (out / "READY.json").write_text(json.dumps({
            "model": "bespokelabs/Bespoke-Nimble-9B-v3", "revision": adapter.name, "base_revision": contract["revision"],
            "adapter_sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()}))
        print(f"merged in {time.time() - t:.0f} s", flush=True)
        del model
        torch.cuda.empty_cache()
    from nimble.scoring.cuda_scorer import CudaCandidateScorer
    scorer = CudaCandidateScorer(model_path=str(out), model_id="bespokelabs/Bespoke-Nimble-9B-v3",
                                 revision=adapter.name, max_input_tokens=8192)
    print("temperature", scorer.temperature, flush=True)

    def judge(r):
        schema = {f: {"type": "boolean", "description": f"Whether the text states all of this claim: {r[f]}"}
                  for f in FORMS}
        fields = scorer.score("\n".join(r["docs"]), schema)["fields"]
        return {f: fields[f]["scores"]["true"] for f in FORMS}
    return judge


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
