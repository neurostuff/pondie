"""Judge the pilot set (`evjudge_build.py`) with an NLI model or Bespoke-MiniCheck, and score the judges.

    python evjudge_run.py nli PILOT.jsonl OUT.jsonl         # DeBERTa-v3-large NLI (transformers, one GPU)
    python evjudge_run.py minicheck PILOT.jsonl OUT.jsonl   # Bespoke-MiniCheck-7B (vLLM; TP from CUDA_VISIBLE_DEVICES)
    python evjudge_run.py nuextract PILOT.jsonl OUT.jsonl   # NuExtract3 zero-shot, a yes/no enum (vLLM)
    python evjudge_run.py score OUT.jsonl [OUT.jsonl ...]

Each row gains "scores": {name: support probability}. NLI: entailment probability of each doc alone (max over docs,
`nli_max`), of all docs joined (`nli_joint`), and the joined docs' contradiction probability (`nli_contra`).
MiniCheck: P("Yes") for all docs joined as one document (`minicheck`), the package's own prompt and read-out.
CLAIM=bare judges the bare claim instead (scores suffixed _bare). Score: AUC for far (states vs not; partly left
out) and for pos vs neg; the cut-off with the best balanced accuracy on pos/neg, and the accuracy it gives on each
set (far states: share accepted; far not: share rejected).
"""
import json
import math
import os
import sys

NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MINICHECK = "bespokelabs/Bespoke-MiniCheck-7B"
# MiniCheck's prompts (github.com/Liyan06/MiniCheck, minicheck/utils.py)
SYSTEM = ("Determine whether the provided claim is consistent with the corresponding document. Consistency in this "
          "context implies that all information presented in the claim is substantiated by the document. If not, it "
          "should be considered inconsistent. Please assess the claim's consistency with the document by responding "
          'with either "Yes" or "No".')
USER = "Document: [DOCUMENT]\nClaim: [CLAIM]"


def load(path):
    return [json.loads(line) for line in open(path)]


CLAIM = os.environ.get("CLAIM", "claim")   # the row field judged: "claim" (entity named) or "bare"
SUFFIX = "" if CLAIM == "claim" else "_" + CLAIM


def run_nli(rows):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI, torch_dtype=torch.float16).cuda().eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    pairs = []
    for r in rows:
        pairs += [(d, r[CLAIM]) for d in r["docs"]] + [("\n".join(r["docs"]), r[CLAIM])]
    probs = []
    with torch.no_grad():
        for b in range(0, len(pairs), 32):
            batch = pairs[b:b + 32]
            enc = tok([p for p, _ in batch], [h for _, h in batch], truncation="only_first", max_length=512,
                      padding=True, return_tensors="pt").to("cuda")
            probs += model(**enc).logits.float().softmax(-1).tolist()
    k = 0
    for r in rows:
        each = probs[k:k + len(r["docs"])]
        joint = probs[k + len(r["docs"])]
        k += len(r["docs"]) + 1
        r.setdefault("scores", {}).update({"nli_max" + SUFFIX: max(p[lab["entailment"]] for p in each),
                                           "nli_joint" + SUFFIX: joint[lab["entailment"]],
                                           "nli_contra" + SUFFIX: joint[lab["contradiction"]]})


def run_minicheck(rows):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(MINICHECK, trust_remote_code=True)
    tp = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    llm = LLM(MINICHECK, tensor_parallel_size=tp, trust_remote_code=True, max_model_len=8192,
              gpu_memory_utilization=0.85, enable_prefix_caching=True,
              enforce_eager=True)  # vLLM 0.28's InternLM2 fails CUDA graph capture (forward() lacks intermediate_tensors)
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": USER.replace("[DOCUMENT]", "\n".join(r["docs"]))
                                         .replace("[CLAIM]", r[CLAIM])}],
                                       add_generation_prompt=True, tokenize=False) for r in rows]
    out = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1, logprobs=5))
    for r, o in zip(rows, out):
        p = sum(math.exp(t.logprob) for t in o.outputs[0].logprobs[0].values()
                if t.decoded_token.strip().lower() == "yes")
        r.setdefault("scores", {})["minicheck" + SUFFIX] = p


NUEXTRACT = "numind/NuExtract3"
NU_TEMPLATE = '{"claim_supported": ["yes", "no"]}'
NU_PROMPT = ("Does the document state everything the claim says? Answer yes only if all of the claim is stated by the "
             "document.\n\nDocument:\n[DOCUMENT]\n\nClaim: [CLAIM]")


def run_nuextract(rows):
    """zero-shot NuExtract3 (structured mode, a yes/no enum); the score is P(yes) / (P(yes) + P(no)) at the answer token"""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(NUEXTRACT)
    tp = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    llm = LLM(NUEXTRACT, tensor_parallel_size=tp, max_model_len=8192, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0})
    prompts = [tok.apply_chat_template([{"role": "user", "content": NU_PROMPT.replace("[DOCUMENT]", "\n".join(r["docs"]))
                                         .replace("[CLAIM]", r[CLAIM])}],
                                       template=NU_TEMPLATE, enable_thinking=False,
                                       add_generation_prompt=True, tokenize=False) for r in rows]
    out = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=24, logprobs=10))
    for r, o in zip(rows, out):
        p = None
        for step in o.outputs[0].logprobs:   # the first position whose candidates include yes or no
            cand = {}
            for t in step.values():
                w = t.decoded_token.strip().strip('"').lower()
                if w in ("yes", "no"):
                    cand[w] = cand.get(w, 0) + math.exp(t.logprob)
            if cand:
                p = cand.get("yes", 0) / sum(cand.values())
                break
        r.setdefault("scores", {})["nuextract" + SUFFIX] = 0.0 if p is None else p


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def score(files):
    rows = {}
    for f in files:
        for r in load(f):
            rows.setdefault((r["set"], r["id"]), {**r, "scores": {}})["scores"].update(r["scores"])
    rows = list(rows.values())
    names = sorted({k for r in rows for k in r["scores"] if not k.startswith("nli_contra")})
    far = [r for r in rows if r["set"] == "far" and r["label"] != 0.5]
    pn = [r for r in rows if r["set"] in ("pos", "neg")]
    print(f"far: {sum(r['label'] == 1 for r in far)} states / {sum(r['label'] == 0 for r in far)} not; "
          f"pos/neg: {len(pn) // 2} pairs")
    print(f"{'judge':<16} {'far AUC':>8} {'pos/neg AUC':>12} {'cut':>5} {'pos acc':>8} {'neg acc':>8} "
          f"{'far states':>10} {'far not':>8}")
    for n in names:
        def acc(rs, cut, want):
            return round(sum((r["scores"][n] > cut) == want for r in rs) / max(1, len(rs)), 3)
        pos, neg = [r for r in pn if r["label"] == 1], [r for r in pn if r["label"] == 0]
        # the cut-off with the best balanced accuracy on pos/neg, then applied to the adjudicated items
        cut = max(sorted({r["scores"][n] for r in pn}), key=lambda c: acc(pos, c, True) + acc(neg, c, False))
        fa = auc([r["scores"][n] for r in far if r["label"] == 1], [r["scores"][n] for r in far if r["label"] == 0])
        pa = auc([r["scores"][n] for r in pos], [r["scores"][n] for r in neg])
        print(f"{n:<16} {fa:>8} {pa:>12} {cut:>5.2f} {acc(pos, cut, True):>8} {acc(neg, cut, False):>8} "
              f"{acc([r for r in far if r['label'] == 1], cut, True):>10} "
              f"{acc([r for r in far if r['label'] == 0], cut, False):>8}")
    print("\nfar items (label: scores)")
    for r in sorted((r for r in rows if r["set"] == "far"), key=lambda r: r["id"]):
        print(f"  {r['id']:>2} {r['label']:<3} " + " ".join(f"{k}={v:.2f}" for k, v in sorted(r["scores"].items()))
              + f"  {r['slot']}")


def main():
    cmd = sys.argv[1]
    if cmd == "score":
        return score(sys.argv[2:])
    rows = load(sys.argv[2])
    {"nli": run_nli, "minicheck": run_minicheck, "nuextract": run_nuextract}[cmd](rows)
    with open(sys.argv[3], "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
