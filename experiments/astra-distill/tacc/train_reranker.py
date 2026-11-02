"""Train an evidence reranker (CrossEncoder) on the train split's (claim, unit, label) pairs (`vista/rerank_data.py
train`), from the sentence-transformers production template (train_cross_encoder_example.py).

    python train_reranker.py RUN BASE_MODEL [--lr 2e-5] [--batch 64] [--accum 1] [--max-length 512]   (SMOKE_TEST=1: 1 step, tiny data)

Data: $W/data/rerank/train/train.jsonl {query, doc, label} (1 gold unit + 5 BM25-hard + 2 random negatives per
claim) and dev.jsonl {query, positive, documents} (300 held-out train papers, BM25 top 50) for
CrossEncoderRerankingEvaluator, which picks the best checkpoint. Loss: BinaryCrossEntropyLoss with pos_weight from
the label counts. The model goes to $W/runs/<RUN>/final; it is not pushed to the Hub (the records are not public;
PUSH_HUB=<repo> pushes it privately). Then the val benchmark is scored as for the zero-shot models
(`rerank_eval.py` -> $W/runs/rerank-zs/<RUN>.metrics.json).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from contextlib import nullcontext

import torch
from datasets import load_dataset
from transformers import EarlyStoppingCallback

from sentence_transformers import CrossEncoder, CrossEncoderModelCardData, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

W = os.environ.get("ASTRA_W", "/work/07723/jdk3232/vista/astra-distill")
DATA = f"{W}/data/rerank"
SMOKE_TEST = os.environ.get("SMOKE_TEST") == "1"


def autocast_ctx():
    """bf16/fp16 autocast for evaluator calls outside the trainer (which has its own autocast)."""
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def setup_logging(run):
    """Configure logging + TF32. Tees to $W/logs/{run}.log and silences HTTP spam."""
    os.makedirs(f"{W}/logs", exist_ok=True)
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
                        handlers=[logging.StreamHandler(), logging.FileHandler(f"{W}/logs/{run}.log")], force=True)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+, no quality loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("base")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)   # per-device batch x accum = effective batch
    ap.add_argument("--max-length", type=int, default=512)
    a = ap.parse_args()
    run = a.run + ("-smoke" if SMOKE_TEST else "")
    out_dir = f"{W}/runs/{run}"
    setup_logging(run)

    logging.info(f"Loading base model: {a.base}")
    model = CrossEncoder(a.base, num_labels=1, max_length=a.max_length, trust_remote_code=True,
                         model_card_data=CrossEncoderModelCardData(language="en",
                                                                   model_name=f"{a.base.split('/')[-1]} evidence reranker"))

    train = load_dataset("json", data_files=f"{DATA}/train/train.jsonl", split="train")
    train = train.select_columns(["query", "doc", "label"])
    dev = [json.loads(l) for l in open(f"{DATA}/train/dev.jsonl")]
    if SMOKE_TEST:
        logging.info("SMOKE_TEST=1: trimmed data; max_steps=1")
        train, dev = train.select(range(512)), dev[:20]
    eval_ds = train.select(range(min(2000, len(train))))     # loss on seen rows only; the dev evaluator picks checkpoints
    n_pos = sum(1 for x in train["label"] if x > 0.5)
    pos_weight = (len(train) - n_pos) / max(n_pos, 1)
    logging.info(f"  train: {len(train):,} rows | positives {n_pos:,} | pos_weight {pos_weight:.2f} | dev queries {len(dev):,}")
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(pos_weight))

    evaluator = CrossEncoderRerankingEvaluator(samples=dev, batch_size=128, name="ev-dev", always_rerank_positives=True)
    logging.info("Baseline evaluation:")
    with autocast_ctx():
        # Must run before deriving metric_key: evaluator(model) mutates primary_metric to add the name_ prefix.
        baseline_eval = evaluator(model)[evaluator.primary_metric]
    metric_key = f"eval_{evaluator.primary_metric}"
    logging.info(f"  baseline {metric_key} = {baseline_eval:.4f}")

    args = CrossEncoderTrainingArguments(
        output_dir=out_dir, num_train_epochs=1, max_steps=1 if SMOKE_TEST else -1,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, weight_decay=0.01, warmup_steps=0.1, lr_scheduler_type="linear", bf16=torch.cuda.is_available(),
        eval_strategy="steps", eval_steps=0.05, save_strategy="steps", save_steps=0.05, save_total_limit=2,
        logging_steps=0.01, logging_first_step=True, load_best_model_at_end=True, metric_for_best_model=metric_key,
        greater_is_better=True, report_to="none", run_name=run, seed=12, dataloader_num_workers=2,
    )
    # CE rerankers typically peak mid-training and then degrade: stop at the best dev checkpoint.
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=train, eval_dataset=eval_ds, loss=loss,
                                  evaluator=evaluator, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()

    logging.info("Post-training evaluation:")
    with autocast_ctx():
        score = evaluator(model)[evaluator.primary_metric]
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    logging.info(f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f}")

    final_dir = f"{out_dir}/final"
    model.save_pretrained(final_dir)
    logging.info(f"Saved final model to {final_dir}")

    if os.environ.get("PUSH_HUB") and not SMOKE_TEST:
        try:
            commit_url = model.push_to_hub(os.environ["PUSH_HUB"], private=True)
            logging.info(f"Pushed model to {commit_url.rsplit('/commit/', 1)[0]}")
        except Exception:
            import traceback
            logging.error(f"Hub push failed:\n{traceback.format_exc()}")

    if not SMOKE_TEST:
        del model, trainer
        torch.cuda.empty_cache()
        subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rerank_eval.py"),
                        f"{DATA}/bench", f"{W}/runs/rerank-zs", f"{run}={final_dir}", "--max-length", str(a.max_length)], check=True)


if __name__ == "__main__":
    main()
