#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
# usage: GPUS=0 sbatch -p gpu-a100-small|gpu-a100-dev -J NAME -t HH:MM:SS -o $WORK/astra-distill/logs/%x-%j.out \
#          --export=ALL,ARGS="<script> <args>" ls6-rerank.sbatch
# Lonestar6 counterpart of rerank.sbatch (Vista): code/<script> with LS6's rerank env (sentence-transformers 6.2,
# torch 2.11 cu128). Models come from Vista's HF cache on the shared $WORK (offline); data/rerank links to Vista's.
# GPUS goes in the environment, not in --export (--export splits on commas).
# ENV=elephant runs it with envs/elephant (OneJev qev, from Vista's patched ext/OneJev; `bin/build-elephant.sh`), the
# decision models under $W/hf-decision, e.g. ARGS="judge_claims.py ROWS.jsonl OUT.jsonl".
set -u
W=/work/07723/jdk3232/ls6/astra-distill; L=$W/logs; C=$W/code; mkdir -p $L $W/runs
ENV=${ENV:-rerank}
export ASTRA_W=$W HF_HOME=/work/07723/jdk3232/vista/astra-distill/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
[ $ENV = elephant ] && export DECISION_HUB=$W/hf-decision/hub HF_HOME=$W/hf-decision
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
echo "node $(hostname) job $SLURM_JOB_ID env $ENV args $ARGS gpus $CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader; nproc; free -g | head -2
cd $C && $W/envs/$ENV/bin/python -I $ARGS
rc=$?; echo "exit $rc $(date)"; exit $rc
