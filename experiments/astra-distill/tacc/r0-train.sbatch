#!/bin/bash
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
W=$WORK/astra-distill; E=$W/envs-aarch64
export UV_CACHE_DIR=$W/cache/uv HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 CC=gcc CXX=g++ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3.8-27B/snapshots/* | head -1)
$E/train/bin/python -I $W/code/r0-train-mem.py --model $Q --mode nf4 --lens 49152,40960 --steps 3 --val $W/data/v5/val.jsonl
echo DONE
