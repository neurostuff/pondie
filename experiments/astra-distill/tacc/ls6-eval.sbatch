#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
# usage: GPUS=<CUDA_VISIBLE_DEVICES> sbatch -p gpu-a100-dev|gpu-h100 -J NAME -t HH:MM:SS \
#          -o $WORK/astra-distill/logs/%x-%j.out --export=ALL,EVAL=<eval config name> ls6-eval.sbatch
# GPUS goes in the environment, not in --export: --export splits on commas, so GPUS=1,2 there arrives as GPUS=1.
# Lonestar6 counterpart of eval.sbatch: eval_record.py with LS6's x86_64 envs. Config `tp` must match the GPUs given.
# Nodes are whole (3x A100 40 GB or 2x H100 80 GB); Slurm rejects --gres. A100 gpu1+gpu2 share a socket.
set -u
W=/work/07723/jdk3232/ls6/astra-distill; L=$W/logs; C=$W/code; mkdir -p $L $W/runs
export ASTRA_W=$W ASTRA_ENVS=$W/envs ASTRA_SCRATCH=$SCRATCH/astra-distill
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# flashinfer's sampler JIT-compiles with nvcc, which LS6 compute nodes lack on PATH (ls6-smoke)
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$SLURM_JOB_ID
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR
export CUDA_VISIBLE_DEVICES=$GPUS
echo "node $(hostname) job $SLURM_JOB_ID eval $EVAL gpus $GPUS $(date)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader; free -g | head -2
$W/envs/serve/bin/python -I $C/eval_record.py $C/configs/$EVAL.json
rc=$?; echo "eval exit $rc $(date)"; exit $rc
