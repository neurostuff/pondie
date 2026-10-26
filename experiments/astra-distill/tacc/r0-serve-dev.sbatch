#!/bin/bash
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
# usage: MODEL=nu|qwen sbatch -J r0-$MODEL -t HH:MM:SS -o ... r0-serve.sbatch   (LIMIT=n for a subset)
set -u
W=$WORK/astra-distill; L=$W/logs; E=$W/envs-aarch64; C=$W/code
export UV_CACHE_DIR=$W/cache/uv HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PATH=$W/bin:$PATH CC=gcc CXX=g++
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
snap(){ ls -d $HF_HOME/hub/models--$1/snapshots/* | head -1; }
if [ "$MODEL" = nu ]; then M=$(snap numind--NuExtract3); N=nuextract3; else M=$(snap Qwen--Qwen3.8-27B-FP8); N=q27fp8; fi
WORKERS=${WORKERS:-24}; LIMIT=${LIMIT:-0}; OUT=${OUT:-$W/out/r0-$MODEL-val.jsonl}
echo "node $(hostname) job $SLURM_JOB_ID model $M"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
S=$(date +%s)
setsid nohup $E/serve/bin/vllm serve $M --served-model-name $N --host 127.0.0.1 --port 8391 --max-model-len 49152 \
  --gpu-memory-utilization 0.80 --max-num-seqs $WORKERS --language-model-only > $L/vllm-r0-$MODEL-$SLURM_JOB_ID.log 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 120); do curl -s -m 3 localhost:8391/v1/models >/dev/null && break; kill -0 $SPID 2>/dev/null || { tail -40 $L/vllm-r0-$MODEL-$SLURM_JOB_ID.log; exit 1; }; sleep 10; done
echo "vllm up in $(( $(date +%s)-S )) s"; grep -E "KV cache|Maximum concurrency|version" $L/vllm-r0-$MODEL-$SLURM_JOB_ID.log | head -6
R=$(date +%s)
$E/serve/bin/python -I $C/r0-infer.py $W/data/v5/val.bodies.jsonl $OUT http://127.0.0.1:8391/v1/chat/completions $N $MODEL $WORKERS $LIMIT
echo "INFER_WALL $(( $(date +%s)-R )) s"
grep -E "throughput|Running:" $L/vllm-r0-$MODEL-$SLURM_JOB_ID.log | tail -5
kill -INT $SPID; wait $SPID 2>/dev/null; echo "TOTAL $(( $(date +%s)-S )) s"
