#!/bin/bash
#SBATCH -J astra-vsmoke
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:15:00
#SBATCH -A IBN26027
#SBATCH -o /work/07723/jdk3232/vista/astra-distill/logs/vsmoke-%j.out
# Vista (GH200, aarch64) smoke: envs, LoRA/QLoRA throughput+memory, vLLM guided JSON. 1 SU/node-h, min charge 15 min.
# Idempotent; pick stages with  STAGES="env t4b t27q t27b serve4b serve27" sbatch vista-smoke.sbatch  (default: all).
# Slurm rejects --gres/--mem. Stage markers: "=== STAGE x OK|FAIL". Code lives in $WORK/astra-distill/code (rsync tacc/*.py there).
set -u
STAGES=${STAGES:-"env t4b t27q t27b serve4b serve27 serve27b"}
W=$WORK/astra-distill; L=$W/logs; E=$W/envs-aarch64; C=$W/code; mkdir -p $L
export UV_CACHE_DIR=$W/cache/uv UV_PYTHON_INSTALL_DIR=$W/python UV_LINK_MODE=copy PIP_CACHE_DIR=$W/cache/pip
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PATH=$W/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# TRAP: the default env sets CC=nvc (NVHPC) and triton rejects its -Wno-psabi; force gcc for triton/vLLM JIT builds
export CC=gcc CXX=g++
T0=$(date +%s); st(){ echo "=== STAGE $1 $2 ($(( $(date +%s)-T0 )) s) $(date +%T)"; }
has(){ [[ " $STAGES " == *" $1 "* ]]; }
snap(){ ls -d $HF_HOME/hub/models--$1/snapshots/* | head -1; }
NU=$(snap numind--NuExtract3); Q27=$(snap Qwen--Qwen3.8-27B); Q27F=$(snap Qwen--Qwen3.8-27B-FP8)

echo "node $(hostname) job $SLURM_JOB_ID arch $(uname -m)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvidia-smi | sed -n 3,4p
free -g | sed -n 2p; nproc --all; df -h /tmp | tail -1; gcc --version | head -1; which nvcc || echo "no nvcc"
for u in https://huggingface.co https://pypi.org https://wheels.vllm.ai https://download.pytorch.org/whl/cu128/; do echo "$u $(curl -sI -m 15 $u | head -1)"; done

mk(){ # name python
  [ -x $E/$1/bin/python ] || $W/bin/uv venv -q --python $2 $E/$1; }
if has env; then
  # torch 2.11 ships triton 3.6, which flash-linear-attention 0.5.2 refuses on Hopper (needs >=3.7.1): use torch 2.13 (triton 3.7.1)
  ( [ -f $E/train/.ok ] || { mk train 3.12 && VIRTUAL_ENV=$E/train uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129 \
    && VIRTUAL_ENV=$E/train uv pip install transformers==5.17.0 peft==0.21.0 bitsandbytes==0.50.2 trl==1.13.0 accelerate==1.15.0 \
       flash-linear-attention==0.5.2 numpy safetensors && touch $E/train/.ok; } ) > $L/venv-train-$SLURM_JOB_ID.log 2>&1 &
  P1=$!
  # PyPI vllm is a CUDA 13 build: use the +cu129 build from wheels.vllm.ai
  ( [ -f $E/serve/.ok ] || { mk serve 3.12 && VIRTUAL_ENV=$E/serve uv pip install "vllm==0.28.0+cu129" transformers==5.16.1 xgrammar==0.2.3 \
    --extra-index-url https://wheels.vllm.ai/0.28.0/cu129 --index-strategy unsafe-best-match --torch-backend cu129 && touch $E/serve/.ok; } ) > $L/venv-serve-$SLURM_JOB_ID.log 2>&1 &
  P2=$!
  wait $P1; wait $P2
  for e in train serve; do [ -f $E/$e/.ok ] && echo "env $e OK" || { echo "env $e FAIL"; tail -15 $L/venv-$e-$SLURM_JOB_ID.log; }; done
  for e in train serve; do [ -x $E/$e/bin/python ] && { echo "== $e"; $E/$e/bin/python -I -c "
import importlib.metadata as m,sys
print(sys.version.split()[0])
for p in ['torch','vllm','transformers','peft','bitsandbytes','trl','accelerate','xgrammar','flashinfer-python','flash-linear-attention','triton','numpy']:
    try: print(' ',p,m.version(p))
    except Exception: pass
import torch; print('  cuda avail',torch.cuda.is_available(),'torch.cuda',torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"; }; done
  st env DONE
fi

train(){ # stage tag model mode lens steps
  [ -f $E/train/.ok ] || { echo "train env missing"; st $1 FAIL; return 1; }
  $E/train/bin/python -I $C/vista-train-smoke.py --model $2 --mode $3 --lens $4 --steps $5 > $L/train-$1-$SLURM_JOB_ID.log 2>&1
  grep -E "LOAD|LORA|fast paths|step|RESULT|SUMMARY|Error|error" $L/train-$1-$SLURM_JOB_ID.log | tail -${6:-25}
  grep -q SUMMARY $L/train-$1-$SLURM_JOB_ID.log && st $1 OK || { tail -25 $L/train-$1-$SLURM_JOB_ID.log; st $1 FAIL; return 1; }
}
has t4b && train t4b $NU bf16 16384,32768 5
if has t27q; then
  train t27q $Q27 nf4 8192,16384 3 || train t27fb $Q27 bf16 4096 3
fi
has t27b && train t27b $Q27 bf16 4096,8192 3

serve(){ # tag model name maxlen
  [ -f $E/serve/.ok ] || { echo "serve env missing"; st $1 FAIL; return 1; }
  # TRAP: ~17.8 GiB of the 95 GiB is unavailable at start (free 77.2 GiB), so 0.90 fails vLLM startup; 0.80 works
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 PATH=$E/serve/bin:$PATH setsid nohup $E/serve/bin/vllm serve $2 --served-model-name $3 --host 127.0.0.1 --port 8391 \
    --max-model-len $4 --gpu-memory-utilization 0.80 --max-num-seqs 16 --language-model-only > $L/vllm-$1-$SLURM_JOB_ID.log 2>&1 < /dev/null &
  SPID=$!; UP=0; S=$(date +%s)
  for i in $(seq 1 120); do curl -s -m 3 localhost:8391/v1/models >/dev/null && { UP=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 10; done
  [ $UP = 1 ] || { tail -30 $L/vllm-$1-$SLURM_JOB_ID.log; st $1 FAIL; kill $SPID 2>/dev/null; return 1; }
  echo "vllm up in $(( $(date +%s)-S )) s"; grep -E "KV cache|Maximum concurrency|Loading weights took|vLLM API|version" $L/vllm-$1-$SLURM_JOB_ID.log | head -8
  $E/serve/bin/python -I $C/vista-client.py http://127.0.0.1:8391/v1/chat/completions $3 2>&1 | grep -E "GUIDED_JSON|DECODE|Error"
  kill -INT $SPID; wait $SPID 2>/dev/null; sleep 5; st $1 OK
}
has serve4b && { serve serve4b $NU nuextract3 32768 || true; }
has serve27 && { serve serve27fp8 $Q27F q27fp8 32768 || true; }
has serve27b && { serve serve27bf16 $Q27 q27bf16 32768 || true; }
echo "ELAPSED $(( $(date +%s)-T0 )) s"
