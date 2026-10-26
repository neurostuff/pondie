#!/bin/bash
#SBATCH -J astra-smoke
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A IBN26027
#SBATCH -o /work/07723/jdk3232/ls6/astra-distill/logs/smoke-%j.out
# Lonestar6 smoke test for astra-distill: stack build, merge, vLLM guided-JSON effects on a slice (vs beast), 8 QLoRA steps.
# Stages print "=== STAGE <name> OK|FAIL" so a failure points at its stage. Idempotent: existing envs/weights/merged are reused.
# Template: change W, the slice, the adapter and the stage block you need. Slurm rejects --gres/--mem; node = 3x A100 40 GB.
set -u
W=$WORK/astra-distill; L=$W/logs; mkdir -p $L
export UV_CACHE_DIR=$W/cache/uv UV_PYTHON_INSTALL_DIR=$W/python UV_LINK_MODE=copy PIP_CACHE_DIR=$W/cache/pip
export HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PATH=$W/bin:$PATH
ROOT=$W/data/slice; CODE=$W/code; AD=$W/models/adapter-ctx2-s0; MM=$W/models/merged-ctx2-s0
T0=$(date +%s); st(){ echo "=== STAGE $1 $2 ($(( $(date +%s)-T0 )) s since start) $(date +%T)"; }
fail(){ st "$1" FAIL; echo "$(date) job $SLURM_JOB_ID failed at $1"; kill $SPID 2>/dev/null; exit 1; }
SPID=""

# --- 0 environment
echo "node $(hostname)  job $SLURM_JOB_ID  OMP_NUM_THREADS=${OMP_NUM_THREADS:-} nproc_all=$(nproc --all)"; taskset -p $$; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi | head -4 | tail -2; module list 2>&1 | tail -3; nproc; free -g | head -2; df -h /tmp | tail -1
if curl -sI -m 15 https://huggingface.co | head -1 | grep -q " 200\| 30"; then NET=1; else NET=0; fi
echo "INTERNET_FROM_COMPUTE_NODE=$NET"; curl -sI -m 15 https://pypi.org | head -1; curl -sI -m 15 https://download.pytorch.org | head -1
st env OK

# --- 1 base weights
if [ -f $W/hf/hub/models--numind--NuExtract3/refs/main ]; then echo "weights already cached"
elif [ $NET = 1 ]; then
  [ -x $W/envs/hfdl/bin/hf ] || { $W/bin/uv venv -q --python 3.12 $W/envs/hfdl && VIRTUAL_ENV=$W/envs/hfdl $W/bin/uv pip install -q huggingface_hub; }
  H=$W/envs/hfdl/bin/hf
  $H download numind/NuExtract3 > $L/hfdl.log 2>&1 || fail weights
else echo "no internet and no cached weights"; fail weights; fi
du -sh $W/hf; st weights OK
export HF_HUB_OFFLINE=1

# --- 2 venvs (versions = beast's; serve = /home/james/vllm-venv, train = venv-train, score = astra-distill/venv tokenizer use)
mkvenv(){ # name python
  [ -x $W/envs/$1/bin/python ] || $W/bin/uv venv -q --python $2 $W/envs/$1 || return 1; }
# NOTE: PyPI's vllm 0.28.0 is a CUDA 13 build (ImportError libcudart.so.13 on the 12.8 driver); beast's is the +cu129 build from wheels.vllm.ai
( [ -f $W/envs/serve/.ok ] || { mkvenv serve 3.12 && VIRTUAL_ENV=$W/envs/serve $W/bin/uv pip install "vllm==0.28.0+cu129" transformers==5.16.1 xgrammar==0.2.3 \
    flashinfer-python==0.6.16.post3 --extra-index-url https://wheels.vllm.ai/0.28.0/cu129 --index-strategy unsafe-best-match \
    --torch-backend cu129 && touch $W/envs/serve/.ok; } ) > $L/venv-serve.log 2>&1 &
P1=$!
( [ -f $W/envs/train/.ok ] || mkvenv train 3.10 && VIRTUAL_ENV=$W/envs/train $W/bin/uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
  && VIRTUAL_ENV=$W/envs/train $W/bin/uv pip install transformers==5.17.0 peft==0.21.0 bitsandbytes==0.50.2 trl==1.13.0 accelerate==1.15.0 \
    liger-kernel==0.8.3 datasets==5.0.1 safetensors==0.8.0 numpy==2.2.6 huggingface-hub flash-linear-attention==0.5.2 && touch $W/envs/train/.ok ) > $L/venv-train.log 2>&1 &
P2=$!
( [ -f $W/envs/score/.ok ] || mkvenv score 3.12 && VIRTUAL_ENV=$W/envs/score $W/bin/uv pip install transformers==5.19.0 numpy && touch $W/envs/score/.ok ) > $L/venv-score.log 2>&1 &
P3=$!
wait $P1; wait $P2; wait $P3
for e in serve train score; do [ -f $W/envs/$e/.ok ] || { tail -5 $L/venv-$e.log; fail venv-$e; }; done
st venvs OK
for e in serve train score; do echo "== $e"; $W/envs/$e/bin/python -I -c "
import importlib.metadata as m,sys
print(sys.version.split()[0])
for p in ['torch','vllm','transformers','peft','bitsandbytes','trl','accelerate','xgrammar','liger-kernel','flashinfer-python','flash-linear-attention','triton','numpy']:
    try: print(' ',p,m.version(p))
    except Exception: pass"; done

# --- 3 merge adapter into base (CPU, train venv), as nx/merge_prose.py
if [ ! -f $MM/config.json ]; then
  CUDA_VISIBLE_DEVICES="" $W/envs/train/bin/python $CODE/merge_prose.py $AD $MM > $L/merge-$SLURM_JOB_ID.log 2>&1 || { tail -8 $L/merge-$SLURM_JOB_ID.log; fail merge; }
fi
tail -3 $L/merge-$SLURM_JOB_ID.log 2>/dev/null; st merge OK

# --- 4 serve merged model on GPU 0 (same flags as nx/serve.sh but 1 GPU, no TP) and run the effects client on the slice
# VLLM_USE_FLASHINFER_SAMPLER=0: flashinfer's sampler JIT-compiles with nvcc, which LS6 compute nodes lack on PATH (RuntimeError: Could not find nvcc)
stage4(){
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$W/envs/serve/bin:$PATH setsid nohup $W/envs/serve/bin/vllm serve $MM \
  --served-model-name nuextract3 --host 127.0.0.1 --port 8381 --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 --max-num-batched-tokens 8192 --language-model-only --no-enable-prefix-caching > $L/vllm-$SLURM_JOB_ID.log 2>&1 < /dev/null &
SPID=$!
UP=0; for i in $(seq 1 120); do curl -s -m 3 localhost:8381/v1/models >/dev/null && { UP=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 10; done
[ $UP = 1 ] || { tail -30 $L/vllm-$SLURM_JOB_ID.log; st serve FAIL; return 1; }
st serve OK
mkdir -p $ROOT/results; rm -rf $ROOT/nx/runs/effects-ls6; NX_ROOT=$ROOT $W/envs/score/bin/python $CODE/nx/effects_nx.py infer ls6 --url http://127.0.0.1:8381/v1/chat/completions \
  --workers 32 > $L/effects-$SLURM_JOB_ID.log 2>&1 || { tail $L/effects-$SLURM_JOB_ID.log; kill -INT $SPID; return 1; }
head -c 600 $ROOT/nx/runs/effects-ls6/raw.jsonl; echo; cat $ROOT/nx/runs/effects-ls6/stats.json; echo
$W/envs/score/bin/python $CODE/tacc/compare_effects.py $ROOT/ref-raw.jsonl $ROOT/nx/runs/effects-ls6/raw.jsonl > $L/compare-$SLURM_JOB_ID.txt; cat $L/compare-$SLURM_JOB_ID.txt
kill -INT $SPID; wait $SPID 2>/dev/null; SPID=""; sleep 5
st effects OK
}
# a failed serve/effects stage must not hide the training check
stage4 || { st effects FAIL; kill -INT $SPID 2>/dev/null; SPID=""; }

# --- 5 QLoRA training, 8 optimizer steps of the beast recipe on GPU 0 (1 card: --accum 4 as beast's per-card accum)
$W/envs/score/bin/python $CODE/tacc/patch_train.py $CODE/nx/train_prose.py $CODE/tacc/train_prose_smoke.py
rm -rf $W/runs/smoke-train
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $W/envs/train/bin/torchrun --nproc_per_node=1 --master_port 29597 \
  $CODE/tacc/train_prose_smoke.py --model numind/NuExtract3 --data $ROOT/train-slice.jsonl --out $W/runs/smoke-train \
  --lr 1e-4 --slow-lr-div 1 --accum 4 --max-tokens 4500 --save-every 100000 --log-every 1 --no-trackio --epochs 0.7 --seed 0 \
  > $L/train-$SLURM_JOB_ID.log 2>&1 || { tail -30 $L/train-$SLURM_JOB_ID.log; fail train; }
grep -v "Loading weights" $L/train-$SLURM_JOB_ID.log | tail -30
st train OK
echo "ELAPSED $(( $(date +%s)-T0 )) s"
