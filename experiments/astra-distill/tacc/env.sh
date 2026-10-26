# sourced by every Vista job script (code lives in $WORK/astra-distill/code, so edits apply when queued jobs start)
W=/work/07723/jdk3232/vista/astra-distill; L=$W/logs; E=$W/envs-aarch64; C=$W/code; mkdir -p $L $W/runs
export UV_CACHE_DIR=$W/cache/uv HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PATH=$W/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CC=gcc CXX=g++                      # trap 1: the default CC=nvc breaks triton's JIT
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0   # traps for vLLM
export ASTRA_W=$W ASTRA_SCRATCH=${SCRATCH:-/scratch/07723/jdk3232}/astra-distill
# Kernel caches on the node, one per job: Triton defaulted to ~/.triton/cache on /home1 (NFS), shared by every job;
# three jobs compiling at once collided and h34-3k died reading a file another rewrote (OSError 116, 2026-10-25).
export TRITON_CACHE_DIR=/tmp/triton-${SLURM_JOB_ID:-$$}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-${SLURM_JOB_ID:-$$}
mkdir -p $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR
