#!/bin/bash
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
# usage: sbatch -J NAME -t HH:MM:SS -o $WORK/astra-distill/logs/%x-%j.out --export=ALL,RUN=<run>[,SMOKE_EVAL=<eval config>] chain.sbatch
# Trains configs/$RUN.json; with SMOKE_EVAL, runs that eval in this same job right after training (smoke tests).
# Continuation chunks and the eval are submitted from the login node with --dependency: Vista does not allow sbatch
# from a compute node. A soft stop (exit 75) fails afterok, so chain chunks with afterany.
set -u
source /work/07723/jdk3232/vista/astra-distill/code/env.sh
echo "node $(hostname) job $SLURM_JOB_ID run $RUN $(date)"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
$E/train/bin/python -I $C/train_record.py $C/configs/$RUN.json
rc=$?; echo "train exit $rc $(date)"
[ $rc -ne 0 ] && exit $rc
if [ -n "${SMOKE_EVAL:-}" ]; then
    $E/serve/bin/python -I $C/eval_record.py $C/configs/$SMOKE_EVAL.json; rc=$?; echo "smoke eval exit $rc $(date)"; exit $rc
fi
exit 0
