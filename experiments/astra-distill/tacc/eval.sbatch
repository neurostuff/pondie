#!/bin/bash
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
# usage: sbatch -J NAME -t HH:MM:SS -o $WORK/astra-distill/logs/%x-%j.out [-p gh-dev] --export=ALL,EVAL=<eval config name> eval.sbatch
# Config: $WORK/astra-distill/code/configs/$EVAL.json (merge adapter -> serve with vLLM -> guided greedy on val -> runs/<run>/val-pred.jsonl)
set -u
source /work/07723/jdk3232/vista/astra-distill/code/env.sh
echo "node $(hostname) job $SLURM_JOB_ID eval $EVAL $(date)"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
$E/serve/bin/python -I $C/eval_record.py $C/configs/$EVAL.json
rc=$?; echo "eval exit $rc $(date)"; exit $rc
