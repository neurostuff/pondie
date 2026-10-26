#!/bin/bash
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
# usage: sbatch -J NAME -t HH:MM:SS -o $WORK/astra-distill/logs/%x-%j.out [-p gh-dev] --export=ALL,RUN=<run> train.sbatch
# The run's hyperparameters are in $WORK/astra-distill/code/configs/$RUN.json, read at start.
set -u
source /work/07723/jdk3232/vista/astra-distill/code/env.sh
echo "node $(hostname) job $SLURM_JOB_ID run $RUN $(date)"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
$E/train/bin/python -I $C/train_record.py $C/configs/$RUN.json
rc=$?; echo "train exit $rc $(date)"; exit $rc
