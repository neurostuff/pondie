#!/bin/bash
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
# Smoke of train_record.py (4B, 27B) and eval_record.py (base NuExtract3; then the 4B smoke adapter through merge+serve).
set -u
source /work/07723/jdk3232/vista/astra-distill/code/env.sh
echo "node $(hostname) job $SLURM_JOB_ID $(date)"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
S="${STAGES:-train4b eval-base eval-merge train27}"
for s in $S; do
  case $s in
    train4b)    $E/train/bin/python -I $C/train_record.py $C/configs/smoke-4b.json ;;
    train27)    $E/train/bin/python -I $C/train_record.py $C/configs/smoke-27b.json ;;
    eval-base)  $E/serve/bin/python -I $C/eval_record.py $C/configs/smoke-eval-base.json ;;
    tput27)     $E/train/bin/python -I $C/train_record.py $C/configs/tput-27b.json ;;
    tput4)      $E/train/bin/python -I $C/train_record.py $C/configs/tput-4b.json ;;
    tput27p)    $E/train/bin/python -I $C/train_record.py $C/configs/tput2-27b.json ;;
    train27b)   $E/train/bin/python -I $C/train_record.py $C/configs/smoke2-27b.json ;;
    eval27)     $E/serve/bin/python -I $C/eval_record.py $C/configs/smoke-eval27.json ;;
    eval-merge) $E/serve/bin/python -I $C/eval_record.py $C/configs/smoke-eval-merge.json ;;
  esac
  echo "=== STAGE $s exit $? $(date)"
done
