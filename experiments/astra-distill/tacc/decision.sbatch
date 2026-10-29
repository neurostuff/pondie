#!/bin/bash
#SBATCH -A IBN26027
#SBATCH -N 1
#SBATCH -n 1
# Decision-model judge of the evidence pilot set (vista/evjudge_decision.py), one model per job:
#   sbatch -p gh-dev -t 00:30:00 -J dec-<m> -o $WORK/astra-distill/logs/%x-%j.out --export=ALL,DEC=<m> decision.sbatch
# DEC: elephant (envs-aarch64/elephant: OneJev qev) | nimble (envs-aarch64/dec517 + its decision-index repo code/) | d3 (dec517)
set -eo pipefail
echo "node $(hostname) job $SLURM_JOB_ID dec $DEC $(date)"
source $WORK/astra-distill/code/env.sh
export DECISION_HUB=$SCRATCH/astra-distill/hf-decision/hub HF_HOME=$SCRATCH/astra-distill/hf-decision
J=$W/data/ev1/judge
case $DEC in
  elephant) PY=$E/elephant/bin/python ;;
  nimble) PY=$E/dec517/bin/python; export PYTHONPATH=$W/ext/nimble-v3-index/code ;;
  d3) PY=$E/dec517/bin/python ;;
  *) echo "DEC must be elephant, nimble or d3"; exit 2 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY $C/vista/evjudge_decision.py $DEC $J/pilot.jsonl $J/$DEC.jsonl
