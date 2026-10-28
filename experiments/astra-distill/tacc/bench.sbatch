#!/bin/bash
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
# usage: sbatch -J bench-27b -t 01:55:00 -o $WORK/astra-distill/logs/%x-%j.out --export=ALL,CONFIGS="a b c d" bench.sbatch
# Inference speed benchmark: configs/bench-27b-<x>.json one after another on one node, sharing one merged model
# (keep_merged). A config that fails does not stop the rest. Per-config predictions: runs/r3-27b-bench/pred-<x>.jsonl.
set -u
source /work/07723/jdk3232/vista/astra-distill/code/env.sh
echo "node $(hostname) job $SLURM_JOB_ID configs ${CONFIGS:-a b c d} $(date)"
for x in ${CONFIGS:-a b c d}; do
  S=$(date +%s); echo "=== CONFIG $x start $(date +%T)"
  $E/serve/bin/python -I $C/eval_record.py $C/configs/bench-27b-$x.json; rc=$?
  echo "=== CONFIG $x exit $rc wall $(( $(date +%s)-S )) s"
  sleep 20   # let the GPU memory of the stopped server return
done
