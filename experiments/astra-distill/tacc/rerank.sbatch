#!/bin/bash
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A IBN26027
# usage: sbatch -J NAME -t HH:MM:SS -o $WORK/astra-distill/logs/%x-%j.out [-p gh-dev] --export=ALL,ARGS="<script> <args>" rerank.sbatch
# Runs code/<script> with the rerank env (sentence-transformers 6.2, torch 2.13 cu129), e.g.
#   ARGS="rerank_eval.py $W/data/rerank/bench $W/runs/rerank-zs bm25 BAAI/bge-reranker-v2-m3"
# ENV=elephant runs it with envs-aarch64/elephant (OneJev qev; models under $SCRATCH/astra-distill/hf-decision), for
#   ARGS="rerank_decision.py $W/data/rerank/bench $W/runs/rerank-zs"
set -u
source /work/07723/jdk3232/vista/astra-distill/code/env.sh
ENV=${ENV:-rerank}
[ $ENV = elephant ] && export DECISION_HUB=$SCRATCH/astra-distill/hf-decision/hub HF_HOME=$SCRATCH/astra-distill/hf-decision
echo "node $(hostname) job $SLURM_JOB_ID env $ENV args $ARGS $(date)"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
cd $C && $E/$ENV/bin/python -I $ARGS
rc=$?; echo "exit $rc $(date)"; exit $rc
