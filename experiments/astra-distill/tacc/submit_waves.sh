#!/bin/bash
# usage: submit_waves.sh PREFIX FIRST LAST [MAX_QUEUED] [TIME]
#   e.g. submit_waves.sh ns-27b-e 1 84 30 01:15:00
# Run from a machine with the `vista` ssh master open. Submits configs/<PREFIX>-sNNN.json as gh jobs named
# ns-e-sNNN, keeping at most MAX_QUEUED of them pending or running (the gh limit is 40 submitted per user,
# shared with other work). Submitted shards are appended to $WORK/astra-distill/logs/<PREFIX>.submitted, which
# is read first, so a restart never submits a shard twice. Exits when every shard is submitted, or on an ssh
# failure.
set -u
PREFIX=$1 FIRST=$2 LAST=$3 MAX=${4:-30} TIME=${5:-01:15:00}
for k in $(seq "$FIRST" "$LAST"); do
  shard=$(printf "s%03d" "$k")
  while :; do
    state=$(timeout 60 ssh -o BatchMode=yes vista "grep -qx $shard \$WORK/astra-distill/logs/$PREFIX.submitted 2>/dev/null && echo done; squeue -h -u \$USER -n \$(printf 'ns-e-s%03d,' \$(seq 0 999) | sed 's/,\$//') -o %i | wc -l; echo ok" 2>/dev/null)
    echo "$state" | grep -q '^ok' || { echo "$(date +%T) ssh failed"; exit 1; }
    echo "$state" | grep -q '^done' && { echo "$(date +%T) $shard already submitted"; continue 2; }
    queued=$(echo "$state" | grep -v '^ok' | tail -1)
    [ "$queued" -lt "$MAX" ] && break
    sleep 300
  done
  out=$(timeout 60 ssh -o BatchMode=yes vista "cd \$WORK/astra-distill/code && sbatch -p gh -J ns-e-$shard -t $TIME -o \$WORK/astra-distill/logs/%x-%j.out --export=ALL,EVAL=$PREFIX-$shard eval.sbatch && echo $shard >> \$WORK/astra-distill/logs/$PREFIX.submitted" 2>&1 | grep -o 'Submitted batch job [0-9]*')
  [ -n "$out" ] || { echo "$(date +%T) $shard: submit failed"; exit 1; }
  echo "$(date +%T) $shard: $out"
done
echo "$(date +%T) all shards $FIRST-$LAST submitted"
