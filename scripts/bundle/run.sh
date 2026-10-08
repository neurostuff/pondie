#!/bin/bash
# Extract records for this bundle's papers through codex: gpt-6.1-sol, medium effort.
#
#   bash run.sh                          # every paper, 12 at a time
#   LIMIT=20 bash run.sh                 # the first 20
#   OFFSET=5000 LIMIT=5000 bash run.sh   # a shard; shards share one RUN
#
# Records land in runs/$RUN/records. Re-running with the same RUN resumes: a stage whose
# inputs are unchanged is skipped. MODEL, STAGE_EFFORT, WORKERS and RUN can be overridden.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -x "$HERE/.venv/bin/python" ] || { echo "no .venv: run bash $HERE/setup.sh first" >&2; exit 1; }

export BUNDLE=$HERE
export RUN=${RUN:-pubget_sol61_medium}
export BACKEND=codex
export MODEL=${MODEL:-gpt-6.1-sol}
export STAGE_EFFORT=${STAGE_EFFORT:-single=medium fill=medium evidence=medium repair=medium}
export WORKERS=${WORKERS:-12}
export PYTHON=$HERE/.venv/bin/python
export PONDIE_SCHEMA_DIR=$HERE/code/study_schema
exec bash "$HERE/code/scripts/run_pubget.sh"
