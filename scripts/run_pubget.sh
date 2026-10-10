#!/bin/bash
# Run pondie over the pubget bundle that scripts/bundle_pubget.py writes.
#
#   BUNDLE=/data/james/pubget-bundle RUN=pubget_all ENV_FILE=path/to/.env \
#     [WORKERS=20] [OFFSET=0] [LIMIT=n] [TIER=flex] [BACKEND=gateway] [MODEL=...] \
#     [STAGE_EFFORT="single=medium ..."] [PYTHON=python] bash scripts/run_pubget.sh
#
# OFFSET and LIMIT take a slice of $BUNDLE/pubget.pmids, so the corpus can be run in
# shards under one RUN. A re-run with the same RUN resumes: each stage whose inputs are
# unchanged is skipped. The run reads its own copy of each paper (runs/$RUN/corpus),
# because the split stage rewrites stage1/analyses.json. TIER is the service
# tier: flex (cheaper, slower), default, priority, or empty for the provider's default.
# BACKEND=codex sends the calls through `codex exec` on the `codex login` account
# instead of the gateway; it needs no ENV_FILE and sends no TIER.
# MODEL names the model (a gateway `@provider/model` name, or a codex one such as
# gpt-6.1-sol); STAGE_EFFORT is the space-separated --stage-effort map.
#
# The workflow is the one scored in experiments/stage-ablation: strict structured outputs,
# sentence-indexed evidence, then build and repair.
set -euo pipefail
: "${BUNDLE:?set BUNDLE to the bundle directory}" "${RUN:?set RUN to a run name}"
BACKEND=${BACKEND:-gateway}
ENV_ARGS=()
if [ "$BACKEND" = gateway ]; then
  : "${ENV_FILE:?set ENV_FILE to the file of API credentials}"
  ENV_ARGS=(--env "$ENV_FILE")
fi
PYTHON=${PYTHON:-python}
MODEL=${MODEL:-@psyc-aid338-ope-333f18/gpt-6-luna}
read -r -a EFFORT <<< "${STAGE_EFFORT:-single=medium fill=low evidence=low repair=medium}"
RUN_DIR=$BUNDLE/runs/$RUN
mkdir -p "$RUN_DIR/corpus"

PMIDS=$RUN_DIR/slice-${OFFSET:-0}-${LIMIT:-all}.pmids
# One awk, not grep | tail | head: under pipefail, head closing the pipe kills the script.
awk -v from="${OFFSET:-0}" -v n="${LIMIT:-}" \
  '!/^#/ && ++i > from && (n == "" || i <= from + n)' "$BUNDLE/pubget.pmids" > "$PMIDS"
while IFS=$'\t' read -r _pmid study _source; do
  [ -e "$RUN_DIR/corpus/$study" ] || cp -r "$BUNDLE/corpus/$study" "$RUN_DIR/corpus/$study"
done < "$PMIDS"
echo "$(date +%T) $RUN: $(wc -l < "$PMIDS") paper(s) from $PMIDS via $BACKEND, $MODEL"

PONDIE_DATA_DIR=$BUNDLE "$PYTHON" -m pondie.cli extract \
  --pmids "$PMIDS" --run "$RUN" --corpus "$RUN_DIR/corpus" --flavour best \
  --model "$MODEL" --backend "$BACKEND" ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
  --stages tables split single fill evidence build repair \
  --structured-outputs --evidence-format indexed \
  --service-tier "${TIER-flex}" --stage-effort "${EFFORT[@]}" \
  --workers "${WORKERS:-20}" --no-progress 2>&1 | tee -a "$RUN_DIR/run.log"
