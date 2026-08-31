#!/usr/bin/env bash
# Stage 1 then extraction for the 10 papers added 2026-08-27.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
echo "=== stage 1: parse tables ==="
$PY -m pondie.extraction.passes.parse_tables --pmids data/labelstudio-add10.pmids \
    --texts data/texts --autonima .tmp_repos/autonima --key-file .env
echo "=== stage 1 rc=$? ==="
echo "=== extraction (demand-driven, recommended config) ==="
mkdir -p data/add10/payloads data/direction-bench/no-examples
$PY -m pondie.extraction.passes.run_extraction --pmids data/labelstudio-add10.pmids \
    --texts data/texts --payloads data/add10/payloads \
    --examples data/direction-bench/no-examples \
    --workflow demand-driven --zero-foci-rule --max-attempts 3 --key-file .env
echo "=== extraction rc=$? ==="
echo ADD10_COMPLETE
