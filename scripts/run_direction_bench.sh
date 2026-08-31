#!/usr/bin/env bash
# k=3 replicates of the recommended configuration over the reviewer-gold papers.
#
# Each replicate gets its own payload tree AND its own record directory. Sharing the record
# directory silently collapses the replicates onto each other: run_extraction writes built
# records to --examples, so three replicates pointed at one directory leave one set of
# records on disk and two replicates unscoreable.
#
# VERSION is the prompt revision under test, so a re-run never overwrites the arm it is
# being compared against.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
VERSION="${1:-v3}"
for k in 1 2 3; do
  out="data/direction-bench/$VERSION-rep$k"
  mkdir -p "$out/payloads" "$out/records"
  echo "=== $VERSION replicate $k -> $out ==="
  $PY -m pondie.extraction.passes.run_extraction \
      --pmids data/gold-direction-16.pmids \
      --texts data/texts \
      --payloads "$out/payloads" \
      --examples "$out/records" \
      --workflow demand-driven --zero-foci-rule --max-attempts 3 \
      --key-file .env > "$out/run.log" 2>&1
  echo "=== $VERSION replicate $k done (rc=$?) ==="
done
echo "ALL REPLICATES COMPLETE"
