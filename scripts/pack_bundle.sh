#!/bin/bash
# Make a bundle portable: write this checkout's code and schema, and the bundle's own
# setup/run scripts, into a directory that bundle_pubget.py fills with the corpus.
#
#   bash scripts/pack_bundle.sh OUT_DIR
#
# The code is `git archive HEAD`, so only committed files travel, and the schema is the
# study_schema submodule at the commit HEAD pins, read from SCHEMA_REPO (default: the
# submodule checkout). MANIFEST.json records both commits.
set -euo pipefail
OUT=${1:?usage: pack_bundle.sh OUT_DIR}
REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
SCHEMA=${SCHEMA_REPO:-$REPO/study_schema}
PINNED=$(git ls-tree HEAD study_schema | awk '{print $3}')
[ "$(git -C "$SCHEMA" rev-parse HEAD)" = "$PINNED" ] || {
  echo "study_schema is not at the commit HEAD pins ($PINNED): git submodule update" >&2
  exit 1
}

mkdir -p "$OUT/code"
git archive HEAD | tar -x -C "$OUT/code"
rm -rf "$OUT/code/study_schema" && mkdir "$OUT/code/study_schema"
git -C "$SCHEMA" archive "$PINNED" | tar -x -C "$OUT/code/study_schema"
cp scripts/bundle/setup.sh scripts/bundle/run.sh scripts/bundle/README.md "$OUT/"
papers=null
[ -f "$OUT/pubget.pmids" ] && papers=$(grep -vc "^#" "$OUT/pubget.pmids")
cat > "$OUT/MANIFEST.json" <<EOF
{
  "code_commit": "$(git rev-parse HEAD)",
  "schema_commit": "$PINNED",
  "packed": "$(date -u +%FT%TZ)",
  "papers": $papers
}
EOF
echo "packed $(git rev-parse --short HEAD) into $OUT"
