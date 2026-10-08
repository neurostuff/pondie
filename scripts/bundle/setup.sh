#!/bin/bash
# Create this bundle's Python environment (.venv) and check that codex can run.
#
#   bash setup.sh            # uv when it is installed, else python3 -m venv + pip
#
# The code is installed editable because it finds its schema (code/study_schema) next to
# its own source. scispacy is the abbreviation detector `build` and `repair` use; without
# it a weaker miner runs and records differ.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

if command -v uv >/dev/null; then
  uv venv --python ">=3.11" "$HERE/.venv"
  uv pip install --python "$HERE/.venv/bin/python" -e "$HERE/code[abbreviations]"
else
  "${PYTHON:-python3}" -m venv "$HERE/.venv"
  "$HERE/.venv/bin/pip" install --quiet --upgrade pip
  "$HERE/.venv/bin/pip" install -e "$HERE/code[abbreviations]"
fi
"$HERE/.venv/bin/python" -c "import pondie, scispacy; print('pondie and scispacy import')"

if ! command -v codex >/dev/null; then
  echo "codex is not installed: npm install -g @openai/codex, then codex login" >&2
  exit 1
fi
codex login status || { echo "codex is not logged in: run codex login" >&2; exit 1; }
echo "ready: bash $HERE/run.sh"
