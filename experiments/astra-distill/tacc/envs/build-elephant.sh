#!/bin/bash
# elephant-1 runtime (OneJev qev @81ce62f + the checkpoint serve.patch), as its setup.sh, in a uv venv with the
# aarch64 torch the other Vista envs use (2.13.0+cu129) instead of the x86 docker image; causal-conv1d (a source
# build) left out. Also clones the Nimble scorer repo for the Nimble job.
set -e
W=$WORK/astra-distill; X=$W/ext; E=$W/envs-aarch64/elephant; UV=$W/bin/uv
export UV_CACHE_DIR=$W/cache/uv UV_PYTHON_INSTALL_DIR=$W/python
S=$(ls -d $SCRATCH/astra-distill/hf-decision/hub/models--baselabs--elephant-1/snapshots/*/)
cd $X
[ -d nimble ] || git clone -q https://github.com/bespokelabsai/nimble.git
echo "nimble $(git -C nimble rev-parse HEAD)"
[ -d OneJev ] || git clone -q https://github.com/OmniJev/OneJev.git
git -C OneJev checkout -q -f 81ce62f1597c91e46767d4d02d6ac2e18534fe94 && git -C OneJev clean -fdq && git -C OneJev apply "$S/serve.patch"
[ -d $E ] || $UV venv -q -p 3.12 $E
grep -v -E "^causal-conv1d==" $S/constraints.txt > constraints-elephant.txt
$UV pip install -q -p $E/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
  -c constraints-elephant.txt torch==2.13.0+cu129 -e "./OneJev[train]" flash-linear-attention
echo installed
