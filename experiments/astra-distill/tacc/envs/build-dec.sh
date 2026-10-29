#!/bin/bash
# d3 + Nimble runtime: transformers 5.17.0 as both pin, with the aarch64 torch the other Vista envs use
set -e
W=$WORK/astra-distill; E=$W/envs-aarch64/dec517; UV=$W/bin/uv
export RAYON_NUM_THREADS=4 UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_BUILDS=2 UV_CONCURRENT_INSTALLS=4
export UV_CACHE_DIR=$W/cache/uv UV_PYTHON_INSTALL_DIR=$W/python
[ -d $E ] || $UV venv -q -p 3.12 $E
$UV pip install -q -p $E/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 \
  torch==2.13.0+cu129 torchvision transformers==5.17.0 peft==0.21.0 accelerate==1.15.0 flash-linear-attention==0.5.2 \
  einops safetensors sentencepiece==0.2.2 pillow opencv-python-headless "huggingface_hub>=1.0"
cd /tmp && PYTHONPATH=$W/ext/nimble $E/bin/python -c "import torch, torchvision, cv2, fla, transformers, peft, nimble.scoring.cuda_scorer; print('dec517 ok', torch.__version__, torchvision.__version__, transformers.__version__)"
