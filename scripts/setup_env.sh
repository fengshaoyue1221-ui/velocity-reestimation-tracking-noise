#!/usr/bin/env bash
# One-shot environment setup for the A100 experiments. No sudo needed.
# Usage (from the kit root):  bash scripts/setup_env.sh
#   Fallback if torch cannot see the GPU with the default cu117 build (old driver):
#   TORCH_FLAVOR=cu113 bash scripts/setup_env.sh
# Produces: conda env "mtr", MTR cloned + patched + compiled in ./MTR, and a check log.
set -eo pipefail   # no -u: conda activation scripts reference unset variables
KIT=$(cd "$(dirname "$0")/.." && pwd)
ENV=${ENV_NAME:-mtr}
LOG=$KIT/setup_env.log
exec > >(tee -a "$LOG") 2>&1
echo "=== setup started $(date)"

# ---- 0. conda (installs Miniconda into $HOME if missing) ----
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found -> installing Miniconda to \$HOME/miniconda3"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
set +u; eval "$(conda shell.bash hook)"; set -u

# ---- 1. env ----
if ! conda env list | grep -qE "^$ENV\s"; then
  conda create -y -n "$ENV" python=3.8
fi
set +u; conda activate "$ENV"; set -u
python -m pip install -q "pip<24.1" setuptools wheel

# ---- 2/3. CUDA compiler + pinned Python packages ----
# default: torch 1.13.1+cu117 from PyPI, nvcc 11.7 (works on driver >= 450.80 via CUDA minor-version compatibility)
# fallback: TORCH_FLAVOR=cu113 -> torch 1.12.1+cu113, nvcc 11.3 (matches drivers that report CUDA 11.4, e.g. 470.xx)
FLAVOR=${TORCH_FLAVOR:-cu117}
if [ "$FLAVOR" = cu113 ]; then CUDA_VER=11.3; CUDA_LABEL=cuda-11.3.1; else CUDA_VER=11.7; CUDA_LABEL=cuda-11.7.1; fi
echo "torch flavor: $FLAVOR (nvcc $CUDA_VER)"
if ! command -v nvcc >/dev/null 2>&1 || ! nvcc --version | grep -q "release $CUDA_VER"; then
  conda install -y -c "nvidia/label/$CUDA_LABEL" cuda-toolkit --repodata-fn repodata.json
fi
export CUDA_HOME=$CONDA_PREFIX
nvcc --version | tail -1
# CUDA 11.x supports gcc <= 11
command -v gcc >/dev/null 2>&1 || conda install -y -c conda-forge gxx_linux-64=11 gcc_linux-64=11 --repodata-fn repodata.json
GCC_MAJOR=$( (command -v gcc >/dev/null && gcc -dumpversion || echo 0) | cut -d. -f1)
if [ "$GCC_MAJOR" -gt 11 ]; then
  echo "gcc $GCC_MAJOR too new for CUDA $CUDA_VER -> installing gcc 11 into the env"
  conda install -y -c conda-forge gxx_linux-64=11 gcc_linux-64=11 --repodata-fn repodata.json
  export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
fi

if [ "$FLAVOR" = cu113 ]; then
  grep -vE '^(torch|nvidia-[a-z0-9-]*cu11)==' "$KIT/scripts/requirements_a100.lock.txt" > /tmp/req_no_torch.txt
  pip install -r /tmp/req_no_torch.txt
  pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
else
  pip install -r "$KIT/scripts/requirements_a100.lock.txt"
fi

# GPU visible to torch? (needed before compiling ops for the right architecture)
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "!!! torch $FLAVOR cannot use the GPU on this driver."
  [ "$FLAVOR" = cu117 ] && echo "!!! Re-run with:  TORCH_FLAVOR=cu113 bash scripts/setup_env.sh"
  exit 1
fi
ARCH=$(python -c "import torch;c=torch.cuda.get_device_capability(0);print(f'{c[0]}.{c[1]}')")
echo "GPU compute capability: $ARCH"

# ---- 4. MTR: clone, patch, compile ----
cd "$KIT"
if [ ! -d MTR ]; then git clone https://github.com/sshaoshuai/MTR.git; fi
cd MTR
if ! grep -q PerceptionNoise mtr/datasets/waymo/waymo_dataset.py; then
  git apply --whitespace=nowarn "$KIT/mtr_perception_noise.patch"
fi
TORCH_CUDA_ARCH_LIST="$ARCH" python setup.py develop

# ---- 5. checks (all must pass) ----
echo "=== checks"
python - <<'EOF'
import numpy as np, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available to torch"
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
t = torch.from_numpy(np.zeros(3, dtype=np.float32)).cuda()   # numpy<->torch ABI check
print("numpy", np.__version__, "interop OK", t.sum().item() == 0)
import mtr.ops.knn.knn_utils, mtr.ops.attention.attention_utils_v2
print("MTR CUDA ops import OK")
import tensorflow as tf
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import motion_metrics_pb2
print("tensorflow", tf.__version__, "+ waymo motion metrics ops OK")
EOF
python tests/test_perception_noise.py
echo "=== setup finished OK $(date)"
