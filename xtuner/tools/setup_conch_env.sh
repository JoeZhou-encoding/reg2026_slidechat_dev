#!/bin/bash
# =============================================================================
# Build the conda env for the CONCH WSI feature dump (REG2026 Metric A).
# Run on a Genkai LOGIN node (has internet). Reference: 20260430_genkai_deployment.md
#
#   bash reg2026_slidechat_dev/xtuner/tools/setup_conch_env.sh
#
# Creates a self-managed py3.11 env with torch(cu124) + CONCH deps (timm /
# transformers / huggingface_hub) + WSI dump deps (tifffile / zarr / imagecodecs /
# h5py). The CONCH code itself is the bundled xtuner/tools/conch (no pip install;
# put on PYTHONPATH). Does NOT touch the existing `reg_2026` env used by
# make_thumbnails.py.
# =============================================================================
set -euo pipefail

module purge 2>/dev/null || true
module load cuda/12.6.1 2>/dev/null || echo "[warn] 'module load cuda/12.6.1' failed (non-Genkai?) — continuing"

# ---- paths (override via env) ----
H_HOME="${H_HOME:-/home/pj24003162/ku40003404/weihao/00}"
REG2026="${REG2026:-$H_HOME/reg_2026}"
SLIDECHAT="${SLIDECHAT:-$H_HOME/reg2026_slidechat_dev}"
ENV_PREFIX="${ENV_PREFIX:-$REG2026/_envs/conch_dump_py311}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"          # H100 -> cu124
PY_VER="${PY_VER:-3.11}"

echo "[cfg] ENV_PREFIX=$ENV_PREFIX"
echo "[cfg] SLIDECHAT=$SLIDECHAT  TORCH_CUDA=$TORCH_CUDA  PY=$PY_VER"

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
[ -n "$CONDA_BASE" ] || { echo "ERROR: conda not found; set CONDA_BASE"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"

mkdir -p "$REG2026/_envs"
[ -d "$ENV_PREFIX" ] || conda create -y -p "$ENV_PREFIX" "python=$PY_VER"
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip wheel setuptools

# torch matching CUDA (mirrors 20260430 runbook: torch 2.6.0 / tv 0.21.0 / cu124)
echo "[pip] torch ($TORCH_CUDA)"
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# CONCH (bundled open_clip_custom) deps + WSI dump deps
echo "[pip] conch + dump deps"
python -m pip install \
  timm transformers huggingface_hub \
  tifffile zarr imagecodecs h5py \
  numpy pillow tqdm

# ---- verify (CPU side; CUDA is checked later on a GPU node) ----
export PYTHONPATH="$SLIDECHAT/xtuner/tools:${PYTHONPATH:-}"
echo; echo "=============== verify ==============="
python - <<'PY'
import importlib
for m in ["torch","torchvision","timm","transformers","tifffile","zarr","imagecodecs","h5py","numpy","PIL"]:
    mod = importlib.import_module(m)
    print(f"{m:14s} {getattr(mod,'__version__','')}")
from conch.open_clip_custom import create_model_from_pretrained  # noqa
print("conch.open_clip_custom import OK")
# tifffile<->zarr version-pair roundtrip (this pair bit us locally; catch at build time)
import numpy as np, tifffile, zarr, tempfile, os
p = os.path.join(tempfile.gettempdir(), "_conchenv_probe.tif")
tifffile.imwrite(p, (np.random.rand(512, 512, 3) * 255).astype("uint8"), tile=(256, 256))
arr = zarr.open(tifffile.imread(p, aszarr=True), mode="r")
print("tifffile+zarr roundtrip OK", arr.shape)
os.remove(p)
PY

echo "[done] env: $ENV_PREFIX"
cat <<EOF

=============== next ===============
1) Prefetch CONCH weights on this LOGIN node (CONCH is gated -> needs HF token):
     export HF_HOME=$REG2026/_cache/huggingface
     hf download MahmoodLab/CONCH --local-dir $REG2026/models/CONCH
   (or scp our local CONCH/pytorch_model.bin to $REG2026/models/CONCH/)
   -> CONCH_CKPT = $REG2026/models/CONCH/pytorch_model.bin

2) Interactive 1-GPU smoke (verify real CONCH on 1 slide + per-slide time):
     pjsub --interact -L rscgrp=b-inter,gpu=1,elapse=1:00:00
     # inside the interactive shell:
     CONCH_CKPT=$REG2026/models/CONCH/pytorch_model.bin \\
       bash $SLIDECHAT/xtuner/tools/smoke_conch_interactive.sh

3) Full dump (b-batch, 1 node, 4 GPU) — set elapse from the smoke timing first:
     cd $REG2026   # submit from /home
     CONCH_CKPT=$REG2026/models/CONCH/pytorch_model.bin \\
       pjsub $SLIDECHAT/xtuner/tools/pjsub_conch_dump.sh
EOF
