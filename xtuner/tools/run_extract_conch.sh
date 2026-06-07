#!/bin/bash
# =============================================================================
# PJM bulk job: CONCH WSI feature dump (tissue-grid, capped) for REG2026 Metric A.
# Each bulk sub-job (PJM_BULKNUM) processes a round-robin shard of the slides on
# 1 GPU. Idempotent: re-submitting resumes (done slides are skipped).
#
# PREREQUISITES (login node, once):
#   1. train_thumb/{id}.tiff.mask.png exists for all slides
#        -> pjsub run_make_thumbnails.sh   (Phase 1, ~5.6 h)
#   2. CONCH checkpoint downloaded:
#        hf download MahmoodLab/CONCH    (or point CONCH_CKPT at pytorch_model.bin)
#   3. conda env `reg_2026` has: torch + tifffile + zarr + h5py + imagecodecs
#        + CONCH deps (einops, ...). The bundled conch is on PYTHONPATH below.
#
# ⚠️ FILL IN the GPU resource line (rscunit / rscgrp) from `pjshowrsc` for this
#    account — left as a placeholder on purpose (do not guess the partition).
# =============================================================================
#PJM -L "rscunit=FILL_ME,rscgrp=FILL_ME-gpu"
#PJM -L "vnode=1"
#PJM -L "vnode-core=8"
#PJM -L "elapse=48:00:00"
#PJM --bulk --sparam "0-7"          # 8 shards: PJM_BULKNUM = 0..7  (must match N_SHARDS)
#PJM -j -N reg2026_conch

set -uo pipefail

# ---- paths (override via env / pjsub -x) ------------------------------------
H_HOME="${H_HOME:-/home/pj24003162/ku40003404/weihao/00}"
REG_ROOT="${REG_ROOT:-$H_HOME/reg_2026/data/reg2026}"
SLIDECHAT="${SLIDECHAT:-$H_HOME/reg_2026/src/reg2026_slidechat_dev}"   # repo on Genkai
CONCH_CKPT="${CONCH_CKPT:?set CONCH_CKPT to the CONCH pytorch_model.bin path}"
N_SHARDS="${N_SHARDS:-8}"
CAP="${CAP:-20480}"
TISSUE_FRAC="${TISSUE_FRAC:-0.1}"
BATCH="${BATCH:-256}"

CONDA_BASE_DIR="${CONDA_BASE_DIR:-/home/pj24003162/ku40003404/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-reg_2026}"

cd "$PJM_O_WORKDIR" 2>/dev/null || true
echo "Job $PJM_JOBID  bulk=$PJM_BULKNUM/$N_SHARDS  host=$(hostname)  $(date)"

# ---- conda ------------------------------------------------------------------
if [ -f "${CONDA_BASE_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found"; exit 1
fi
conda activate "${CONDA_ENV_NAME}" || { echo "conda activate failed"; exit 1; }

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1                       # compute node has no internet
# bundled CONCH (xtuner/tools/conch) must be importable as `conch`
export PYTHONPATH="${SLIDECHAT}/xtuner/tools:${PYTHONPATH:-}"

echo "REG_ROOT=$REG_ROOT  CONCH_CKPT=$CONCH_CKPT"
echo "N_SHARDS=$N_SHARDS CAP=$CAP TISSUE_FRAC=$TISSUE_FRAC BATCH=$BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- run --------------------------------------------------------------------
python "${SLIDECHAT}/xtuner/tools/extract_conch_features.py" \
    --root "$REG_ROOT" \
    --conch "$CONCH_CKPT" \
    --device cuda \
    --cap "$CAP" \
    --tissue-frac "$TISSUE_FRAC" \
    --batch "$BATCH" \
    --shard "$PJM_BULKNUM" \
    --n-shards "$N_SHARDS" \
    --out "$REG_ROOT/train_feat_conch"

echo "Done bulk=$PJM_BULKNUM at $(date)"
