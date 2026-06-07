#!/bin/bash
# =============================================================================
# PJM job: generate thumbnails + tissue masks for all REG2026 train slides.
# Prerequisite (Phase 1) for the CONCH dump — the dump reads {id}.tiff.mask.png.
# CPU-only (tifffile/zarr streaming, multiprocessing). Idempotent: existing
# outputs are skipped, so re-submitting resumes. ~5.6 h for 11220 slides.
#
# NOTE vs reg_2026/run_make_thumbnails.sh: that copy is MISSING `#PJM -L rscgrp`,
# so `pjsub` rejects it (GENKAI1003). This fixed copy sets the CPU group.
#
# Submit from /home or /fast:
#   cd /home/.../weihao/00/reg_2026
#   pjsub /home/.../weihao/00/reg2026_slidechat_dev/xtuner/tools/run_make_thumbnails.sh
#
# ⚠️ rscgrp=a-batch is the CPU-preprocessing group per 20260430_genkai_deployment.md.
#    Verify with `pjshowrsc --rg`; if your CPU group differs, override on the CLI:
#      pjsub -L rscgrp=<your-cpu-group> run_make_thumbnails.sh
# =============================================================================
#PJM -L rscgrp=a-batch
#PJM -L vnode-core=32
#PJM -L elapse=08:00:00
#PJM -j
#PJM -S
#PJM -o make_thumbnails.%j.out
# NOTE: no subdir in `-o` — PJM creates this file at submit time relative to the
# submit dir; `logs/...` would fail (GENKAI 0028) unless logs/ already exists.

set -uo pipefail
cd "${PJM_O_WORKDIR:-$PWD}" || true

# ---- paths (override via env / pjsub -x KEY=VAL) ----
H_HOME="${H_HOME:-/home/pj24003162/ku40003404/weihao/00}"
REG2026="${REG2026:-$H_HOME/reg_2026}"
SLIDECHAT="${SLIDECHAT:-$H_HOME/reg2026_slidechat_dev}"
REG_ROOT="${REG_ROOT:-$REG2026/data/reg2026}"            # holds train/ ; masks -> train_thumb/
ENV_PREFIX="${ENV_PREFIX:-$REG2026/_envs/conch_dump_py311}"   # has tifffile/zarr/imagecodecs

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export PYTHONUNBUFFERED=1
NCPU=$(nproc)
WORKERS=$(( NCPU > 32 ? 32 : (NCPU > 2 ? NCPU - 2 : 1) ))

echo "Job ${PJM_JOBID:-local}  host=$(hostname)  NCPU=$NCPU  WORKERS=$WORKERS  $(date)"
echo "REG_ROOT=$REG_ROOT"
[ -d "$REG_ROOT/train" ] || { echo "ERR: $REG_ROOT/train missing — set REG_ROOT"; exit 2; }

python "$SLIDECHAT/xtuner/tools/make_thumbnails.py" \
  --root "$REG_ROOT" \
  --workers "$WORKERS" \
  --target-long-side 2048 \
  --chunk 8192

echo "done $(date)"
echo "masks: $(ls "$REG_ROOT/train_thumb"/*.mask.png 2>/dev/null | wc -l) (expect ~11220)"
