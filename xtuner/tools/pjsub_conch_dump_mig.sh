#!/bin/bash
# =============================================================================
# CONCH WSI feature dump — MIG variant (when full b-batch H100s are unavailable).
# PJM bulk job: each sub-job grabs ONE MIG slice and processes one round-robin
# shard. Shards run concurrently as MIG slices free up; idempotent + resumable.
#
# Same outputs as the 4-GPU version -> you can later switch to pjsub_conch_dump.sh
# (full b-batch) and it RESUMES (already-done .h5 are skipped). Mix freely.
#
# Submit from /home or /fast:
#   cd $REG2026
#   CONCH_CKPT=$REG2026/models/CONCH/pytorch_model.bin \
#     pjsub $SLIDECHAT/xtuner/tools/pjsub_conch_dump_mig.sh
#
# ⚠️ N_SHARDS below MUST equal the --sparam range size. Set elapse from the MIG
#    smoke timing. Lower BATCH if a MIG slice OOMs.
# =============================================================================
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM --bulk --sparam 0-7
#PJM -j
#PJM -S
#PJM -o conch_dump_mig.%j.out

set -uo pipefail
cd "${PJM_O_WORKDIR:-$PWD}" || true

module purge
module load cuda/12.6.1

H_HOME="${H_HOME:-/home/pj24003162/ku40003404/weihao/00}"
REG2026="${REG2026:-$H_HOME/reg_2026}"
SLIDECHAT="${SLIDECHAT:-$H_HOME/reg2026_slidechat_dev}"
REG_ROOT="${REG_ROOT:-$REG2026/data/reg2026}"
ENV_PREFIX="${ENV_PREFIX:-$REG2026/_envs/conch_dump_py311}"
CONCH_CKPT="${CONCH_CKPT:?set CONCH_CKPT to the CONCH pytorch_model.bin path}"
OUT="${OUT:-$REG_ROOT/train_feat_conch}"
LOGDIR="${LOGDIR:-$REG2026/logs}"
N_SHARDS="${N_SHARDS:-8}"          # MUST match the --sparam range size (0-7 => 8)
CAP="${CAP:-20480}"
TISSUE_FRAC="${TISSUE_FRAC:-0.1}"
BATCH="${BATCH:-128}"             # MIG has less memory than a full H100; lower if OOM
NWORKERS="${NWORKERS:-8}"         # parallel patch-decode workers (decode is the bottleneck)

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export PYTHONPATH="$SLIDECHAT/xtuner/tools:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_HOME="$REG2026/_cache/huggingface"
export TMPDIR="${PJM_SSD_DIR:-$REG2026/tmp}"
mkdir -p "$OUT" "$LOGDIR" "$TMPDIR"

SHARD="${PJM_BULKNUM:-0}"         # bulk index = this shard
echo "Job ${PJM_JOBID:-local} bulk=$SHARD/$N_SHARDS host=$(hostname) BATCH=$BATCH $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Pre-flight (fail fast)
[ -d "$REG_ROOT/train" ]       || { echo "ERR: $REG_ROOT/train missing"; exit 2; }
[ -d "$REG_ROOT/train_thumb" ] || { echo "ERR: $REG_ROOT/train_thumb (masks) missing"; exit 2; }
[ -f "$CONCH_CKPT" ]           || { echo "ERR: CONCH_CKPT not found"; exit 2; }

# One MIG slice = one visible device -> --device cuda. One shard, one process.
python "$SLIDECHAT/xtuner/tools/extract_conch_features.py" \
  --root "$REG_ROOT" --conch "$CONCH_CKPT" --device cuda \
  --cap "$CAP" --tissue-frac "$TISSUE_FRAC" --batch "$BATCH" --num-workers "$NWORKERS" \
  --shard "$SHARD" --n-shards "$N_SHARDS" --out "$OUT" \
  2>&1 | tee "$LOGDIR/conch_dump_mig.shard${SHARD}of${N_SHARDS}.${PJM_JOBID:-local}.log"
rc=${PIPESTATUS[0]}

echo "shard $SHARD done rc=$rc  h5_total=$(ls "$OUT"/*.h5 2>/dev/null | wc -l)  $(date)"
exit $rc
