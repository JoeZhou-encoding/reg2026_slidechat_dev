#!/bin/bash
# =============================================================================
# CONCH dump — MIG, HALF 0 of 2  (this job + pjsub_conch_dump_mig_h1.sh = full set).
# 8 MIG sub-jobs handle global shards 0..7 of N_SHARDS=16 (the FIRST half).
# pjsub_conch_dump_mig_h1.sh handles shards 8..15 (the SECOND half).
# Submit BOTH (two 8-MIG bulk jobs); disjoint shards, idempotent, resumable.
#
# SELF-CONTAINED: every variable is hard-defaulted in this file. Do NOT rely on
# `VAR=... pjsub` (PJM does NOT propagate the submit shell's environment).
#
#   cd /home/pj24003162/ku40003404/weihao/00/reg_2026     # submit from /home
#   pjsub <abs path>/pjsub_conch_dump_mig_h0.sh
#   pjsub <abs path>/pjsub_conch_dump_mig_h1.sh
# =============================================================================
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM --bulk --sparam 0-7
#PJM -j
#PJM -S
#PJM -o conch_dump_mig_h0.%j.out

set -uo pipefail
cd "${PJM_O_WORKDIR:-$PWD}" || true

module purge
module load cuda/12.6.1

# ---- this half (hard-coded; the only difference vs h1) ----
N_SHARDS=16              # global shards across BOTH jobs (8 here + 8 in h1)
SHARD_OFFSET=0          # h0 -> global shards 0..7 ; h1 sets 8

# ---- all paths hard-coded (no dependence on login-node env) ----
H_HOME=/home/pj24003162/ku40003404/weihao/00
REG2026=$H_HOME/reg_2026
SLIDECHAT=$H_HOME/reg2026_slidechat_dev
REG_ROOT=$REG2026/data/reg2026
ENV_PREFIX=$REG2026/_envs/conch_dump_py311
CONCH_CKPT=$REG2026/models/CONCH/pytorch_model.bin
OUT=$REG_ROOT/train_feat_conch
LOGDIR=$REG2026/logs
CAP=20480
TISSUE_FRAC=0.1
BATCH=256               # fits MIG 1g.12gb (verified); lower if OOM
NWORKERS=4              # MIG gpu=1 ~ 4 CPU cores

# ---- conda base, self-contained (don't assume `conda` is on PATH) ----
CONDA_BASE=""
for cb in "$(conda info --base 2>/dev/null || true)" "$H_HOME/miniconda3" /home/pj24003162/ku40003404/miniconda3; do
  [ -n "$cb" ] && [ -f "$cb/etc/profile.d/conda.sh" ] && { CONDA_BASE="$cb"; break; }
done
[ -n "$CONDA_BASE" ] || { echo "ERR: conda base not found"; exit 2; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX" || { echo "ERR: conda activate $ENV_PREFIX failed"; exit 2; }

export PYTHONPATH="$SLIDECHAT/xtuner/tools:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_HOME="$REG2026/_cache/huggingface"
export TMPDIR="${PJM_SSD_DIR:-$REG2026/tmp}"
mkdir -p "$OUT" "$LOGDIR" "$TMPDIR"

SHARD=$(( ${PJM_BULKNUM:-0} + SHARD_OFFSET ))     # global shard
echo "Job ${PJM_JOBID:-local} bulk=${PJM_BULKNUM:-0} -> global shard=$SHARD/$N_SHARDS host=$(hostname) BATCH=$BATCH $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Pre-flight (fail fast)
[ -d "$REG_ROOT/train" ]       || { echo "ERR: $REG_ROOT/train missing"; exit 2; }
[ -d "$REG_ROOT/train_thumb" ] || { echo "ERR: $REG_ROOT/train_thumb (masks) missing"; exit 2; }
[ -f "$CONCH_CKPT" ]           || { echo "ERR: CONCH_CKPT not found: $CONCH_CKPT"; exit 2; }

python "$SLIDECHAT/xtuner/tools/extract_conch_features.py" \
  --root "$REG_ROOT" --conch "$CONCH_CKPT" --device cuda \
  --cap "$CAP" --tissue-frac "$TISSUE_FRAC" --batch "$BATCH" --num-workers "$NWORKERS" \
  --shard "$SHARD" --n-shards "$N_SHARDS" --out "$OUT" \
  2>&1 | tee "$LOGDIR/conch_dump_mig.shard${SHARD}of${N_SHARDS}.${PJM_JOBID:-local}.log"
rc=${PIPESTATUS[0]}

echo "global shard $SHARD done rc=$rc  h5_total=$(ls "$OUT"/*.h5 2>/dev/null | wc -l)  $(date)"
exit $rc
