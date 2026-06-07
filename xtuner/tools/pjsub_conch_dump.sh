#!/bin/bash
# =============================================================================
# CONCH WSI feature dump — full run on Genkai b-batch, 1 node, 4x H100.
# Launches 4 data-parallel shards (one process pinned per GPU, round-robin slides).
# Idempotent: re-submitting resumes (done slides skipped). Reference:
# 20260430_genkai_deployment.md (b-batch / gpu=4 / module load cuda/12.6.1).
#
# Submit from /home or /fast:
#   cd /home/.../weihao/00/reg_2026
#   CONCH_CKPT=$PWD/models/CONCH/pytorch_model.bin \
#     pjsub reg2026_slidechat_dev/xtuner/tools/pjsub_conch_dump.sh   # (use abs path)
#
# ⚠️ Set `elapse` from the interactive smoke timing before submitting the full run.
# =============================================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=4
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S
#PJM -o logs/conch_dump.%j.out

set -uo pipefail

module purge
module load cuda/12.6.1

# ---- paths (override via env / pjsub -x KEY=VAL) ----
H_HOME="${H_HOME:-/home/pj24003162/ku40003404/weihao/00}"
REG2026="${REG2026:-$H_HOME/reg_2026}"
SLIDECHAT="${SLIDECHAT:-$H_HOME/reg2026_slidechat_dev}"
REG_ROOT="${REG_ROOT:-$REG2026/data/reg2026}"           # contains train/ + train_thumb/
ENV_PREFIX="${ENV_PREFIX:-$REG2026/_envs/conch_dump_py311}"
CONCH_CKPT="${CONCH_CKPT:?set CONCH_CKPT to the CONCH pytorch_model.bin path}"
OUT="${OUT:-$REG_ROOT/train_feat_conch}"
NGPU="${NGPU:-4}"
CAP="${CAP:-20480}"
TISSUE_FRAC="${TISSUE_FRAC:-0.1}"
BATCH="${BATCH:-256}"

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export PYTHONPATH="$SLIDECHAT/xtuner/tools:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_HOME="$REG2026/_cache/huggingface"
export TMPDIR="${PJM_SSD_DIR:-$REG2026/tmp}"            # per-job local SSD scratch
mkdir -p "$OUT" logs "$TMPDIR"

echo "Job ${PJM_JOBID:-local}  host=$(hostname)  NGPU=$NGPU  $(date)"
echo "REG_ROOT=$REG_ROOT  OUT=$OUT  CONCH_CKPT=$CONCH_CKPT"
echo "CAP=$CAP TISSUE_FRAC=$TISSUE_FRAC BATCH=$BATCH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Pre-flight: data + masks must exist (fail fast, not 4 GPUs into the run).
[ -d "$REG_ROOT/train" ]       || { echo "ERR: $REG_ROOT/train missing"; exit 2; }
[ -d "$REG_ROOT/train_thumb" ] || { echo "ERR: $REG_ROOT/train_thumb (masks) missing"; exit 2; }
[ -f "$CONCH_CKPT" ]           || { echo "ERR: CONCH_CKPT not found"; exit 2; }

# ---- launch one shard per GPU ----
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="$g" python "$SLIDECHAT/xtuner/tools/extract_conch_features.py" \
    --root "$REG_ROOT" --conch "$CONCH_CKPT" --device cuda \
    --cap "$CAP" --tissue-frac "$TISSUE_FRAC" --batch "$BATCH" \
    --shard "$g" --n-shards "$NGPU" --out "$OUT" \
    > "logs/conch_dump.${PJM_JOBID:-local}.gpu${g}.log" 2>&1 &
  pids+=($!)
  echo "launched shard $g/$NGPU on GPU $g (pid ${pids[-1]})  -> logs/conch_dump.${PJM_JOBID:-local}.gpu${g}.log"
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

n_h5=$(ls "$OUT"/*.h5 2>/dev/null | wc -l)
echo "all shards done rc=$rc  h5_files=$n_h5  $(date)"
# Aggregate per-shard status counts (no_mask/no_tissue/err) for a quick health view.
for s in "$OUT"/_status_shard*of*.jsonl; do
  [ -f "$s" ] && echo "  $(basename "$s"): $(python -c "import sys,json,collections;print(dict(collections.Counter(json.loads(l)['status'].split(':')[0] for l in open(sys.argv[1]))))" "$s")"
done
exit $rc
