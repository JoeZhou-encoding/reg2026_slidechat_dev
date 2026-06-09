#!/bin/bash
# =============================================================================
# Run the CONCH dump audit on Genkai (login node; NO GPU needed, read-only).
# Self-contained: every path hard-coded (matches the dump job). Just:
#   bash /home/pj24003162/ku40003404/weihao/00/reg2026_slidechat_dev/xtuner/tools/run_verify_dump.sh
# (ensure verify_dump.py is already on the server next to this file: git pull / scp)
# =============================================================================
set -uo pipefail

# ---- paths (identical to the dump job) ----
H_HOME=/home/pj24003162/ku40003404/weihao/00
REG2026=$H_HOME/reg_2026
SLIDECHAT=$H_HOME/reg2026_slidechat_dev
REG_ROOT=$REG2026/data/reg2026
ENV_PREFIX=$REG2026/_envs/conch_dump_py311
LOGDIR=$REG2026/logs
COT=$REG_ROOT/train_CoT.json          # adjust if train_CoT.json is elsewhere (coverage check is optional)

# ---- activate the dump env (PREFIX env -> activate by full path, not -n) ----
CONDA_BASE=""
for cb in "$(conda info --base 2>/dev/null || true)" "$H_HOME/miniconda3" /home/pj24003162/ku40003404/miniconda3; do
  [ -n "$cb" ] && [ -f "$cb/etc/profile.d/conda.sh" ] && { CONDA_BASE="$cb"; break; }
done
[ -n "$CONDA_BASE" ] || { echo "ERR: conda base not found"; exit 2; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX" || { echo "ERR: conda activate $ENV_PREFIX failed"; exit 2; }

# ---- sanity: env has the deps ----
python -c "import numpy, h5py; print('numpy', numpy.__version__, ' h5py', h5py.__version__)" \
  || { echo "ERR: numpy/h5py missing in $ENV_PREFIX"; exit 2; }
[ -f "$COT" ] || echo "NOTE: $COT not found -> id-coverage check will be skipped (harmless)"

# ---- run the audit, tee to a timestamped report ----
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)
REPORT="$LOGDIR/verify_dump.$TS.txt"
python "$SLIDECHAT/xtuner/tools/verify_dump.py" \
  --out   "$REG_ROOT/train_feat_conch" \
  --train "$REG_ROOT/train" \
  --masks "$REG_ROOT/train_thumb" \
  --logs  "$LOGDIR" \
  --submit-dir "$REG2026" \
  --cot   "$COT" \
  --n-shards 16 --feature-dim 512 --cap 20480 --train-sample-num 10240 --sample 24 \
  2>&1 | tee "$REPORT"
rc=${PIPESTATUS[0]}
echo "report -> $REPORT   (verify_dump exit rc=$rc)"
exit $rc
