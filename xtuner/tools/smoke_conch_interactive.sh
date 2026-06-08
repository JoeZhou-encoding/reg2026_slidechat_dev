#!/bin/bash
# =============================================================================
# Interactive-node smoke for the CONCH WSI feature dump (REG2026 Metric A).
# Verifies REAL CONCH on a few real slides end-to-end + reports per-slide time,
# WITHOUT submitting a batch job. Run INSIDE an interactive GPU job:
#
#   pjsub --interact -L rscgrp=b-inter,gpu=1,elapse=1:00:00
#   # then inside:
#   CONCH_CKPT=<path/to/pytorch_model.bin> \
#     bash reg2026_slidechat_dev/xtuner/tools/smoke_conch_interactive.sh
#
# Output goes to a throwaway dir ($OUT), NOT the real train_feat_conch.
# =============================================================================
set -uo pipefail

module purge 2>/dev/null || true
module load cuda/12.6.1 2>/dev/null || echo "[warn] cuda module load failed"

H_HOME="${H_HOME:-/home/pj24003162/ku40003404/weihao/00}"
REG2026="${REG2026:-$H_HOME/reg_2026}"
SLIDECHAT="${SLIDECHAT:-$H_HOME/reg2026_slidechat_dev}"
REG_ROOT="${REG_ROOT:-$REG2026/data/reg2026}"          # contains train/ + train_thumb/
ENV_PREFIX="${ENV_PREFIX:-$REG2026/_envs/conch_dump_py311}"
CONCH_CKPT="${CONCH_CKPT:-$REG2026/models/CONCH/pytorch_model.bin}"
OUT="${OUT:-$REG2026/tmp/conch_smoke_out}"
LIMIT="${LIMIT:-3}"                                     # smoke: first few slides
BATCH="${BATCH:-256}"                                   # lower (e.g. 64) on a MIG slice

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export PYTHONPATH="$SLIDECHAT/xtuner/tools:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_HOME="$REG2026/_cache/huggingface"

echo "================= pre-checks ================="
echo "REG_ROOT=$REG_ROOT  CONCH_CKPT=$CONCH_CKPT  OUT=$OUT  LIMIT=$LIMIT"
[ -d "$REG_ROOT/train" ]       || { echo "ERR: $REG_ROOT/train missing — set REG_ROOT to the dir holding train/"; exit 2; }
[ -d "$REG_ROOT/train_thumb" ] || { echo "ERR: $REG_ROOT/train_thumb (masks) missing — run make_thumbnails.py first"; exit 2; }
[ -f "$CONCH_CKPT" ]           || { echo "ERR: CONCH_CKPT not found: $CONCH_CKPT"; exit 2; }
echo "train slides : $(ls "$REG_ROOT/train"/*.tiff 2>/dev/null | wc -l)"
echo "masks        : $(ls "$REG_ROOT/train_thumb"/*.mask.png 2>/dev/null | wc -l)"
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" || exit 2

echo "================= dump $LIMIT slide(s) ================="
rm -rf "$OUT"; mkdir -p "$OUT"
t0=$(date +%s)
python "$SLIDECHAT/xtuner/tools/extract_conch_features.py" \
  --root "$REG_ROOT" --conch "$CONCH_CKPT" --device cuda \
  --cap 20480 --tissue-frac 0.1 --batch "$BATCH" \
  --shard 0 --n-shards 1 --limit "$LIMIT" --out "$OUT"
dt=$(( $(date +%s) - t0 ))
echo "elapsed: ${dt}s for up to ${LIMIT} slides"

echo "================= verify h5 ================="
python - "$OUT" <<'PY'
import sys, glob, h5py, numpy as np
fs = sorted(glob.glob(sys.argv[1] + "/*.h5"))
assert fs, "no .h5 produced (all no_mask/no_tissue? check train_thumb masks)"
ok = True
for f in fs:
    with h5py.File(f, "r") as h:
        x = h["features"][:]; c = h["coords"][:]
        x32 = x.astype(np.float32)
        nan = bool(np.isnan(x32).any()); inf = bool(np.isinf(x32).any())
        print(f"{f.split('/')[-1]:40s} feat={x.shape} {x.dtype} coords={c.shape} "
              f"NaN={nan} Inf={inf} mean={x32.mean():.3f} std={x32.std():.3f} "
              f"norm={np.linalg.norm(x32,axis=1).mean():.2f} n={int(h.attrs['n_patches'])} "
              f"proj_contrast={bool(h.attrs['proj_contrast'])}")
        ok &= (x.shape[1] == 512 and x.shape[0] > 0 and not nan and not inf)
        # coords must be in-bounds and on the patch grid
        H, W, ps = int(h.attrs["H"]), int(h.attrs["W"]), int(h.attrs["patch_size"])
        ok &= bool((c[:,0].max() + ps <= H) and (c[:,1].max() + ps <= W)) if len(c) else True
print("SMOKE OK" if ok else "SMOKE FAILED")
sys.exit(0 if ok else 1)
PY
rc=$?
echo "================= done (rc=$rc) ================="
echo "Per-slide time ~ $(python -c "print(f'{$dt/$LIMIT:.1f}')")s  ->  11220 slides / 4 GPU ≈ "
python -c "print(f'{11220*$dt/$LIMIT/4/3600:.1f} h (rough; big PIT_02 slides are the tail)')"
exit $rc
