#!/bin/bash
# =============================================================================
# Build the REG2026 Metric A TRAINING env the AUTHORS' way: recreate EXACTLY from
# SlideChat's environment.yaml (torch 2.4.0 cu121 + flash-attn 2.5.8 + all 304 pinned
# deps — their tested combo). Then add the ONLY two things the yaml omits:
#   - xtuner itself (editable repo, not captured in the export)
#   - h5py (our .h5 feature loader branch)
#
# Long job (flash-attn 2.5.8 may COMPILE against torch 2.4 -> tens of minutes). Run in tmux.
#   bash setup_train_env.sh
# =============================================================================
set -uo pipefail

H_HOME=/home/pj24003162/ku40003404/weihao/00
REG2026=$H_HOME/reg_2026
SLIDECHAT=$H_HOME/reg2026_slidechat_dev
ENV_PREFIX=$REG2026/_envs/sc_train          # NEW env (leave the old reg2026_train_py310 alone)

module purge 2>/dev/null || true
module load cuda/12.6.1 2>/dev/null || true   # nvcc for flash-attn build

# ---- conda base ----
CONDA_BASE=""
for cb in "$(conda info --base 2>/dev/null || true)" "$H_HOME/miniconda3" /home/pj24003162/ku40003404/miniconda3; do
  [ -n "$cb" ] && [ -f "$cb/etc/profile.d/conda.sh" ] && { CONDA_BASE="$cb"; break; }
done
[ -n "$CONDA_BASE" ] || { echo "ERR: conda base not found"; exit 2; }
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- keep pip temp + cache on the SAME filesystem (avoids flash-attn 'Invalid cross-device
#      link' during its wheel install) ----
export TMPDIR="$REG2026/tmp" PIP_CACHE_DIR="$REG2026/.pipcache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# ---- recreate the env EXACTLY from the authors' environment.yaml ----
echo ">>> conda env create -f environment.yaml -p $ENV_PREFIX  (this is the long step)"
conda env create -p "$ENV_PREFIX" -f "$SLIDECHAT/environment.yaml" \
  || { echo "ERR: conda env create failed. If it complained about name/prefix, remove the"
       echo "    'name:' and 'prefix:' lines from a COPY of environment.yaml and retry with that."; exit 2; }

conda activate "$ENV_PREFIX" || { echo "ERR: activate $ENV_PREFIX failed"; exit 2; }

# ---- the two omissions: xtuner (editable) + h5py (our loader) ----
( cd "$SLIDECHAT" && pip install -e . --no-deps ) || { echo "ERR: pip install -e . failed"; exit 2; }
pip install h5py

# ---- verify ----
echo "================= VERIFY ================="
python - <<'PY'
import torch, flash_attn, deepspeed, peft, timm, pandas, h5py, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| flash_attn", flash_attn.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| deepspeed", deepspeed.__version__, "| h5py", h5py.__version__)
from xtuner.model import LLaVAModel
from xtuner.model.torchscale.model.LongNet import make_longnet_from_name
from xtuner.dataset import LLaVADataset
print("xtuner LLaVAModel + LongNet + dataset imports OK")
PY
echo "ENV_PREFIX=$ENV_PREFIX"
echo "next: S8 on GPU -> pjsub $SLIDECHAT/xtuner/tools/pjsub_s8_smoke_mig.sh  (edit ENV_PREFIX in it to sc_train)"
