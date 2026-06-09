#!/bin/bash
# =============================================================================
# STRICT authors' env: recreate EXACTLY from SlideChat's environment.yaml
# (torch 2.4.0 cu121 + flash-attn 2.5.8 + all 304 pinned deps — their tested combo).
#
# The yaml ships CUDA 12.1 *runtime* but NO nvcc, and flash-attn 2.5.8 has no prebuilt wheel
# for torch 2.4 -> it must COMPILE. We make that self-contained by installing cuda-nvcc=12.1
# (matching torch's cuda 12.1) into the env, then compiling flash. (~20-40 min; run in tmux.)
#
#   bash setup_train_env.sh
# =============================================================================
set -uo pipefail

H_HOME=/home/pj24003162/ku40003404/weihao/00
REG2026=$H_HOME/reg_2026
SLIDECHAT=$H_HOME/reg2026_slidechat_dev
ENV=xtuner-env

# ---- conda base ----
CONDA_BASE=""
for cb in "$(conda info --base 2>/dev/null || true)" "$H_HOME/miniconda3" /home/pj24003162/ku40003404/miniconda3; do
  [ -n "$cb" ] && [ -f "$cb/etc/profile.d/conda.sh" ] && { CONDA_BASE="$cb"; break; }
done
[ -n "$CONDA_BASE" ] || { echo "ERR: conda base not found"; exit 2; }
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- pip temp+cache on /home (avoids flash 'Invalid cross-device link') ----
export TMPDIR="$REG2026/tmp" PIP_CACHE_DIR="$REG2026/.pipcache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# ---- remove the broken xtuner-env (torch 2.12) if present ----
conda deactivate 2>/dev/null || true
conda env remove -n "$ENV" -y 2>/dev/null || true

# ---- sanitized yaml: drop flash (compile separately) + name/prefix lines ----
SAN="$TMPDIR/env_noflash.yaml"
grep -vE "flash-attn|^name:|^prefix:" "$SLIDECHAT/environment.yaml" > "$SAN"

# ---- create env with all 304 deps EXCEPT flash (torch 2.4 + deepspeed 0.16.1 + ...) ----
echo ">>> conda env create -n $ENV  (all authors' deps except flash; long)"
conda env create -n "$ENV" -f "$SAN" || { echo "ERR: conda env create failed"; exit 2; }
conda activate "$ENV"

# ---- nvcc 12.1 to match torch's cuda 12.1, for the flash compile ----
conda install -y -c nvidia cuda-nvcc=12.1 || conda install -y -c nvidia cuda-nvcc || \
  { echo "ERR: could not install cuda-nvcc=12.1"; exit 2; }
export CUDA_HOME="$CONDA_PREFIX"

# ---- flash-attn 2.5.8 (authors' version) — compile for H100 (sm90) ----
echo ">>> compiling flash-attn==2.5.8 (H100 sm90; ~20-40 min)"
export TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS="${MAX_JOBS:-4}"
pip install flash-attn==2.5.8 --no-build-isolation \
  || { echo "ERR: flash-attn compile failed -- paste the error"; exit 2; }

# ---- the two yaml omissions: xtuner (editable) + h5py (our loader) ----
( cd "$SLIDECHAT" && pip install -e . --no-deps ) || { echo "ERR: pip install -e . failed"; exit 2; }
pip install h5py

# ---- verify ----
echo "================= VERIFY ================="
python - <<'PY'
import torch, flash_attn, deepspeed, peft, timm, pandas, h5py, transformers
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available(),
      "| flash_attn", flash_attn.__version__, "| deepspeed", deepspeed.__version__,
      "| transformers", transformers.__version__, "| peft", peft.__version__)
from xtuner.model import LLaVAModel
from xtuner.model.torchscale.model.LongNet import make_longnet_from_name
from xtuner.dataset import LLaVADataset
print("xtuner LLaVAModel + LongNet + dataset imports OK")
PY
echo "next: pjsub $SLIDECHAT/xtuner/tools/pjsub_s8_smoke_mig.sh"
