#!/bin/bash
# =============================================================================
# Build the REG2026 Metric A TRAINING env (SlideChat xtuner fork + deepspeed + h5py).
# Separate from conch_dump_py311 (dump-only). Self-contained; run on Genkai.
#   bash setup_train_env.sh
# Then train:
#   NPROC_PER_NODE=4 xtuner train \
#     $SLIDECHAT/xtuner/configs/slidechat/metric_a_sft.py \
#     --deepspeed $SLIDECHAT/xtuner/configs/deepspeed/deepspeed_zero2.json
#
# Notes: README uses python=3.10; requirements pin transformers<=4.42.4 (Qwen2.5 OK via
# Qwen2 arch). torch installed as cu124 BEFORE `pip install -e .` so it isn't pulled as CPU.
# h5py is OUR addition (the .h5 feature loader branch in dataset/llava.py + utils.py).
# =============================================================================
set -uo pipefail

H_HOME=/home/pj24003162/ku40003404/weihao/00
REG2026=$H_HOME/reg_2026
SLIDECHAT=$H_HOME/reg2026_slidechat_dev
ENV_PREFIX=$REG2026/_envs/reg2026_train_py310

module purge 2>/dev/null || true
module load cuda/12.6.1 2>/dev/null || true

# ---- conda base (self-contained) ----
CONDA_BASE=""
for cb in "$(conda info --base 2>/dev/null || true)" "$H_HOME/miniconda3" /home/pj24003162/ku40003404/miniconda3; do
  [ -n "$cb" ] && [ -f "$cb/etc/profile.d/conda.sh" ] && { CONDA_BASE="$cb"; break; }
done
[ -n "$CONDA_BASE" ] || { echo "ERR: conda base not found"; exit 2; }
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- create env ----
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.10 || { echo "ERR: conda create failed"; exit 2; }
fi
conda activate "$ENV_PREFIX" || { echo "ERR: activate failed"; exit 2; }
python -m pip install -U pip

# ---- torch cu124 FIRST (so editable install doesn't pull CPU torch) ----
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
  || { echo "ERR: torch install failed"; exit 2; }

# ---- SlideChat xtuner fork (editable) -> pulls mmengine / transformers<=4.42.4 / peft / deepspeed ----
( cd "$SLIDECHAT" && pip install -e . ) || { echo "ERR: pip install -e . failed"; exit 2; }
# requirements pin transformers<=4.42.4 but peft is unbounded -> latest peft needs
# transformers>=4.43 (EncoderDecoderCache) and breaks. Pin peft to a 4.42-compatible version.
pip install "peft==0.11.1"

# ---- our extras ----
pip install h5py                       # .h5 WSI feature loader branch (B4)
# deepspeed: the `-e .` requirements/deepspeed.txt also lists mpi4py-mpich, which needs an MPI
# toolchain and often aborts the whole deepspeed install. Single-node/4-GPU uses NCCL (no MPI),
# so install deepspeed directly and skip mpi4py-mpich.
pip install deepspeed || echo "WARN: deepspeed install failed -- retry: pip install deepspeed"
# requirements.txt is INCOMPLETE: the model/dataset code imports these but they are NOT declared
# (torchscale/LongNet+CONCH need timm; the csv loader needs pandas; etc.). Versions from the
# authors' environment.yaml -- install them all up front to avoid one-by-one ModuleNotFoundError.
pip install timm==1.0.12 accelerate==1.2.0 pandas==2.2.3 matplotlib==3.9.3 tqdm==4.67.1 opencv-python==4.10.0.84

# ---- verify ----
echo "================= VERIFY ================="
python - <<'PY'
import torch, h5py, transformers, deepspeed, peft, timm, pandas, accelerate, xtuner
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| deepspeed", deepspeed.__version__, "| timm", timm.__version__, "| h5py", h5py.__version__)
from xtuner.model import LLaVAModel                          # exercises the full model __init__
from xtuner.model.torchscale.model.LongNet import make_longnet_from_name
from xtuner.model.modules import ProjectorConfig, ProjectorModel
from xtuner.dataset import LLaVADataset
print("xtuner LLaVAModel + LongNet + projector + dataset imports OK")
PY
echo "xtuner CLI: $(command -v xtuner || echo MISSING)"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "next: run S8 smoke -> python $SLIDECHAT/xtuner/tools/forward_smoke_longnet.py"
