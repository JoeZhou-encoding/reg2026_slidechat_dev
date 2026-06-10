#!/usr/bin/env bash
# =============================================================================
# TSUBAME: build the Metric A (SlideChat/xtuner) TRAINING env — MIRRORS the verified
# Genkai stack exactly: torch 2.6.0+cu124 + flash-attn 2.7.4.post1 (prebuilt cp310 wheel,
# NO compile) + deepspeed + transformers 4.42.4 + peft 0.11.1 + timm + h5py + xtuner (editable).
#
# Run on the TSUBAME LOGIN node (has network; pip install needs no GPU). TSUBAME ships conda
# (you are in (base)); this probes it, does not reinstall miniconda. Separate from the data env
# (reg_dl_py311) and Metric B's reg_b_py311.
#
#   bash setup_tsubame_train_env_metric_a.sh
#
# PREREQ: slidechat_dev must already be on TSUBAME at $SLIDECHAT (step [1] of the playbook).
# Mirrors reg_b_sft/setup_tsubame_train_env.sh conventions. py3.10 + cp310 flash wheel to match
# the EXACT validated Genkai combo (reg2026_train_py310); do not bump to py311 (flash wheel is cp310).
# =============================================================================
set -uo pipefail

# ---- overridable params ----
PROJECT="${PROJECT:-/gs/bs/tgh-26IDE/ethanx/reg_data_2026}"          # group disk (confirmed)
SLIDECHAT="${SLIDECHAT:-$PROJECT/src/reg2026_slidechat_dev}"         # where step [1] put the code
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/_envs/reg_a_train_py310}"          # NEW metric-A training env
HF_HOME="${HF_HOME:-$PROJECT/_cache/huggingface}"
TMPDIR_BUILD="${TMPDIR_BUILD:-$PROJECT/tmp}"
# pinned to the verified Genkai combo:
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0 torchvision==0.21.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
FLASH_WHL="${FLASH_WHL:-flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
FLASH_URL="${FLASH_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$FLASH_WHL}"

echo "[cfg] PROJECT=$PROJECT"
echo "[cfg] SLIDECHAT=$SLIDECHAT"
echo "[cfg] ENV_PREFIX=$ENV_PREFIX"
echo "[cfg] (PROJECT/SLIDECHAT wrong? Ctrl-C and export overrides, then rerun)"
mkdir -p "$PROJECT/_envs" "$PROJECT/src" "$HF_HOME" "$PROJECT/runs" "$PROJECT/logs" "$TMPDIR_BUILD"
export TMPDIR="$TMPDIR_BUILD" PIP_CACHE_DIR="$PROJECT/.pipcache"
mkdir -p "$PIP_CACHE_DIR"

[ -d "$SLIDECHAT/xtuner" ] || { echo "ERR: slidechat_dev not at $SLIDECHAT (do step [1] code transfer first)"; exit 2; }

# ---- conda base (TSUBAME has conda active in (base); probe is fine on login node) ----
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
[ -n "$CONDA_BASE" ] || { echo "ERR: conda not on PATH (should be (base) on TSUBAME). Activate conda first."; exit 2; }
echo "[conda] base=$CONDA_BASE"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
  echo "[conda] create $ENV_PREFIX (python 3.10)"
  conda create -y -p "$ENV_PREFIX" python=3.10 || { echo "ERR: conda create failed"; exit 2; }
fi
conda activate "$ENV_PREFIX" || { echo "ERR: activate failed"; exit 2; }
python -m pip install --upgrade pip wheel setuptools ninja packaging

# ---- torch 2.6.0 cu124 (prebuilt; self-contained CUDA runtime) ----
echo "[pip] $TORCH_SPEC ($TORCH_INDEX)"
python -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX" || { echo "ERR: torch install failed"; exit 2; }
# sanity: must be cxx11abi=False for the prebuilt flash wheel below
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI)
assert torch.__version__.startswith("2.6.0"), "expected torch 2.6.0 for the cp310/torch2.6 flash wheel"
assert torch._C._GLIBCXX_USE_CXX11_ABI is False, "flash wheel is cxx11abiFALSE; torch ABI mismatch"
PY

# ---- prebuilt flash-attn 2.7.4.post1 (NO compile; matches torch2.6/cu12/cp310/abiFALSE) ----
echo "[flash] prebuilt wheel: $FLASH_WHL"
( cd "$TMPDIR_BUILD" && wget -q "$FLASH_URL" -O "$FLASH_WHL" ) || { echo "ERR: flash wheel download failed ($FLASH_URL)"; exit 2; }
python -m pip install --no-deps "$TMPDIR_BUILD/$FLASH_WHL" || { echo "ERR: flash wheel install failed"; exit 2; }

# ---- xtuner (editable) WITH its OWN declared deps -> mmengine/datasets/einops/tiktoken/
#      bitsandbytes/lagent/openpyxl/transformers_stream_generator/... Use xtuner's requirement list,
#      NOT a hand-picked subset. Done BEFORE the authoritative re-pin below so our pins win. ----
( cd "$SLIDECHAT" && python -m pip install -e . ) || { echo "ERR: xtuner -e . failed"; exit 2; }

# ---- AUTHORITATIVE re-pin to the verified Genkai versions (xtuner's resolver may pull others).
#      Reinstall torch 2.6 + the flash wheel too, so `pip install -e .` cannot leave a torch/flash
#      mismatch (flash wheel is built for torch 2.6 exactly). ----
python -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX" || { echo "ERR: torch re-pin failed"; exit 2; }
python -m pip install --no-deps "$TMPDIR_BUILD/$FLASH_WHL" || { echo "ERR: flash re-pin failed"; exit 2; }
python -m pip install \
  "transformers==4.42.4" "peft==0.11.1" "deepspeed==0.19.1" "timm==1.0.12" \
  "h5py" "scikit-image" "pandas" "matplotlib" "opencv-python" \
  "fairscale" \
  || { echo "ERR: re-pin failed"; exit 2; }
# fairscale is imported by the VENDORED torchscale/LongNet code (decoder.py) but is NOT in xtuner's
# setup.py -> `pip install -e .` won't pull it. The full-import verify below is the real backstop:
# any other vendored dep surfaces HERE on the login node, not in a qsub job.

# ---- verify (cheap, no GPU/CUDA_HOME needed) ----
echo; echo "=============== VERIFY ==============="
python - <<'PY'
import torch, flash_attn, transformers, h5py
print("torch", torch.__version__, "| flash_attn", flash_attn.__version__,
      "| transformers", transformers.__version__, "| h5py", h5py.__version__)
import peft, timm, deepspeed
print("peft", peft.__version__, "| timm", timm.__version__, "| deepspeed", deepspeed.__version__)
# FULL import chain — the real backstop. This is what train.py imports; any missing vendored dep
# (fairscale, ...) fails HERE on the login node instead of wasting a qsub job. deepspeed import
# works on the TSUBAME login node via the system CUDA (/apps/t4/.../cuda).
from xtuner.model import LLaVAModel
from xtuner.model.torchscale.model.LongNet import make_longnet_from_name
from xtuner.dataset import LLaVADataset
print("FULL IMPORT OK (xtuner LLaVAModel + LongNet + dataset)")
PY

cat <<EOF

=============== NEXT ===============
1) Activate:  source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_PREFIX
2) xtuner + flash forward path is validated by the S8 smoke ON A GPU NODE (qsub), NOT here:
     python $SLIDECHAT/xtuner/tools/forward_smoke_longnet.py
   On TSUBAME the GPU job MUST 'module load <cuda>' so deepspeed finds CUDA_HOME (run 'module avail cuda').
3) Models (login node, has network):
     export HF_HOME=$HF_HOME HF_HUB_ENABLE_HF_TRANSFER=1
     hf download Qwen/Qwen2.5-7B-Instruct --local-dir $PROJECT/models/Qwen2.5-7B-Instruct
     hf download General-Medical-AI/SlideChat_Weight stage2_pth/mp_rank_00_model_states.pt --local-dir $PROJECT/models/SlideChat_Weight
     python $SLIDECHAT/xtuner/tools/extract_stage2_model.py \\
       $PROJECT/models/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt \\
       $PROJECT/models/SlideChat_Weight/stage2_model.pth
EOF
echo "[done] metric-A training env: $ENV_PREFIX"
