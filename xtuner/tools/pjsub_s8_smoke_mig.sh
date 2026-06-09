#!/bin/bash
# =============================================================================
# S8 forward smoke on a b-batch-MIG slice (1g.12gb ~11GB) — verifies the GPU forward
# path: LongNet(512->512, variable N) + Projector(512->H_llm). NO LLM, tiny (<1GB).
# Self-contained. Submit from /home or /fast:
#   pjsub /home/pj24003162/ku40003404/weihao/00/reg2026_slidechat_dev/xtuner/tools/pjsub_s8_smoke_mig.sh
# =============================================================================
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=00:20:00
#PJM -j
#PJM -S
#PJM -o s8_smoke_mig.%j.out

set -uo pipefail
cd "${PJM_O_WORKDIR:-$PWD}" || true
module purge 2>/dev/null || true
module load cuda/12.6.1 2>/dev/null || true

H_HOME=/home/pj24003162/ku40003404/weihao/00
REG2026=$H_HOME/reg_2026
SLIDECHAT=$H_HOME/reg2026_slidechat_dev
# xtuner-env: STRICT authors' env (environment.yaml) — torch 2.4.0 cu121 + flash-attn 2.5.8 (compiled).
ENV_ACT=xtuner-env

CONDA_BASE=""
for cb in "$(conda info --base 2>/dev/null || true)" "$H_HOME/miniconda3" /home/pj24003162/ku40003404/miniconda3; do
  [ -n "$cb" ] && [ -f "$cb/etc/profile.d/conda.sh" ] && { CONDA_BASE="$cb"; break; }
done
[ -n "$CONDA_BASE" ] || { echo "ERR: conda base not found"; exit 2; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_ACT" || { echo "ERR: activate $ENV_ACT failed"; exit 2; }

echo "host=$(hostname)  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# full N range on GPU (MIG has plenty for LongNet+projector only)
python "$SLIDECHAT/xtuner/tools/forward_smoke_longnet.py"
echo "S8 smoke done rc=$?  $(date)"
