#!/bin/bash
# =============================================================================
# Metric A continue-SFT — SMOKE (largest-64 WSIs). Genkai b-batch, 1 node, 4x H100, ZeRO-2.
# FULLY SELF-CONTAINED: every path/var hardcoded below; does NOT inherit any login-shell env.
# Small + short elapse -> backfills/queues sooner than the full job and finishes first, so its
# result can validate the path while the full job is still queued (no wasted wall-clock).
#
#   pjsub /home/pj24003162/ku40003404/weihao/00/reg2026_slidechat_dev/xtuner/tools/pjsub_metric_a_smoke.sh
#
# Prereq: the subset json must exist (make_smoke_subset.py -> largest-N WSIs, so the smoke hits
# the worst-case 10240-visual-token peak-memory path). Twin of pjsub_metric_a_full.sh
# (ONLY differences: elapse / DATA_PATH / WORK_DIR / MASTER_PORT / -o name + derived log/echo).
# =============================================================================
#PJM -L rscgrp=b-batch
#PJM -L gpu=4
#PJM -L elapse=00:30:00
#PJM -j
#PJM -S
#PJM -o metric_a_smoke.%j.out

set -uo pipefail
cd "${PJM_O_WORKDIR:-$PWD}" || true
module purge
module load cuda/12.6.1

# ---- hardcoded paths (NO login-shell env inheritance) ----
H_HOME=/home/pj24003162/ku40003404/weihao/00
REG2026=$H_HOME/reg_2026
SLIDECHAT=$H_HOME/reg2026_slidechat_dev
ENV_PREFIX=$REG2026/_envs/reg2026_train_py310
CONFIG=$SLIDECHAT/xtuner/configs/slidechat/metric_a_sft.py
DS_CONFIG=$SLIDECHAT/xtuner/configs/deepspeed/deepspeed_zero2.json
TRAIN_PY=$SLIDECHAT/xtuner/tools/train.py
PRETRAINED=$REG2026/models/SlideChat_Weight/stage2_model.pth
NGPU=4
LOGDIR=$REG2026/logs
DATA_PATH=$REG2026/data/reg2026/metric_a/sft_metric_a_train.SMOKE64.json
STAMP=${PJM_JOBID:-local}                 # PJM runtime var (set by scheduler, not login env)
WORK_DIR=$REG2026/outputs/experiments/exp_metricA_sft_SMOKE_${STAMP}
MASTER_PORT=29555

# ---- conda base (filesystem probe over hardcoded candidates; no `conda info`/PATH reads) ----
CONDA_BASE=""
for cb in "$H_HOME/miniconda3" /home/pj24003162/ku40003404/miniconda3; do
  [ -f "$cb/etc/profile.d/conda.sh" ] && { CONDA_BASE="$cb"; break; }
done
[ -n "$CONDA_BASE" ] || { echo "ERR: conda base not found (edit CONDA_BASE candidates)"; exit 2; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX" || { echo "ERR: activate $ENV_PREFIX failed"; exit 2; }

# ---- runtime env (offline; torchrun gets addr/port explicitly below) ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$REG2026/_cache/huggingface
export TMPDIR="${PJM_SSD_DIR:-$REG2026/tmp}"      # per-job SSD if provided, else /home tmp
export OMP_NUM_THREADS=8
mkdir -p "$WORK_DIR" "$LOGDIR" "$TMPDIR"

echo "SMOKE Job $STAMP  host=$(hostname)  NGPU=$NGPU  $(date)"
echo "CONFIG=$CONFIG"
echo "DATA_PATH=$DATA_PATH  (largest-64 WSIs subset)"
echo "WORK_DIR=$WORK_DIR"
echo "DS_CONFIG=$DS_CONFIG  (ZeRO-2; accum/micro/clip/bf16 auto-filled)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ---- pre-flight (fail fast) ----
[ -f "$DATA_PATH" ] || { echo "ERR: SMOKE subset missing: $DATA_PATH -- run make_smoke_subset.py first"; exit 2; }
[ -f "$PRETRAINED" ] || { echo "ERR: stage2_model.pth missing: $PRETRAINED"; exit 2; }
[ -d "$REG2026/models/Qwen2.5-7B-Instruct" ] || { echo "ERR: Qwen2.5-7B-Instruct dir missing"; exit 2; }
[ -d "$REG2026/data/reg2026/train_feat_conch" ] || { echo "ERR: train_feat_conch dir missing"; exit 2; }
[ -f "$DS_CONFIG" ] || { echo "ERR: deepspeed config missing: $DS_CONFIG"; exit 2; }
[ -f "$TRAIN_PY" ] || { echo "ERR: train.py missing: $TRAIN_PY"; exit 2; }
python -c "import flash_attn" 2>/dev/null || { echo "ERR: flash_attn not importable in $ENV_PREFIX"; exit 2; }

# ---- pre-flight: confirm pretrained_pth keys overlap the model submodules ----
# model/llava.py loads pretrained_pth with strict=False and discards missing/unexpected keys, so a
# mis-prefixed checkpoint would SILENTLY train the visual stack from scratch. Log the key match (mmap
# = cheap, keys only). Only the unambiguous 0-key case aborts; any introspection error just warns.
python - "$PRETRAINED" <<'PY'
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=True)
except Exception as e:
    print(f"[preflight] WARN: could not introspect keys ({e}); leaving the load to training"); sys.exit(0)
sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
keys = list(sd.keys())
pref = {}
for k in keys:
    p = k.split(".")[0]; pref[p] = pref.get(p, 0) + 1
print("[preflight] pretrained_pth submodule key counts:", dict(sorted(pref.items(), key=lambda x: -x[1])))
zero = [m for m in ("llm", "LongNet_encoder", "projector") if sum(k.startswith(m + ".") for k in keys) == 0]
if zero:
    print(f"[preflight] FATAL: zero keys for {zero} -> strict=False would train these from scratch"); sys.exit(7)
print("[preflight] OK: llm.* + LongNet_encoder.* + projector.* all present")
PY
[ "$?" = "7" ] && { echo "ERR: pretrained_pth key mismatch (see [preflight] above) -- aborting"; exit 2; }

# ---- save exact command (reproducibility) ----
cat > "$WORK_DIR/command.sh" <<EOF
#!/usr/bin/env bash
module load cuda/12.6.1
source $CONDA_BASE/etc/profile.d/conda.sh; conda activate $ENV_PREFIX
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$NGPU \\
  --master_addr=127.0.0.1 --master_port=$MASTER_PORT \\
  $TRAIN_PY $CONFIG \\
  --deepspeed $DS_CONFIG \\
  --cfg-options train_dataloader.dataset.data_path=$DATA_PATH \\
  --work-dir $WORK_DIR \\
  --launcher pytorch
EOF

# ---- launch: call torchrun DIRECTLY. The `xtuner train` wrapper (entry_point.py cli()) runs
#      subprocess.run(torchrun) WITHOUT check=True and returns None, so it exits 0 even when
#      training crashes -> ${PIPESTATUS[0]} would mask a failed run. This reproduces exactly what
#      `xtuner train` builds (entry_point.py:284-293) but propagates torchrun's true rc. ----
torchrun --nnodes=1 --node_rank=0 --nproc_per_node="$NGPU" \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  "$TRAIN_PY" "$CONFIG" \
  --deepspeed "$DS_CONFIG" \
  --cfg-options train_dataloader.dataset.data_path="$DATA_PATH" \
  --work-dir "$WORK_DIR" \
  --launcher pytorch \
  2>&1 | tee "$LOGDIR/metric_a_smoke.${STAMP}.log"
rc=${PIPESTATUS[0]}
echo "SMOKE torchrun done rc=$rc  $(date)"
echo "  work-dir: $WORK_DIR"
ls -lh "$WORK_DIR" 2>/dev/null | head -20
exit $rc
