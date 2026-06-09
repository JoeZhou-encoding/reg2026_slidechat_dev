#!/bin/bash
# =============================================================================
# Metric A continue-SFT — SMOKE (largest-64 WSIs). TSUBAME, 1 full node (node_f = 4x H100), ZeRO-2.
# FULLY SELF-CONTAINED: every path/var hardcoded; does NOT inherit login-shell env. TSUBAME
# conventions from logs/run_tsubame.sh (Grid Engine #$, cuda/12.8.0, conda base under S_HOME).
# Launch logic = the reviewed Genkai twin (torchrun-direct so rc propagates + key-overlap preflight).
#
#   SUBMIT (group required, per TSUBAME):
#     qsub -g tgh-26IDE /gs/bs/tgh-26IDE/ethanx/reg_data_2026/src/reg2026_slidechat_dev/xtuner/tools/qsub_metric_a_smoke.sh
#
# Twin of qsub_metric_a_full.sh (ONLY differences: h_rt / DATA_PATH / WORK_DIR / MASTER_PORT / -o name).
# Prereq: SMOKE64 subset json exists (make_smoke_subset.py). node_f assumed = 4 GPUs (confirm via nvidia-smi).
# =============================================================================
#$ -S /bin/bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=00:30:00
#$ -p -5
#$ -j y
#$ -o /gs/bs/tgh-26IDE/ethanx/reg_data_2026/logs/metric_a_smoke.$JOB_ID.out

set -uo pipefail
module unload openmpi/5.0.2-intel cuda 2>/dev/null || true
module load cuda/12.8.0

# ---- hardcoded paths (NO login-shell env inheritance) ----
S_HOME=/gs/bs/tgh-26IDE/ethanx
PROJECT=$S_HOME/reg_data_2026
SLIDECHAT=$PROJECT/src/reg2026_slidechat_dev
ENV_PREFIX=$PROJECT/_envs/reg_a_train_py310
CONFIG=$SLIDECHAT/xtuner/configs/slidechat/metric_a_sft.py
DS_CONFIG=$SLIDECHAT/xtuner/configs/deepspeed/deepspeed_zero2.json
TRAIN_PY=$SLIDECHAT/xtuner/tools/train.py
PRETRAINED=$PROJECT/models/SlideChat_Weight/stage2_model.pth
CONDA_BASE=$S_HOME/miniconda3
NGPU=4
LOGDIR=$PROJECT/logs
DATA_PATH=$PROJECT/data/reg2026/metric_a/sft_metric_a_train.SMOKE64.json
STAMP=${JOB_ID:-local}                    # UGE runtime var (set by scheduler, not login env)
WORK_DIR=$PROJECT/outputs/experiments/exp_metricA_sft_SMOKE_${STAMP}
MASTER_PORT=29555

# ---- the config reads $REG2026 to resolve TSUBAME paths (Genkai default otherwise) ----
export REG2026=$PROJECT

# ---- conda (hardcoded base per run_tsubame.sh) ----
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || { echo "ERR: conda base not at $CONDA_BASE"; exit 2; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX" || { echo "ERR: activate $ENV_PREFIX failed"; exit 2; }

# ---- runtime env (compute node = NO network -> offline; NCCL/CUDA stability per run_tsubame.sh) ----
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$PROJECT/_cache/huggingface
export TMPDIR=$PROJECT/tmp
export OMP_NUM_THREADS=8 CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export LIBRARY_PATH=$CONDA_PREFIX/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$WORK_DIR" "$LOGDIR" "$TMPDIR"

echo "SMOKE Job $STAMP  host=$(hostname)  NGPU=$NGPU  REG2026=$REG2026  $(date)"
echo "CONFIG=$CONFIG"
echo "DATA_PATH=$DATA_PATH  (largest-64 WSIs subset)"
echo "WORK_DIR=$WORK_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ---- pre-flight (fail fast) ----
[ -f "$DATA_PATH" ] || { echo "ERR: SMOKE subset missing: $DATA_PATH -- run make_smoke_subset.py first"; exit 2; }
[ -f "$PRETRAINED" ] || { echo "ERR: stage2_model.pth missing: $PRETRAINED"; exit 2; }
[ -d "$PROJECT/models/Qwen2.5-7B-Instruct" ] || { echo "ERR: Qwen2.5-7B-Instruct dir missing"; exit 2; }
[ -d "$PROJECT/data/reg2026/train_feat_conch" ] || { echo "ERR: train_feat_conch dir missing"; exit 2; }
[ -f "$DS_CONFIG" ] || { echo "ERR: deepspeed config missing: $DS_CONFIG"; exit 2; }
[ -f "$TRAIN_PY" ] || { echo "ERR: train.py missing: $TRAIN_PY"; exit 2; }
python -c "import flash_attn" 2>/dev/null || { echo "ERR: flash_attn not importable in $ENV_PREFIX"; exit 2; }

# ---- pre-flight: confirm pretrained_pth keys overlap the model submodules (strict=False guard) ----
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
module load cuda/12.8.0
source $CONDA_BASE/etc/profile.d/conda.sh; conda activate $ENV_PREFIX
export REG2026=$PROJECT HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$NGPU \\
  --master_addr=127.0.0.1 --master_port=$MASTER_PORT \\
  $TRAIN_PY $CONFIG \\
  --deepspeed $DS_CONFIG \\
  --cfg-options train_dataloader.dataset.data_path=$DATA_PATH \\
  --work-dir $WORK_DIR \\
  --launcher pytorch
EOF

# ---- launch: torchrun DIRECTLY (the `xtuner train` wrapper swallows torchrun's rc; this
#      reproduces what it builds but propagates the true rc via ${PIPESTATUS[0]}). ----
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
