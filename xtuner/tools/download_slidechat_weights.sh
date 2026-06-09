#!/bin/bash
# =============================================================================
# Download SlideChat continue-SFT weights to the server.
#   General-Medical-AI/SlideChat_Weight  (LongNet + projector [+ LLM])
#   Qwen/Qwen2.5-7B-Instruct             (base LLM, ~15GB)
# (CONCH already present at models/CONCH.)
#
# Per project rule: DRY-RUN by default (prints config + exits). Add --yes to download.
#   python  <...>/check_weights.py                      # confirm absent first
#   bash    <...>/download_slidechat_weights.sh         # review config (dry run)
#   bash    <...>/download_slidechat_weights.sh --yes   # actually download
#
# Run in an env that has huggingface_hub (conch_dump_py311). For gated repos:
#   export HF_TOKEN=hf_xxx   (never hard-coded here; .env has it)
# =============================================================================
set -uo pipefail

H_HOME=/home/pj24003162/ku40003404/weihao/00
REG2026=$H_HOME/reg_2026
MODELS=$REG2026/models
ENV_PREFIX=$REG2026/_envs/conch_dump_py311

SLIDECHAT_REPO=General-Medical-AI/SlideChat_Weight
QWEN_REPO=Qwen/Qwen2.5-7B-Instruct

GO=0; [ "${1:-}" = "--yes" ] && GO=1

# ---- activate an env with huggingface_hub (self-contained) ----
CONDA_BASE=""
for cb in "$(conda info --base 2>/dev/null || true)" "$H_HOME/miniconda3" /home/pj24003162/ku40003404/miniconda3; do
  [ -n "$cb" ] && [ -f "$cb/etc/profile.d/conda.sh" ] && { CONDA_BASE="$cb"; break; }
done
[ -n "$CONDA_BASE" ] || { echo "ERR: conda base not found"; exit 2; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX" || { echo "ERR: conda activate $ENV_PREFIX failed"; exit 2; }
# newer huggingface_hub ships `hf` (huggingface-cli is deprecated/no-op); prefer hf, fall back.
HF_BIN=""
if command -v hf >/dev/null 2>&1; then HF_BIN=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF_BIN=huggingface-cli
else echo "ERR: neither 'hf' nor 'huggingface-cli' found in $ENV_PREFIX"; exit 2; fi
echo "  hf cli: $HF_BIN"

# token (optional, for gated repos) — from env only
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
unset HF_HUB_OFFLINE 2>/dev/null || true   # ensure online for the download

mkdir -p "$MODELS"
echo "=============== DOWNLOAD CONFIG (review) ==============="
echo "  $SLIDECHAT_REPO  ->  $MODELS/SlideChat_Weight"
echo "  $QWEN_REPO        ->  $MODELS/Qwen2.5-7B-Instruct  (~15 GB)"
echo "  env:    $ENV_PREFIX"
echo "  token:  ${HF_TOKEN:+set}"; [ -z "${HF_TOKEN:-}" ] && echo "          (HF_TOKEN not set; public repos download fine without it)"
echo "  free space at $MODELS:"; df -h "$MODELS" 2>/dev/null | sed 's/^/    /'
echo "======================================================="
if [ "$GO" != "1" ]; then
  echo "DRY RUN — nothing downloaded. Re-run with --yes to download."
  exit 0
fi

dl () {  # $1 repo, $2 dest
  echo ">>> $1  ->  $2"
  "$HF_BIN" download "$1" --local-dir "$2" \
    || { echo "ERR: download of $1 failed"; return 1; }
}
rc=0
dl "$SLIDECHAT_REPO" "$MODELS/SlideChat_Weight"   || rc=1
dl "$QWEN_REPO"      "$MODELS/Qwen2.5-7B-Instruct" || rc=1
echo "----------------------------------------------"
echo "download rc=$rc.  verify with:  python $(dirname "$0")/check_weights.py"
exit $rc
