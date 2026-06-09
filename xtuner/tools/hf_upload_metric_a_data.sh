#!/usr/bin/env bash
# =============================================================================
# Relay Metric A processed data Genkai -> HF Hub (PRIVATE dataset). Run on the GENKAI
# LOGIN node (has network). TSUBAME then downloads via hf_download_metric_a_data.sh.
#
# Uploads: train_feat_conch/ (~43GB, 11109 .h5)  +  metric_a/sft_metric_a_train.json (61MB).
# Models (Qwen / SlideChat) are NOT relayed — re-download those from public HF on TSUBAME.
#
#   HF_REPO=<your-hf-username>/reg2026-metric-a-data \
#     bash hf_upload_metric_a_data.sh
#
# ⚠️ This PUBLISHES data to an external service (HF). Repo is created PRIVATE. Confirm you are
#    OK uploading the competition-derived features before running. Token is read from .env,
#    never hardcoded/printed.
# =============================================================================
set -uo pipefail

REG2026="${REG2026:-/home/pj24003162/ku40003404/weihao/00/reg_2026}"
: "${HF_REPO:?set HF_REPO=<hf-username>/reg2026-metric-a-data}"
FEAT_DIR="${FEAT_DIR:-$REG2026/data/reg2026/train_feat_conch}"
SFT_JSON="${SFT_JSON:-$REG2026/data/reg2026/metric_a/sft_metric_a_train.json}"

# ---- token from .env (never commit/print) ----
[ -f "$REG2026/.env" ] && source "$REG2026/.env"
: "${HF_TOKEN:?HF_TOKEN not set (put it in $REG2026/.env)}"
export HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=1

# ---- pre-flight ----
[ -d "$FEAT_DIR" ] || { echo "ERR: feature dir missing: $FEAT_DIR"; exit 2; }
[ -f "$SFT_JSON" ] || { echo "ERR: SFT json missing: $SFT_JSON"; exit 2; }
n_h5=$(ls "$FEAT_DIR"/*.h5 2>/dev/null | wc -l)
echo "repo=$HF_REPO  feat_dir=$FEAT_DIR (.h5=$n_h5)  sft_json=$SFT_JSON"
python -c "import huggingface_hub, hf_transfer" 2>/dev/null \
  || { echo "ERR: pip install 'huggingface_hub>=0.24' hf_transfer  (in this env) first"; exit 2; }

# ---- create private repo + upload (resumable: re-run continues where it left off) ----
python - "$HF_REPO" "$FEAT_DIR" "$SFT_JSON" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, feat_dir, sft_json = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
print(f"[hf] repo ready (private): {repo}")
api.upload_file(path_or_fileobj=sft_json, path_in_repo="metric_a/sft_metric_a_train.json",
                repo_id=repo, repo_type="dataset", commit_message="sft json")
print("[hf] uploaded metric_a/sft_metric_a_train.json")
# 43GB / 11k files: upload_large_folder is resumable + parallel + multi-commit
api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=feat_dir,
                        allow_patterns=["*.h5", "*.jsonl"])
print("[hf] uploaded train_feat_conch/*.h5 (+ status jsonl) -> repo root")
PY
echo "[done] uploaded to https://huggingface.co/datasets/$HF_REPO (PRIVATE)"
echo "next on TSUBAME: HF_REPO=$HF_REPO bash hf_download_metric_a_data.sh"
