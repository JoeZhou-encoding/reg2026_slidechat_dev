#!/usr/bin/env bash
# =============================================================================
# Download Metric A processed data from HF Hub (PRIVATE dataset) onto TSUBAME. Run on the
# TSUBAME LOGIN node (has network). Counterpart of hf_upload_metric_a_data.sh (run on Genkai).
#
# Places:  train_feat_conch/*.h5 -> $PROJECT/data/reg2026/train_feat_conch/
#          sft json              -> $PROJECT/data/reg2026/metric_a/sft_metric_a_train.json
#
#   HF_REPO=<your-hf-username>/reg2026-metric-a-data \
#     bash hf_download_metric_a_data.sh
# =============================================================================
set -uo pipefail

PROJECT="${PROJECT:-/gs/bs/tgh-26IDE/ethanx/reg_data_2026}"
: "${HF_REPO:?set HF_REPO=<hf-username>/reg2026-metric-a-data}"
DATA_ROOT="$PROJECT/data/reg2026"
FEAT_DIR="$DATA_ROOT/train_feat_conch"

# ---- token from .env (private repo needs it) ----
[ -f "$PROJECT/.env" ] && source "$PROJECT/.env"
: "${HF_TOKEN:?HF_TOKEN not set (put it in $PROJECT/.env)}"
export HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$FEAT_DIR" "$DATA_ROOT/metric_a"

python -c "import huggingface_hub, hf_transfer" 2>/dev/null \
  || { echo "ERR: pip install 'huggingface_hub>=0.24' hf_transfer  (in the data env) first"; exit 2; }

python - "$HF_REPO" "$FEAT_DIR" "$DATA_ROOT" <<'PY'
import os, sys
from huggingface_hub import snapshot_download, hf_hub_download
repo, feat_dir, data_root = sys.argv[1], sys.argv[2], sys.argv[3]
tok = os.environ["HF_TOKEN"]
# features: *.h5 (+ status jsonl) straight into train_feat_conch (resumable)
snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=["*.h5", "*.jsonl"],
                  local_dir=feat_dir, token=tok)
# sft json -> data_root/metric_a/sft_metric_a_train.json
hf_hub_download(repo_id=repo, repo_type="dataset", filename="metric_a/sft_metric_a_train.json",
                local_dir=data_root, token=tok)
print("[hf] download complete")
PY

n_h5=$(ls "$FEAT_DIR"/*.h5 2>/dev/null | wc -l)
echo "[done] train_feat_conch .h5 = $n_h5  (expect 11109)"
echo "       sft json: $DATA_ROOT/metric_a/sft_metric_a_train.json"
ls -lh "$DATA_ROOT/metric_a/sft_metric_a_train.json" 2>/dev/null || echo "  (WARN: sft json missing)"
