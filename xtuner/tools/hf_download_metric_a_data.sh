#!/usr/bin/env bash
# =============================================================================
# Download Metric A processed data from HF Hub (PRIVATE dataset) onto TSUBAME. Run on the
# TSUBAME LOGIN node (has network). Counterpart of hf_upload_metric_a_data.sh (run on Genkai).
#
# Pulls the tar parts (feat_archive/) + sft json, reassembles + extracts to:
#   $PROJECT/data/reg2026/train_feat_conch/*.h5
#   $PROJECT/data/reg2026/metric_a/sft_metric_a_train.json
#
#   HF_REPO=zzqsb/reg2026-metric-a-data  bash hf_download_metric_a_data.sh
#
# Disk: transient ~43GB parts + ~43GB extracted (rm parts after). Set PARTS_DIR=/path to override.
# =============================================================================
set -uo pipefail

PROJECT="${PROJECT:-/gs/bs/tgh-26IDE/ethanx/reg_data_2026}"
HF_REPO="${HF_REPO:-zzqsb/reg2026-metric-a-data-feats}"   # feature tar parts + sft json (option A)
DATA_ROOT="$PROJECT/data/reg2026"
FEAT_DIR="$DATA_ROOT/train_feat_conch"
PARTS_DIR="${PARTS_DIR:-$PROJECT/tmp/feat_parts_dl}"

# ---- token from .env (private repo) ----
[ -f "$PROJECT/.env" ] && source "$PROJECT/.env"
export HF_TOKEN="${HF_TOKEN:-}"      # PUBLIC repo: token optional (kept if present in .env)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"   # 0=robust, 1=fast (may hang)
mkdir -p "$FEAT_DIR" "$DATA_ROOT/metric_a" "$PARTS_DIR"

python -c "import huggingface_hub" 2>/dev/null \
  || { echo "ERR: pip install 'huggingface_hub>=0.24'  (in the data env) first"; exit 2; }

# ---- download parts (feat_archive/) + json ----
python - "$HF_REPO" "$PARTS_DIR" "$DATA_ROOT" <<'PY'
import os, sys
from huggingface_hub import snapshot_download, hf_hub_download
repo, parts_dir, data_root = sys.argv[1], sys.argv[2], sys.argv[3]
tok = os.environ.get("HF_TOKEN") or None    # public repo: None is fine
snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=["feat_archive/*"],
                  local_dir=parts_dir, token=tok)   # -> parts_dir/feat_archive/*
# sft json is NOT in the zip-only repo by default; fetch if present, else note (transfer separately)
try:
    hf_hub_download(repo_id=repo, repo_type="dataset", filename="metric_a/sft_metric_a_train.json",
                    local_dir=data_root, token=tok)
    print("[hf] sft json fetched")
except Exception:
    print("[hf] note: sft json not in repo (zip-only) -> transfer it separately")
print("[hf] download complete")
PY

AR="$PARTS_DIR/feat_archive"
[ -d "$AR" ] || { echo "ERR: feat_archive not downloaded under $PARTS_DIR"; exit 2; }
echo "[manifest]"; cat "$AR/MANIFEST.txt" 2>/dev/null || echo "(no MANIFEST.txt)"

# ---- reassemble + extract (auto-detect plain vs gzip parts) ----
echo "[untar] extracting -> $FEAT_DIR (transient ~43GB)"
if ls "$AR"/train_feat_conch.tar.gz.part* >/dev/null 2>&1; then
  GZ="$(command -v pigz || command -v gzip)"
  cat "$AR"/train_feat_conch.tar.gz.part* | "$GZ" -d | tar -xf - -C "$DATA_ROOT" \
    || { echo "ERR: cat|gunzip|tar failed"; exit 2; }
elif ls "$AR"/train_feat_conch.tar.part* >/dev/null 2>&1; then
  cat "$AR"/train_feat_conch.tar.part* | tar -xf - -C "$DATA_ROOT" \
    || { echo "ERR: cat|tar failed"; exit 2; }
else
  echo "ERR: no tar parts found under $AR"; exit 2
fi

# ---- verify count against manifest ----
n_h5=$(ls "$FEAT_DIR"/*.h5 2>/dev/null | wc -l)
exp=$(sed -n 's/^n_h5=//p' "$AR/MANIFEST.txt" 2>/dev/null)
echo "[done] train_feat_conch .h5 = $n_h5  (manifest expected: ${exp:-?})"
[ -n "$exp" ] && [ "$n_h5" != "$exp" ] && echo "  WARN: count mismatch ($n_h5 != $exp) -- re-check download/extract"
ls -lh "$DATA_ROOT/metric_a/sft_metric_a_train.json" 2>/dev/null || echo "  WARN: sft json missing"
echo "  (rm -rf $PARTS_DIR to reclaim ~43GB once verified)"
