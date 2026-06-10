#!/usr/bin/env bash
# =============================================================================
# Relay Metric A processed data Genkai -> HF Hub (PRIVATE dataset). Run on the GENKAI
# LOGIN node (has network). TSUBAME then downloads via hf_download_metric_a_data.sh.
#
# 11109 individual .h5 are TOO MANY for a smooth HF upload, so we tar train_feat_conch into
# ~5GB SPLIT PARTS (a few files HF handles easily, resumable) + upload the SFT json.
# Default: plain tar (fp16 features barely gzip; the win is file-count, not bytes).
#   COMPRESS=gzip  -> pipe through pigz/gzip (smaller, slower). PART_SIZE=5G by default.
#
#   HF_REPO=zzqsb/reg2026-metric-a-data  bash hf_upload_metric_a_data.sh
#
# ⚠️ PUBLISHES data to HF (repo created PRIVATE). Confirm before running. Token from .env only.
# Disk: needs ~43GB free at PARTS_DIR (override PARTS_DIR=/path if /home is tight).
# =============================================================================
set -uo pipefail

REG2026="${REG2026:-/home/pj24003162/ku40003404/weihao/00/reg_2026}"
HF_REPO="${HF_REPO:-zzqsb/reg2026-metric-a-data}"
DATA_ROOT="${DATA_ROOT:-$REG2026/data/reg2026}"
FEAT_DIR="${FEAT_DIR:-$DATA_ROOT/train_feat_conch}"
SFT_JSON="${SFT_JSON:-$DATA_ROOT/metric_a/sft_metric_a_train.json}"
PARTS_DIR="${PARTS_DIR:-$REG2026/tmp/feat_parts}"
PART_SIZE="${PART_SIZE:-5G}"
COMPRESS="${COMPRESS:-none}"          # none | gzip

# ---- token from .env (never commit/print) ----
[ -f "$REG2026/.env" ] && source "$REG2026/.env"
: "${HF_TOKEN:?HF_TOKEN not set (put it in $REG2026/.env)}"
export HF_TOKEN
# hf_transfer can hang silently on some networks -> default OFF (standard upload shows progress,
# is robust + resumable). Opt in for speed with HF_HUB_ENABLE_HF_TRANSFER=1 if your link is clean.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# ---- pre-flight ----
[ -d "$FEAT_DIR" ] || { echo "ERR: feature dir missing: $FEAT_DIR"; exit 2; }
[ -f "$SFT_JSON" ] || { echo "ERR: SFT json missing: $SFT_JSON"; exit 2; }
python -c "import huggingface_hub" 2>/dev/null \
  || { echo "ERR: pip install 'huggingface_hub>=0.24'  (in this env) first"; exit 2; }
echo "[hf] HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER (0=standard/robust, 1=fast)"
n_h5=$(ls "$FEAT_DIR"/*.h5 2>/dev/null | wc -l)
echo "repo=$HF_REPO  feat_dir=$FEAT_DIR (.h5=$n_h5)  compress=$COMPRESS  part_size=$PART_SIZE"

# ---- tar -> split parts ----
rm -rf "$PARTS_DIR"; mkdir -p "$PARTS_DIR"
echo "[tar] bundling train_feat_conch -> $PARTS_DIR (this writes ~43GB; needs free space)"
if [ "$COMPRESS" = "gzip" ]; then
  GZ="$(command -v pigz || command -v gzip)"; echo "[tar] compress via $GZ"
  tar -cf - -C "$DATA_ROOT" train_feat_conch | "$GZ" \
    | split -b "$PART_SIZE" -d -a 3 - "$PARTS_DIR/train_feat_conch.tar.gz.part" \
    || { echo "ERR: tar|gzip|split failed"; exit 2; }
else
  tar -cf - -C "$DATA_ROOT" train_feat_conch \
    | split -b "$PART_SIZE" -d -a 3 - "$PARTS_DIR/train_feat_conch.tar.part" \
    || { echo "ERR: tar|split failed"; exit 2; }
fi
# manifest: expected .h5 count + part list (download verifies against it)
echo "n_h5=$n_h5" > "$PARTS_DIR/MANIFEST.txt"
( cd "$PARTS_DIR" && ls -1 train_feat_conch.tar* >> MANIFEST.txt && du -sh . )
echo "[tar] parts:"; ls -lh "$PARTS_DIR"/train_feat_conch.tar* | head

# ---- create private repo + upload parts (folder = few files now) + json ----
python - "$HF_REPO" "$PARTS_DIR" "$SFT_JSON" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, parts_dir, sft_json = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
print(f"[hf] repo ready (private): {repo}")
api.upload_file(path_or_fileobj=sft_json, path_in_repo="metric_a/sft_metric_a_train.json",
                repo_id=repo, repo_type="dataset", commit_message="sft json")
print("[hf] uploaded metric_a/sft_metric_a_train.json")
api.upload_folder(folder_path=parts_dir, path_in_repo="feat_archive",
                  repo_id=repo, repo_type="dataset", commit_message="feature tar parts")
print("[hf] uploaded feat_archive/ (tar parts + MANIFEST.txt)")
PY
echo "[done] -> https://huggingface.co/datasets/$HF_REPO (PRIVATE)"
echo "next on TSUBAME: HF_REPO=$HF_REPO bash hf_download_metric_a_data.sh"
echo "(you may now rm -rf $PARTS_DIR to reclaim space once upload verified)"
