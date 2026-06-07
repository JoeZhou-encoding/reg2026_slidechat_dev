#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONCH WSI feature dump for REG2026 Metric A — option D: tissue-grid tiling, capped.

Reads single-layer BigTIFF WSIs via tifffile+zarr (NO OpenSlide — the REG2026
slides have no pyramid / no mpp), tiles ONLY the tissue regions (using the
precomputed mask from make_thumbnails.py), runs CONCH ViT-B/16 to get 512-d
per-patch features, caps the per-slide patch count, and writes one fp16 HDF5
per slide.

Pipeline (per slide):
  1. zarr-open the BigTIFF lazily            -> (H, W, 3) uint8 RGB
  2. load tissue mask  {tiff}.mask.png       -> thumbnail-res, 255 = tissue
  3. area-average the mask onto the patch grid (H//P, W//P) -> per-cell tissue frac
  4. keep cells with tissue_frac >= TISSUE_FRAC -> patch top-left coords (Y, X)
  5. if #patches > CAP, uniform (linspace) stride-subsample to CAP
  6. decode each patch (zarr only touches its 256x256 JPEG tiles) -> CONCH
     encode_image(proj_contrast=False, normalize=False) -> (N, 512)
  7. save fp16 HDF5: 'features' (N,512), 'coords' (N,2), + attrs

Why this matches SlideChat: SlideChat stores ALL tissue patches per WSI and
downsamples to sample_num (~20480) at train/infer time (xtuner/dataset/llava.py,
tools/test_full.py). Tiling the tissue grid and capping at CAP makes the STORED
set equal to what SlideChat would actually consume, while bounding disk/compute.
CONCH features use proj_contrast=False to match SlideChat's expected 512-d input.

Mask is REQUIRED: slides with a missing mask are SKIPPED and recorded (a missing
mask must not silently fall back to full-slide tiling, which would explode N).

Idempotent: a slide whose output .h5 already exists is skipped. Shardable for PJM
bulk jobs via --shard / --n-shards (round-robin so big slides spread across GPUs).

Smoke (login node, 1 slide):
  python extract_conch_features.py --root <data/reg2026> --conch <ckpt.bin> --limit 1
PJM bulk shard:
  python extract_conch_features.py --root ... --conch ... --shard $PJM_BULKNUM --n-shards 8

The data/IO/selection logic is unit-tested with a mock encoder in
test_extract_conch_features.py (no GPU / CONCH needed); real CONCH correctness is
verified on Genkai on one real slide.
"""

import argparse
import glob
import json
import os
import time
import traceback

import numpy as np
from PIL import Image

# Pillow resample constant (BOX = area-average); name differs across versions.
try:
    _RESAMPLE_BOX = Image.Resampling.BOX
except AttributeError:  # Pillow < 9.1
    _RESAMPLE_BOX = Image.BOX

FEATURE_DIM = 512
DEFAULT_PATCH = 256
DEFAULT_CAP = 20480
DEFAULT_TISSUE_FRAC = 0.1


# ---------------------------------------------------------------------------
# Patch selection (mask logic) — pure, unit-testable, no torch / IO of the WSI
# ---------------------------------------------------------------------------
def select_tissue_patches(mask, H, W, patch_size=DEFAULT_PATCH,
                          tissue_frac=DEFAULT_TISSUE_FRAC, cap=DEFAULT_CAP):
    """Pick non-overlapping `patch_size` grid cells whose tissue fraction (from the
    thumbnail mask) is >= `tissue_frac`; if more than `cap`, uniformly subsample.

    Args:
        mask: 2-D uint8 array, thumbnail resolution, >0 = tissue.
        H, W: full-resolution slide height / width.
    Returns:
        (N, 2) int64 array of full-resolution patch top-left coords (Y, X),
        each satisfying Y+patch_size <= H and X+patch_size <= W. Row-major order;
        deterministic.
    """
    n_y, n_x = H // patch_size, W // patch_size
    if n_y < 1 or n_x < 1:
        return np.zeros((0, 2), dtype=np.int64)

    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    # Area-average the mask onto the (n_y, n_x) patch grid -> per-cell tissue frac.
    grid = np.asarray(
        Image.fromarray(binary).resize((n_x, n_y), _RESAMPLE_BOX)
    ).astype(np.float32) / 255.0

    keep = np.argwhere(grid >= tissue_frac)          # (k, 2) -> (gy, gx)
    if keep.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    coords = np.stack([keep[:, 0] * patch_size, keep[:, 1] * patch_size], axis=1)

    if cap is not None and len(coords) > cap:
        idx = np.linspace(0, len(coords) - 1, cap).astype(np.int64)
        coords = coords[idx]
    return coords.astype(np.int64)


# ---------------------------------------------------------------------------
# Decode + encode — `preprocess` and `encode_fn` are injected (real CONCH or mock)
# ---------------------------------------------------------------------------
def decode_and_encode(arr, coords, patch_size, preprocess, encode_fn, batch_size=256):
    """Decode each patch from the lazy zarr array and encode in batches.

    `preprocess`: PIL.Image -> preprocessed patch (torch tensor for real CONCH).
    `encode_fn` : list[preprocessed] -> np.ndarray (B, FEATURE_DIM) float.
    Patches that fail to decode (corrupt tile) or have the wrong shape are skipped;
    returned features and coords stay aligned.
    """
    feats, kept = [], []
    buf, buf_coords = [], []

    def flush():
        if buf:
            feats.append(np.asarray(encode_fn(buf), dtype=np.float32))
            kept.extend(buf_coords)
            buf.clear()
            buf_coords.clear()

    for (Y, X) in coords:
        Y, X = int(Y), int(X)
        try:
            patch = np.asarray(arr[Y:Y + patch_size, X:X + patch_size])
        except Exception:
            continue
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue
        if patch.ndim == 2:
            patch = np.stack([patch] * 3, axis=-1)
        if patch.shape[2] == 4:
            patch = patch[:, :, :3]
        buf.append(preprocess(Image.fromarray(patch.astype(np.uint8))))
        buf_coords.append((Y, X))
        if len(buf) >= batch_size:
            flush()
    flush()

    if feats:
        F = np.concatenate(feats, axis=0).astype(np.float32)
    else:
        F = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return F, np.asarray(kept, dtype=np.int64).reshape(-1, 2)


def write_features_h5(out_path, feats, coords, slide_id, H, W,
                      patch_size, cap, tissue_frac):
    """Write fp16 features + int32 coords + provenance attrs. Atomic (tmp+rename)."""
    import h5py
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("features", data=feats.astype(np.float16),
                         compression="gzip", compression_opts=4)
        f.create_dataset("coords", data=coords.astype(np.int32))
        f.attrs["slide_id"] = slide_id
        f.attrs["n_patches"] = int(len(feats))
        f.attrs["patch_size"] = int(patch_size)
        f.attrs["cap"] = int(cap) if cap is not None else -1
        f.attrs["tissue_frac"] = float(tissue_frac)
        f.attrs["proj_contrast"] = False          # SlideChat-style pre-projection feats
        f.attrs["feature_dim"] = int(FEATURE_DIM)
        f.attrs["H"] = int(H)
        f.attrs["W"] = int(W)
        f.attrs["source"] = "extract_conch_features.py"
    os.replace(tmp, out_path)


def extract_one_wsi(tiff_path, mask_path, out_path, encode_fn, preprocess,
                    patch_size=DEFAULT_PATCH, cap=DEFAULT_CAP,
                    tissue_frac=DEFAULT_TISSUE_FRAC, batch_size=256):
    """Extract+save features for one slide. Returns a status dict (never raises)."""
    import tifffile
    import zarr
    slide_id = os.path.basename(tiff_path)

    if os.path.exists(out_path):
        return {"slide": slide_id, "status": "skip"}
    if not os.path.exists(mask_path):
        return {"slide": slide_id, "status": "no_mask"}

    store = None
    try:
        store = tifffile.imread(tiff_path, aszarr=True)
        arr = zarr.open(store, mode="r")
        H, W = arr.shape[:2]

        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        coords = select_tissue_patches(mask, H, W, patch_size, tissue_frac, cap)
        if len(coords) == 0:
            return {"slide": slide_id, "status": "no_tissue", "H": H, "W": W}

        feats, kept = decode_and_encode(arr, coords, patch_size,
                                        preprocess, encode_fn, batch_size)
        if len(feats) == 0:
            return {"slide": slide_id, "status": "decode_fail",
                    "n_candidates": int(len(coords))}

        write_features_h5(out_path, feats, kept, slide_id, H, W,
                          patch_size, cap, tissue_frac)
        return {"slide": slide_id, "status": "ok", "n": int(len(feats)),
                "n_candidates": int(len(coords)),
                "cap_hit": bool(cap is not None and len(coords) >= cap)}
    except Exception as e:
        traceback.print_exc()
        return {"slide": slide_id, "status": f"err:{type(e).__name__}:{e}"}
    finally:
        # Release the WSI file handle (Windows locks it; matters across 11220 slides).
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Real CONCH loader
# ---------------------------------------------------------------------------
def build_conch(ckpt, device="cuda", model_cfg="conch_ViT-B-16"):
    """Load CONCH and return (encode_fn, preprocess). Mirrors tools/test_full.py."""
    import torch
    from conch.open_clip_custom import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained(
        model_cfg, checkpoint_path=ckpt, device=device)
    model = model.eval()
    use_half = "cuda" in str(device)
    if use_half:
        model = model.to(device, dtype=torch.float16)

    def encode_fn(buf):
        t = torch.stack(buf).to(device)
        if use_half:
            t = t.to(torch.float16)
        with torch.no_grad():
            f = model.encode_image(t, proj_contrast=False, normalize=False)
        return f.float().cpu().numpy()

    return encode_fn, preprocess


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CONCH WSI feature dump (tissue-grid, capped)")
    p.add_argument("--root", default=os.environ.get(
        "REG2026_ROOT", "/home/pj24003162/ku40003404/weihao/00/reg_2026/data/reg2026"))
    p.add_argument("--slide-subdir", default="train")
    p.add_argument("--mask-subdir", default="train_thumb",
                   help="dir with {tiff}.mask.png from make_thumbnails.py")
    p.add_argument("--out", default=None, help="default: <root>/train_feat_conch")
    p.add_argument("--conch", required=True, help="CONCH checkpoint path (e.g. pytorch_model.bin)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--patch-size", type=int, default=DEFAULT_PATCH)
    p.add_argument("--cap", type=int, default=DEFAULT_CAP)
    p.add_argument("--tissue-frac", type=float, default=DEFAULT_TISSUE_FRAC)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="process first N slides (smoke)")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root
    slide_dir = os.path.join(root, args.slide_subdir)
    mask_dir = os.path.join(root, args.mask_subdir)
    out_dir = args.out or os.path.join(root, "train_feat_conch")
    os.makedirs(out_dir, exist_ok=True)

    slides = sorted(glob.glob(os.path.join(slide_dir, "*.tiff")))
    if args.limit:
        slides = slides[:args.limit]
    mine = [s for i, s in enumerate(slides) if i % args.n_shards == args.shard]

    print(f"root        : {root}")
    print(f"slides dir  : {slide_dir}  (#total={len(slides)})")
    print(f"mask dir    : {mask_dir}")
    print(f"out dir     : {out_dir}")
    print(f"shard       : {args.shard}/{args.n_shards}  -> {len(mine)} slides this shard")
    print(f"patch={args.patch_size} cap={args.cap} tissue_frac={args.tissue_frac} "
          f"batch={args.batch} device={args.device}", flush=True)

    encode_fn, preprocess = build_conch(args.conch, device=args.device)
    print("CONCH loaded.", flush=True)

    status_log = os.path.join(out_dir, f"_status_shard{args.shard}of{args.n_shards}.jsonl")
    counts = {}
    t0 = time.time()
    with open(status_log, "w", encoding="utf-8") as logf:
        for i, tiff in enumerate(mine, 1):
            tiff_name = os.path.basename(tiff)
            stem = os.path.splitext(tiff_name)[0]
            mask_path = os.path.join(mask_dir, tiff_name + ".mask.png")
            out_path = os.path.join(out_dir, stem + ".h5")

            r = extract_one_wsi(tiff, mask_path, out_path, encode_fn, preprocess,
                                patch_size=args.patch_size, cap=args.cap,
                                tissue_frac=args.tissue_frac, batch_size=args.batch)
            st = r["status"] if r["status"] in ("ok", "skip", "no_mask", "no_tissue",
                                                "decode_fail") else "err"
            counts[st] = counts.get(st, 0) + 1
            logf.write(json.dumps(r, ensure_ascii=False) + "\n")
            logf.flush()
            if st in ("no_mask", "err", "no_tissue", "decode_fail"):
                print(f"  !! {r}", flush=True)
            if i % 20 == 0 or i == len(mine):
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                eta = (len(mine) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(mine)}] {counts}  "
                      f"elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)

    print(f"\nDONE shard {args.shard}/{args.n_shards}  {counts}  "
          f"total={(time.time()-t0)/60:.1f}m")
    print(f"status log: {status_log}")
    no_mask = counts.get("no_mask", 0)
    if no_mask:
        print(f"WARNING: {no_mask} slides had NO mask and were SKIPPED — "
              f"run make_thumbnails.py for them first.")


if __name__ == "__main__":
    main()
