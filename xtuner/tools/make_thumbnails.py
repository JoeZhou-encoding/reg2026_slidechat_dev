# Vendored into the slidechat monorepo from reg_data_2026 (author: ku40003404)
# so REG2026 Metric-A preprocessing (tissue mask + CONCH dump) lives together.
# The reg_data_2026 copy is the source of truth — keep in sync if it changes.
"""
Generate ~2K thumbnails + binary tissue masks for each slide in train/.

Approach:
  Slides up to 87K x 187K x 3 (~48 GB uncompressed) can't be imread()'d into
  RAM. We open the BigTIFF lazily via zarr and stream-decode in 8K x 8K blocks,
  mean-pooling each block to the target ratio before assembling the output.
  Memory per worker stays bounded at ~few hundred MB regardless of slide size.

Output layout (default --out-dir = <root>/train_thumb):
  {slide_id}.png       — RGB thumbnail, max(H,W) == TARGET_LONG_SIDE
  {slide_id}.mask.png  — uint8 binary mask (255 = tissue, 0 = background)

Idempotent: slides whose outputs already exist are skipped (use --force to redo).
"""

import argparse
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import zarr
from PIL import Image

DEFAULT_TARGET_LONG_SIDE = 2048
DEFAULT_CHUNK = 8192
TISSUE_THRESHOLD = 220   # gray < this → tissue. Tune for staining variance.


def _aligned_chunk(chunk: int, ratio: int) -> int:
    """Force the streaming chunk size to be a multiple of `ratio` so that
    block reshape-based mean-pooling is exact."""
    c = (chunk // ratio) * ratio
    return c if c > 0 else ratio


def downsample_streamed(arr: zarr.Array, ratio: int, chunk: int) -> np.ndarray:
    """Mean-pool `arr` by an integer `ratio`, streaming `chunk x chunk` blocks.

    Trailing pixels (h % ratio, w % ratio) are dropped — at ratio<=100 the
    cropped border is < 100 px in the original, < 1 px in the thumbnail.
    """
    h, w = arr.shape[:2]
    out_h, out_w = h // ratio, w // ratio
    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    cs = _aligned_chunk(chunk, ratio)

    for y0 in range(0, out_h * ratio, cs):
        y1 = min(y0 + cs, out_h * ratio)
        for x0 in range(0, out_w * ratio, cs):
            x1 = min(x0 + cs, out_w * ratio)
            block = np.asarray(arr[y0:y1, x0:x1])      # HWC uint8 RGB
            bh, bw = block.shape[:2]
            oh, ow = bh // ratio, bw // ratio
            block = block[: oh * ratio, : ow * ratio]
            # Mean-pool: reshape into (oh, ratio, ow, ratio, 3) then mean over the
            # ratio axes. uint16 cast avoids overflow during sum.
            pooled = (
                block.astype(np.uint16)
                .reshape(oh, ratio, ow, ratio, 3)
                .mean(axis=(1, 3))
                .astype(np.uint8)
            )
            out[y0 // ratio : y0 // ratio + oh, x0 // ratio : x0 // ratio + ow] = pooled
    return out


def tissue_mask(thumb: np.ndarray, threshold: int = TISSUE_THRESHOLD) -> np.ndarray:
    """Naive but effective: gray darker than threshold → tissue.
    HE-stained pathology background is ~245-255 brightness; tissue is < 230."""
    gray = thumb.mean(axis=-1)
    return (gray < threshold).astype(np.uint8) * 255


def process_one(args) -> Tuple[str, str, float, float]:
    slide_path, out_dir, target_long, chunk, force = args
    slide_id = slide_path.name
    out_thumb = out_dir / f"{slide_id}.png"
    out_mask = out_dir / f"{slide_id}.mask.png"
    if not force and out_thumb.exists() and out_mask.exists():
        return slide_id, "skip", 0.0, 0.0

    t0 = time.time()
    try:
        # series=0, level=0: some slides are pyramidal/multi-series -> default aszarr
        # returns a zarr Group (no .shape); this forces the full-res base Array.
        store = tifffile.imread(str(slide_path), aszarr=True, series=0, level=0)
        arr = zarr.open(store, mode="r")
        h, w = arr.shape[:2]
        ratio = max(1, max(h, w) // target_long)
        thumb = downsample_streamed(arr, ratio, chunk)
        mask = tissue_mask(thumb)
        Image.fromarray(thumb).save(out_thumb, optimize=False)
        Image.fromarray(mask).save(out_mask, optimize=False)
        tissue_frac = float(mask.mean() / 255.0)
        return slide_id, "ok", time.time() - t0, tissue_frac
    except Exception as e:
        traceback.print_exc()
        return slide_id, f"err:{type(e).__name__}:{e}", time.time() - t0, 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=os.environ.get(
            "REG2026_ROOT",
            "/home/pj24003162/ku40003404/weihao/00/reg_2026/data/reg2026",
        ),
    )
    p.add_argument("--out-dir", default=None,
                   help="Default: <root>/train_thumb")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--target-long-side", type=int, default=DEFAULT_TARGET_LONG_SIDE)
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N slides (smoke test)")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "train_thumb"
    out_dir.mkdir(parents=True, exist_ok=True)

    slides = sorted((root / "train").glob("*.tiff"))
    if args.limit:
        slides = slides[: args.limit]
    print(f"root          : {root}")
    print(f"out_dir       : {out_dir}")
    print(f"#slides       : {len(slides)}")
    print(f"workers       : {args.workers}")
    print(f"target_long   : {args.target_long_side}")
    print(f"force         : {args.force}", flush=True)

    tasks = [
        (p, out_dir, args.target_long_side, args.chunk, args.force)
        for p in slides
    ]
    t0 = time.time()
    ok = skip = err = 0
    tissue_fracs = []

    with Pool(args.workers) as pool:
        for i, (slide_id, status, dt, frac) in enumerate(
            pool.imap_unordered(process_one, tasks), 1
        ):
            if status == "ok":
                ok += 1
                tissue_fracs.append(frac)
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"!! {slide_id} : {status}", flush=True)
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(tasks) - i) / rate if rate > 0 else 0
                tf_mean = float(np.mean(tissue_fracs)) if tissue_fracs else 0.0
                print(
                    f"  [{i:5d}/{len(tasks)}] ok={ok} skip={skip} err={err}  "
                    f"avg_tissue={tf_mean:.2%}  "
                    f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                    flush=True,
                )

    print(f"\nDONE  ok={ok} skip={skip} err={err}  "
          f"total={(time.time() - t0) / 60:.1f}m")
    if tissue_fracs:
        a = np.array(tissue_fracs)
        print(f"tissue fraction: mean={a.mean():.2%} median={np.median(a):.2%} "
              f"p5={np.percentile(a,5):.2%} p95={np.percentile(a,95):.2%}")


if __name__ == "__main__":
    main()
