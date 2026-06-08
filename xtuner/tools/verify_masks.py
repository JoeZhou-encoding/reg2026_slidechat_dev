#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify the make_thumbnails.py output (tissue masks) and estimate how many CONCH
patches the dump will produce — BEFORE running the (expensive) CONCH dump.

Three checks:
  1. COMPLETENESS  — every train/*.tiff has a matching train_thumb/{id}.tiff.mask.png
     (and a thumbnail). Lists any missing.
  2. VALIDITY      — tissue fraction of every mask (mask-only, fast): distribution
     + count of near-empty (<1%, would be no_tissue) and suspicious (>60%) masks.
  3. PATCH ESTIMATE— on a random SAMPLE, open each TIFF for its shape ONLY (no pixel
     decode) and reuse extract_conch_features.select_tissue_patches to count the
     tissue patches the dump would keep; extrapolate to the full set + disk (fp16).

Run on Genkai (login node is fine — no GPU needed):
  conda activate $ENV_PREFIX        # conch_dump_py311 (has tifffile/zarr/PIL)
  python reg2026_slidechat_dev/xtuner/tools/verify_masks.py --root $REG_ROOT
"""
import argparse
import glob
import os
import random

import numpy as np
from PIL import Image

import extract_conch_features as E   # reuse the exact patch-selection logic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get(
        "REG_ROOT", "/home/pj24003162/ku40003404/weihao/00/reg_2026/data/reg2026"))
    ap.add_argument("--slide-subdir", default="train")
    ap.add_argument("--mask-subdir", default="train_thumb")
    ap.add_argument("--sample", type=int, default=300, help="#slides for patch estimate (0=all)")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--tissue-frac", type=float, default=0.1)
    ap.add_argument("--cap", type=int, default=20480)
    args = ap.parse_args()

    tr = os.path.join(args.root, args.slide_subdir)
    th = os.path.join(args.root, args.mask_subdir)
    slides = sorted(glob.glob(os.path.join(tr, "*.tiff")))
    mask_of = lambda p: os.path.join(th, os.path.basename(p) + ".mask.png")
    thumb_of = lambda p: os.path.join(th, os.path.basename(p) + ".png")

    print(f"root         : {args.root}")
    print(f"train slides : {len(slides)}")

    # ---- 1. completeness ----
    miss_mask = [p for p in slides if not os.path.exists(mask_of(p))]
    miss_thumb = [p for p in slides if not os.path.exists(thumb_of(p))]
    print("\n[1] COMPLETENESS")
    print(f"  masks present  : {len(slides) - len(miss_mask)} / {len(slides)}   missing={len(miss_mask)}")
    print(f"  thumbs present : {len(slides) - len(miss_thumb)} / {len(slides)}   missing={len(miss_thumb)}")
    for p in miss_mask[:10]:
        print(f"    MISSING mask: {os.path.basename(p)}")
    if len(miss_mask) > 10:
        print(f"    ... and {len(miss_mask) - 10} more")

    # ---- 2. validity: tissue fraction of every mask (mask-only, fast) ----
    fr = []
    bad = []
    for p in slides:
        m = mask_of(p)
        if not os.path.exists(m):
            continue
        a = np.asarray(Image.open(m))
        if a.ndim == 3:
            a = a[:, :, 0]
        fr.append(float((a > 0).mean()))
    fr = np.array(fr) if fr else np.zeros(0)
    print("\n[2] VALIDITY (per-mask tissue fraction)")
    if len(fr):
        pct = lambda x: np.percentile(fr, x)
        print(f"  mean={fr.mean():.1%} median={np.median(fr):.1%} "
              f"p5={pct(5):.1%} p95={pct(95):.1%} min={fr.min():.1%} max={fr.max():.1%}")
        print(f"  near-empty (<1% tissue, -> no_tissue at dump): {(fr < 0.01).sum()}")
        print(f"  suspicious (>60% tissue, check staining/threshold): {(fr > 0.60).sum()}")
        print(f"  (sanity: smoke run reported ~13% mean; far from that => mask threshold issue)")
    else:
        print("  no masks to score!")

    # ---- 3. patch estimate (sample; open TIFF for shape only) ----
    import tifffile
    import zarr
    have = [p for p in slides if os.path.exists(mask_of(p))]
    samp = have if args.sample in (0, None) or args.sample >= len(have) \
        else random.Random(0).sample(have, args.sample)
    print(f"\n[3] PATCH ESTIMATE (sample={len(samp)}, patch={args.patch_size}, "
          f"tissue_frac={args.tissue_frac}, cap={args.cap})")
    counts, capped, errs = [], 0, 0
    for p in samp:
        try:
            store = tifffile.imread(p, aszarr=True)
            arr = zarr.open(store, mode="r")
            H, W = arr.shape[:2]
            store.close()
        except Exception:
            errs += 1
            continue
        mk = np.asarray(Image.open(mask_of(p)))
        if mk.ndim == 3:
            mk = mk[:, :, 0]
        c = E.select_tissue_patches(mk, H, W, patch_size=args.patch_size,
                                    tissue_frac=args.tissue_frac, cap=None)
        counts.append(len(c))
        if len(c) > args.cap:
            capped += 1
    counts = np.array(counts) if counts else np.zeros(0, dtype=int)
    if len(counts):
        eff = np.minimum(counts, args.cap)            # patches actually stored (after cap)
        tot = eff.mean() * len(slides)                # extrapolate to all slides
        print(f"  patches/slide (pre-cap): mean={counts.mean():.0f} median={np.median(counts):.0f} "
              f"p95={np.percentile(counts,95):.0f} max={counts.max()}")
        print(f"  zero-tissue slides in sample : {(counts == 0).sum()}")
        print(f"  slides hitting cap(>{args.cap}) : {capped} / {len(counts)} "
              f"({capped/len(counts):.1%})")
        print(f"  tiff-open errors in sample   : {errs}")
        print(f"  => est TOTAL stored patches over {len(slides)} slides: {tot/1e6:.1f}M")
        print(f"  => est disk (fp16, 512-d)               : {tot*512*2/1e9:.1f} GB")
        print(f"  => est avg per-slide .h5 size           : {eff.mean()*512*2/1e6:.1f} MB")
    else:
        print("  no slides scored (all tiff-open failures?)")


if __name__ == "__main__":
    main()
