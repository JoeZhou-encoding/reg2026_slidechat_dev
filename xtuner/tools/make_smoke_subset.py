#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_smoke_subset.py — carve a tiny subset of the SFT json for a 1-epoch training smoke.

Picks the N examples whose feature .h5 has the LARGEST patch count (and that actually exist
under image_folder), so the smoke deterministically exercises the WORST-CASE memory path: the
loader downsamples only when patches >= sample_num (10240) -> the biggest WSIs produce the full
10240-visual-token sequence, which is exactly where peak GPU memory / OOM would bite. A smoke
that selected the first-64-with-h5 could miss this entirely (review finding, medium).

  python xtuner/tools/make_smoke_subset.py \
    --full   $REG2026/data/reg2026/metric_a/sft_metric_a_train.json \
    --image-folder $REG2026/data/reg2026/train_feat_conch \
    --out    $REG2026/data/reg2026/metric_a/sft_metric_a_train.SMOKE64.json \
    --n 64 --sample-num 10240
"""
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--sample-num", type=int, default=10240,
                    help="loader downsample cap; we report how many picked WSIs reach it")
    args = ap.parse_args()

    import h5py  # only to read exact patch counts for the PICKED few (the report)

    data = json.load(open(args.full, encoding="utf-8"))
    # Rank by FILE SIZE (one stat per WSI) as a proxy for patch count: .h5 holds (N, 512) fp16, so file
    # size grows with N. This avoids OPENING all ~11k HDF5 files, which is metadata-bound and painfully
    # slow on a parallel FS (Lustre/GPFS) -> seconds instead of 10+ minutes.
    size_cache = {}           # h5 path -> file size (dedup: many questions share one WSI)
    cand, missing = [], 0     # (size, example)
    for ex in data:
        imgs = ex.get("image") or []
        if not imgs:
            continue
        p = os.path.join(args.image_folder, imgs[0])
        s = size_cache.get(p, -1)
        if s == -1:
            s = os.path.getsize(p) if os.path.isfile(p) else None
            size_cache[p] = s
        if s is None:
            missing += 1
            continue
        cand.append((s, ex))

    if not cand:
        print(f"ERR: no example with a readable .h5 under {args.image_folder} (missing={missing})")
        sys.exit(2)

    # largest files first -> largest-N WSIs -> exercises the >=sample_num downsample / peak-mem path
    cand.sort(key=lambda t: t[0], reverse=True)
    picked = [ex for _, ex in cand[:args.n]]

    json.dump(picked, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # exact patch counts for the report: open ONLY the picked few (fast)
    picked_n = []
    for ex in picked:
        try:
            with h5py.File(os.path.join(args.image_folder, ex["image"][0]), "r") as f:
                picked_n.append(int(f["features"].shape[0]))
        except Exception:
            pass
    print(f"wrote {len(picked)} examples -> {args.out}  (skipped {missing} with missing .h5)")
    if picked_n:
        n_ge = sum(1 for n in picked_n if n >= args.sample_num)
        print(f"  patch count in subset: min={min(picked_n)} max={max(picked_n)} "
              f"(>= sample_num {args.sample_num}: {n_ge}/{len(picked_n)})")
        if n_ge == 0:
            print(f"  NOTE: no picked WSI reaches sample_num={args.sample_num}; peak is "
                  f"max={max(picked_n)} tokens (no train slide is that large). Informational, not an error.")


if __name__ == "__main__":
    main()
