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

    import h5py  # train env has it (B4 loader); fail loudly if not

    data = json.load(open(args.full, encoding="utf-8"))
    shape_cache = {}          # h5 path -> patch count (dedup: many questions share one WSI)
    cand, missing = [], 0     # (n_patch, example)
    for ex in data:
        imgs = ex.get("image") or []
        if not imgs:
            continue
        p = os.path.join(args.image_folder, imgs[0])
        n = shape_cache.get(p, -1)
        if n == -1:
            if not os.path.isfile(p):
                shape_cache[p] = None
                missing += 1
                continue
            try:
                with h5py.File(p, "r") as f:
                    n = int(f["features"].shape[0])   # cheap: reads shape metadata only
            except Exception as e:
                print(f"  warn: cannot read {p}: {e}")
                n = None
            shape_cache[p] = n
        if n is None:
            missing += 1
            continue
        cand.append((n, ex))

    if not cand:
        print(f"ERR: no example with a readable .h5 under {args.image_folder} (missing={missing})")
        sys.exit(2)

    # largest-N WSIs first -> guarantees the >=sample_num downsample path is exercised
    cand.sort(key=lambda t: t[0], reverse=True)
    picked = [ex for _, ex in cand[:args.n]]
    picked_n = [n for n, _ in cand[:args.n]]

    json.dump(picked, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_ge = sum(1 for n in picked_n if n >= args.sample_num)
    print(f"wrote {len(picked)} examples -> {args.out}")
    print(f"  patch count in subset: min={min(picked_n)} max={max(picked_n)} "
          f"(>= sample_num {args.sample_num}: {n_ge}/{len(picked_n)})")
    print(f"  global max patch count over all candidates = {cand[0][0]} "
          f"(stem={cand[0][1]['image'][0]})")
    if n_ge == 0:
        print(f"  NOTE: no picked WSI reaches sample_num={args.sample_num}; the smoke will NOT "
              f"exercise the 10240-token peak (no train slide is that large) -> peak is "
              f"max={max(picked_n)} tokens. This is informational, not an error.")


if __name__ == "__main__":
    main()
