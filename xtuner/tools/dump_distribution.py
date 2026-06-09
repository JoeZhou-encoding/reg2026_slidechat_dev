#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_distribution.py — detailed distribution of the CONCH feature dump (read-only).

  A. N (patches/slide): full percentiles, histogram buckets, #capped, #tiny, total patches, GB.
  B. by series (PIT_0X prefix) and by organ (if --cot): N stats per group.
  C. feature health (sample K slides, streaming accumulation):
       overall mean/std, per-DIM std distribution + #dead dims, global value percentiles,
       per-patch L2 norm distribution.

Run with the dump env (login node):
  python dump_distribution.py --out $REG_ROOT/train_feat_conch --cot $REG_ROOT/train_CoT.json
"""
import argparse, glob, json, os, random
import numpy as np
import h5py


def pctl(a, ps):
    q = np.percentile(a, ps)
    return "  ".join(f"p{p}={int(v)}" if v == int(v) else f"p{p}={v:.3f}" for p, v in zip(ps, q))


def stem_of(b):
    b = os.path.basename(b)
    for suf in (".h5", ".tiff"):
        if b.endswith(suf):
            return b[:-len(suf)]
    return os.path.splitext(b)[0]


def series_of(stem):
    # PIT_01_00019_01 -> PIT_01
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def group_stats(name, groups):
    print(f"\n  by {name}:")
    print(f"    {'group':12} {'#slide':>7} {'medN':>7} {'meanN':>7} {'maxN':>7} {'%capped':>8} {'%>10240':>8}")
    for g in sorted(groups):
        a = np.array(groups[g])
        capped = 100 * (a >= 20480).mean()
        over = 100 * (a > 10240).mean()
        print(f"    {g:12} {len(a):>7} {int(np.median(a)):>7} {a.mean():>7.0f} {a.max():>7} "
              f"{capped:>7.1f}% {over:>7.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dir with .h5")
    ap.add_argument("--cot", default=None, help="train_CoT.json for per-organ split (optional)")
    ap.add_argument("--cap", type=int, default=20480)
    ap.add_argument("--sample-num", type=int, default=10240)
    ap.add_argument("--sample", type=int, default=64, help="#slides for feature-health stats")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    h5s = sorted(glob.glob(os.path.join(a.out, "*.h5")))
    if not h5s:
        print("no .h5 found"); return
    print(f"#h5 = {len(h5s):,}   (dir: {a.out})")

    # ---------- A. N distribution (attr-only, all) ----------
    organ = {}
    if a.cot and os.path.isfile(a.cot):
        for c in json.load(open(a.cot, encoding="utf-8")):
            organ[stem_of(c["id"])] = c.get("organ", "?")
    Ns, by_series, by_organ = [], {}, {}
    for p in h5s:
        st = stem_of(p)
        with h5py.File(p, "r") as f:
            n = int(f.attrs.get("n_patches", f["features"].shape[0]))
        Ns.append(n)
        by_series.setdefault(series_of(st), []).append(n)
        if organ:
            by_organ.setdefault(organ.get(st, "?"), []).append(n)
    Ns = np.array(Ns)
    total = int(Ns.sum())
    gb = total * 512 * 2 / 1e9  # fp16
    print("\n" + "=" * 70 + "\nA. N (patches / slide)\n" + "=" * 70)
    print("  " + pctl(Ns, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]))
    print(f"  mean={Ns.mean():.0f}  std={Ns.std():.0f}")
    buckets = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5000),
               (5000, 10240), (10240, 20480), (20480, 10**9)]
    print("  histogram:")
    for lo, hi in buckets:
        m = (Ns >= lo) & (Ns < hi) if hi < 10**9 else (Ns >= lo)
        c = int(m.sum())
        bar = "#" * int(60 * c / len(Ns))
        lbl = f"[{lo:>5}, cap]" if lo == 20480 else f"[{lo:>5},{hi:>6})"
        print(f"    {lbl:16} {c:>6} ({100*c/len(Ns):4.1f}%) {bar}")
    n_cap = int((Ns >= a.cap).sum()); n_tiny = int((Ns < 100).sum())
    print(f"  == cap({a.cap}): {n_cap} ({100*n_cap/len(Ns):.1f}%)   "
          f"< 100 patches: {n_tiny} ({100*n_tiny/len(Ns):.1f}%)   "
          f"> sample_num({a.sample_num}): {int((Ns>a.sample_num).sum())} "
          f"({100*(Ns>a.sample_num).mean():.1f}%)")
    print(f"  TOTAL patches = {total:,}   feature size (fp16) ~ {gb:.1f} GB")

    # ---------- B. by series / organ ----------
    print("\n" + "=" * 70 + "\nB. N by group\n" + "=" * 70)
    group_stats("series (PIT_0X)", by_series)
    if by_organ:
        group_stats("organ", by_organ)

    # ---------- C. feature health (sample, streaming) ----------
    print("\n" + "=" * 70 + f"\nC. feature health (sample {min(a.sample,len(h5s))} slides)\n" + "=" * 70)
    rng = random.Random(a.seed)
    sample = rng.sample(h5s, min(a.sample, len(h5s)))
    D = 512
    s = np.zeros(D); ss = np.zeros(D); cnt = 0
    l2s = []
    val_chunks = []  # subsampled raw values for global percentiles
    gmin, gmax = np.inf, -np.inf
    for p in sample:
        with h5py.File(p, "r") as f:
            x = f["features"][:].astype(np.float32)
        s += x.sum(0); ss += (x * x).sum(0); cnt += x.shape[0]
        l2s.append(np.linalg.norm(x, axis=1))
        gmin = min(gmin, float(x.min())); gmax = max(gmax, float(x.max()))
        val_chunks.append(x[::97].ravel())  # sparse subsample for value histogram
    per_dim_mean = s / cnt
    per_dim_std = np.sqrt(np.maximum(ss / cnt - per_dim_mean**2, 0))
    l2 = np.concatenate(l2s)
    vals = np.concatenate(val_chunks)
    dead = int((per_dim_std < 1e-3).sum())
    print(f"  pooled patches = {cnt:,}  over {len(sample)} slides")
    print(f"  overall: mean={per_dim_mean.mean():+.4f}  std={per_dim_std.mean():.4f}  "
          f"value range=[{gmin:.2f}, {gmax:.2f}]")
    print(f"  per-DIM std:  min={per_dim_std.min():.3f}  p50={np.median(per_dim_std):.3f}  "
          f"max={per_dim_std.max():.3f}   dead dims(std<1e-3)={dead}/512")
    print(f"  per-DIM mean: min={per_dim_mean.min():+.3f}  p50={np.median(per_dim_mean):+.3f}  "
          f"max={per_dim_mean.max():+.3f}")
    print(f"  value percentiles: " + pctl(vals, [0, 1, 50, 99, 100]))
    print(f"  per-patch L2 norm: " + pctl(l2, [0, 1, 50, 99, 100]) + f"   (sqrt(512)={np.sqrt(512):.1f})")
    print("\n  (mean~0 / std~1 / L2~sqrt(512) / 0 dead dims = healthy ViT/CONCH embeddings)")


if __name__ == "__main__":
    main()
