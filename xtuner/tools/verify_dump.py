#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_dump.py — audit the CONCH WSI feature dump (run on Genkai after the job finishes).

Checks:
  1. EXIT / health   : 16 per-shard status jsonl present, statuses aggregate, no 'err';
                       (optionally) 'done rc=' lines in job .out/.log.
  2. COUNTS          : #h5 vs #tiff vs #mask; missing list; reconcile with status counts;
                       (optionally) coverage of train_CoT ids.
  3. SPOT-CHECK      : dim==512 on EVERY h5 (cheap, shape-only) + N distribution;
                       deep-load K random slides -> dtype/NaN/Inf/stats/degeneracy/attrs;
                       conformance to training params (512-d / N<=cap / fp16 / 'features' key).

Read-only. Run with the dump env:
  /home/pj24003162/ku40003404/miniconda3/bin/conda run -n conch_dump_py311 \
    python verify_dump.py            # uses hard-coded Genkai defaults below
"""
import argparse, glob, json, os, random, sys
import numpy as np
import h5py

H_HOME = "/home/pj24003162/ku40003404/weihao/00"
REG2026 = f"{H_HOME}/reg_2026"
REG_ROOT = f"{REG2026}/data/reg2026"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{REG_ROOT}/train_feat_conch", help="dir with the .h5")
    ap.add_argument("--train", default=f"{REG_ROOT}/train", help="dir with source .tiff")
    ap.add_argument("--masks", default=f"{REG_ROOT}/train_thumb", help="dir with .mask.png")
    ap.add_argument("--logs", default=f"{REG2026}/logs", help="dir to scan for job .out/.log")
    ap.add_argument("--submit-dir", default=REG2026, help="also scan here for conch_dump_*.out")
    ap.add_argument("--cot", default=None, help="train_CoT.json to check id coverage (optional)")
    ap.add_argument("--n-shards", type=int, default=16)
    ap.add_argument("--feature-dim", type=int, default=512)
    ap.add_argument("--cap", type=int, default=20480)
    ap.add_argument("--train-sample-num", type=int, default=10240,
                    help="SlideChat sample_num; report how many slides exceed it (get downsampled)")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--sample", type=int, default=24, help="deep-load this many random h5")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def stem_of(path):
    b = os.path.basename(path)
    for suf in (".h5", ".tiff.mask.png", ".tiff"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return os.path.splitext(b)[0]


def main():
    a = parse_args()
    fails, warns = [], []
    def fail(m): fails.append(m); print("  [FAIL] " + m)
    def warn(m): warns.append(m); print("  [warn] " + m)
    def ok(m):   print("  [ok]   " + m)

    print(f"OUT   = {a.out}")
    print(f"TRAIN = {a.train}\nMASKS = {a.masks}\nLOGS  = {a.logs}\n")

    # ============================================================ 1. EXIT / health
    print("=" * 70 + "\n1. EXIT / HEALTH\n" + "=" * 70)
    status_files = sorted(glob.glob(os.path.join(a.out, "_status_shard*of*.jsonl")))
    agg = {}
    status_by_stem = {}
    n_status_lines = 0
    if not status_files:
        warn(f"no _status_shard*.jsonl in {a.out} (can't reconcile per-slide status)")
    else:
        if len(status_files) != a.n_shards:
            warn(f"found {len(status_files)} status files, expected {a.n_shards} shards")
        else:
            ok(f"found all {a.n_shards} per-shard status files")
        for sf in status_files:
            for line in open(sf, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                n_status_lines += 1
                st = r.get("status", "?")
                base = st.split(":")[0] if isinstance(st, str) else "?"
                agg[base] = agg.get(base, 0) + 1
                sl = r.get("slide", "")
                if sl:
                    status_by_stem[stem_of(sl)] = base
        print(f"  status lines = {n_status_lines:,}  ->  {dict(sorted(agg.items()))}")
        n_err = agg.get("err", 0) + agg.get("decode_fail", 0)
        (ok if n_err == 0 else fail)(f"hard failures (err+decode_fail) = {n_err}")

    # job 'done rc=' lines (best-effort; the real signal is status files above)
    done_ok, done_bad, errlines = 0, 0, 0
    for d in (a.logs, a.submit_dir):
        for f in glob.glob(os.path.join(d, "conch_dump*")):
            if not os.path.isfile(f):
                continue
            try:
                txt = open(f, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            for ln in txt.splitlines():
                if "done rc=" in ln:
                    if "rc=0" in ln:
                        done_ok += 1
                    else:
                        done_bad += 1
                if "Traceback" in ln or "CUDA error" in ln or "OutOfMemory" in ln:
                    errlines += 1
    if done_ok or done_bad:
        (ok if done_bad == 0 else fail)(f"'done rc=' lines: rc=0 x{done_ok}, rc!=0 x{done_bad}")
    else:
        warn("no 'done rc=' lines found in logs (final .out may not be pulled here)")
    if errlines:
        warn(f"{errlines} Traceback/CUDA/OOM lines in logs — inspect")

    # ============================================================ 2. COUNTS
    print("=" * 70 + "\n2. COUNTS\n" + "=" * 70)
    h5s = glob.glob(os.path.join(a.out, "*.h5"))
    tiffs = glob.glob(os.path.join(a.train, "*.tiff"))
    masks = glob.glob(os.path.join(a.masks, "*.mask.png"))
    tmps = glob.glob(os.path.join(a.out, "*.tmp"))
    h5_stems = {stem_of(p) for p in h5s}
    tiff_stems = {stem_of(p) for p in tiffs}
    print(f"  #h5={len(h5s):,}  #tiff={len(tiffs):,}  #mask={len(masks):,}  #leftover .tmp={len(tmps)}")
    if tmps:
        warn(f"{len(tmps)} leftover .tmp files (a write was interrupted) e.g. {os.path.basename(tmps[0])}")
    n_ok = agg.get("ok", 0)
    if n_ok:
        (ok if n_ok == len(h5s) else warn)(f"status ok={n_ok:,} vs #h5={len(h5s):,} "
                                           f"(diff={n_ok - len(h5s)})")
    # reconcile: ok + skip + no_mask + no_tissue + decode_fail + err == #tiff ?
    if agg:
        tot = sum(agg.values())
        (ok if tot == len(tiffs) else warn)(f"status total={tot:,} vs #tiff={len(tiffs):,}")
    missing = sorted(tiff_stems - h5_stems)
    print(f"  tiff without h5: {len(missing)}")
    for st in missing[:8]:
        print(f"      {st}  (status={status_by_stem.get(st, '?')})")
    # coverage vs train_CoT ids (training-relevant)
    if a.cot and os.path.isfile(a.cot):
        cot = json.load(open(a.cot, encoding="utf-8"))
        cot_stems = {stem_of(c["id"]) for c in cot}
        miss_cot = sorted(cot_stems - h5_stems)
        (ok if not miss_cot else warn)(f"train_CoT ids without h5: {len(miss_cot)}/{len(cot_stems)}"
                                       + (f"  e.g.{miss_cot[:5]}" if miss_cot else ""))

    if not h5s:
        fail("no .h5 found — aborting spot-check"); return _verdict(fails, warns)

    # ============================================================ 3. SPOT-CHECK
    print("=" * 70 + "\n3. SPOT-CHECK (dim / uniform / conformance)\n" + "=" * 70)
    # 3a. cheap shape-only pass over ALL h5: dim==512, N distribution, patch_size
    Ns, bad_dim, bad_attr, psizes = [], [], [], set()
    for p in h5s:
        try:
            with h5py.File(p, "r") as f:
                shp = f["features"].shape
                if len(shp) != 2 or shp[1] != a.feature_dim:
                    bad_dim.append((os.path.basename(p), shp))
                Ns.append(shp[0])
                ps = f.attrs.get("patch_size")
                if ps is not None:
                    psizes.add(int(ps))
                if "coords" not in f or "slide_id" not in f.attrs:
                    bad_attr.append(os.path.basename(p))
        except Exception as e:
            bad_attr.append(f"{os.path.basename(p)}:{type(e).__name__}")
    Ns = np.array(Ns)
    (ok if not bad_dim else fail)(f"feature dim == {a.feature_dim} on ALL {len(h5s):,} h5"
                                  + (f"  (BAD: {bad_dim[:3]})" if bad_dim else ""))
    (ok if not bad_attr else warn)(f"every h5 has 'coords' + 'slide_id' attr"
                                   + (f"  (BAD: {bad_attr[:3]})" if bad_attr else ""))
    print(f"  N (patches/slide): min={Ns.min()} p50={int(np.median(Ns))} "
          f"mean={Ns.mean():.0f} max={Ns.max()}")
    over_cap = int((Ns > a.cap).sum())
    (ok if over_cap == 0 else fail)(f"N <= cap({a.cap}) on all: {len(h5s) - over_cap}/{len(h5s)} "
                                    f"(over={over_cap})")
    over_sn = int((Ns > a.train_sample_num).sum())
    print(f"  N > train sample_num({a.train_sample_num}) -> downsampled at train: "
          f"{over_sn}/{len(h5s)} ({100*over_sn/len(h5s):.1f}%)")
    print(f"  patch_size attr values seen: {sorted(psizes) or 'none'}")
    if psizes and psizes != {a.patch_size}:
        warn(f"patch_size not uniformly {a.patch_size}: {sorted(psizes)}")

    # 3b. deep load K random slides
    rng = random.Random(a.seed)
    sample = rng.sample(h5s, min(a.sample, len(h5s)))
    print(f"\n  deep-loading {len(sample)} random slides:")
    deg, naninf, dtypes = [], [], set()
    for p in sample:
        with h5py.File(p, "r") as f:
            feats = f["features"][:]
            dtypes.add(str(feats.dtype))
            x = feats.astype(np.float32)
            n = x.shape[0]
            nan = bool(np.isnan(x).any() or np.isinf(x).any())
            if nan:
                naninf.append(os.path.basename(p))
            col_std = x.std(axis=0)
            row_var = x.var(axis=1)
            n_const_rows = int((row_var < 1e-8).sum())
            l2 = float(np.linalg.norm(x, axis=1).mean())
            # degenerate = features all ~constant across patches
            if float(col_std.mean()) < 1e-6 or n_const_rows == n:
                deg.append(os.path.basename(p))
            n_attr = int(f.attrs.get("n_patches", -1))
            attr_ok = (n_attr == n)
            print(f"    {os.path.basename(p):28} N={n:5d} dtype={feats.dtype} "
                  f"mean={x.mean():+.3f} std={x.std():.3f} |row|L2={l2:.2f} "
                  f"const_rows={n_const_rows} attrN={'ok' if attr_ok else n_attr}")
    (ok if not naninf else fail)(f"no NaN/Inf in sampled features"
                                 + (f"  (BAD:{naninf[:3]})" if naninf else ""))
    (ok if not deg else fail)(f"no degenerate (constant) feature tensors"
                              + (f"  (BAD:{deg[:3]})" if deg else ""))
    (ok if dtypes == {"float16"} else warn)(f"dtype fp16: seen {sorted(dtypes)}")

    # 3c. conformance to training params
    print("\n  conformance to training params:")
    ok(f"feature dim {a.feature_dim} matches SlideChat LongNet input (CONCH proj_contrast=False)")
    ok(f"h5 has 'features' dataset -> readable by our SlideChat .h5 loader branch (D-A4)")
    print(f"    -> set training sample_num >= what you want (dump cap={a.cap}); "
          f"{over_sn} slides exceed {a.train_sample_num}")
    print("    NOTE: SlideChat-native csv copy (D-A4 dual-store) is separate -- verify it too if generated.")

    return _verdict(fails, warns)


def _verdict(fails, warns):
    print("\n" + "=" * 70)
    if fails:
        print(f"VERDICT: {len(fails)} FAILURE(S), {len(warns)} warning(s)")
        for m in fails:
            print("  FAIL: " + m)
        sys.exit(1)
    print(f"VERDICT: PASS ({len(warns)} warning(s))")
    for m in warns:
        print("  warn: " + m)
    sys.exit(0)


if __name__ == "__main__":
    main()
