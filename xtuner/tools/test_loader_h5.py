#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_loader_h5.py — verify the .h5 feature loader branch == the .csv branch (B4).

Replicates the EXACT logic added to dataset/llava.py __getitem__ and dataset/utils.py
load_image (xtuner isn't importable locally). Asserts: a feature set saved as both
SlideChat-format .csv and our .h5 loads to the SAME tensor, with the SAME linspace
downsampling when N >= sample_num. Run: python test_loader_h5.py
"""
import os, sys, tempfile, shutil
import numpy as np, h5py
# NOTE: pandas is binary-broken in this LOCAL env (numpy bump), so the csv side uses numpy I/O
# mirroring the SAME data flow as pd.read_csv().iloc[:, :512] (first 512 cols). The downsample
# is identical to the library branch. On the server the real loader uses pandas on the same csv.


def load_csv(path, sample_num):                       # mirrors llava.py / utils.py .csv branch
    image = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(512))  # = iloc[:, :512]
    total = image.shape[0]
    if total >= sample_num:
        idx = np.linspace(0, total - 1, sample_num, dtype=int)
        image = image[idx][:sample_num]
    return image.astype(np.float32)


def load_h5(path, sample_num):                        # mirrors the new .h5 branch
    with h5py.File(path, 'r') as f:
        arr = f['features'][:]
    total = arr.shape[0]
    if total >= sample_num:
        idx = np.linspace(0, total - 1, sample_num, dtype=int)
        arr = arr[idx]
    return np.ascontiguousarray(arr, dtype=np.float32)


fails = 0
for N, sample_num in [(300, 10240), (15000, 10240), (10240, 10240), (37, 10240)]:
    feats = np.random.default_rng(0).standard_normal((N, 512)).astype(np.float16)
    d = tempfile.mkdtemp(prefix='ldtest_')
    try:
        h5p, csvp = os.path.join(d, 'x.h5'), os.path.join(d, 'x.csv')
        with h5py.File(h5p, 'w') as f:
            f.create_dataset('features', data=feats)
        # SlideChat-format csv: header (0..511,patch_name) + rows of 512 %.4f floats + patch name
        with open(csvp, 'w') as f:
            f.write(",".join(str(i) for i in range(512)) + ",patch_name\n")
            for r in range(N):
                f.write(",".join(f"{v:.4f}" for v in feats[r].astype(np.float32)) + f",{r}_{r}.jpeg\n")

        a_csv, a_h5 = load_csv(csvp, sample_num), load_h5(h5p, sample_num)
        exp_rows = min(N, sample_num)
        ok_shape = a_csv.shape == a_h5.shape == (exp_rows, 512)
        ok_close = np.allclose(a_csv, a_h5, atol=2e-3)   # csv=%.4f(round), h5=fp16 -> ~5e-5 apart
        ok = ok_shape and ok_close
        fails += (not ok)
        maxd = float(np.abs(a_csv - a_h5).max()) if ok_shape else -1
        print(f"[{'PASS' if ok else 'FAIL'}] N={N:>6} sn={sample_num} -> "
              f"rows={a_h5.shape[0]} (csv {a_csv.shape}, h5 {a_h5.shape}) maxabsdiff={maxd:.2e}")
    finally:
        shutil.rmtree(d, ignore_errors=True)

print("=" * 50)
print("ALL LOADER TESTS PASSED (h5 branch == csv branch)" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
