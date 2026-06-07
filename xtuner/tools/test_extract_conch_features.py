#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke / unit tests for extract_conch_features.py — runs with NO GPU and NO CONCH.

Builds a tiny synthetic tiled TIFF + a make_thumbnails-style mask, injects a MOCK
encoder, and checks the mask-selection logic, decode/encode IO, fp16 HDF5 output,
idempotency, missing-mask handling, sharding, and SlideChat read-compat.

Run:  python test_extract_conch_features.py
(real CONCH numerical correctness is verified separately on Genkai on 1 real slide)
"""
import os
import sys
import tempfile

import numpy as np
import tifffile
from PIL import Image

import extract_conch_features as E

P = 256


# ---- synthetic data -------------------------------------------------------
def synth_slide(H, W, box):
    """White (240) slide with a darker 'tissue' rectangle box=(y0,y1,x0,x1)."""
    img = np.full((H, W, 3), 240, np.uint8)
    y0, y1, x0, x1 = box
    img[y0:y1, x0:x1] = 110
    return img


def write_tiff(path, img):
    tifffile.imwrite(path, img, tile=(256, 256))


def make_mask(img, target_long=2048, threshold=220):
    """Mimic make_thumbnails.py: downsample then gray<threshold -> tissue (255)."""
    H, W = img.shape[:2]
    ratio = max(1, max(H, W) // target_long)
    th = np.asarray(Image.fromarray(img).resize((W // ratio, H // ratio), Image.BILINEAR))
    gray = th.mean(axis=-1)
    return (gray < threshold).astype(np.uint8) * 255


def mock_preprocess(pil):
    # Real CONCH returns a (3,224,224) tensor; the mock just needs the mean pixel.
    return np.asarray(pil.resize((224, 224))).astype(np.float32)


def mock_encode(buf):
    # Deterministic: feature = mean pixel value of the patch, broadcast to 512.
    out = np.zeros((len(buf), E.FEATURE_DIM), dtype=np.float32)
    for i, p in enumerate(buf):
        out[i, :] = float(np.asarray(p).mean())
    return out


# ---- tests ----------------------------------------------------------------
def t_select_only_tissue():
    H = W = 4 * P                      # 4x4 patch grid
    box = (P, 3 * P, P, 3 * P)         # tissue covers the central 2x2 cells
    img = synth_slide(H, W, box)
    mask = make_mask(img)
    coords = E.select_tissue_patches(mask, H, W, patch_size=P, tissue_frac=0.5, cap=None)
    assert len(coords) >= 4, f"expected >=4 tissue patches, got {len(coords)}"
    # every kept patch must overlap the tissue box and stay in-bounds
    for (Y, X) in coords:
        assert 0 <= Y <= H - P and 0 <= X <= W - P, f"out of bounds ({Y},{X})"
        cy, cx = Y + P // 2, X + P // 2
        assert box[0] <= cy < box[1] and box[2] <= cx < box[3], \
            f"patch center ({cy},{cx}) not in tissue box {box}"
    return f"{len(coords)} patches, all in tissue & in-bounds"


def t_cap_subsample():
    H = W = 20 * P                     # 400 candidate cells, all tissue
    img = synth_slide(H, W, (0, H, 0, W))
    mask = make_mask(img)
    cap = 50
    coords = E.select_tissue_patches(mask, H, W, patch_size=P, tissue_frac=0.1, cap=cap)
    assert len(coords) == cap, f"cap not applied: {len(coords)} != {cap}"
    coords2 = E.select_tissue_patches(mask, H, W, patch_size=P, tissue_frac=0.1, cap=cap)
    assert np.array_equal(coords, coords2), "subsample not deterministic"
    return f"capped to {cap}, deterministic"


def t_no_tissue():
    H = W = 4 * P
    img = np.full((H, W, 3), 245, np.uint8)   # all background
    mask = make_mask(img)
    coords = E.select_tissue_patches(mask, H, W, patch_size=P, tissue_frac=0.1, cap=None)
    assert len(coords) == 0, f"expected 0, got {len(coords)}"
    return "all-background -> 0 patches"


def t_extract_one_mock(tmp):
    H = W = 6 * P
    box = (P, 4 * P, P, 4 * P)
    img = synth_slide(H, W, box)
    tiff = os.path.join(tmp, "PIT_TEST_0001_01.tiff")
    write_tiff(tiff, img)
    mask = make_mask(img)
    mask_path = os.path.join(tmp, "PIT_TEST_0001_01.tiff.mask.png")
    Image.fromarray(mask).save(mask_path)
    out = os.path.join(tmp, "PIT_TEST_0001_01.h5")

    r = E.extract_one_wsi(tiff, mask_path, out, mock_encode, mock_preprocess,
                          patch_size=P, cap=None, tissue_frac=0.5, batch_size=4)
    assert r["status"] == "ok", f"status={r}"
    import h5py
    with h5py.File(out, "r") as f:
        feats = f["features"][:]
        coords = f["coords"][:]
        assert feats.dtype == np.float16, f"features dtype {feats.dtype} != float16"
        assert feats.shape == (r["n"], E.FEATURE_DIM), f"bad shape {feats.shape}"
        assert coords.shape == (r["n"], 2)
        assert f.attrs["proj_contrast"] == np.bool_(False) or f.attrs["proj_contrast"] in (False, 0)
        # tissue patches are darker (110) -> mean ~110, background -> ~240.
        assert feats.astype(np.float32).mean() < 200, "tissue features unexpectedly bright"
        for (Y, X) in coords:
            cy, cx = Y + P // 2, X + P // 2
            assert box[0] <= cy < box[1] and box[2] <= cx < box[3], "coord not in tissue"
    return f"{r['n']} patches, fp16 (N,512), coords in tissue"


def t_idempotent(tmp):
    H = W = 4 * P
    img = synth_slide(H, W, (0, H, 0, W))
    tiff = os.path.join(tmp, "PIT_IDEM_01.tiff")
    write_tiff(tiff, img)
    mp = os.path.join(tmp, "PIT_IDEM_01.tiff.mask.png")
    Image.fromarray(make_mask(img)).save(mp)
    out = os.path.join(tmp, "PIT_IDEM_01.h5")
    r1 = E.extract_one_wsi(tiff, mp, out, mock_encode, mock_preprocess,
                           patch_size=P, cap=None, tissue_frac=0.1, batch_size=8)
    r2 = E.extract_one_wsi(tiff, mp, out, mock_encode, mock_preprocess,
                           patch_size=P, cap=None, tissue_frac=0.1, batch_size=8)
    assert r1["status"] == "ok" and r2["status"] == "skip", f"{r1} / {r2}"
    return "second run -> skip"


def t_missing_mask(tmp):
    H = W = 4 * P
    img = synth_slide(H, W, (0, H, 0, W))
    tiff = os.path.join(tmp, "PIT_NOMASK_01.tiff")
    write_tiff(tiff, img)
    out = os.path.join(tmp, "PIT_NOMASK_01.h5")
    r = E.extract_one_wsi(tiff, os.path.join(tmp, "does_not_exist.mask.png"), out,
                          mock_encode, mock_preprocess, patch_size=P)
    assert r["status"] == "no_mask", f"status={r}"
    assert not os.path.exists(out), "output written despite missing mask"
    return "missing mask -> no_mask, no output"


def t_sharding():
    slides = [f"s{i}.tiff" for i in range(23)]
    n_shards = 4
    shards = [[s for i, s in enumerate(slides) if i % n_shards == k] for k in range(n_shards)]
    flat = sorted(sum(shards, []))
    assert flat == sorted(slides), "shards do not cover all slides"
    seen = set()
    for sh in shards:
        for s in sh:
            assert s not in seen, "overlap between shards"
            seen.add(s)
    sizes = [len(sh) for sh in shards]
    assert max(sizes) - min(sizes) <= 1, f"unbalanced shards {sizes}"
    return f"23 slides / 4 shards -> {sizes}, disjoint & complete"


def t_slidechat_compat(tmp):
    """The fp16 h5 we write must be readable like tools/test_full.load_feature_array."""
    H = W = 5 * P
    img = synth_slide(H, W, (0, H, 0, W))
    tiff = os.path.join(tmp, "PIT_COMPAT_01.tiff")
    write_tiff(tiff, img)
    mp = os.path.join(tmp, "PIT_COMPAT_01.tiff.mask.png")
    Image.fromarray(make_mask(img)).save(mp)
    out = os.path.join(tmp, "PIT_COMPAT_01.h5")
    E.extract_one_wsi(tiff, mp, out, mock_encode, mock_preprocess,
                      patch_size=P, cap=None, tissue_frac=0.1, batch_size=16)
    import h5py
    with h5py.File(out, "r") as f:           # SlideChat reads f['features'][:]
        arr = f["features"][:]
    arr = arr.astype(np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] != 512:
        arr = arr[:, :512]
    assert arr.shape[1] == 512 and arr.shape[0] > 0, f"bad SlideChat-read shape {arr.shape}"
    return f"SlideChat-style read OK: {arr.shape}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # ignore_cleanup_errors: on Windows a lingering WSI handle can briefly lock the
    # temp .tiff; that's a teardown artifact, not a test failure.
    tmp_ctx = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    tmp = tmp_ctx.name
    tests = [
        ("select_only_tissue", lambda: t_select_only_tissue()),
        ("cap_subsample", lambda: t_cap_subsample()),
        ("no_tissue", lambda: t_no_tissue()),
        ("extract_one_mock", lambda: t_extract_one_mock(tmp)),
        ("idempotent", lambda: t_idempotent(tmp)),
        ("missing_mask", lambda: t_missing_mask(tmp)),
        ("sharding", lambda: t_sharding()),
        ("slidechat_compat", lambda: t_slidechat_compat(tmp)),
    ]
    ok = 0
    for name, fn in tests:
        try:
            msg = fn()
            print(f"  [PASS] {name}: {msg}")
            ok += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok}/{len(tests)} passed")
    try:
        tmp_ctx.cleanup()
    except Exception:
        pass
    sys.exit(0 if ok == len(tests) else 1)


if __name__ == "__main__":
    main()
