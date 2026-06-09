#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_stage2_model.py — turn the DeepSpeed stage-2 model checkpoint into a plain
state_dict that xtuner's guess_load_checkpoint can load via the FILE path.

Why: we download ONLY stage2_pth/mp_rank_00_model_states.pt (15.3GB) and skip the 91.6GB
optim states. That file is a DeepSpeed dict {'module': <weights>, 'optimizer': ..., ...}.
guess_load_checkpoint(file) only unwraps a 'state_dict' key, so feeding it raw would match
0 params (silent failure). We extract ckpt['module'] and re-save as {'state_dict': module}.

Then set pretrained_pth = <out .pth> in the training config.

Run (in an env with torch):
  python extract_stage2_model.py \
    <models>/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt \
    <models>/SlideChat_Weight/stage2_model.pth
"""
import sys
import torch


def main():
    if len(sys.argv) != 3:
        print("usage: python extract_stage2_model.py <mp_rank_00_model_states.pt> <out.pth>")
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu")
    if isinstance(ckpt, dict) and "module" in ckpt:
        sd = ckpt["module"]
        print(f"unwrapped DeepSpeed 'module' ({len(sd)} tensors)")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        print(f"already has 'state_dict' ({len(sd)} tensors)")
    else:
        sd = ckpt
        print(f"using top-level dict as state_dict ({len(sd)} tensors)")

    # quick sanity: how many keys per submodule (expect llm.* + LongNet_encoder.* + projector.*)
    pref = {}
    for k in sd:
        p = k.split(".")[0]
        pref[p] = pref.get(p, 0) + 1
    print("top-level submodule key counts:", dict(sorted(pref.items(), key=lambda x: -x[1])))
    for must in ("llm", "LongNet_encoder", "projector"):
        hit = any(k.startswith(must + ".") or k == must for k in sd)
        print(f"  has {must}.* : {'YES' if hit else 'NO (!) check key names'}")

    torch.save({"state_dict": sd}, dst)
    print(f"WROTE {dst}  -> set pretrained_pth='{dst}' in the training config")


if __name__ == "__main__":
    main()
