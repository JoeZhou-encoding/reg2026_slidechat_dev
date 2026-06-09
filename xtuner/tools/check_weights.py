#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_weights.py — confirm whether the SlideChat continue-SFT weights are on the server.

Checks (read-only) the expected model dirs and reports present / missing / incomplete:
  models/CONCH                 (patch encoder, frozen; we already have it)
  models/SlideChat_Weight      (LongNet + projector [+ LLM]; General-Medical-AI/SlideChat_Weight)
  models/Qwen2.5-7B-Instruct   (base LLM; Qwen/Qwen2.5-7B-Instruct)

Run:  python check_weights.py            # uses hard-coded Genkai default models dir
      python check_weights.py --models <dir>
"""
import argparse, glob, os

H_HOME = "/home/pj24003162/ku40003404/weihao/00"
REG2026 = f"{H_HOME}/reg_2026"


def report(name, path, needs_config=True):
    if not os.path.isdir(path):
        print(f"[MISSING ] {name:22} {path}  (directory absent)")
        return False
    allf = [f for f in glob.glob(os.path.join(path, "**", "*"), recursive=True) if os.path.isfile(f)]
    weights = [f for f in allf if f.endswith((".safetensors", ".bin", ".pth", ".pt"))]
    cfg = os.path.exists(os.path.join(path, "config.json"))
    size = sum(os.path.getsize(f) for f in allf) / 1e9
    complete = bool(weights) and (cfg or not needs_config)
    tag = "OK      " if complete else "INCOMPLT"
    print(f"[{tag}] {name:22} {len(allf):>4} files, {len(weights):>2} weight, "
          f"{size:6.1f} GB, config.json={'Y' if cfg else 'N'}  ({path})")
    for f in sorted(os.path.basename(x) for x in allf)[:10]:
        print(f"             {f}")
    return complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=f"{REG2026}/models")
    a = ap.parse_args()
    M = a.models
    print(f"models dir: {M}\n")
    r_conch = report("CONCH", f"{M}/CONCH")
    r_slide = report("SlideChat_Weight", f"{M}/SlideChat_Weight")
    r_qwen = report("Qwen2.5-7B-Instruct", f"{M}/Qwen2.5-7B-Instruct")
    print("\n" + "=" * 60)
    need = [n for n, ok in [("SlideChat_Weight", r_slide), ("Qwen2.5-7B-Instruct", r_qwen)] if not ok]
    if need:
        print(f"TO DOWNLOAD: {', '.join(need)}  -> run download_slidechat_weights.sh")
    else:
        print("All continue-SFT weights present.")
    print(f"(CONCH {'present' if r_conch else 'MISSING — also needed for dump'})")


if __name__ == "__main__":
    main()
