#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forward_smoke_longnet.py (S8) — resolve the LongNet/Projector dims + LongNet variable-length.

Runs a real forward through the SAME pieces model/llava.py builds:
  LongNet_2_layers_512_dim   : token_embeddings [N,1,512] -> encoder_out  (expect dim 512)
  ProjectorModel(512 -> H_llm): [1,N,512] -> [1,N,H_llm]   (H_llm = Qwen2.5-7B hidden = 3584)

Purpose:
  - settle whether the stale '1024'/'768'/'4096' comments are real (code says 512 throughout).
  - confirm LongNet accepts VARIABLE N (37 .. 10240) without crashing (P0 LongNet varlen / M5).

Run in the TRAINING env (needs torch + xtuner + torchscale), from a dir where `xtuner` imports:
  cd <slidechat_dev>/xtuner && python tools/forward_smoke_longnet.py
  # or: PYTHONPATH=<slidechat_dev> python tools/forward_smoke_longnet.py
"""
import argparse, sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-llm", type=int, default=3584, help="LLM hidden size (Qwen2.5-7B=3584)")
    ap.add_argument("--visual-dim", type=int, default=512, help="CONCH/LongNet dim")
    ap.add_argument("--ns", type=int, nargs="+", default=[37, 256, 1024, 5000, 10240],
                    help="patch counts N to test (LongNet variable length)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    try:
        from xtuner.model.modules import ProjectorConfig, ProjectorModel
        from xtuner.model.torchscale.model.LongNet import make_longnet_from_name
    except Exception as e:
        print(f"IMPORT FAILED ({type(e).__name__}: {e})\n"
              f"-> run inside the SlideChat training env, from <slidechat_dev>/xtuner "
              f"(or set PYTHONPATH=<slidechat_dev>).")
        sys.exit(2)

    dev = torch.device(args.device)
    D = args.visual_dim
    print(f"device={dev}  visual_dim={D}  H_llm={args.h_llm}")

    # ---- LongNet: variable length, output dim ----
    ln = make_longnet_from_name("LongNet_{}_layers_{}_dim".format(2, D)).to(dev).eval()
    print("\n[LongNet] token_embeddings [N,1,%d] -> encoder_out  (mirrors model/llava.py:334)" % D)
    ln_dim = None
    bad = []
    for N in args.ns:
        try:
            x = torch.randn(N, 1, D, device=dev)             # (seq, batch=1, dim)
            with torch.no_grad():
                out = ln(src_tokens=None, token_embeddings=x)["encoder_out"]
            ln_dim = out.shape[-1]
            finite = bool(torch.isfinite(out).all())
            print(f"  N={N:>6}: in {tuple(x.shape)} -> encoder_out {tuple(out.shape)}  "
                  f"finite={finite}")
            if out.shape[-1] != D or not finite:
                bad.append((N, tuple(out.shape), finite))
        except Exception as e:
            bad.append((N, f"{type(e).__name__}: {e}", False))
            print(f"  N={N:>6}: FAILED {type(e).__name__}: {e}")

    # ---- Projector: 512 -> H_llm ----
    print("\n[Projector] ProjectorConfig(visual_hidden_size=%d, llm_hidden_size=%d, depth=2)" % (D, args.h_llm))
    cfg = ProjectorConfig(visual_hidden_size=D, llm_hidden_size=args.h_llm, depth=2)
    proj = ProjectorModel(cfg).to(dev).float().eval()
    proj_ok = True
    for N in (37, 10240):
        x = torch.randn(1, N, D, device=dev)
        with torch.no_grad():
            out = proj(x)
        print(f"  N={N:>6}: in {tuple(x.shape)} -> {tuple(out.shape)}")
        if out.shape[-1] != args.h_llm or out.shape[1] != N:
            proj_ok = False

    # ---- verdict ----
    print("\n" + "=" * 60)
    print(f"LongNet output dim observed = {ln_dim}  (code expects {D}; stale comments say 1024)")
    if not bad and ln_dim == D and proj_ok:
        print(f"PASS: LongNet {D}->{D} accepts variable N {args.ns} (varlen OK); "
              f"Projector {D}->{args.h_llm}. Stale 768/1024/4096 comments confirmed wrong.")
        print("=> S8 resolves to 512 (not a blocker). LongNet varlen (P0) confirmed.")
        sys.exit(0)
    print(f"FAIL/REVIEW: bad={bad}  proj_ok={proj_ok}")
    print("=> if LongNet dim != 512 or some N crashed, ESCALATE S8 to blocker.")
    sys.exit(1)


if __name__ == "__main__":
    main()
