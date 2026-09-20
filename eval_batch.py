#!/usr/bin/env python3
"""Batch evaluation of the two-stage pose pipeline, with per-stage timing.

Loads the .ply and the codebook ONCE and reuses them across all queries (a bash
loop would re-parse 414k gaussians per image, which dominates the runtime).

Stages timed separately:
  setup   : .ply parse + upload to GPU, codebook load (one-time)
  coarse  : silhouette match against the codebook (PTAD chamfer or IoU)
  refine  : gsplat SE(3) silhouette refinement of K hypotheses + selection

Ground truth comes from the filename `<...>_<radius>_<azim>_<elev>_<...>.png`.
NOTE the azimuth sign: `render_*` filenames run opposite to the verified RT
convention, so use --gt-azim-sign -1 (default) for them.

Usage
-----
    python eval_batch.py --ply shape_8430/point_cloud/8430.ply \
        --codebook cb_refs20.npz --query-dir shape_8430/images \
        --hyps 3 --iters 80
"""

import argparse
import glob
import os
import time

import numpy as np

from codebook import Codebook, pose_from_azel, _isnum
from gsplat_refine import (load_gaussians_torch, build_targets, optimise_pose,
                           cv_viewmat_to_gl_w2c, rot_geodesic_deg, cam_center,
                           object_mask)


def gt_from_name(path, target, azim_sign):
    nums = [float(x) for x in os.path.splitext(os.path.basename(path))[0].split("_")
            if _isnum(x)]
    if len(nums) < 3:
        return None, None
    r, az, el = nums[-3:]
    return pose_from_azel(r, azim_sign * az, el, target), (r, az, el)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--metric", default="chamfer", choices=["chamfer", "iou"])
    ap.add_argument("--hyps", type=int, default=3, help="hypotheses to refine per query")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--w-sil", type=float, default=1.0)
    ap.add_argument("--w-dt", type=float, default=5.0)
    ap.add_argument("--scale-mult", type=float, default=1.0)
    ap.add_argument("--white-thresh", type=int, default=235)
    ap.add_argument("--gt-azim-sign", type=float, default=-1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-dir", default=None, help="save aligned renders here")
    ap.add_argument("--rot-target", type=float, default=10.0)
    ap.add_argument("--trans-target", type=float, default=0.1)
    args = ap.parse_args()

    import torch
    import cv2
    dev = torch.device(args.device)

    t = time.time()
    cb = Codebook(args.codebook)
    t_cb = time.time() - t

    t = time.time()
    gauss, sh_deg = load_gaussians_torch(args.ply, dev, args.scale_mult)
    means0, quats0 = gauss["means"].clone(), gauss["quats"].clone()
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t_ply = time.time() - t

    queries = sorted(glob.glob(os.path.join(args.query_dir, "*.png")))
    print(f"setup: codebook {cb.N} templates in {t_cb:.2f}s | "
          f".ply {len(means0)} gaussians in {t_ply:.2f}s")
    print(f"{len(queries)} queries | metric={args.metric} | hyps={args.hyps} "
          f"| iters={args.iters}\n")

    hdr = (f"{'query':44s} {'rot°':>7} {'trans':>7} {'coarse_ms':>10} "
           f"{'refine_s':>9} {'sel':>4} {'ok':>3}")
    print(hdr); print("-" * len(hdr))

    rots, transs, t_coarse, t_refine, npass = [], [], [], [], 0
    for qp in queries:
        t0 = time.time()
        hyps, score, order, f_q, W = cb.match(qp, args.metric, args.white_thresh,
                                              topk=args.hyps)
        tc = (time.time() - t0) * 1000

        qimg = cv2.imread(qp)
        H, Wq = qimg.shape[:2]
        K = torch.tensor([[f_q, 0, Wq / 2], [0, f_q, H / 2], [0, 0, 1]],
                         dtype=torch.float32, device=dev)
        tgt = build_targets(qimg, object_mask(qimg, args.white_thresh), dev)

        t0 = time.time()
        best = None
        for k, h in enumerate(hyps):
            fl, V = optimise_pose(gauss, means0, quats0, sh_deg, K, H, Wq, tgt,
                                  h["RT"], iters=args.iters, lr=args.lr,
                                  w_sil=args.w_sil, w_dt=args.w_dt,
                                  tag=f"h{k}", verbose=False)
            if best is None or fl < best[0]:
                best = (fl, V, k)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        tr = time.time() - t0

        RT_est = cv_viewmat_to_gl_w2c(best[1])
        RT_gt, azel = gt_from_name(qp, cb.target, args.gt_azim_sign)
        rot = rot_geodesic_deg(RT_est[:3, :3], RT_gt[:3, :3])
        trans = float(np.linalg.norm(cam_center(RT_est[:3, :3], RT_est[:3, 3])
                                     - cam_center(RT_gt[:3, :3], RT_gt[:3, 3])))
        ok = rot < args.rot_target and trans < args.trans_target
        npass += ok
        rots.append(rot); transs.append(trans); t_coarse.append(tc); t_refine.append(tr)
        print(f"{os.path.basename(qp)[:44]:44s} {rot:7.3f} {trans:7.4f} {tc:10.1f} "
              f"{tr:9.2f} {best[2]:>4d} {'Y' if ok else 'N':>3}")

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            with torch.no_grad():
                gauss["means"], gauss["quats"] = means0, quats0
                from gsplat_refine import render, gl_w2c_to_cv_viewmat
                Vt = torch.tensor(best[1], dtype=torch.float32, device=dev)
                rgb, _, _ = render(gauss, Vt, K, H, Wq, sh_deg)
            out = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
            cv2.imwrite(os.path.join(args.save_dir,
                                     os.path.splitext(os.path.basename(qp))[0]
                                     + "_aligned.png"), out)

    r, tr_, tc_, trf = (np.asarray(rots), np.asarray(transs),
                        np.asarray(t_coarse), np.asarray(t_refine))
    print("\n" + "=" * 72)
    print(f"ACCURACY over {len(r)} queries "
          f"(targets: rot<{args.rot_target}°, trans<{args.trans_target})")
    print(f"  rotation °   : median={np.median(r):7.3f} mean={r.mean():7.3f} "
          f"max={r.max():7.3f}   <target: {(r<args.rot_target).sum()}/{len(r)}")
    print(f"  translation  : median={np.median(tr_):7.4f} mean={tr_.mean():7.4f} "
          f"max={tr_.max():7.4f}   <target: {(tr_<args.trans_target).sum()}/{len(tr_)}")
    print(f"  BOTH pass    : {npass}/{len(r)}")
    print(f"\nTIMING")
    print(f"  setup (one-time): codebook {t_cb:.2f}s + .ply {t_ply:.2f}s "
          f"= {t_cb+t_ply:.2f}s")
    print(f"  coarse match    : median={np.median(tc_):7.1f} ms  mean={tc_.mean():7.1f} ms")
    print(f"  refine ({args.hyps} hyp)  : median={np.median(trf):7.2f} s   "
          f"mean={trf.mean():7.2f} s   (= {trf.mean()/args.hyps:.2f} s per hypothesis)")
    print(f"  per query total : {np.median(tc_)/1000 + np.median(trf):7.2f} s (median)")
    print(f"  whole batch     : {tc_.sum()/1000 + trf.sum():7.1f} s + setup")


if __name__ == "__main__":
    main()
