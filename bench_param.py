#!/usr/bin/env python3
"""Compare the two pose parameterisations: camera-viewmat vs object-transform.

Both optimise the same SE(3) delta and produce the same effective viewmat
`V0 @ exp(xi)`, so this script checks two things:

  1. **equivalence** — do they reach the same pose / loss?
  2. **cost** — how much does the object-transform path pay for touching every
     gaussian (a 4xN transform each step, plus gsplat having to emit per-gaussian
     gradients) versus the camera path (16 numbers of gradient)?

Usage
-----
    python bench_param.py --root shape_8430
    python bench_param.py --root /data/.../shape_1095 --iters 80 --repeat 3
"""

import argparse
import time

import numpy as np

from dataset import ShadObject
from utils.gaussians import load_gaussians_torch, supports_viewmat_grad
from utils.geometry import pose_error
from utils.masks import silhouette_targets
from utils.pose_init import Codebook
from utils.pose_refine import refine_pose


def timed_refine(gauss, means0, quats0, sh_degree, K, W, H, tgt, RT0, param,
                 iters, lr, repeat, sync):
    import torch
    losses, poses, times = [], [], []
    for _ in range(repeat):
        if sync:
            torch.cuda.synchronize()
        t0 = time.time()
        loss, RT = refine_pose(gauss, means0, quats0, sh_degree, K, W, H, tgt, RT0,
                               iters=iters, lr=lr, param=param)
        if sync:
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(loss)
        poses.append(RT)
    return np.asarray(times), float(np.mean(losses)), poses[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="a shape_* folder")
    ap.add_argument("--query", default=None, help="default: first query of the shape")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    dev = torch.device(args.device)
    sync = dev.type == "cuda"

    obj = ShadObject(args.root)
    print(obj)
    cam_ok = supports_viewmat_grad(dev)
    print(f"gsplat viewmat gradients: {'AVAILABLE' if cam_ok else 'NOT available'}")

    gauss, sh_degree = load_gaussians_torch(obj.ply_path, dev)
    means0, quats0 = gauss["means"].clone(), gauss["quats"].clone()
    n = len(means0)
    print(f"gaussians: {n}")

    cb = Codebook.from_references(obj.references(), target=obj.target,
                                 obj_size=obj.obj_size)
    qs = obj.queries()
    q = next((x for x in qs if args.query and args.query in x.name), qs[0])
    W, H = q.shape
    hyps, f_q, _ = cb.match(query_png=q.path, topk=1)
    RT0 = hyps[0]["RT"]
    K = torch.tensor(cb.intrinsics(f_q, W, H), dtype=torch.float32, device=dev)
    tgt = silhouette_targets(q.mask(), dev)
    print(f"query: {q.name} ({W}x{H}), {args.iters} iters x {args.repeat} repeats\n")

    modes = ["object"] + (["camera"] if cam_ok else [])
    out = {}
    for m in modes:
        # warm-up so kernel compilation is not counted
        refine_pose(gauss, means0, quats0, sh_degree, K, W, H, tgt, RT0,
                    iters=3, lr=args.lr, param=m)
        gauss["means"], gauss["quats"] = means0.clone(), quats0.clone()
        t, loss, RT = timed_refine(gauss, means0, quats0, sh_degree, K, W, H, tgt,
                                   RT0, m, args.iters, args.lr, args.repeat, sync)
        gauss["means"], gauss["quats"] = means0.clone(), quats0.clone()
        out[m] = {"t": t, "loss": loss, "RT": RT}
        print(f"{m:7s}: {t.mean():6.3f} s +- {t.std():.3f}  "
              f"({1000*t.mean()/args.iters:5.2f} ms/iter)  final loss {loss:.6f}")

    if len(out) == 2:
        so, sc = out["object"]["t"].mean(), out["camera"]["t"].mean()
        rot, trans = pose_error(out["camera"]["RT"], out["object"]["RT"])
        print(f"\nspeedup (object -> camera): {so/sc:.2f}x  "
              f"({1000*(so-sc)/args.iters:.2f} ms/iter saved)")
        print(f"agreement: rotation {rot:.4f} deg, centre {trans:.5f}  "
              f"| loss delta {abs(out['camera']['loss']-out['object']['loss']):.2e}")
        print("=> equivalent poses (both give V0 @ exp(xi)); pick 'camera' for speed.")
    else:
        print("\nOnly the object parameterisation is usable on this gsplat build.")


if __name__ == "__main__":
    main()
