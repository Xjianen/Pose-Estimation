#!/usr/bin/env python3
"""Align query images to each object's 3DGS model, export the aligned reference
mask + depth, and report timing / accuracy / errors.

(The name follows the project convention; nothing is trained here — the pipeline
is inference only: coarse silhouette retrieval followed by differentiable
refinement.)

Per query
---------
  1. coarse : match the query silhouette against a template codebook -> top-K poses
  2. refine : gsplat SE(3) silhouette refinement of each hypothesis, keep the
              lowest final loss
  3. export : render at the estimated pose and save the aligned MASK and DEPTH
              (what the downstream anomaly stage consumes)

Codebook modes
--------------
  refs   only the reference views shipped with the shape (fair vs PTAD, which uses
         those 20 views)
  dense  a rendered azimuth x elevation grid from the .ply (~2 ms/view)

Examples
--------
    # one shape
    python train.py --data-root /data/xje/datasets/brokenchairs180k/shapes \
        --shapes shape_1095 --out-dir runs/fair

    # several / all shapes
    python train.py --data-root /data/xje/datasets/brokenchairs180k/shapes \
        --shapes "shape_10*" --out-dir runs/batch
    python train.py --data-root /data/xje/datasets/brokenchairs180k/shapes \
        --max-shapes 20 --out-dir runs/batch

    # ablations
    python train.py --data-root ... --shapes shape_1095 --hyps 1   # no multi-hypothesis
    python train.py --data-root ... --shapes shape_1095 --w-dt 0   # no distance transform
"""

import argparse
import json
import os

import numpy as np

from dataset import ShadDataset, ShadObject
from utils.gaussians import load_gaussians_torch, render_at_pose
from utils.geometry import pose_error_symmetric, verify_pose_convention
from utils.masks import silhouette_targets
from utils.pose_init import Codebook
from utils.pose_refine import (refine_multi_hypothesis,
                               refine_multi_hypothesis_batched, resolve_param)
from utils.stats import PoseMetrics, StageTimer

EXTRA_COLS = ("coarse_ms", "refine_s", "iters", "sel")


def build_codebook(args, obj, gauss, sh_degree, timer):
    cache = None
    if args.codebook_cache_dir:
        os.makedirs(args.codebook_cache_dir, exist_ok=True)
        cache = os.path.join(args.codebook_cache_dir,
                             f"{obj.name}_{args.codebook_mode}.npz")
    if cache and os.path.exists(cache):
        with timer("codebook load"):
            cb = Codebook.load(cache)
        print(f"  codebook: {cb.N} templates (cached)")
        return cb

    with timer("codebook build"):
        if args.codebook_mode == "refs":
            cb = Codebook.from_references(obj.references(), mask_res=args.mask_res,
                                          white_thresh=args.white_thresh,
                                          target=obj.target, obj_size=obj.obj_size)
        else:
            cb = Codebook.render(gauss, sh_degree, obj.target,
                                 azim_step=args.azim_step,
                                 elev_range=(args.elev_min, args.elev_max),
                                 elev_step=args.elev_step, radius=args.cb_radius,
                                 res=args.cb_res, focal=args.cb_focal,
                                 mask_res=args.mask_res, obj_size=obj.obj_size,
                                 device=args.device, verbose=args.verbose)
    print(f"  codebook: {cb.N} templates ({args.codebook_mode})")
    if cache:
        cb.save(cache)
    return cb


def run_object(args, obj, timer, metrics, records, batched=False):
    import torch
    import cv2
    dev = torch.device(args.device)

    refs = obj.references()
    dR, dt_, ok = verify_pose_convention(refs, obj.target)
    print(f"\n{obj}")
    print(f"  pose convention on {len(refs)} refs: max|dR|={dR:.1e} "
          f"max|dt|={dt_:.1e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("  [skip] pose convention mismatch")
        return

    with timer("gaussian load"):
        gauss, sh_degree = load_gaussians_torch(obj.ply_path, dev, args.scale_mult)
    means0, quats0 = gauss["means"].clone(), gauss["quats"].clone()
    cb = build_codebook(args, obj, gauss, sh_degree, timer)

    queries = obj.queries()
    if args.max_queries:
        queries = queries[:args.max_queries]
    mask_dir = os.path.join(args.out_dir, obj.name, "mask")
    depth_dir = os.path.join(args.out_dir, obj.name, "depth")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    print(f"  {len(queries)} queries")

    for qs in queries:
        W, H = qs.shape
        with timer("coarse"):
            hyps, f_q, _ = cb.match(query_png=qs.path, metric=args.metric,
                                    white_thresh=args.white_thresh, topk=args.hyps)
        t_coarse = timer.last("coarse") * 1000

        K = torch.tensor(cb.intrinsics(f_q, W, H), dtype=torch.float32, device=dev)
        tgt = silhouette_targets(qs.mask(), dev,
                                 gray_np=qs.gray() if args.w_edge > 0 else None)

        with timer("refine"):
            common = dict(lr=args.lr, w_sil=args.w_sil, w_dt=args.w_dt,
                          w_edge=args.w_edge, patience=args.patience,
                          min_delta=args.min_delta, verbose=args.verbose)
            if batched:
                loss, RT_est, sel, all_losses, n_it = refine_multi_hypothesis_batched(
                    gauss, sh_degree, K, W, H, tgt, hyps, iters=args.iters, **common)
            else:
                loss, RT_est, sel, all_losses, n_it = refine_multi_hypothesis(
                    gauss, means0, quats0, sh_degree, K, W, H, tgt, hyps,
                    iters=args.iters, param=args.param, **common)
        t_refine = timer.last("refine")

        with timer("export"):
            gauss["means"], gauss["quats"] = means0, quats0
            rgb, alpha, depth = render_at_pose(gauss, RT_est, K, W, H, sh_degree)
            stem = os.path.splitext(qs.name)[0]
            cv2.imwrite(os.path.join(mask_dir, stem + "_mask.png"),
                        (alpha > 0.5).astype(np.uint8) * 255)
            np.save(os.path.join(depth_dir, stem + "_depth.npy"),
                    np.where(alpha > 0.5, depth, 0.0).astype(np.float32))
            if args.save_rgb:
                cv2.imwrite(os.path.join(args.out_dir, obj.name, stem + "_aligned.png"),
                            (rgb * 255).astype(np.uint8)[..., ::-1])

        rec = {"shape": obj.name, "query": qs.name, "sel": sel, "loss": loss,
               "hyp_losses": all_losses, "coarse_ms": t_coarse,
               "refine_s": t_refine, "iters_used": n_it, "RT_est": RT_est.tolist()}
        RT_gt = qs.gt_pose()
        if RT_gt is None:
            print(f"  {qs.name[:44]:44s}  (no GT in filename; pose exported only)")
        else:
            rot, trans, k = pose_error_symmetric(RT_est, RT_gt, obj.target,
                                                 args.sym_azim)
            metrics.add(f"{obj.name}/{qs.name}", rot, trans,
                        extra={"coarse_ms": t_coarse, "refine_s": t_refine,
                               "iters": n_it, "sel": sel})
            metrics.print_row(extra_cols=EXTRA_COLS)
            rec.update({"rot_deg": rot, "trans": trans, "sym_k": k})
        records.append(rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("data")
    src.add_argument("--data-root", default="/data/xje/datasets/brokenchairs180k/shapes",
                     help="directory containing the shape_* folders")
    src.add_argument("--shapes", default=None,
                     help="comma-separated names/globs; default = all usable shapes")
    src.add_argument("--root", default=None,
                     help="shortcut: a single shape folder (overrides --data-root)")
    src.add_argument("--shape-pattern", default="shape_*")
    src.add_argument("--max-shapes", type=int, default=0)
    src.add_argument("--max-queries", type=int, default=0, help="per shape")
    src.add_argument("--ref-subdir", default="mv_images")
    src.add_argument("--query-subdir", default="images")

    ap.add_argument("--out-dir", default="runs/out")
    ap.add_argument("--device", default="cuda")

    cbg = ap.add_argument_group("codebook")
    cbg.add_argument("--codebook-mode", default="refs", choices=["refs", "dense"])
    cbg.add_argument("--codebook-cache-dir", default=None)
    cbg.add_argument("--metric", default="chamfer", choices=["chamfer", "iou"])
    cbg.add_argument("--mask-res", type=int, default=160)
    cbg.add_argument("--azim-step", type=float, default=3.0)
    cbg.add_argument("--elev-min", type=float, default=15.0)
    cbg.add_argument("--elev-max", type=float, default=45.0)
    cbg.add_argument("--elev-step", type=float, default=2.0)
    cbg.add_argument("--cb-radius", type=float, default=3.5)
    cbg.add_argument("--cb-res", type=int, default=256)
    cbg.add_argument("--cb-focal", type=float, default=355.0)

    rf = ap.add_argument_group("refinement")
    rf.add_argument("--hyps", type=int, default=3, help="hypotheses refined per query")
    rf.add_argument("--iters", type=int, default=80)
    rf.add_argument("--lr", type=float, default=0.01)
    rf.add_argument("--w-sil", type=float, default=1.0)
    rf.add_argument("--w-dt", type=float, default=5.0)
    rf.add_argument("--w-edge", type=float, default=0.0)
    rf.add_argument("--scale-mult", type=float, default=1.0)
    rf.add_argument("--batched", action="store_true",
                    help="render all hypotheses in one rasterisation call. Measured as "
                         "NO net win: 1.33x cheaper per camera-render but early stopping "
                         "becomes whole-batch instead of per-hypothesis, which cancels it "
                         "out (1.79s vs 1.74s). Off by default; kept for re-testing on "
                         "other gsplat builds")
    rf.add_argument("--patience", type=int, default=10,
                    help="early stopping: stop after this many iterations without the "
                         "loss improving by more than --min-delta. The default is the "
                         "best measured setting (faster AND slightly more accurate than "
                         "running the full iteration cap); 0 disables it")
    rf.add_argument("--min-delta", type=float, default=1e-4)
    rf.add_argument("--param", default="auto", choices=["auto", "camera", "object"],
                    help="pose parameterisation: 'camera' puts the SE(3) delta on the "
                         "viewmat (no per-gaussian work; needs gsplat viewmat grads), "
                         "'object' transforms the gaussians (works everywhere), "
                         "'auto' probes and prefers camera")

    ev = ap.add_argument_group("evaluation")
    ev.add_argument("--white-thresh", type=int, default=235)
    ev.add_argument("--azim-sign", type=float, default=-1.0,
                    help="-1 for render_* query filenames (see dataset docstring)")
    ev.add_argument("--sym-azim", type=float, default=0.0,
                    help="object azimuth symmetry period in degrees; 0 = none. "
                         "Use 180 for symmetric objects such as shape_8430")
    ev.add_argument("--rot-target", type=float, default=10.0)
    ev.add_argument("--trans-target", type=float, default=0.1)
    ev.add_argument("--save-rgb", action="store_true")
    ev.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import torch
    timer = StageTimer(sync=(torch.device(args.device).type == "cuda"))

    if args.root:
        objects = [ShadObject(args.root, args.azim_sign, args.white_thresh,
                              args.ref_subdir, args.query_subdir)]
        print(f"single shape: {args.root}")
    else:
        ds = ShadDataset(args.data_root, args.shape_pattern, args.azim_sign,
                         args.white_thresh, args.ref_subdir, args.query_subdir)
        print(ds)
        if ds.incomplete:
            for n, miss in list(ds.incomplete.items())[:5]:
                print(f"  skipped {n}: missing {', '.join(miss)}")
            if len(ds.incomplete) > 5:
                print(f"  ... and {len(ds.incomplete)-5} more incomplete shapes")
        objects = ds.select(args.shapes, limit=args.max_shapes)
        if not objects:
            raise SystemExit("no shapes selected")

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = PoseMetrics(args.rot_target, args.trans_target)
    records = []
    mode = resolve_param(args.param, torch.device(args.device))
    batched = args.batched and mode == "camera"
    print(f"\n{len(objects)} shape(s) | metric={args.metric} | hyps={args.hyps} "
          f"| iters={args.iters} | w_dt={args.w_dt} | sym_azim={args.sym_azim} "
          f"| param={args.param}->{mode}"
          + (f" | early-stop({args.patience}, {args.min_delta})"
             if args.patience else "")
          + (" | batched" if batched else " | sequential"))
    metrics.header(extra_cols=EXTRA_COLS)

    for obj in objects:
        run_object(args, obj, timer, metrics, records, batched)

    summary = metrics.report() if metrics.rows else {}
    if metrics.rows:
        metrics.hypothesis_summary()
    timer.report(per_query_stages=("coarse", "refine", "export"))

    print(f"\nexported per-query aligned reference under {args.out_dir}/<shape>/")
    print("  mask/<query>_mask.png    uint8 0/255")
    print("  depth/<query>_depth.npy  float32, 0 outside the object")
    print("\nNOTE filename ground truth is rounded (azim/elev 1°, radius 0.1), so "
          "errors\n     below ~0.5° / ~0.05 sit at the label-quantisation floor.")

    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "timing": dict(timer.totals), "per_query": records}, f, indent=2)
    print(f"  metrics -> {out_json}")


if __name__ == "__main__":
    main()
