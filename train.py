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
import time

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


def render_depth_map(depth_tensor, colormap="turbo"):
    """Color each depth image using its own valid foreground min/max range."""
    from matplotlib import colormaps
    if hasattr(depth_tensor, "detach"):
        depth_array = depth_tensor.detach().cpu().numpy()
    else:
        depth_array = np.asarray(depth_tensor)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    if depth_array.ndim != 2:
        raise ValueError("Depth must have shape HxW or HxWx1")
    valid = np.isfinite(depth_array) & (depth_array > 0)
    if not np.any(valid):
        return np.zeros((*depth_array.shape, 3), dtype=np.uint8)
    near = float(depth_array[valid].min())
    far = float(depth_array[valid].max())
    if far <= near:
        valid_color = np.zeros((*depth_array.shape, 3), dtype=np.uint8)
        valid_color[valid] = 255
        return valid_color[..., ::-1].copy()
    normalized = np.zeros(depth_array.shape, dtype=np.float32)
    normalized[valid] = (far - depth_array[valid]) / (far - near)
    colored = (colormaps[colormap](normalized)[..., :3] * 255).astype(np.uint8)
    colored[~valid] = 0
    return colored[..., ::-1].copy()


def save_pose_comparison(path, query_bgr, initial_mask, final_mask, depth_bgr):
    """Save the four panels with an explicit noninteractive plotting backend."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10, 6.4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    panels = [(query_bgr[..., ::-1], "query RGB"),
              (initial_mask, "initial pose (coarse top-1)"),
              (final_mask, "final pose"),
              (depth_bgr[..., ::-1], "final pose depth")]
    for ax, (image, title) in zip(axes.flat, panels):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=150)
    fig.clear()


def resolve_cb_geometry(args, queries):
    """Codebook rendering radius/focal; 'auto' derives them from the queries.

    The auto focal is the CALIBRATED query focal rescaled to the codebook
    resolution, so the templates are rendered under the same perspective as
    the queries (the query FOV can differ from the references — see
    dataset/loader.py). The auto radius is the median filename label, i.e.
    the distance the queries were actually rendered from.
    """
    if args.cb_radius != "auto":
        radius = float(args.cb_radius)
    else:
        labels = [q.params[0] for q in queries if q.params is not None]
        radius = float(np.median(labels)) if labels else 3.5
    if args.cb_focal != "auto":
        focal = float(args.cb_focal)
    else:
        q0 = queries[0]
        focal = float(q0.K[0, 0]) * args.cb_res / float(q0.shape[0])
    return radius, focal


def build_codebook(args, obj, gauss, sh_degree, timer, queries):
    cache = None
    if args.codebook_cache_dir:
        os.makedirs(args.codebook_cache_dir, exist_ok=True)
        suffix = ""
        if args.codebook_mode == "dense":
            radius, focal = resolve_cb_geometry(args, queries)
            suffix = f"_{focal:.0f}_{radius * 100:.0f}"
        cache = os.path.join(args.codebook_cache_dir,
                             f"{obj.name}_{args.codebook_mode}{suffix}.npz")
    if cache and os.path.exists(cache):
        with timer("codebook load"):
            cb = Codebook.load(cache)
        if not args.quiet:
            print(f"  codebook: {cb.N} templates (cached)")
        return cb

    with timer("codebook build"):
        if args.codebook_mode == "refs":
            cb = Codebook.from_references(obj.references(), mask_res=args.mask_res,
                                          white_thresh=args.white_thresh,
                                          target=obj.target, obj_size=obj.obj_size)
        else:
            radius, focal = resolve_cb_geometry(args, queries)
            if not args.quiet:
                print(f"  dense codebook: radius={radius:.2f} focal={focal:.1f}"
                      f" @ {args.cb_res}px (auto from queries)")
            cb = Codebook.render(gauss, sh_degree, obj.target,
                                 azim_step=args.azim_step,
                                 elev_range=(args.elev_min, args.elev_max),
                                 elev_step=args.elev_step, radius=radius,
                                 res=args.cb_res, focal=focal,
                                 mask_res=args.mask_res, obj_size=obj.obj_size,
                                 device=args.device, verbose=args.verbose)
    if not args.quiet:
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
    if not args.quiet:
        print(f"\n{obj}")
        print(f"  pose convention on {len(refs)} refs: max|dR|={dR:.1e} "
              f"max|dt|={dt_:.1e} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise ValueError("pose convention mismatch")

    with timer("gaussian load"):
        gauss, sh_degree = load_gaussians_torch(obj.ply_path, dev, args.scale_mult)
    means0, quats0 = gauss["means"].clone(), gauss["quats"].clone()

    queries = obj.queries()
    if args.max_queries:
        queries = queries[:args.max_queries]
    cb = build_codebook(args, obj, gauss, sh_degree, timer, queries)
    mask_dir = os.path.join(args.out_dir, obj.name, "mask")
    reference_dir = (os.path.join(args.reference_out_dir, obj.name)
                     if args.reference_out_dir
                     else os.path.join(args.out_dir, obj.name, "reference_camera"))
    depth_dir = None if args.lean else os.path.join(args.out_dir, obj.name, "depth")
    compare_dir = None if args.lean else os.path.join(args.out_dir, obj.name, "compare_camera")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)
    if depth_dir:
        os.makedirs(depth_dir, exist_ok=True)
    if compare_dir:
        os.makedirs(compare_dir, exist_ok=True)
    if not args.quiet:
        print(f"  {len(queries)} queries")

    # Guard against silent filename mis-parsing. The old "last three numbers"
    # rule reads render_<obj>_<anomaly_id>_2.5_120_35_5_anomaly as
    # (radius, azim, elev) = (120, 35, 5), so the GT radius becomes the azimuth
    # label and every reported error is garbage (trans ~300 with r=2.5, where
    # the max possible is 5). Verified against such a run: logged errors
    # reproduced the mis-parse exactly.
    ref_radius = float(np.median([ref["radius"] for ref in refs]))
    gt_radii = [qs.params[0] for qs in queries if qs.params is not None]
    if gt_radii:
        q_radius = float(np.median(gt_radii))
        if not args.quiet:
            print(f"  GT from filenames: radius median {q_radius:.2f} "
                  f"(references: {ref_radius:.2f})")
        if not 0.4 <= q_radius / ref_radius <= 2.5:
            raise ValueError(
                f"query GT radius median {q_radius:.2f} is incompatible with the "
                f"reference radius {ref_radius:.2f} -- query filenames were "
                f"almost certainly mis-parsed (stale dataset/shad.py?); refusing "
                f"to evaluate against garbage ground truth. Update dataset/shad.py "
                f"so parse_view_params takes the first decimal field + its two "
                f"successors, then rerun.")

    for qs in queries:
        W, H = qs.shape
        with timer("coarse"):
            hyps, f_q, _ = cb.match(query_mask=qs.mask(), metric=args.metric,
                                    query_focal=qs.K[0, 0],
                                    white_thresh=args.white_thresh, topk=args.hyps)
        t_coarse = timer.last("coarse") * 1000

        K = torch.tensor(qs.K, dtype=torch.float32, device=dev)
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
            _, initial_alpha, _ = render_at_pose(
                gauss, hyps[0]["RT"], K, W, H, sh_degree)
            stem = os.path.splitext(qs.name)[0]
            final_mask = (alpha > 0.5).astype(np.uint8) * 255
            initial_mask = (initial_alpha > 0.5).astype(np.uint8) * 255
            depth_values = np.where(alpha > 0.5, depth, 0.0).astype(np.float32)
            depth_colored = render_depth_map(
                depth_values)
            depth_colored[alpha <= 0.5] = 0
            cv2.imwrite(os.path.join(mask_dir, stem + "_mask.png"),
                        final_mask)
            if depth_dir:
                np.save(os.path.join(depth_dir, stem + "_depth.npy"),
                        depth_values)
            cv2.imwrite(os.path.join(reference_dir, stem + "_depth.png"), depth_colored)
            np.save(os.path.join(reference_dir, stem + "_depth.npy"), depth_values)
            if compare_dir:
                save_pose_comparison(os.path.join(compare_dir, qs.name),
                                     qs.image, initial_mask, final_mask, depth_colored)
            if args.save_rgb:
                cv2.imwrite(os.path.join(args.out_dir, obj.name, stem + "_aligned.png"),
                            (rgb * 255).astype(np.uint8)[..., ::-1])

        rec = {"shape": obj.name, "query": qs.name, "sel": sel, "loss": loss,
               "hyp_losses": all_losses, "coarse_ms": t_coarse,
               "refine_s": t_refine, "iters_used": n_it, "RT_est": RT_est.tolist(),
               "query_focal": float(qs.K[0, 0]), "radius_scale": qs.radius_scale,
               "hypotheses": [{"azim": h["azim"], "elev": h["elev"],
                               "radius": h["radius"], "score": h["score"]}
                              for h in hyps]}
        RT_gt = qs.gt_pose()
        if RT_gt is None:
            if not args.quiet:
                print(f"  {qs.name[:44]:44s}  (no GT in filename; pose exported only)")
        else:
            rot, trans, k = pose_error_symmetric(RT_est, RT_gt, obj.target,
                                                 args.sym_azim)
            metrics.add(f"{obj.name}/{qs.name}", rot, trans,
                        extra={"coarse_ms": t_coarse, "refine_s": t_refine,
                               "iters": n_it, "sel": sel})
            if not args.quiet:
                metrics.print_row(extra_cols=EXTRA_COLS)
            rec.update({"rot_deg": rot, "trans": trans, "sym_k": k})
        records.append(rec)
    if args.skip_if_done:
        open(os.path.join(reference_dir, ".done"), "w").close()


def _print_shape_summary(name, rows, elapsed, args):
    """One-line per-shape summary for --quiet (batch) mode."""
    if not rows:
        print(f"{name}: no GT rows; {elapsed:.0f}s")
        return
    rots = np.array([r["rot"] for r in rows])
    trans = np.array([r["trans"] for r in rows])
    rpass = int((rots < args.rot_target).sum())
    tpass = int((trans < args.trans_target).sum())
    both = int(sum(1 for r in rows if r["ok"]))
    n = len(rows)
    print(f"{name}: {n}q rot {rpass}/{n}(med{np.median(rots):.1f} max{rots.max():.1f}) "
          f"trans {tpass}/{n}(med{np.median(trans):.3f} max{trans.max():.3f}) "
          f"both {both}/{n}={100*both/n:.0f}% {elapsed:.0f}s")


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
    src.add_argument("--max-refs", type=int, default=0,
                    help="randomly subsample N reference views (0 = all); "
                         "pair with --ref-seed for reproducibility")
    src.add_argument("--ref-seed", type=int, default=0,
                    help="seed for --max-refs subsampling")
    src.add_argument("--ref-subdir", default="mv_images")
    src.add_argument("--query-subdir", default="images")

    ap.add_argument("--out-dir", default="runs/out")
    ap.add_argument("--device", default="cuda")

    cbg = ap.add_argument_group("codebook")
    cbg.add_argument("--codebook-mode", default="dense", choices=["refs", "dense"],
                     help="'dense' renders an azimuth x elevation grid from the 3DGS "
                          "model (recommended: templates at the queries' own elevations "
                          "and perspective, measured to cut the initial rotation error "
                          "from ~11 deg to ~0); 'refs' uses only the 20 shipped "
                          "reference views (the PTAD-fair ablation)")
    cbg.add_argument("--codebook-cache-dir", default=None)
    cbg.add_argument("--metric", default="chamfer", choices=["chamfer", "iou"])
    cbg.add_argument("--mask-res", type=int, default=160)
    cbg.add_argument("--azim-step", type=float, default=3.0)
    cbg.add_argument("--elev-min", type=float, default=15.0)
    cbg.add_argument("--elev-max", type=float, default=45.0)
    cbg.add_argument("--elev-step", type=float, default=2.0)
    cbg.add_argument("--cb-radius", default="auto",
                     help="codebook render radius; 'auto' = median query filename "
                          "label (3.5 previously)")
    cbg.add_argument("--cb-res", type=int, default=256)
    cbg.add_argument("--cb-focal", default="auto",
                     help="codebook render focal at --cb-res; 'auto' = the calibrated "
                          "query focal rescaled to --cb-res (355.0 previously, which "
                          "mismatches queries whose FOV differs from the references)")

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
    ev.add_argument("--quiet", action="store_true",
                    help="minimal stdout: one summary line per shape (rot/trans "
                         "pass, medians, elapsed) instead of the per-query table, "
                         "convention/intrinsics/codebook/timing blocks. For batch "
                         "runs where the per-shape results.json holds the detail")
    ev.add_argument("--reference-out-dir", default=None,
                    help="write reference_camera to DIR/<shape>/ instead of under "
                         "--out-dir; use when batch-rendering a dataset-wide "
                         "reference_camera folder")
    ev.add_argument("--lean", action="store_true",
                    help="lean export: only mask + reference_camera (skip the "
                         "duplicate depth/ dir and the compare_camera panels); "
                         "for batch dataset generation")
    ev.add_argument("--skip-if-done", action="store_true",
                    help="skip a shape whose reference_camera dir already has a "
                         ".done sentinel (resumable batch runs)")
    ev.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import torch
    timer = StageTimer(sync=(torch.device(args.device).type == "cuda"))

    failures = []
    if args.root:
        object_paths = [args.root]
        if not args.quiet:
            print(f"single shape: {args.root}")
    else:
        ds = ShadDataset(args.data_root, args.shape_pattern, args.azim_sign,
                         args.white_thresh, args.ref_subdir, args.query_subdir)
        if not args.quiet:
            print(ds)
        if ds.incomplete:
            failures.extend({"shape": n, "stage": "assets", "error": ", ".join(miss)}
                            for n, miss in ds.incomplete.items())
            for n, miss in list(ds.incomplete.items())[:5]:
                print(f"  skipped {n}: missing {', '.join(miss)}")
            if len(ds.incomplete) > 5:
                print(f"  ... and {len(ds.incomplete)-5} more incomplete shapes")
        object_paths = ds.selected_paths(args.shapes, limit=args.max_shapes)
        if not object_paths:
            raise SystemExit("no shapes selected")

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = PoseMetrics(args.rot_target, args.trans_target)
    records = []
    mode = resolve_param(args.param, torch.device(args.device))
    batched = args.batched and mode == "camera"
    if not args.quiet:
        print(f"\n{len(object_paths)} shape(s) | metric={args.metric} | hyps={args.hyps} "
              f"| iters={args.iters} | w_dt={args.w_dt} | sym_azim={args.sym_azim} "
              f"| param={args.param}->{mode}"
              + (f" | early-stop({args.patience}, {args.min_delta})"
                 if args.patience else "")
              + (" | batched" if batched else " | sequential"))
        metrics.header(extra_cols=EXTRA_COLS)

    out_json = os.path.join(args.out_dir, "results.json")
    def save_progress(summary=None):
        temporary = out_json + ".tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "summary": summary or {},
                       "timing": dict(timer.totals), "per_query": records,
                       "failures": failures}, f, indent=2)
        os.replace(temporary, out_json)

    # Instantiate one shape at a time; do not preload the entire dataset.
    for index, path in enumerate(object_paths, 1):
        name = os.path.basename(os.path.normpath(path))
        if args.skip_if_done:
            ref_dir_name = (os.path.join(args.reference_out_dir, name)
                            if args.reference_out_dir
                            else os.path.join(args.out_dir, name, "reference_camera"))
            if os.path.exists(os.path.join(ref_dir_name, ".done")):
                print(f"{name}: skip (done)" if args.quiet
                      else f"[{index}/{len(object_paths)}] {name} -> skip (done)")
                continue
        if not args.quiet:
            print(f"\n[{index}/{len(object_paths)}] {name}")
        rows_before = len(metrics.rows)
        t0 = time.time()
        try:
            obj = ShadObject(path, args.azim_sign, args.white_thresh,
                             args.ref_subdir, args.query_subdir,
                             max_refs=args.max_refs, ref_seed=args.ref_seed,
                             verbose=not args.quiet)
            if args.max_refs and not args.quiet:
                print(f"  references: subsampled {args.max_refs} (seed={args.ref_seed})")
            run_object(args, obj, timer, metrics, records, batched)
        except Exception as exc:
            if args.root:
                raise
            failures.append({"shape": name, "stage": "pose_estimation",
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"{name} FAILED: {type(exc).__name__}: {exc}" if args.quiet
                  else f"  [failed] {name}: {exc}")
        else:
            if args.quiet:
                _print_shape_summary(name, metrics.rows[rows_before:],
                                     time.time() - t0, args)
        finally:
            save_progress()

    summary = metrics.report(quiet=args.quiet) if metrics.rows else {}
    if metrics.rows and not args.quiet:
        metrics.hypothesis_summary()
    if not args.quiet:
        timer.report(per_query_stages=("coarse", "refine", "export"))
        print(f"\nexported per-query aligned reference under {args.out_dir}/<shape>/")
        print("  mask/<query>_mask.png    uint8 0/255")
        print("  depth/<query>_depth.npy  float32, 0 outside the object")
        print("  reference_camera/<query>_depth.png + _depth.npy  colored depth + values")
        print("  compare_camera/<query>.png  query / initial mask / final mask / depth")
        print("\nNOTE filename ground truth is rounded (azim/elev 1°, radius 0.1), so "
              "errors\n     below ~0.5° / ~0.05 sit at the label-quantisation floor.")

    save_progress(summary)
    if not args.quiet:
        print(f"  failed/incomplete shapes: {len(failures)} (see failures in results.json)")
        print(f"  metrics -> {out_json}")


if __name__ == "__main__":
    main()
