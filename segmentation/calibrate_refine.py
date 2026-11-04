"""Fit the global boundary-calibration constant of refine.py on a dev split.

This is NOT training a network: refine.py has one isotropic scale (plus an
optional translation) that compensates the ~1% FOV/rasterisation mismatch
between the query render and the hard-edged ``seg_new_*`` label render. This
script fits those few scalars on a *disjoint* calibration split and reports the
held-out mIoU, so the number quoted in the README is not fitted on the test
images.

Both layouts are supported (defaults follow the server shape layout)::

    server: <root>/<shape>/{images,mask}/...  + --gt-root .../annotations
    bucket: <root>/{images,masks}/<bucket>/... + --layout bucket --mask-subdir masks

Examples
--------
    python segmentation/calibrate_refine.py \
        --root /data/xje/datasets/brokenchairs180k/shapes \
        --gt-root /data/xje/datasets/brokenchairs180k/annotations \
        --limit 400 --json refine_calib.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from core import imread
    from refine import remove_floor_plane, calibrate_boundary
    from eval_miou import scan_subdir, query_key
except ImportError:
    from .core import imread
    from .refine import remove_floor_plane, calibrate_boundary
    from .eval_miou import scan_subdir, query_key


def iou(a, b):
    u = int(np.logical_or(a, b).sum())
    return np.logical_and(a, b).sum() / u if u else 0.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt-root", default=None)
    ap.add_argument("--gt-subdir", default="annotations")
    ap.add_argument("--gt-pattern", default="seg_new_*.png")
    ap.add_argument("--layout", choices=["server", "bucket"], default="server")
    ap.add_argument("--images-subdir", default="images")
    ap.add_argument("--mask-subdir", default="mask",
                    help="shipped masks (local bucket release uses 'masks')")
    ap.add_argument("--limit", type=int, default=400,
                    help="number of labelled queries to sample (0 = all)")
    ap.add_argument("--split", type=float, default=0.5,
                    help="fraction of sampled queries used for calibration")
    ap.add_argument("--scales", default="1.0,1.005,1.0075,1.01,1.0125,1.015")
    ap.add_argument("--shifts", default="-2,0,2")
    ap.add_argument("--prior-threshold", type=int, default=200,
                    help="loosen the colour prior to min(rgb)<T before floor "
                         "removal (200 keeps the shipped mask)")
    ap.add_argument("--json", default=None)
    # argparse treats "-2,0,2" as an option token; accept both
    # ``--shifts -2,0,2`` and ``--shifts=-2,0,2``.
    argv = sys.argv[1:]
    fixed, i = [], 0
    while i < len(argv):
        if argv[i] == "--shifts" and i + 1 < len(argv) \
                and argv[i + 1].startswith("-"):
            fixed.append("--shifts=" + argv[i + 1])
            i += 2
        else:
            fixed.append(argv[i])
            i += 1
    args = ap.parse_args(fixed)

    t0 = time.time()
    images = scan_subdir(args.root, args.images_subdir, args.layout, "*.png")
    masks = scan_subdir(args.root, args.mask_subdir, args.layout, "*.png")
    gt_root = args.gt_root or args.root
    gt_sub = "" if args.gt_root else args.gt_subdir
    gts = scan_subdir(gt_root, gt_sub, args.layout, args.gt_pattern,
                      recursive_gt=True)
    keys = sorted(set(images) & set(masks) & set(gts))
    if args.limit:
        step = max(1, len(keys) // args.limit)
        keys = keys[::step][:args.limit]
    n_cal = max(1, int(len(keys) * args.split))
    calib, test = keys[:n_cal], keys[n_cal:]
    print(f"indexed {len(images)} images / {len(gts)} GT; "
          f"calib={len(calib)} test={len(test)}", flush=True)
    if not calib:
        print("no labelled queries found", file=sys.stderr)
        return 1

    data = {}
    for k in keys:
        rgb = cv2.cvtColor(imread(images[k], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        old = imread(masks[k], cv2.IMREAD_GRAYSCALE) > 127
        gt = imread(gts[k], cv2.IMREAD_GRAYSCALE) > 0
        if args.prior_threshold > 200:
            old = old | (rgb.min(2) < args.prior_threshold)
        clean, _ = remove_floor_plane(rgb, old)
        data[k] = (gt, clean, rgb)

    scales = [float(x) for x in args.scales.split(",")]
    shifts = [float(x) for x in args.shifts.split(",")]
    grid = list(itertools.product(scales, shifts, shifts,
                                  ["centre", "centroid"]))
    best = None
    for s, dx, dy, origin in grid:
        vals = [iou(gt, calibrate_boundary(c, s, origin, dx, dy) > 0)
                for gt, c, _ in (data[k] for k in calib)]
        v = float(np.mean(vals))
        if best is None or v > best[0]:
            best = (v, s, dx, dy, origin)
    _, s, dx, dy, origin = best

    def score(ks, params=None):
        ss, ddx, ddy, oo = params or (s, dx, dy, origin)
        return float(np.mean([iou(gt, calibrate_boundary(c, ss, oo, ddx, ddy) > 0)
                              for gt, c, _ in (data[k] for k in ks)]))

    base_cal = float(np.mean([iou(gt, c > 0) for gt, c, _ in (data[k] for k in calib)]))
    base_test = float(np.mean([iou(gt, c > 0) for gt, c, _ in (data[k] for k in test)])) if test else 0.0
    old_test = float(np.mean([iou(gt, imread(masks[k], cv2.IMREAD_GRAYSCALE) > 127)
                              for k in test for gt in [data[k][0]]])) if test else 0.0
    cal_miou, test_miou = score(calib), (score(test) if test else 0.0)
    simple_params = (1.01, 0.0, 0.0, "centre")
    print("\n" + "=" * 74)
    print(f"best      : scale={s}  dx={dx}  dy={dy}  origin={origin}")
    print(f"calib     : floor-cleaned {base_cal:.4f} -> {cal_miou:.4f}")
    print(f"held-out  : shipped {old_test:.4f} | floor-cleaned {base_test:.4f} "
          f"| calibrated {test_miou:.4f}")
    if test:
        print(f"held-out  : fixed (1.01, centre) = {score(test, simple_params):.4f}")
    print("=" * 74)

    out = {"n_calib": len(calib), "n_test": len(test),
           "best": {"scale": s, "dx": dx, "dy": dy, "origin": origin},
           "calib_miou": cal_miou, "test_miou": test_miou,
           "test_shipped": old_test, "test_floor_clean": base_test,
           "test_fixed_1p01": (score(test, simple_params) if test else 0.0),
           "elapsed_s": round(time.time() - t0, 1)}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print("saved", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
