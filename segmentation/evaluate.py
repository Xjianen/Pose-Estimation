"""Compare shipped masks against the regenerated ones and report quality.

No pixel-perfect ground truth is available for the query views: the
``render/`` and ``reference/`` buffers in this dataset come from a DIFFERENT
camera (39.66 deg FOV vs 50 deg for queries). We therefore report
unsupervised, pose-relevant proxies:

  edge_align      boundary pixels on a strong RGB gradient (higher = better)
  shadow_residue  achromatic, smooth lower-half pixels inside the mask
                  (lower = better)
  speckles        tiny connected components / salt-and-pepper noise
  iou / changed   per-pixel agreement between old and new masks

Usage:
    python segmentation/evaluate.py --root /data/.../shapes \
        --pred-subdir mask_clean --csv seg_eval.csv --panel seg_eval.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from core import imread, imwrite, load_rgb, quality_metrics
except ImportError:
    from .core import imread, imwrite, load_rgb, quality_metrics


def _panel(root, pred_subdir, names, out, limit=8):
    rows = []
    for shape, name in names[:limit]:
        rgb = load_rgb(root / shape / "images" / name)
        old = imread(root / shape / "mask" / name, cv2.IMREAD_GRAYSCALE) > 127
        new = imread(root / shape / pred_subdir / name, cv2.IMREAD_GRAYSCALE) > 127
        diff = np.dstack([((old ^ new) * 255).astype(np.uint8)] * 3)
        tiles = [rgb, np.dstack([(old * 255).astype(np.uint8)] * 3),
                 np.dstack([(new * 255).astype(np.uint8)] * 3), diff]
        rows.append([cv2.resize(t, (224, 224)) for t in tiles])
    if not rows:
        return
    panel = np.vstack([np.hstack(r) for r in rows])
    strip = np.full((32, panel.shape[1], 3), 255, np.uint8)
    for j, label in enumerate(["RGB", "old mask", "clean mask", "diff"]):
        cv2.putText(strip, label, (j * 224 + 6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (0, 0, 0), 1, cv2.LINE_AA)
    imwrite(out, cv2.cvtColor(np.vstack([strip, panel]), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--images-subdir", default="images")
    ap.add_argument("--old-subdir", default="mask")
    ap.add_argument("--pred-subdir", default="mask_clean")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--limit", type=int, default=0, help="limit number of shapes")
    args = ap.parse_args()

    root = Path(args.root)
    shapes = sorted(p for p in root.iterdir() if p.is_dir()
                    and (p / args.pred_subdir).is_dir()
                    and (p / args.images_subdir).is_dir())
    if args.limit:
        shapes = shapes[:args.limit]
    if not shapes:
        print(f"no shape dirs with '{args.pred_subdir}/' under {root}", file=sys.stderr)
        return 1

    rows, names, buckets = [], [], {"old": [], "new": []}
    for s in shapes:
        for p in sorted((s / args.images_subdir).glob("*.png")):
            old_p = s / args.old_subdir / p.name
            new_p = s / args.pred_subdir / p.name
            if not old_p.is_file() or not new_p.is_file():
                continue
            rgb = load_rgb(p)
            old = imread(old_p, cv2.IMREAD_GRAYSCALE) > 127
            new = imread(new_p, cv2.IMREAD_GRAYSCALE) > 127
            mo, mn = quality_metrics(rgb, old), quality_metrics(rgb, new)
            buckets["old"].append(mo)
            buckets["new"].append(mn)
            inter, union = (old & new).sum(), (old | new).sum()
            rows.append(dict(shape=s.name, image=p.name,
                             old_edge=round(mo.edge_align, 4),
                             new_edge=round(mn.edge_align, 4),
                             old_shadow=round(mo.shadow_residue, 4),
                             new_shadow=round(mn.shadow_residue, 4),
                             old_speckles=mo.speckles, new_speckles=mn.speckles,
                             iou=round(inter / max(union, 1), 4),
                             removed_px=int((old & ~new).sum()),
                             added_px=int((new & ~old).sum())))
            names.append((s.name, p.name))

    if not rows:
        print("no matched image/mask pairs found", file=sys.stderr)
        return 1

    def mean(bucket, attr):
        return float(np.mean([getattr(m, attr) for m in buckets[bucket]]))

    print(f"matched pairs: {len(rows)}  shapes: {len(shapes)}\n")
    print(f"{'metric':20s} {'old mask':>12s} {'clean mask':>12s} {'delta':>12s}")
    print("-" * 58)
    for attr, label in [("edge_align", "edge_align (up)"),
                        ("shadow_residue", "shadow_residue (down)"),
                        ("speckles", "speckles (down)")]:
        a, b = mean("old", attr), mean("new", attr)
        print(f"{label:20s} {a:12.4f} {b:12.4f} {b - a:+12.4f}")
    print("-" * 58)
    print(f"{'mean IoU(old,new)':20s} {np.mean([r['iou'] for r in rows]):12.4f}")
    print(f"{'mean removed px':20s} {np.mean([r['removed_px'] for r in rows]):12.0f}")
    print(f"{'mean added px':20s} {np.mean([r['added_px'] for r in rows]):12.0f}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\ncsv -> {args.csv}")
    if args.panel:
        # sample across shapes so the panel is representative
        picks, per = [], 8
        by_shape = {}
        for item in names:
            by_shape.setdefault(item[0], []).append(item)
        for s in sorted(by_shape)[:8]:
            picks.extend(by_shape[s][:1])
        _panel(root, args.pred_subdir, picks, args.panel)
        print(f"panel -> {args.panel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
