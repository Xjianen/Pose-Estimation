"""mIoU evaluation for query foreground masks against the official annotations.

Dataset ground truth
--------------------
``annotations/**/seg_new_<key>.png`` are *part-label* images (0 = background,
38 = anomalous part, 42 = intact parts).  Binarising with ``>0`` yields the
pixel-perfect object silhouette of the query view.  Do NOT use ``mask_new_*``
/ ``mask_old_*`` (those are small anomaly-region masks) nor ``seg_old_*``
(the pre-anomaly silhouette).  Only ``*_anomaly`` queries have annotations,
so coverage is always reported alongside the scores.

Why several mIoU numbers
------------------------
The GT is a hard-edged renderer label (no anti-aliasing) while the queries are
anti-aliased RGB renders.  On a 120-image sample, 68% of all disagreed pixels
lie within 2.5 px of the GT boundary and stripping a 2 px band raises IoU from
0.908 to 0.946.  A single raw IoU therefore mostly measures label quantisation
rather than real mistakes.  On a 40-query sample, 84-88% of FN and 66-79% of
FP are inside that band, for both the shipped masks and the SAM masks.  We
report:

  * ``mIoU``        - standard, strict, comparable with the literature
  * ``mIoU@trim2``  - GT-boundary band excluded; isolates structural errors
                      (floor-shadow blobs, filled arches, speckles)
  * ``micro IoU``   - pooled intersection / union, avoids small-object bias
  * ``shape-macro`` - per-shape mIoU averaged, so shapes with many queries do
                      not dominate the dataset mean
  * ``fp/fn_band_frac``  - fraction of errors inside the GT boundary band;
                      high values mean the raw score is quantisation-dominated
  * ``by_structure``- mIoU split by how thin / frame-like the GT is
  * paired win/tie/lose against a baseline (much more sensitive than two
    independent means)

``--dilate N`` applies a morphological boundary correction to *predictions*
before scoring.  This is diagnostic, not a score to report unmodified: masks
derived from anti-aliased RGB tend to sit ~1 px inside a hard-edged GT, so
``--dilate 1`` reveals how much of the gap is pure boundary offset.  Always
report ``--dilate 0`` as the primary result.

Layout
------
The default (and all documented examples) follow the **server layout**::

    <root>/shape_1095/images/render_1095_..._anomaly.png
    <root>/shape_1095/mask/render_1095_..._anomaly.png
    <root>/shape_1095/mask_clean/render_1095_..._anomaly.png

``--layout bucket`` additionally accepts the flattened local release layout
(``<root>/<subdir>/<bucket>/<name>.png``).  Ground truth is indexed by filename
key with a recursive scan, so ``--gt-root`` works for either a bucketed
``annotations/`` tree or per-shape annotation folders.

Examples (server)
-----------------
    # compare shipped masks with the regenerated clean masks
    python segmentation/eval_miou.py \
        --root /data/xje/datasets/brokenchairs180k/shapes \
        --gt-root /data/xje/datasets/brokenchairs180k/annotations \
        --pred-subdir mask --pred-subdir mask_clean \
        --csv miou.csv --json miou.json --panel miou.png

    # no mask_clean on disk yet: run the segmenter on the fly
    python segmentation/eval_miou.py --root ... --gt-root ... \
        --predict --limit 200 --csv miou.csv

    # diagnostic: is the gap just a 1 px boundary offset?
    python segmentation/eval_miou.py --root ... --gt-root ... \
        --pred-subdir mask --pred-subdir mask_clean --dilate 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Filename prefixes stripped to obtain the shared query key.  Order matters:
# longer / more specific prefixes must come first.
_PREFIXES = ("seg_new_", "seg_old_", "mask_new_", "mask_old_",
             "render_", "info_", "mask_", "seg_")


def imread(path, flags=cv2.IMREAD_UNCHANGED):
    """Unicode-safe read (Windows paths with non-ASCII characters)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise IOError(f"empty file: {path}")
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"cannot decode: {path}")
    return img


def imwrite(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"cannot encode: {path}")
    buf.tofile(str(path))


def query_key(filename):
    """``seg_new_1293_13834_2.5_260_30_5_anomaly.png`` -> ``1293_13834_...``."""
    stem = Path(filename).stem
    for p in _PREFIXES:
        if stem.startswith(p):
            stem = stem[len(p):]
            break
    return stem


def scan_subdir(root, subdir, layout, file_pattern, shape_pattern="shape_*",
                shape_filter=None, recursive_gt=False):
    """Return ``{query_key: Path}`` for one image/mask subdir."""
    root = Path(root)
    out = {}

    if recursive_gt:
        # GT may live in bucket folders, per-shape folders, or flat.
        base = root / subdir if (root / subdir).is_dir() else root
        for p in sorted(base.rglob(file_pattern)):
            if p.is_file():
                out.setdefault(query_key(p.name), p)
        return out

    if layout == "bucket":
        base = root / subdir
        if not base.is_dir():
            return out
        for p in sorted(base.rglob(file_pattern)):
            if p.is_file():
                out.setdefault(query_key(p.name), p)
        return out

    # server layout: <root>/<shape>/<subdir>/<file>
    for shape_dir in sorted(root.glob(shape_pattern)):
        if not shape_dir.is_dir():
            continue
        if shape_filter and shape_dir.name not in shape_filter:
            continue
        d = shape_dir / subdir
        if not d.is_dir():
            continue
        for p in sorted(d.glob(file_pattern)):
            if p.is_file():
                out.setdefault(query_key(p.name), p)
    return out


def as_binary(img, thresh, shape=None):
    if img.ndim == 3:
        img = img[..., :3].max(axis=2)
    if shape is not None and img.shape[:2] != shape:
        img = cv2.resize(img, (shape[1], shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return img > thresh


def boundary_valid(gt, radius):
    """Pixels farther than ``radius`` from the GT contour (trimap-style)."""
    if radius <= 0:
        return np.ones(gt.shape, bool)
    g = gt.astype(np.uint8)
    band = cv2.dilate(g, np.ones((3, 3), np.uint8)) ^ \
        cv2.erode(g, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform((~band.astype(bool)).astype(np.uint8),
                                 cv2.DIST_L2, 3)
    return dist > radius + 0.5


def blob_stats(mask, min_area=100):
    """Count / size of connected error blobs (shadow chunks, speckles)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    areas = [int(stats[i, cv2.CC_STAT_AREA])
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return len(areas), (max(areas) if areas else 0)


def score(gt, pred, trim_radius, min_blob_area):
    """All metrics for one (GT, prediction) pair."""
    inter = int((gt & pred).sum())
    union = int((gt | pred).sum())
    fp = int((pred & ~gt).sum())
    fn = int((gt & ~pred).sum())
    gt_area, pred_area = int(gt.sum()), int(pred.sum())

    valid = boundary_valid(gt, trim_radius)
    band = ~valid
    gi, pi = gt & valid, pred & valid
    vi, vu = int((gi & pi).sum()), int((gi | pi).sum())

    fp_mask, fn_mask = pred & ~gt, gt & ~pred
    fp_blobs, fp_biggest = blob_stats(fp_mask, min_blob_area)
    fn_blobs, fn_biggest = blob_stats(fn_mask, min_blob_area)

    return {
        "iou": inter / union if union else 0.0,
        "iou_trim": vi / vu if vu else 0.0,
        "dice": 2 * inter / (gt_area + pred_area)
        if (gt_area + pred_area) else 0.0,
        "inter": inter, "union": union,
        "gt_area": gt_area, "pred_area": pred_area,
        "fp": fp, "fn": fn,
        "gt_band_frac": int((gt & band).sum()) / gt_area if gt_area else 0.0,
        "fp_band_frac": int((fp_mask & band).sum()) / fp if fp else 0.0,
        "fn_band_frac": int((fn_mask & band).sum()) / fn if fn else 0.0,
        "fp_blobs": fp_blobs, "fp_biggest": fp_biggest,
        "fn_blobs": fn_blobs, "fn_biggest": fn_biggest,
    }


def pct(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def summarise(rows, methods, baseline, trim_radius):
    """rows: list of {'shape','key', method -> metrics}."""
    summary = {"n_queries": len(rows), "trim_radius": trim_radius,
               "baseline": baseline, "methods": {}}

    for m in methods:
        vals = [r[m] for r in rows if m in r]
        if not vals:
            continue
        ious = [v["iou"] for v in vals]
        trims = [v["iou_trim"] for v in vals]
        tin = sum(v["inter"] for v in vals)
        tun = sum(v["union"] for v in vals)
        by_shape = defaultdict(list)
        for r in rows:
            if m in r:
                by_shape[r["shape"]].append(r[m]["iou"])
        d = {
            "n": len(vals),
            "miou": float(np.mean(ious)),
            "miou_trim": float(np.mean(trims)),
            "miou_median": pct(ious, 50),
            "miou_p10": pct(ious, 10),
            "miou_p90": pct(ious, 90),
            "micro_iou": tin / tun if tun else 0.0,
            "shape_macro_miou": float(np.mean(
                [np.mean(v) for v in by_shape.values()])) if by_shape else 0.0,
            "n_shapes": len(by_shape),
            "mean_fp": float(np.mean([v["fp"] for v in vals])),
            "mean_fn": float(np.mean([v["fn"] for v in vals])),
            "mean_fp_band_frac": float(np.mean(
                [v["fp_band_frac"] for v in vals])),
            "mean_fn_band_frac": float(np.mean(
                [v["fn_band_frac"] for v in vals])),
            "mean_fp_blobs": float(np.mean([v["fp_blobs"] for v in vals])),
            "mean_fp_biggest": float(np.mean([v["fp_biggest"] for v in vals])),
        }
        if baseline and baseline != m:
            diffs = [r[m]["iou"] - r[baseline]["iou"] for r in rows
                     if m in r and baseline in r]
            d["paired_delta"] = float(np.mean(diffs)) if diffs else 0.0
            d["paired_delta_trim"] = float(np.mean(
                [r[m]["iou_trim"] - r[baseline]["iou_trim"] for r in rows
                 if m in r and baseline in r])) if diffs else 0.0
            d["win"] = sum(x > 1e-6 for x in diffs)
            d["tie"] = sum(abs(x) <= 1e-6 for x in diffs)
            d["lose"] = sum(x < -1e-6 for x in diffs)
            d["win_rate"] = d["win"] / len(diffs) if diffs else 0.0
        summary["methods"][m] = d

    # Stratify by how much of the GT lies inside its own boundary band.
    # A high value means a thin/frame-like object, for which any boundary
    # trim discards most of the signal (and SAM tends to break thin parts).
    thin = []
    for r in rows:
        for m in methods:
            if m in r:
                thin.append((r["key"], r[m]["gt_band_frac"]))
                break
    summary["by_structure"] = {}
    if len(thin) >= 6:
        q1, q2 = np.percentile([v for _, v in thin], [33, 67])
        groups = {"solid": [], "medium": [], "thin": []}
        for key, v in thin:
            groups["solid" if v <= q1 else
                   "thin" if v >= q2 else "medium"].append(key)
        for label, keyset in groups.items():
            ks = set(keyset)
            summary["by_structure"][label] = {
                "n": len(keyset),
                **{m: float(np.mean([r[m]["iou"] for r in rows
                                     if r["key"] in ks and m in r]))
                   for m in methods if any(r["key"] in ks and m in r
                                           for r in rows)},
            }
    return summary


def print_table(summary, coverage):
    print("\n" + "=" * 88)
    print(f"queries scored : {summary['n_queries']}   "
          f"GT coverage    : {coverage['gt_keys']}/{coverage['query_keys']} "
          f"({100 * coverage['ratio']:.1f}% of query images with annotation)")
    print(f"trim radius    : {summary['trim_radius']} px   "
          f"baseline: {summary['baseline'] or 'none'}")
    print("=" * 88)
    hdr = (f"{'method':<18} {'n':>5} {'mIoU':>8} {'trim':>8} "
           f"{'micro':>8} {'shpMac':>8} {'med':>7} {'p10':>7} "
           f"{'FP':>7} {'FN':>7} {'blobs':>6}")
    print(hdr)
    print("-" * 88)
    for m, d in summary["methods"].items():
        print(f"{m:<18} {d['n']:>5} {d['miou']:>8.4f} "
              f"{d['miou_trim']:>8.4f} {d['micro_iou']:>8.4f} "
              f"{d['shape_macro_miou']:>8.4f} {d['miou_median']:>7.4f} "
              f"{d['miou_p10']:>7.4f} {d['mean_fp']:>7.0f} "
              f"{d['mean_fn']:>7.0f} {d['mean_fp_blobs']:>6.2f}")
    print("-" * 88)
    for m, d in summary["methods"].items():
        print(f"  {m:<16} errors inside the GT-boundary band: "
              f"FP {100 * d['mean_fp_band_frac']:.1f}%  "
              f"FN {100 * d['mean_fn_band_frac']:.1f}%   "
              f"largest FP blob {d['mean_fp_biggest']:.0f} px")
    print("-" * 88)
    b = summary["baseline"]
    for m, d in summary["methods"].items():
        if b and m != b and "paired_delta" in d:
            print(f"{m} vs {b}: d_mIoU {d['paired_delta']:+.4f}  "
                  f"d_trim {d['paired_delta_trim']:+.4f}  "
                  f"win/tie/lose {d['win']}/{d['tie']}/{d['lose']} "
                  f"({100 * d['win_rate']:.1f}%)")
    if summary.get("by_structure"):
        print("-" * 88)
        print("mIoU by GT structure (solid / medium / thin-frame objects):")
        for label, d in summary["by_structure"].items():
            vals = "  ".join(f"{k}={v:.4f}" for k, v in d.items()
                            if k != "n")
            print(f"  {label:<8} n={d['n']:<4} {vals}")
    print("=" * 88)


def make_panel(rows, images, gts, paths, baseline, best, panel_path,
               n_show=6, trim_radius=2):
    """RGB | GT | baseline | best, for the largest wins and losses (ASCII text)."""
    paired = [(r[best]["iou"] - r[baseline]["iou"], r) for r in rows
              if best in r and baseline in r]
    if not paired:
        return
    paired.sort(key=lambda x: x[0])
    picks = paired[:n_show // 2] + paired[-n_show // 2:]
    tiles, strip = [], np.full((30, 4 * 256, 3), 255, np.uint8)
    for j, label in enumerate(["RGB", "ground truth", baseline, best]):
        cv2.putText(strip, label, (j * 256 + 6, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1,
                    cv2.LINE_AA)
    for delta, r in picks:
        key = r["key"]
        if key not in images or key not in gts:
            continue
        rgb = cv2.cvtColor(imread(images[key], cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        gt = imread(gts[key], cv2.IMREAD_GRAYSCALE) > 0
        a = imread(paths[baseline][key], cv2.IMREAD_GRAYSCALE) > 127 \
            if key in paths.get(baseline, {}) else np.zeros(gt.shape, bool)
        b = imread(paths[best][key], cv2.IMREAD_GRAYSCALE) > 127 \
            if key in paths.get(best, {}) else np.zeros(gt.shape, bool)
        row = []
        for t in (rgb, np.dstack([gt * 255] * 3),
                  np.dstack([a * 255] * 3), np.dstack([b * 255] * 3)):
            t = cv2.resize(t.astype(np.uint8), (256, 256))
            row.append(t)
        # annotate the delta on the RGB tile
        cv2.putText(row[0], f"dIoU {delta:+.3f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1,
                    cv2.LINE_AA)
        tiles.append(np.hstack(row))
    if tiles:
        panel = np.vstack([strip] + tiles)
        imwrite(panel_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"panel -> {panel_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="mIoU evaluation of query foreground masks vs seg_new_* GT")
    ap.add_argument("--root", required=True,
                    help="dataset root (server: <root>/<shape>/<subdir>/...)")
    ap.add_argument("--gt-root", default=None,
                    help="annotation root; recursive scan for --gt-pattern "
                         "(default: <root> itself, per-shape --gt-subdir)")
    ap.add_argument("--gt-subdir", default="annotations",
                    help="per-shape GT folder when --gt-root is omitted")
    ap.add_argument("--gt-pattern", default="seg_new_*.png")
    ap.add_argument("--gt-thresh", type=int, default=0,
                    help="GT binarisation: value > thresh is foreground "
                         "(0 keeps both part ids 38 and 42)")
    ap.add_argument("--layout", choices=["server", "bucket"],
                    default="server",
                    help="server = <root>/<shape>/<subdir>; "
                         "bucket = <root>/<subdir>/<bucket>")
    ap.add_argument("--shape-pattern", default="shape_*")
    ap.add_argument("--shape", action="append", default=None,
                    help="only these shape names (repeatable)")
    ap.add_argument("--images-subdir", default="images")
    ap.add_argument("--pred-subdir", action="append", default=None,
                    help="mask subdir to score (repeatable). "
                         "Default: mask, mask_clean")
    ap.add_argument("--pred-thresh", type=int, default=127)
    ap.add_argument("--dilate", type=int, default=0,
                    help="boundary correction applied to predictions: +N "
                         "dilates by N px (compensates a mask that is 1 px "
                         "smaller than the hard-edged GT), -N erodes. "
                         "Use 0 for the unmodified, standard mIoU.")
    ap.add_argument("--baseline", default=None,
                    help="method name used for paired comparison "
                         "(default: first --pred-subdir)")
    ap.add_argument("--predict", action="store_true",
                    help="add an on-the-fly 'clean' method by running the "
                         "segmenter on each RGB (no mask_clean on disk needed)")
    ap.add_argument("--predict-out", default=None,
                    help="where --predict writes its masks (default: a temp "
                         "dir, printed at the end)")
    ap.add_argument("--mode", choices=["auto", "sam", "cv"], default="auto")
    ap.add_argument("--model-dir", default=os.path.join(here, "models"))
    ap.add_argument("--trim-radius", type=int, default=2,
                    help="GT-boundary band ignored by mIoU@trim (0 disables)")
    ap.add_argument("--min-blob-area", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0,
                    help="score at most N GT queries (0 = all)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--panel", default=None)
    args = ap.parse_args()

    pred_subdirs = args.pred_subdir or ["mask", "mask_clean"]
    methods = list(dict.fromkeys(pred_subdirs +
                                 (["clean"] if args.predict else [])))
    baseline = args.baseline or (pred_subdirs[0] if pred_subdirs else None)

    t0 = time.time()
    shape_filter = set(args.shape) if args.shape else None

    # ---- build indexes -------------------------------------------------
    images = scan_subdir(args.root, args.images_subdir, args.layout, "*.png",
                         args.shape_pattern, shape_filter)
    gt_root = args.gt_root or args.root
    gt_subdir = "" if args.gt_root else args.gt_subdir
    gts = scan_subdir(gt_root, gt_subdir, args.layout, args.gt_pattern,
                      args.shape_pattern, shape_filter, recursive_gt=True)
    print(f"indexed {len(images)} query images, {len(gts)} GT labels")

    paths = {}
    for m in pred_subdirs:
        paths[m] = scan_subdir(args.root, m, args.layout, "*.png",
                               args.shape_pattern, shape_filter)
        print(f"  {m:<18} {len(paths[m]):>7} masks")

    # ---- optional on-the-fly segmenter ---------------------------------
    sam = None
    if args.predict:
        sys.path.insert(0, here)
        try:
            from core import SAMSegmenter, load_rgb, segment_image
        except ImportError:
            from .core import SAMSegmenter, load_rgb, segment_image
        if args.mode in ("auto", "sam"):
            try:
                sam = SAMSegmenter(args.model_dir)
            except Exception as exc:
                if args.mode == "sam":
                    raise
                print(f"[warn] SAM unavailable ({exc}); falling back to CV")
        predict_out = Path(args.predict_out) if args.predict_out else \
            Path(tempfile.mkdtemp(prefix="miou_pred_"))
        paths["clean"] = {}       # key -> written mask path (filled as we go)

        def _predict_one(key, src_path):
            mask = segment_image(load_rgb(src_path), sam,
                                 mode=("cv" if sam is None else args.mode))["mask"]
            dest = predict_out / ("shape_" + key.split("_")[0]) / \
                ("render_" + key + ".png")
            imwrite(dest, mask)
            paths["clean"][key] = dest
            return mask
    else:
        def _predict_one(key, src_path):
            return imread(src_path, cv2.IMREAD_GRAYSCALE)

    # ---- score ---------------------------------------------------------
    keys = sorted(set(gts) & set(images))
    if args.limit:
        step = max(1, len(keys) // args.limit)
        keys = keys[::step][:args.limit]
    coverage = {"query_keys": len(images), "gt_keys": len(gts),
                "ratio": len(gts) / len(images) if images else 0.0}

    rows, skipped = [], 0
    for i, key in enumerate(keys):
        try:
            gt = as_binary(imread(gts[key], cv2.IMREAD_GRAYSCALE),
                           args.gt_thresh)
        except Exception as exc:
            print(f"[warn] GT {key}: {exc}")
            continue
        if not gt.any():
            skipped += 1
            continue
        rec = {"shape": "shape_" + key.split("_")[0], "key": key}
        for m in methods:
            is_predict = (m == "clean" and args.predict)
            src = images if is_predict else paths.get(m, {})
            if key not in src:
                continue
            try:
                pred = _predict_one(key, src[key]) > args.pred_thresh \
                    if is_predict else \
                    as_binary(imread(src[key], cv2.IMREAD_GRAYSCALE),
                              args.pred_thresh, gt.shape)
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred.astype(np.uint8),
                                      (gt.shape[1], gt.shape[0]),
                                      interpolation=cv2.INTER_NEAREST) > 0
                if args.dilate:
                    r = abs(args.dilate)
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                    op = cv2.dilate if args.dilate > 0 else cv2.erode
                    pred = op(pred.astype(np.uint8), kernel) > 0
                rec[m] = score(gt, pred, args.trim_radius,
                               args.min_blob_area)
            except Exception as exc:
                print(f"[warn] {m}/{key}: {exc}")
        if len(rec) > 2:
            rows.append(rec)
        if args.predict and (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(keys)}] scored, elapsed "
                  f"{time.time() - t0:.0f}s", flush=True)

    if not rows:
        print("ERROR: no query had both a GT label and a prediction. "
              "Check --root / --gt-root / --layout / --pred-subdir.",
              file=sys.stderr)
        return 1

    summary = summarise(rows, methods, baseline, args.trim_radius)
    summary["coverage"] = coverage
    summary["skipped_empty_gt"] = skipped
    summary["config"] = {
        "root": args.root, "gt_root": gt_root, "gt_pattern": args.gt_pattern,
        "gt_thresh": args.gt_thresh, "layout": args.layout,
        "pred_subdirs": pred_subdirs, "predict": args.predict,
        "pred_thresh": args.pred_thresh, "dilate": args.dilate,
        "limit": args.limit, "min_blob_area": args.min_blob_area,
    }
    print_table(summary, coverage)

    # ---- outputs -------------------------------------------------------
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        fields = ["shape", "key", "method", "iou", "iou_trim", "dice",
                  "gt_area", "pred_area", "fp", "fn", "fp_band_frac",
                  "fn_band_frac", "fp_blobs", "fp_biggest", "fn_blobs",
                  "fn_biggest"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                for m in methods:
                    if m in r:
                        row = {"shape": r["shape"], "key": r["key"],
                               "method": m}
                        row.update({k: r[m][k] for k in fields[3:]})
                        w.writerow(row)
        print(f"csv   -> {args.csv}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"json  -> {args.json}")

    if args.panel and baseline:
        best = max((m for m in methods if m != baseline),
                   key=lambda m: summary["methods"].get(m, {}).get("miou", 0),
                   default=None)
        if best:
            try:
                make_panel(rows, images, gts, paths, baseline, best,
                           args.panel, trim_radius=args.trim_radius)
            except Exception as exc:
                print(f"[warn] panel failed: {exc}")
    if args.predict and not args.predict_out:
        print(f"predicted masks kept at -> {predict_out}")
    print(f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
