"""Batch-regenerate clean foreground masks for every shape in the dataset.

Reads ``<shape>/images/*.png`` and writes binary masks (0/255 PNG) to
``<shape>/mask_clean/*.png`` by default. Filenames are preserved 1:1, so the
pose pipeline can consume them by setting ``--mask-dir mask_clean``.

Examples
--------
    # one-time weight download (~14 MB, no training)
    python segmentation/download_models.py

    # dry run on the two local shapes (or any dataset root)
    python segmentation/run_segmentation.py --root /data/xje/datasets/brokenchairs180k/shapes \
        --limit 2 --compare --report seg_report.json

    # full dataset, CPU, 8 parallel shape workers
    python segmentation/run_segmentation.py --root /data/xje/datasets/brokenchairs180k/shapes \
        --workers 8 --report /data/xje/datasets/brokenchairs180k/runs/seg_report.json

    # fast deterministic pass, no model needed
    python segmentation/run_segmentation.py --root ... --mode cv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from core import SAMSegmenter, imread, imwrite, load_rgb, segment_image
except ImportError:
    from .core import SAMSegmenter, imread, imwrite, load_rgb, segment_image


_SAM = None
_ARGS = None


def _init_worker(args):
    """One ONNX session per process (spawn-safe)."""
    global _SAM, _ARGS
    _ARGS = args
    _SAM = None
    if args.mode in ("auto", "sam"):
        try:
            _SAM = SAMSegmenter(args.model_dir)
        except Exception as exc:
            if args.mode == "sam":
                raise
            print(f"[warn] SAM unavailable ({exc}); using CV mode", flush=True)


def _out_dir(args, shape_root):
    if args.mask_root:
        return Path(args.mask_root) / shape_root.name
    return shape_root / args.out_subdir


def _compare_panel(shape_root, out_dir, args, names, limit=6):
    """Side-by-side RGB | old mask | new mask, ASCII labels only."""
    rows = []
    for name in names[:limit]:
        rgb = load_rgb(shape_root / args.images_subdir / name)
        old_p = shape_root / "mask" / name
        new_p = out_dir / name
        old = imread(old_p, cv2.IMREAD_GRAYSCALE) if old_p.is_file() \
            else np.zeros(rgb.shape[:2], np.uint8)
        new = imread(new_p, cv2.IMREAD_GRAYSCALE) if new_p.is_file() \
            else np.zeros(rgb.shape[:2], np.uint8)
        tiles = [rgb, np.dstack([old] * 3), np.dstack([new] * 3)]
        rows.append([cv2.resize(t, (256, 256)) for t in tiles])
    if not rows:
        return
    panel = np.vstack([np.hstack(r) for r in rows])
    strip = np.full((32, panel.shape[1], 3), 255, np.uint8)
    for j, label in enumerate(["RGB", "old mask", "clean mask"]):
        cv2.putText(strip, label, (j * 256 + 6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (0, 0, 0), 1, cv2.LINE_AA)
    imwrite(out_dir / "compare.png", cv2.cvtColor(
        np.vstack([strip, panel]), cv2.COLOR_RGB2BGR))


def _process_shape(task):
    """Worker: segment all query images of one shape."""
    shape_root = Path(task["root"])
    args = _ARGS
    img_dir = shape_root / args.images_subdir
    out_dir = _out_dir(args, shape_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(img_dir.glob("*.png"))
    if not files:
        return {"shape": shape_root.name, "n": 0, "skipped": 0,
                "methods": {}, "metrics": [], "error": "no images"}

    done, skipped, methods, metrics = 0, 0, {}, []
    for p in files:
        dest = out_dir / p.name
        if dest.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            rgb = load_rgb(p)
            res = segment_image(rgb, _SAM, mode=("cv" if _SAM is None else args.mode))
            mask = res["mask"]
            if args.dilate:
                r = abs(args.dilate)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                mask = cv2.dilate(mask, kernel) if args.dilate > 0 \
                    else cv2.erode(mask, kernel)
            imwrite(dest, mask)
            methods[res["method"]] = methods.get(res["method"], 0) + 1
            metrics.append(dict(name=p.name, method=res["method"],
                                **res["metrics"].as_dict()))
            done += 1
        except Exception as exc:
            metrics.append(dict(name=p.name, method="FAILED", error=str(exc)))
    if args.compare and done:
        try:
            _compare_panel(shape_root, out_dir, args, [p.name for p in files])
        except Exception as exc:
            print(f"[warn] compare panel failed for {shape_root.name}: {exc}",
                  flush=True)
    return {"shape": shape_root.name, "n": len(files), "done": done,
            "skipped": skipped, "methods": methods, "metrics": metrics}


def _summarise(results, elapsed, args, report_path):
    total = sum(r["n"] for r in results)
    done = sum(r.get("done", 0) for r in results)
    skipped = sum(r.get("skipped", 0) for r in results)
    failed = sum(m.get("method") == "FAILED"
                 for r in results for m in r.get("metrics", []))
    methods = {}
    for r in results:
        for k, v in r.get("methods", {}).items():
            methods[k] = methods.get(k, 0) + v
    allm = [m for r in results for m in r.get("metrics", [])
            if m.get("method") != "FAILED" and "edge_align" in m]
    agg = {}
    for key, label in [("edge_align", "edge_align"),
                       ("shadow_residue", "shadow_residue"),
                       ("speckles", "speckles")]:
        if allm:
            agg[label] = round(float(np.mean([m[key] for m in allm])), 4)

    print("\n" + "=" * 62)
    print(f"shapes: {len(results)}   images: {total} "
          f"(written {done}, skipped {skipped}, failed {failed})")
    print(f"methods: {methods or 'n/a'}")
    print(f"mean metrics: {agg or 'n/a'}")
    print(f"elapsed: {elapsed:.1f}s")
    print("=" * 62)

    if report_path:
        payload = {
            "config": {"root": args.root, "mode": args.mode,
                       "dilate": args.dilate,
                       "out_subdir": args.out_subdir,
                       "mask_root": args.mask_root,
                       "workers": args.workers,
                       "images_subdir": args.images_subdir,
                       "model_dir": args.model_dir},
            "summary": {"shapes": len(results), "images": total,
                        "written": done, "skipped": skipped, "failed": failed,
                        "methods": methods, "mean_metrics": agg,
                        "elapsed_s": round(elapsed, 1)},
            "shapes": results,
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"report -> {report_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="dataset root containing shape_*/ directories")
    ap.add_argument("--images-subdir", default="images")
    ap.add_argument("--out-subdir", default="mask_clean",
                    help="output subdir INSIDE each shape (default mask_clean)")
    ap.add_argument("--mask-root", default=None,
                    help="if set, write to <mask-root>/<shape>/ instead")
    ap.add_argument("--mode", choices=["auto", "sam", "cv"], default="auto")
    ap.add_argument("--dilate", type=int, default=0,
                    help="optional boundary compensation on the OUTPUT masks: "
                         "+1 dilates 1 px. Masks derived from anti-aliased RGB "
                         "sit ~1 px inside hard-edged renderer GT; enabling "
                         "this aligns them for mIoU scoring. Default 0 "
                         "(unmodified). See segmentation/README.md sec. 8.")
    ap.add_argument("--model-dir", default=os.path.join(here, "models"))
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel shape workers (each loads its own ONNX session)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only first N shapes")
    ap.add_argument("--shape", default=None, help="only this shape name")
    ap.add_argument("--compare", action="store_true",
                    help="write compare.png (RGB | old mask | clean mask)")
    ap.add_argument("--report", default=None, help="write JSON report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    shapes = sorted(p for p in root.iterdir() if p.is_dir()
                    and (p / args.images_subdir).is_dir())
    if args.shape:
        shapes = [p for p in shapes if p.name == args.shape]
    if args.limit:
        shapes = shapes[:args.limit]
    if not shapes:
        print(f"no shape dirs with '{args.images_subdir}/' under {root}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"shapes: {len(shapes)}  mode: {args.mode}  workers: {args.workers}")
        print(f"output: {args.mask_root or '<shape>/' + args.out_subdir}")
    tasks = [{"root": str(p)} for p in shapes]
    t0 = time.time()

    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(args,)) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_process_shape, tasks), 1):
                results.append(r)
                if not args.quiet:
                    print(f"[{i}/{len(tasks)}] {r['shape']:16s} "
                          f"images {r['n']:3d}  written {r.get('done', 0):3d}  "
                          f"skipped {r.get('skipped', 0):3d}  {r.get('methods', {})}",
                          flush=True)
    else:
        _init_worker(args)
        results = []
        for i, task in enumerate(tasks, 1):
            r = _process_shape(task)
            results.append(r)
            if not args.quiet:
                print(f"[{i}/{len(tasks)}] {r['shape']:16s} "
                      f"images {r['n']:3d}  written {r.get('done', 0):3d}  "
                      f"skipped {r.get('skipped', 0):3d}  {r.get('methods', {})}",
                      flush=True)

    _summarise(results, time.time() - t0, args, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
