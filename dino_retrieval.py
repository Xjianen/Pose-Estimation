#!/usr/bin/env python3
"""DINOv2 template retrieval: find the correct-side coarse pose for a query image.

Why this exists
---------------
Silhouette/mask supervision cannot tell a symmetric object's front from its back
(the outline is the same), so mask-only pose optimization (3DGS/gsplat) flips to
the back. DINOv2 features encode *internal structure* (handle, flap, ladder, ...)
and are texture-invariant, so nearest-neighbour retrieval against a set of posed
template renders returns the correct side. That pose is a flip-free initialization
for the differentiable (gsplat) refinement, and the good init also cuts iterations.

Renderer-agnostic: templates are (image + pose) pairs in the same K/RT json format
as the reference views. Render them however you like (gsplat / mesh / Blender) over
a viewpoint grid; this module only reads images + poses.

Preprocessing: the object sits on a white background, which would dilute the global
feature, so we mask the white bg, crop to the object bbox, and resize -> DINOv2 sees
the object, scale/position-normalized. Applied to BOTH templates and query.

Usage
-----
    # quick flip-resolution check using the EXISTING references as templates:
    python dino_retrieval.py --template-dir .../data/8340 \
        --query .../data/render_8340_4.2_69_40_Broken_anomaly.png --topk 5
"""

import argparse
import glob
import json
import os

import numpy as np
import cv2

# self-contained helpers (no external module needed)
def rotation_geodesic_deg(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def w2c_camera_center(R, t):
    return -R.T @ t


def load_gt(js):
    d = json.load(open(js))
    RT = np.asarray(d["RT"], dtype=np.float64)
    return np.asarray(d["K"], dtype=np.float64), RT[:3, :3], RT[:3, 3], d

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------- #
def white_bbox_crop(img_bgr, white_thresh=235, pad=0.12, out=224):
    """Mask the white background, crop to the object's square bbox (white-padded),
    resize to out x out. Returns RGB uint8 (out,out,3)."""
    H, W = img_bgr.shape[:2]
    fg = img_bgr.min(axis=2) < white_thresh
    ys, xs = np.nonzero(fg)
    if len(xs) < 10:
        crop = img_bgr
    else:
        cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0 * (1 + pad)
        X0, Y0 = int(round(cx - half)), int(round(cy - half))
        X1, Y1 = int(round(cx + half)), int(round(cy + half))
        canvas = np.full((Y1 - Y0, X1 - X0, 3), 255, np.uint8)
        sx0, sy0 = max(X0, 0), max(Y0, 0)
        sx1, sy1 = min(X1, W), min(Y1, H)
        canvas[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = img_bgr[sy0:sy1, sx0:sx1]
        crop = canvas
    return cv2.cvtColor(cv2.resize(crop, (out, out)), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
def _load_dinov2(torch, name):
    """Load DINOv2 without needing the network when it is already cached.

    torch.hub.load() pings github to resolve the default branch even for a cached
    repo, which fails on a flaky/offline box. Prefer the local hub cache; only fall
    back to github (with the ref pinned so no branch lookup is needed).
    """
    local = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if os.path.isdir(local):
        print(f"loading DINOv2 from local hub cache: {local}")
        return torch.hub.load(local, name, source="local")
    print("local hub cache not found; fetching from github (needs network)")
    return torch.hub.load("facebookresearch/dinov2:main", name, skip_validation=True)


class Dino:
    def __init__(self, device, name="dinov2_vitl14", mode="spatial"):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.device = device
        self.mode = mode
        self.model = _load_dinov2(torch, name).to(device).eval()

    def features(self, rgb_uint8_list, batch=8):
        """rgb_uint8_list: list of (H,W,3) uint8. Returns L2-normalized (N,D) float32.

        mode='spatial' keeps the patch grid: each patch token is L2-normalized then
        the grid is flattened, so a dot product equals the MEAN PER-PATCH cosine.
        This preserves spatial layout (where the handle/flap sits), which is what
        encodes viewpoint — mean-pooling destroys exactly that signal.
        """
        torch, F = self.torch, self.F
        feats = []
        for i in range(0, len(rgb_uint8_list), batch):
            chunk = rgb_uint8_list[i:i + batch]
            x = np.stack(chunk).astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(self.device)
            with torch.no_grad():
                out = self.model.forward_features(x)
                cls = out["x_norm_clstoken"]
                patch = out["x_norm_patchtokens"]
                if self.mode == "cls":
                    f = cls
                elif self.mode == "mean":
                    f = torch.cat([cls, patch.mean(1)], dim=-1)
                else:                                   # spatial
                    f = F.normalize(patch, dim=-1).flatten(1)
                f = F.normalize(f, dim=-1)
            feats.append(f.float().cpu().numpy())
        return np.concatenate(feats, 0)


# --------------------------------------------------------------------------- #
def parse_azel(path):
    """Parse (radius, azim, elev) = the LAST three numeric parts of the stem.
    Handles '8430_3.0_108_20' and 'render_8430_3.8_31_29_anomaly'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = []
    for p in stem.split("_"):
        try:
            nums.append(float(p))
        except ValueError:
            pass
    return tuple(nums[-3:]) if len(nums) >= 3 else None


def azim_diff(a, b):
    """Smallest absolute azimuth difference in degrees (mod 360)."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def load_templates(template_dir):
    tpls = []
    for png in sorted(glob.glob(os.path.join(template_dir, "*.png"))):
        js = os.path.splitext(png)[0] + ".json"
        if not os.path.exists(js):
            continue
        d = json.load(open(js))
        RT = np.asarray(d["RT"], dtype=np.float64)
        tpls.append({"png": png, "RT": RT,
                     "R": RT[:3, :3], "t": RT[:3, 3],
                     "azel": parse_azel(png)})
    return tpls


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    base = "/home/xiangjianen/projects/Depth-Anything-3/Anomaly/data"
    ap.add_argument("--template-dir", default=f"{base}/8340",
                    help="dir of template renders (*.png + *.json with K/RT)")
    ap.add_argument("--query", default=f"{base}/render_8340_4.2_69_40_Broken_anomaly.png")
    ap.add_argument("--query-dir", default=None,
                    help="batch mode: retrieve for every *.png in this dir and "
                         "report azimuth-error / flip statistics")
    ap.add_argument("--query-gt", default=None, help="query json (optional; default: stem.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--white-thresh", type=int, default=235)
    ap.add_argument("--feat", default="spatial", choices=["spatial", "mean", "cls"],
                    help="spatial keeps the patch grid (viewpoint-discriminative); "
                         "mean/cls pool it away")
    ap.add_argument("--sym-azim", type=float, default=180.0,
                    help="object azimuth symmetry period in degrees (this bag is "
                         "~180-symmetric: theta and theta+180 render near-identically, "
                         "L1=1.9/255). Errors are also reported modulo this. 0=none")
    ap.add_argument("--out-init", default=None,
                    help="write the best template's {K,RT} json here (feed to "
                         "gsplat_refine.py --init-json)")
    args = ap.parse_args()

    tpls = load_templates(args.template_dir)
    if not tpls:
        raise SystemExit(f"no templates in {args.template_dir}")
    print(f"{len(tpls)} templates from {args.template_dir}")

    def crop(path):
        return white_bbox_crop(cv2.imread(path), args.white_thresh, out=args.crop)

    dino = Dino(args.device, mode=args.feat)
    print(f"feature mode = {args.feat}")
    tpl_feats = dino.features([crop(t["png"]) for t in tpls])

    queries = ([args.query] if not args.query_dir
               else sorted(glob.glob(os.path.join(args.query_dir, "*.png"))))
    print(f"{len(queries)} query image(s)")

    az_errs, az_errs_sym, flips = [], [], 0
    for qi, qpath in enumerate(queries):
        q_feat = dino.features([crop(qpath)])[0]
        sims = tpl_feats @ q_feat                   # cosine (features L2-normalized)
        order = np.argsort(-sims)
        qazel = parse_azel(qpath)

        # GT pose scoring if a matching json exists
        gt_js = args.query_gt if (len(queries) == 1 and args.query_gt) else \
            os.path.splitext(qpath)[0] + ".json"
        have_gt = os.path.exists(gt_js)
        Rq = Cq = None
        if have_gt:
            _, Rq, tq, _ = load_gt(gt_js)
            Cq = w2c_camera_center(Rq, tq)

        print(f"\n[{qi}] {os.path.basename(qpath)}  gt azel={qazel}"
              f"{'' if have_gt else '   (no GT json -> azimuth check only)'}")
        print(f"{'rank':>4} {'sim':>6} {'template':<26} {'azel':>18}"
              + (f" {'rot_err°':>9} {'center_err':>11}" if have_gt else ""))
        for k in range(min(args.topk, len(tpls))):
            i = order[k]
            t = tpls[i]
            azel = (f"({t['azel'][0]:.1f},{t['azel'][1]:.0f},{t['azel'][2]:.0f})"
                    if t["azel"] else "n/a")
            line = f"{k:>4} {sims[i]:6.3f} {os.path.basename(t['png']):<26} {azel:>18}"
            if have_gt:
                rot = rotation_geodesic_deg(t["R"], Rq)
                ctr = float(np.linalg.norm(w2c_camera_center(t["R"], t["t"]) - Cq))
                line += f" {rot:9.3f} {ctr:11.4f}"
            print(line)

        best = tpls[order[0]]
        if qazel and best["azel"]:
            ae = azim_diff(best["azel"][1], qazel[1])
            ae_sym = min(ae, abs(args.sym_azim - ae)) if args.sym_azim > 0 else ae
            az_errs.append(ae)
            az_errs_sym.append(ae_sym)
            flipped = ae > 90.0
            flips += int(flipped)
            note = ("symmetry-equivalent (OK)" if flipped and ae_sym < 20
                    else "correct side" if not flipped else "genuinely wrong")
            print(f"  top-1 azimuth {best['azel'][1]:.0f}° vs gt {qazel[1]:.0f}° "
                  f"-> |Δazim| = {ae:.1f}°, mod-{args.sym_azim:.0f}° = {ae_sym:.1f}°"
                  f"  [{note}]")

        if args.out_init and len(queries) == 1:
            bj = json.load(open(os.path.splitext(best["png"])[0] + ".json"))
            json.dump({"K": bj["K"], "RT": bj["RT"]}, open(args.out_init, "w"), indent=2)
            print(f"  wrote init pose -> {args.out_init}")

    if az_errs:
        a = np.asarray(az_errs)
        s = np.asarray(az_errs_sym)
        print("\n" + "=" * 66)
        print(f"queries: {len(a)}   feature={args.feat}")
        print(f"  raw azimuth error        : median={np.median(a):5.1f}° "
              f"mean={a.mean():5.1f}° max={a.max():5.1f}°   >90°: {flips}/{len(a)}")
        print(f"  mod-{args.sym_azim:.0f}° (symmetry-aware): median={np.median(s):5.1f}° "
              f"mean={s.mean():5.1f}° max={s.max():5.1f}°   "
              f"<10°: {(s<10).sum()}/{len(s)}  <20°: {(s<20).sum()}/{len(s)}")
        print("\nNOTE: this object is ~180°-azimuth symmetric in APPEARANCE "
              "(L1 1.9/255 vs 21.8\n      for an 18° neighbour), so theta vs theta+180 "
              "is genuinely unresolvable from\n      a single image. Judge by the "
              "symmetry-aware row; the raw row's flips are\n      equivalent solutions, "
              "not errors.")


if __name__ == "__main__":
    main()
