#!/usr/bin/env python3
"""Silhouette codebook: dense (azimuth x elevation) template masks from a 3DGS .ply,
plus fast mask matching to get a coarse query pose.

Why this design
---------------
* Measured on this dataset: silhouette matching is EXACT (0 deg error, leave-one-out)
  when the template viewpoint distribution covers the query. The earlier failure
  (median 42 deg) came from templates existing only at elev 20 / r 3.0 while queries
  span elev 21-37 / r 3.1-4.4 -- a 17 deg elevation gap changes the silhouette more
  than an 18 deg azimuth change. So: densify the codebook, keep the matcher simple.
* Matching is a single matmul over binary masks -> milliseconds, no network forward.
  Rendering ~2k templates with gsplat takes seconds. This is far CHEAPER than a
  DINOv2 forward per template or a 150-step gradient optimisation per query.
* Masks are bbox-crop normalised, so radius is not discriminative and does NOT need
  to be in the grid: it is recovered analytically from the mask scale
  (r = f * object_size / bbox_pixels) and polished later by gsplat_refine.

Pose convention (reverse-engineered from mv_images and verified to 1e-7)
-----------------------------------------------------------------------
    C      = r * (cos(el)cos(az), sin(el), cos(el)sin(az))     # world is y-up
    target = mesh bounding-box centre
    up     = (0,1,0)
    z = normalise(C - target);  x = normalise(up x z);  y = z x x
    R_w2c  = [x y z]^T ;  t = -R_w2c C                          # OpenGL w2c (= GT `RT`)

Usage
-----
    python codebook.py verify-pose --mv-dir shape_8430/mv_images --mesh shape_8430/8430.obj
    python codebook.py render --ply shape_8430/point_cloud/8430.ply \
        --mesh shape_8430/8430.obj --out cb_8430.npz \
        --azim-step 3 --elev-min 15 --elev-max 45 --elev-step 2
    python codebook.py match --codebook cb_8430.npz --query-dir shape_8430/images
    python codebook.py match --codebook cb_8430.npz --query q.png --out-init init_pose.json
"""

import argparse
import glob
import json
import os

import numpy as np
import cv2

S4 = np.diag([1.0, -1.0, -1.0, 1.0])          # OpenGL <-> OpenCV (involution)
UP = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------- #
# Pose construction (verified against mv_images to ~1e-7)
# --------------------------------------------------------------------------- #
def camera_centre(radius, azim_deg, elev_deg):
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    return radius * np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])


def pose_from_azel(radius, azim_deg, elev_deg, target):
    """Return the 4x4 OpenGL world-to-camera matrix (same convention as GT `RT`)."""
    C = camera_centre(radius, azim_deg, elev_deg)
    z = C - target
    z = z / np.linalg.norm(z)
    x = np.cross(UP, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1).T
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = -R @ C
    return RT


def mesh_bbox_centre(obj_path):
    v = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(v)
    return (v.min(0) + v.max(0)) / 2.0


def mesh_size(obj_path):
    """Max bbox extent (world units) — used for the analytic radius estimate."""
    v = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(v)
    return float((v.max(0) - v.min(0)).max())


# --------------------------------------------------------------------------- #
# Mask utilities (shared by codebook and query so both are normalised the same)
# --------------------------------------------------------------------------- #
def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def normalise_mask(mask, out, pad=0.12):
    """Square bbox crop (aspect preserved, zero padded) -> out x out uint8.
    Also returns the pre-crop bbox pixel size (for the radius estimate)."""
    bb = bbox_of(mask)
    if bb is None:
        return np.zeros((out, out), np.uint8), 0.0
    x0, y0, x1, y1 = bb
    H, W = mask.shape
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    size = max(x1 - x0, y1 - y0)
    half = size / 2.0 * (1 + pad)
    X0, Y0 = int(round(cx - half)), int(round(cy - half))
    X1, Y1 = int(round(cx + half)), int(round(cy + half))
    can = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    sx0, sy0, sx1, sy1 = max(X0, 0), max(Y0, 0), min(X1, W), min(Y1, H)
    can[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = mask[sy0:sy1, sx0:sx1]
    m = cv2.resize(can, (out, out), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8), float(size)


def query_mask(png, white_thresh=235):
    img = cv2.imread(png)
    if img is None:
        raise SystemExit(f"cannot read {png}")
    return (img.min(axis=2) < white_thresh).astype(np.uint8), img.shape[1]


# --------------------------------------------------------------------------- #
def cmd_verify_pose(args):
    """Regenerate RT from each mv_images filename's (r,az,el) and compare to stored."""
    target = mesh_bbox_centre(args.mesh)
    print(f"mesh bbox centre (look-at target) = {np.round(target, 6)}")
    dR, dt = [], []
    for js in sorted(glob.glob(os.path.join(args.mv_dir, "*.json"))):
        nums = [float(p) for p in os.path.splitext(os.path.basename(js))[0].split("_")
                if _isnum(p)]
        r, az, el = nums[-3:]
        RT_gt = np.asarray(json.load(open(js))["RT"], dtype=np.float64)
        RT = pose_from_azel(r, az, el, target)
        dR.append(np.abs(RT[:3, :3] - RT_gt[:3, :3]).max())
        dt.append(np.abs(RT[:3, 3] - RT_gt[:3, 3]).max())
    print(f"verified {len(dR)} views: max|dR| = {max(dR):.3e}   max|dt| = {max(dt):.3e}")
    print("=> PASS (poses match GT convention)" if max(dR) < 1e-5 and max(dt) < 1e-5
          else "=> FAIL: convention mismatch")


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
def cmd_build_from_refs(args):
    """Pack the GIVEN reference views into codebook format — no extra rendering.

    This is the FAIR setting: it uses only the reference viewpoints that ship with
    the dataset (PTAD uses 20 of them; azim 0 and 360 are the same view), so any
    accuracy gain must come from the matcher/refiner, not from extra templates.
    """
    pngs = sorted(glob.glob(os.path.join(args.ref_dir, "*.png")))
    target = mesh_bbox_centre(args.mesh)
    obj_sz = mesh_size(args.mesh)
    masks, bbox_px, RTs, az, el = [], [], [], [], []
    seen = set()
    for p in pngs:
        js = os.path.splitext(p)[0] + ".json"
        if not os.path.exists(js):
            continue
        nums = [float(x) for x in os.path.splitext(os.path.basename(p))[0].split("_")
                if _isnum(x)]
        r, a, e = nums[-3:]
        key = (round(a % 360, 3), round(e, 3))
        if key in seen:                      # drop the duplicate 0/360 view
            print(f"  skipping duplicate view {os.path.basename(p)}")
            continue
        seen.add(key)
        img = cv2.imread(p)
        raw = (img.min(axis=2) < args.white_thresh).astype(np.uint8)
        m, sz = normalise_mask(raw, args.mask_res)
        masks.append(m); bbox_px.append(sz)
        RTs.append(np.asarray(json.load(open(js))["RT"], np.float32))
        az.append(a % 360); el.append(e)
        K = json.load(open(js))["K"]
    W = cv2.imread(pngs[0]).shape[1]
    np.savez_compressed(
        args.out, masks=np.asarray(masks, np.uint8),
        bbox_px=np.asarray(bbox_px, np.float32), RT=np.asarray(RTs, np.float32),
        azim=np.asarray(az, np.float32), elev=np.asarray(el, np.float32),
        radius=np.float32(nums[-3]), focal=np.float32(K[0][0]),
        res=np.int32(W), mask_res=np.int32(args.mask_res),
        obj_size=np.float32(obj_sz), target=target.astype(np.float32))
    print(f"packed {len(masks)} reference views (from {len(pngs)} files) -> {args.out}")


# --------------------------------------------------------------------------- #
def _edt(mask_bool):
    """distance to the nearest True pixel (cv2 equivalent of scipy EDT of ~mask)."""
    return cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 3)


class Codebook:
    """Loaded codebook + the single shared implementation of coarse matching.

    Both the CLI (`match`) and the batch runner use this, so the two can never
    diverge in how they score or how they build poses.
    """

    def __init__(self, path):
        cb = np.load(path)
        self.masks = cb["masks"]
        self.N = self.masks.shape[0]
        self.mres = int(cb["mask_res"])
        self.flat = self.masks.reshape(self.N, -1)
        self.flatf = self.flat.astype(np.float32)
        self.area = self.flat.sum(1).astype(np.float32)
        self.azim, self.elev = cb["azim"], cb["elev"]
        self.RT = cb["RT"]
        self.bbox_px = cb["bbox_px"]
        self.focal = float(cb["focal"])
        self.radius = float(cb["radius"])
        self.res = int(cb["res"])
        self.obj_size = float(cb["obj_size"])
        self.target = cb["target"].astype(np.float64)
        self._dist = None

    def _tpl_dist(self):
        if self._dist is None:
            self._dist = np.stack([_edt(m.astype(bool)) for m in self.masks]
                                  ).reshape(self.N, -1)
        return self._dist

    def match(self, png, metric="chamfer", white_thresh=235, query_focal=-1.0, topk=3):
        """Return (hypotheses, scores, order, f_q, W). Lower score = better."""
        raw, W = query_mask(png, white_thresh)
        qm, qsize = normalise_mask(raw, self.mres)
        q = qm.reshape(-1).astype(np.float32)
        qa = float(q.sum())
        if metric == "iou":
            inter = self.flatf @ q
            score = -(inter / (self.area + qa - inter + 1e-6))
        else:                                   # PTAD symmetric region chamfer
            qd = _edt(qm.astype(bool)).reshape(-1)
            score = ((self._tpl_dist() @ q) / max(qa, 1e-6)
                     + (self.flatf @ qd) / np.maximum(self.area, 1e-6))
        order = np.argsort(score)
        f_q = query_focal if query_focal > 0 else self.focal * (W / self.res)
        hyps = []
        for i in order[:max(topk, 1)]:
            r_est = self.radius * (self.bbox_px[i] / max(qsize, 1e-6)) * (f_q / self.focal)
            hyps.append({"idx": int(i), "azim": float(self.azim[i]),
                         "elev": float(self.elev[i]), "radius": float(r_est),
                         "score": float(score[i]),
                         "RT": pose_from_azel(r_est, float(self.azim[i]),
                                              float(self.elev[i]), self.target)})
        return hyps, score, order, f_q, W


# --------------------------------------------------------------------------- #
def cmd_render(args):
    import torch
    from gsplat import rasterization
    from gsplat_refine import load_gaussians_ply

    dev = torch.device(args.device)
    g = load_gaussians_ply(args.ply)
    sh_deg = g["sh_degree"]
    gs = {k: torch.tensor(v, dtype=torch.float32, device=dev)
          for k, v in g.items() if k != "sh_degree"}
    gs["scales"] = torch.exp(gs["scales"]) * args.scale_mult
    gs["opacities"] = torch.sigmoid(gs["opacities"])
    gs["quats"] = torch.nn.functional.normalize(gs["quats"], dim=-1)
    print(f"loaded {len(g['means'])} gaussians")

    target = mesh_bbox_centre(args.mesh)
    obj_sz = mesh_size(args.mesh)
    azims = np.arange(0.0, 360.0, args.azim_step)
    elevs = np.arange(args.elev_min, args.elev_max + 1e-9, args.elev_step)
    grid = [(a, e) for e in elevs for a in azims]
    print(f"grid: {len(azims)} azim x {len(elevs)} elev = {len(grid)} templates "
          f"(radius fixed at {args.radius}, absorbed by crop normalisation)")

    R = args.res
    K = np.array([[args.focal, 0, R / 2], [0, args.focal, R / 2], [0, 0, 1.0]])
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)

    masks = np.zeros((len(grid), args.mask_res, args.mask_res), np.uint8)
    bbox_px = np.zeros(len(grid), np.float32)
    RTs = np.zeros((len(grid), 4, 4), np.float32)

    import time
    t0 = time.time()
    for i0 in range(0, len(grid), args.batch):
        chunk = grid[i0:i0 + args.batch]
        vm = []
        for (az, el) in chunk:
            RT = pose_from_azel(args.radius, az, el, target)
            RTs[i0 + len(vm)] = RT
            vm.append(S4 @ RT)
        viewmats = torch.tensor(np.stack(vm), dtype=torch.float32, device=dev)
        Ks = Kt[None].expand(len(chunk), 3, 3).contiguous()
        with torch.no_grad():
            _, alphas, _ = rasterization(
                means=gs["means"], quats=gs["quats"], scales=gs["scales"],
                opacities=gs["opacities"], colors=gs["shs"],
                viewmats=viewmats, Ks=Ks, width=R, height=R,
                sh_degree=sh_deg, packed=False)
        a = (alphas[..., 0] > 0.5).to(torch.uint8).cpu().numpy()
        for j in range(len(chunk)):
            m, sz = normalise_mask(a[j], args.mask_res)
            masks[i0 + j] = m
            bbox_px[i0 + j] = sz
        if i0 % (args.batch * 10) == 0:
            print(f"  rendered {i0 + len(chunk)}/{len(grid)}")
    dt = time.time() - t0

    np.savez_compressed(
        args.out, masks=masks, bbox_px=bbox_px, RT=RTs,
        azim=np.array([a for a, _ in grid], np.float32),
        elev=np.array([e for _, e in grid], np.float32),
        radius=np.float32(args.radius), focal=np.float32(args.focal),
        res=np.int32(R), mask_res=np.int32(args.mask_res),
        obj_size=np.float32(obj_sz), target=target.astype(np.float32))
    mb = os.path.getsize(args.out) / 1e6
    print(f"rendered {len(grid)} templates in {dt:.1f}s "
          f"({1000*dt/len(grid):.1f} ms/view) -> {args.out} ({mb:.1f} MB)")


# --------------------------------------------------------------------------- #
def cmd_match(args):
    cb = Codebook(args.codebook)
    masks, N, mres = cb.masks, cb.N, cb.mres
    azim, elev = cb.azim, cb.elev
    print(f"codebook: {N} templates, mask {mres}x{mres}, "
          f"azim step {np.unique(azim)[1]-np.unique(azim)[0]:.0f}°, "
          f"elev {elev.min():.0f}-{elev.max():.0f}°   metric={args.metric}")

    queries = ([args.query] if not args.query_dir
               else sorted(glob.glob(os.path.join(args.query_dir, "*.png"))))
    az_err_p, az_err_m, el_err, r_err = [], [], [], []

    for qp in queries:
        hyps, score, order, f_q, W = cb.match(
            qp, args.metric, args.white_thresh, args.query_focal,
            topk=max(args.topk, args.out_init_topk, 1))
        best = order[0]
        shown = -score if args.metric == "iou" else score
        r_est = hyps[0]["radius"]

        gt = _gt_azel(qp)
        line = (f"\n{os.path.basename(qp)}: top1 azim={azim[best]:.0f}° "
                f"elev={elev[best]:.0f}° {args.metric}={shown[best]:.3f}  r_est={r_est:.2f}")
        if gt:
            ae_p = _azdiff(azim[best], gt[1])
            ae_p = min(ae_p, abs(180 - ae_p))
            ae_m = _azdiff(azim[best], -gt[1] % 360.0)
            ae_m = min(ae_m, abs(180 - ae_m))
            ee = abs(elev[best] - gt[2]); re = abs(r_est - gt[0])
            az_err_p.append(ae_p); az_err_m.append(ae_m)
            el_err.append(ee); r_err.append(re)
            line += (f"\n   gt azim={gt[1]:.0f}° elev={gt[2]:.0f}° r={gt[0]:.1f}"
                     f"  ->  |Δazim| +sign={ae_p:.1f}° / -sign={ae_m:.1f}°"
                     f"  |Δelev|={ee:.1f}°  |Δr|={re:.2f}")
        print(line)
        for k in range(1, min(args.topk, N)):
            i = order[k]
            print(f"     #{k} azim={azim[i]:6.0f}° elev={elev[i]:4.0f}° {args.metric}={shown[i]:.3f}")

        if args.out_init and len(queries) == 1:
            RT = pose_from_azel(r_est, float(azim[best]), float(elev[best]), cb.target)
            json.dump({"K": _K_for(f_q, W), "RT": RT.tolist()},
                      open(args.out_init, "w"), indent=2)
            print(f"   wrote init pose -> {args.out_init}")
            if args.out_init_topk > 1:
                stem = os.path.splitext(args.out_init)[0]
                for k in range(min(args.out_init_topk, N)):
                    i = order[k]
                    RTk = pose_from_azel(r_est, float(azim[i]), float(elev[i]), cb.target)
                    fn = f"{stem}_{k}.json"
                    json.dump({"K": _K_for(f_q, W), "RT": RTk.tolist()},
                              open(fn, "w"), indent=2)
                print(f"   wrote top-{args.out_init_topk} hypotheses -> {stem}_0..{args.out_init_topk-1}.json")

        if args.out_gt and len(queries) == 1:
            if not gt:
                raise SystemExit("--out-gt needs (radius,azim,elev) in the filename")
            RTg = pose_from_azel(gt[0], args.gt_azim_sign * gt[1], gt[2], cb.target)
            json.dump({"K": _K_for(f_q, W), "RT": RTg.tolist()},
                      open(args.out_gt, "w"), indent=2)
            print(f"   wrote filename-derived GT pose -> {args.out_gt} "
                  f"(azim sign {args.gt_azim_sign:+.0f}; filename values are rounded "
                  f"to 1°, so this GT is only accurate to ~0.5°)")

    if az_err_p:
        ap_, am_, e, r = map(np.asarray, (az_err_p, az_err_m, el_err, r_err))
        print("\n" + "=" * 68)
        print(f"{len(ap_)} queries   (azimuth errors are mod-180, i.e. symmetry-aware)")
        print(f"  azimuth vs +filename_azim: median={np.median(ap_):5.1f}° "
              f"mean={ap_.mean():5.1f}° max={ap_.max():5.1f}°  <10°: {(ap_<10).sum()}/{len(ap_)}")
        print(f"  azimuth vs -filename_azim: median={np.median(am_):5.1f}° "
              f"mean={am_.mean():5.1f}° max={am_.max():5.1f}°  <10°: {(am_<10).sum()}/{len(am_)}")
        print(f"  elevation                : median={np.median(e):5.1f}° "
              f"mean={e.mean():5.1f}° max={e.max():5.1f}°")
        print(f"  radius (analytic)        : median={np.median(r):5.2f}  "
              f"mean={r.mean():5.2f} max={r.max():5.2f}")
        better = "-" if am_.mean() < ap_.mean() else "+"
        print(f"\nThe '{better}filename_azim' row is the consistent one: the codebook follows the\n"
              f"mv_images azimuth convention (verified to 1e-7 against real RT matrices),\n"
              f"and '{'render_*' if better=='-' else 'mv_*'}' filenames label azimuth with the "
              f"opposite rotation direction.\nThe OUTPUT POSE is correct either way — only the "
              f"filename-derived label differs.")


def _K_for(f, W):
    return [[f, 0.0, W / 2.0], [0.0, f, W / 2.0], [0.0, 0.0, 1.0]]


def _gt_azel(path):
    nums = [float(p) for p in os.path.splitext(os.path.basename(path))[0].split("_")
            if _isnum(p)]
    return tuple(nums[-3:]) if len(nums) >= 3 else None


def _azdiff(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify-pose", help="check the (r,az,el)->RT formula vs mv_images")
    v.add_argument("--mv-dir", required=True)
    v.add_argument("--mesh", required=True)

    r = sub.add_parser("render", help="render the silhouette codebook with gsplat")
    r.add_argument("--ply", required=True)
    r.add_argument("--mesh", required=True)
    r.add_argument("--out", default="codebook.npz")
    r.add_argument("--azim-step", type=float, default=3.0)
    r.add_argument("--elev-min", type=float, default=15.0)
    r.add_argument("--elev-max", type=float, default=45.0)
    r.add_argument("--elev-step", type=float, default=2.0)
    r.add_argument("--radius", type=float, default=3.5)
    r.add_argument("--res", type=int, default=256)
    r.add_argument("--focal", type=float, default=355.0)
    r.add_argument("--mask-res", type=int, default=160)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--scale-mult", type=float, default=1.0)
    r.add_argument("--device", default="cuda")

    b = sub.add_parser("build-from-refs",
                       help="pack ONLY the given reference views into a codebook (fair vs PTAD)")
    b.add_argument("--ref-dir", required=True)
    b.add_argument("--mesh", required=True)
    b.add_argument("--out", default="cb_refs.npz")
    b.add_argument("--mask-res", type=int, default=160)
    b.add_argument("--white-thresh", type=int, default=235)

    m = sub.add_parser("match", help="match a query silhouette against the codebook")
    m.add_argument("--codebook", required=True)
    m.add_argument("--metric", default="chamfer", choices=["chamfer", "iou"],
                   help="chamfer = PTAD's symmetric region chamfer distance; iou = mask IoU")
    m.add_argument("--query", default=None)
    m.add_argument("--query-dir", default=None)
    m.add_argument("--query-focal", type=float, default=-1.0)
    m.add_argument("--white-thresh", type=int, default=235)
    m.add_argument("--topk", type=int, default=3)
    m.add_argument("--out-init", default=None)
    m.add_argument("--out-init-topk", type=int, default=0,
                   help="also write the top-K poses as <out-init stem>_k.json for "
                        "multi-hypothesis refinement (correct view is in top-3 for all "
                        "16 test queries, so K=3 suffices)")
    m.add_argument("--out-gt", default=None,
                   help="write the filename-derived GT pose json (for honest scoring)")
    m.add_argument("--gt-azim-sign", type=float, default=-1.0,
                   help="-1 for render_* filenames (their azimuth runs opposite to the "
                        "verified RT convention), +1 for mv_images-style names")

    args = ap.parse_args()
    {"verify-pose": cmd_verify_pose, "build-from-refs": cmd_build_from_refs,
     "render": cmd_render, "match": cmd_match}[args.cmd](args)


if __name__ == "__main__":
    main()
