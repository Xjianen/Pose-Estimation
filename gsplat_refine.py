#!/usr/bin/env python3
"""gsplat pose refinement: optimize the query camera pose against a 3DGS .ply.

Role in the pipeline
--------------------
1. dino_retrieval.py -> correct-side COARSE pose (fixes the front/back flip).
2. THIS script       -> differentiable gsplat render + SE(3) refinement to the
                        strict target (rot <10deg, center <0.1).

Key ideas
---------
* Pose is optimized as a small SE(3) delta `xi in R^6` applied to the GAUSSIANS
  (means + quats) while the camera viewmat stays fixed at the init. This is
  equivalent to moving the camera (V_est = V0 @ exp(xi)) but only needs gradients
  through means/quats, which every gsplat version supports -> version-robust.
* Loss is TEXTURE-INVARIANT (query is textured/colored, the .ply from an untextured
  mesh is gray): soft-silhouette (alpha vs query object mask) + grayscale edge/
  gradient loss (+ optional DINOv2 feature-metric). Silhouette alone is symmetric
  and would re-introduce the flip, so we rely on the DINOv2 init for the side and
  the edge/feature term to lock internal structure.

Conventions
-----------
GT `RT` is OpenGL w2c (looks -z); gsplat/OpenCV viewmat looks +z. They differ by
S=diag(1,-1,-1). We convert init OpenGL->OpenCV for rendering and OpenCV->OpenGL
for scoring. The .ply is assumed to live in the SAME world frame as the GT poses
(true if built from the mesh); `--verify` checks this by rendering a known
reference pose and comparing silhouette IoU to that reference image.

Usage
-----
    python gsplat_refine.py --selftest                       # numpy-only math check
    python gsplat_refine.py --ply obj.ply --verify \
        --verify-ref data/8340/8340_3.5_108_20              # convention/frame gate
    python gsplat_refine.py --ply obj.ply \
        --init-json <retrieved_template>.json --iters 60     # refine from DINO init
"""

import argparse
import json
import os

import numpy as np

S3 = np.diag([1.0, -1.0, -1.0])
S4 = np.diag([1.0, -1.0, -1.0, 1.0])          # OpenGL<->OpenCV (involution)


# --------------------------------------------------------------------------- #
# Convention helpers (numpy; used by selftest and scoring)
# --------------------------------------------------------------------------- #
def gl_w2c_to_cv_viewmat(RT_gl):
    return S4 @ RT_gl


def cv_viewmat_to_gl_w2c(V_cv):
    return S4 @ V_cv


def rot_geodesic_deg(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def cam_center(R, t):
    return -R.T @ t


# --------------------------------------------------------------------------- #
# Standard 3DGS .ply loader
# --------------------------------------------------------------------------- #
def load_gaussians_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while True:
            line = f.readline(); hdr += line
            if line.strip() == b"end_header":
                break
        txt = hdr.decode("latin1").splitlines()
        n = next(int(l.split()[-1]) for l in txt if l.startswith("element vertex"))
        props = [l.split()[-1] for l in txt if l.startswith("property")]
        assert "binary_little_endian" in "".join(txt), "expected binary_little_endian ply"
        data = np.frombuffer(f.read(n * len(props) * 4), dtype=np.float32).reshape(n, len(props))
    col = {name: i for i, name in enumerate(props)}

    def g(*names):
        return np.stack([data[:, col[nm]] for nm in names], axis=1)

    means = g("x", "y", "z")
    scales = g("scale_0", "scale_1", "scale_2")
    quats = g("rot_0", "rot_1", "rot_2", "rot_3")
    opac = data[:, col["opacity"]]
    f_dc = g("f_dc_0", "f_dc_1", "f_dc_2")                    # (N,3)
    rest_names = sorted([p for p in props if p.startswith("f_rest_")],
                        key=lambda s: int(s.split("_")[-1]))
    if rest_names:
        f_rest = np.stack([data[:, col[p]] for p in rest_names], 1)   # (N, 3*15)
        k = f_rest.shape[1] // 3
        f_rest = f_rest.reshape(-1, 3, k).transpose(0, 2, 1)          # (N,k,3)
        shs = np.concatenate([f_dc[:, None, :], f_rest], axis=1)      # (N, k+1, 3)
    else:
        shs = f_dc[:, None, :]
    sh_degree = int(round(np.sqrt(shs.shape[1]))) - 1
    return {"means": means, "scales": scales, "quats": quats,
            "opacities": opac, "shs": shs, "sh_degree": sh_degree}


# --------------------------------------------------------------------------- #
# Torch / gsplat pieces (imported lazily so --selftest needs no torch)
# --------------------------------------------------------------------------- #
def se3_exp(xi):
    """xi: (6,) torch, [omega(3), nu(3)] -> 4x4 SE(3) matrix (torch)."""
    import torch
    omega, nu = xi[:3], xi[3:]
    theta = torch.linalg.norm(omega) + 1e-12
    K = torch.zeros(3, 3, device=xi.device, dtype=xi.dtype)
    K[0, 1], K[0, 2], K[1, 0] = -omega[2], omega[1], omega[2]
    K[1, 2], K[2, 0], K[2, 1] = -omega[0], -omega[1], omega[0]
    K = K / theta
    R = (torch.eye(3, device=xi.device, dtype=xi.dtype)
         + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K))
    V = (torch.eye(3, device=xi.device, dtype=xi.dtype)
         + (1 - torch.cos(theta)) / theta * K
         + (theta - torch.sin(theta)) / theta * (K @ K))
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3, 3] = V @ nu
    return T


def quat_mul(a, b):
    import torch
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def mat_to_quat(R):
    import torch
    w = torch.sqrt(torch.clamp(1 + R[0, 0] + R[1, 1] + R[2, 2], min=1e-8)) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return torch.stack([w, x, y, z])


def render(gauss_t, viewmat, K, H, W, sh_degree):
    """gsplat rasterization -> (rgb[H,W,3], alpha[H,W,1], depth[H,W,1])."""
    import torch
    from gsplat import rasterization
    colors, alphas, meta = rasterization(
        means=gauss_t["means"], quats=gauss_t["quats"], scales=gauss_t["scales"],
        opacities=gauss_t["opacities"], colors=gauss_t["shs"],
        viewmats=viewmat[None], Ks=K[None], width=W, height=H,
        sh_degree=sh_degree, render_mode="RGB+ED", packed=False)
    rgb = colors[0, ..., :3]
    depth = colors[0, ..., 3:4]
    return rgb, alphas[0], depth


def sobel_edges(gray):
    import torch
    import torch.nn.functional as F
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=gray.dtype, device=gray.device)[None, None]
    ky = kx.transpose(-1, -2)
    g = gray[None, None]
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)[0, 0]


# --------------------------------------------------------------------------- #
def object_mask(img_bgr, white_thresh=235):
    return (img_bgr.min(axis=2) < white_thresh).astype(np.float32)


def load_gaussians_torch(ply_path, device, scale_mult=1.0):
    """Load a .ply once and return (gauss_tensors, sh_degree). Reuse across queries."""
    import torch
    import torch.nn.functional as F
    g = load_gaussians_ply(ply_path)
    gauss = {k: torch.tensor(v, dtype=torch.float32, device=device)
             for k, v in g.items() if k != "sh_degree"}
    gauss["scales"] = torch.exp(gauss["scales"]) * scale_mult
    gauss["opacities"] = torch.sigmoid(gauss["opacities"])
    gauss["quats"] = F.normalize(gauss["quats"], dim=-1)
    return gauss, g["sh_degree"]


def build_targets(qimg_bgr, q_mask_np, device):
    """Silhouette target + inside/outside distance fields (long-range gradients)."""
    import torch
    import cv2
    H, W = q_mask_np.shape
    m_u8 = (q_mask_np > 0.5).astype(np.uint8)
    diag = float(np.hypot(H, W))
    dt_out = cv2.distanceTransform(1 - m_u8, cv2.DIST_L2, 5) / diag
    dt_in = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5) / diag
    gray = cv2.cvtColor(qimg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    return {"mask": torch.tensor(q_mask_np, device=device),
            "dto": torch.tensor(dt_out, device=device),
            "dti": torch.tensor(dt_in, device=device),
            "gray": torch.tensor(gray, device=device)}


def optimise_pose(gauss, means0, quats0, sh_deg, K, H, W, tgt, RT_gl0,
                  iters=80, lr=0.01, w_sil=1.0, w_dt=5.0, w_edge=0.0,
                  tag="", verbose=True):
    """Refine one initial pose against the silhouette. Returns (final_loss, V_est_cv).

    Pose is an SE(3) delta applied to the GAUSSIANS (means+quats) with the viewmat
    fixed at the init -- equivalent to moving the camera (V_est = V0 @ exp(xi)) but
    only needs gradients through means/quats, which every gsplat version supports.
    """
    import torch
    import torch.nn.functional as F
    dev = means0.device
    V0 = torch.tensor(gl_w2c_to_cv_viewmat(RT_gl0), dtype=torch.float32, device=dev)
    xi = torch.zeros(6, device=dev, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    last = float("inf")
    for it in range(iters):
        opt.zero_grad()
        T = se3_exp(xi)
        gauss["means"] = (T[:3, :3] @ means0.T).T + T[:3, 3]
        gauss["quats"] = quat_mul(mat_to_quat(T[:3, :3])[None].expand_as(quats0), quats0)
        rgb, alpha, _ = render(gauss, V0, K, H, W, sh_deg)
        a = alpha[..., 0]

        loss, logs = 0.0, {}
        scales = (1, 2, 4)
        for s in scales:
            if s == 1:
                A, Mq, Do, Di = a, tgt["mask"], tgt["dto"], tgt["dti"]
            else:
                def pool(z, s=s):
                    return F.avg_pool2d(z[None, None], s)[0, 0]
                A, Mq, Do, Di = (pool(a), pool(tgt["mask"]),
                                 pool(tgt["dto"]), pool(tgt["dti"]))
            dice = 1 - (2 * (A * Mq).sum() + 1) / (A.sum() + Mq.sum() + 1)
            l1 = F.l1_loss(A, Mq)
            l_dt = (A * Do).mean() + ((1 - A) * Di).mean()
            loss = loss + (w_sil * (dice + l1) + w_dt * l_dt) / len(scales)
            if s == 1:
                logs = {"dice": dice.item(), "dt": l_dt.item()}
        if w_edge > 0:
            loss = loss + w_edge * F.l1_loss(
                sobel_edges(rgb.mean(-1) * tgt["mask"]),
                sobel_edges(tgt["gray"] * tgt["mask"]))
        loss.backward()
        opt.step()
        last = loss.item()
        if verbose and (it % 20 == 0 or it == iters - 1):
            print(f"  [{tag}] it{it:03d} loss={last:.4f} dice={logs['dice']:.4f} "
                  f"dt={logs['dt']:.5f}")
    with torch.no_grad():
        T = se3_exp(xi).cpu().numpy()
    return last, gl_w2c_to_cv_viewmat(RT_gl0) @ T


def refine(args):
    import torch
    import cv2

    dev = torch.device(args.device)
    gauss, sh_deg = load_gaussians_torch(args.ply, dev, args.scale_mult)
    print(f"loaded {len(gauss['means'])} gaussians, sh_degree={sh_deg}")

    qimg = cv2.imread(args.query)
    H, W = qimg.shape[:2]
    Kq = np.asarray(json.load(open(args.query_gt or os.path.splitext(args.query)[0]
                                   + ".json"))["K"], dtype=np.float64)
    K = torch.tensor(Kq, dtype=torch.float32, device=dev)

    if args.query_mask:
        q_mask_np = (cv2.imread(args.query_mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
    else:
        q_mask_np = object_mask(qimg, args.white_thresh)
    tgt = build_targets(qimg, q_mask_np, dev)
    print(f"query {W}x{H}, object mask covers {q_mask_np.mean()*100:.1f}% of pixels")

    means0, quats0 = gauss["means"].clone(), gauss["quats"].clone()

    # multi-hypothesis: refine from each init and keep the lowest final loss.
    # The silhouette loss is a valid selector here; it cannot resolve the 180 deg
    # symmetry, but those hypotheses are equivalent anyway.
    inits = [p.strip() for p in args.init_json.split(",") if p.strip()]
    results = []
    for p in inits:
        RT0 = np.asarray(json.load(open(p))["RT"], dtype=np.float64)
        fl, V = optimise_pose(gauss, means0, quats0, sh_deg, K, H, W, tgt, RT0,
                              iters=args.iters, lr=args.lr, w_sil=args.w_sil,
                              w_dt=args.w_dt, w_edge=args.w_edge,
                              tag=os.path.basename(p))
        results.append((fl, V, p))
        print(f"  -> {os.path.basename(p)}: final loss {fl:.5f}")
    results.sort(key=lambda x: x[0])
    best_loss, V_est_cv, best_init = results[0]
    if len(inits) > 1:
        print(f"\nselected hypothesis {os.path.basename(best_init)} "
              f"(loss {best_loss:.5f} of {len(inits)})")

    RT_gl_est = cv_viewmat_to_gl_w2c(V_est_cv)
    R_est, t_est = RT_gl_est[:3, :3], RT_gl_est[:3, 3]
    C_est = cam_center(R_est, t_est)

    print("\n" + "=" * 60)
    _, Rg, tg, _ = _load_gt(args.query_gt or os.path.splitext(args.query)[0] + ".json")
    Cg = cam_center(Rg, tg)
    rot = rot_geodesic_deg(R_est, Rg)
    trans = float(np.linalg.norm(C_est - Cg))
    print(f"rotation error = {rot:.3f}°   translation error = {trans:.4f}")
    print(f"=> {'PASS' if rot < 10 and trans < 0.1 else 'FAIL'} (target <10°/<0.1)")

    if args.save_render:
        import cv2
        with torch.no_grad():
            gauss["means"] = means0
            gauss["quats"] = quats0
            V_est = torch.tensor(V_est_cv, dtype=torch.float32, device=dev)
            rgb, alpha, _ = render(gauss, V_est, K, H, W, sh_deg)
        out = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        cv2.imwrite(args.save_render, out)
        print(f"saved aligned reference render at estimated pose -> {args.save_render}")


def verify(args):
    """Render the .ply at a known reference pose; compare silhouette IoU to the
    real reference image. High IoU => convention + world frame are consistent."""
    import torch
    import cv2
    dev = torch.device(args.device)
    g = load_gaussians_ply(args.ply)
    gauss = {k: torch.tensor(v, dtype=torch.float32, device=dev)
             for k, v in g.items() if k != "sh_degree"}
    gauss["scales"] = torch.exp(gauss["scales"]) * args.scale_mult
    gauss["opacities"] = torch.sigmoid(gauss["opacities"])
    gauss["quats"] = torch.nn.functional.normalize(gauss["quats"], dim=-1)

    d = json.load(open(args.verify_ref + ".json"))
    RT_gl = np.asarray(d["RT"], np.float64)
    K = np.asarray(d["K"], np.float64)
    ref = cv2.imread(args.verify_ref + ".png")
    H, W = ref.shape[:2]
    V = torch.tensor(gl_w2c_to_cv_viewmat(RT_gl), dtype=torch.float32, device=dev)
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)
    with torch.no_grad():
        rgb, alpha, _ = render(gauss, V, Kt, H, W, g["sh_degree"])
    a = (alpha[..., 0].cpu().numpy() > 0.5)
    m = object_mask(ref) > 0.5
    iou = (a & m).sum() / ((a | m).sum() + 1e-6)
    cv2.imwrite("verify_render.png",
                (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
    print(f"silhouette IoU(render vs real ref) = {iou:.3f}  "
          f"({'OK convention/frame' if iou > 0.8 else 'MISMATCH -> frame/convention wrong'})")
    print("wrote verify_render.png for visual check")


def _load_gt(js):
    d = json.load(open(js))
    RT = np.asarray(d["RT"], np.float64)
    return np.asarray(d["K"], np.float64), RT[:3, :3], RT[:3, 3], d


# --------------------------------------------------------------------------- #
def selftest():
    """numpy-only: verify pose-composition + OpenGL<->OpenCV round-trip."""
    rng = np.random.RandomState(0)

    def rand_rot():
        Q, _ = np.linalg.qr(rng.randn(3, 3))
        return Q if np.linalg.det(Q) > 0 else Q @ np.diag([1, 1, -1.0])

    RT_gl0 = np.eye(4); RT_gl0[:3, :3] = rand_rot(); RT_gl0[:3, 3] = [0.1, 0.2, -3.5]
    V0 = gl_w2c_to_cv_viewmat(RT_gl0)
    # a known delta T; effective cv viewmat = V0 @ T; convert back to GL
    T = np.eye(4); T[:3, :3] = rand_rot(); T[:3, 3] = rng.randn(3) * 0.1
    V_est = V0 @ T
    RT_gl_est = cv_viewmat_to_gl_w2c(V_est)
    # round-trip: applying S4 twice is identity
    assert np.allclose(cv_viewmat_to_gl_w2c(gl_w2c_to_cv_viewmat(RT_gl0)), RT_gl0)
    # with T=I, estimate must equal init
    assert np.allclose(cv_viewmat_to_gl_w2c(V0 @ np.eye(4)), RT_gl0, atol=1e-9)
    print("[selftest] PASS — OpenGL<->OpenCV round-trip + pose composition correct")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    base = "/home/xiangjianen/projects/Depth-Anything-3/Anomaly/data"
    ap.add_argument("--ply", help="standard 3DGS .ply")
    ap.add_argument("--query", default=f"{base}/render_8340_4.2_69_40_Broken_anomaly.png")
    ap.add_argument("--query-gt", default=None)
    ap.add_argument("--init-json", help="init pose json(s), comma-separated for "
                                        "multi-hypothesis refinement (best final loss wins)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--w-sil", type=float, default=1.0)
    ap.add_argument("--w-dt", type=float, default=5.0,
                    help="weight of the distance-transform boundary term (gives "
                         "long-range gradients; the main driver of convergence)")
    ap.add_argument("--w-edge", type=float, default=0.0,
                    help="edge term only helps for a SHADED model; keep 0 for a "
                         "geometry-only (white) .ply where render has no internal edges")
    ap.add_argument("--scale-mult", type=float, default=1.0,
                    help="multiply gaussian scales (this .ply has sub-pixel splats; "
                         "raise to 2-3 if the rendered silhouette looks sparse/holey)")
    ap.add_argument("--query-mask", default=None,
                    help="explicit query object mask png (e.g. annotations/"
                         "render_*_mask.png); default = white-background threshold")
    ap.add_argument("--white-thresh", type=int, default=235)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-ref", help="reference stem (no ext) for --verify")
    ap.add_argument("--save-render", default=None,
                    help="save the clean render at the estimated pose (aligned "
                         "reference for anomaly comparison)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
    elif args.verify:
        verify(args)
    else:
        assert args.ply and args.init_json, "need --ply and --init-json"
        refine(args)


if __name__ == "__main__":
    main()
