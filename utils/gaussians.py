"""3D Gaussian model: standard 3DGS .ply loading and gsplat rendering.

Notes on this dataset's .ply
---------------------------
It is geometry-only: every gaussian is pure white (base colour [1,1,1], std 0)
and higher SH coefficients are all zero — it was sampled from the mesh rather
than trained on images. So the only signal a render provides that is comparable
to a textured query is the **silhouette (alpha)**; the RGB carries no internal
structure. Depth is still meaningful and is what the downstream anomaly stage
consumes.

The splats are also sub-pixel (median max-axis ~0.0038 world units, i.e. ~0.7 px
at 512), but there are 414k of them so coverage is solid. `scale_mult` is there
in case a coarser render is wanted.
"""

import numpy as np


def load_gaussians_ply(path):
    """Parse a standard 3DGS .ply (binary_little_endian, all float32 properties)."""
    with open(path, "rb") as f:
        hdr = b""
        while True:
            line = f.readline()
            hdr += line
            if line.strip() == b"end_header":
                break
        txt = hdr.decode("latin1").splitlines()
        n = next(int(l.split()[-1]) for l in txt if l.startswith("element vertex"))
        props = [l.split()[-1] for l in txt if l.startswith("property")]
        assert "binary_little_endian" in "".join(txt), "expected binary_little_endian ply"
        data = np.frombuffer(f.read(n * len(props) * 4),
                             dtype=np.float32).reshape(n, len(props))
    col = {name: i for i, name in enumerate(props)}

    def g(*names):
        return np.stack([data[:, col[nm]] for nm in names], axis=1)

    f_dc = g("f_dc_0", "f_dc_1", "f_dc_2")
    rest = sorted([p for p in props if p.startswith("f_rest_")],
                  key=lambda s: int(s.split("_")[-1]))
    if rest:
        fr = np.stack([data[:, col[p]] for p in rest], 1)          # (N, 3*k)
        k = fr.shape[1] // 3
        fr = fr.reshape(-1, 3, k).transpose(0, 2, 1)               # (N, k, 3)
        shs = np.concatenate([f_dc[:, None, :], fr], axis=1)
    else:
        shs = f_dc[:, None, :]
    return {"means": g("x", "y", "z"),
            "scales": g("scale_0", "scale_1", "scale_2"),
            "quats": g("rot_0", "rot_1", "rot_2", "rot_3"),
            "opacities": data[:, col["opacity"]],
            "shs": shs,
            "sh_degree": int(round(np.sqrt(shs.shape[1]))) - 1}


def load_gaussians_torch(ply_path, device, scale_mult=1.0):
    """Load once, activate the parameters, move to GPU. Returns (gauss, sh_degree).

    Parsing 414k gaussians is the slowest single step, so callers should do this
    once and reuse the result across all queries.
    """
    import torch
    import torch.nn.functional as F
    g = load_gaussians_ply(ply_path)
    gauss = {k: torch.tensor(v, dtype=torch.float32, device=device)
             for k, v in g.items() if k != "sh_degree"}
    gauss["scales"] = torch.exp(gauss["scales"]) * scale_mult
    gauss["opacities"] = torch.sigmoid(gauss["opacities"])
    gauss["quats"] = F.normalize(gauss["quats"], dim=-1)
    return gauss, g["sh_degree"]


def render(gauss, viewmat_cv, K, width, height, sh_degree):
    """gsplat rasterisation. Returns (rgb[H,W,3], alpha[H,W,1], depth[H,W,1]).

    `viewmat_cv` and `K` are torch tensors; viewmat is OpenCV world-to-camera.
    render_mode="RGB+ED" puts expected depth in the 4th colour channel.
    """
    from gsplat import rasterization
    colors, alphas, _ = rasterization(
        means=gauss["means"], quats=gauss["quats"], scales=gauss["scales"],
        opacities=gauss["opacities"], colors=gauss["shs"],
        viewmats=viewmat_cv[None], Ks=K[None], width=width, height=height,
        sh_degree=sh_degree, render_mode="RGB+ED", packed=False)
    return colors[0, ..., :3], alphas[0], colors[0, ..., 3:4]


def render_batch(gauss, viewmats_cv, K, width, height, sh_degree):
    """Rasterise SEVERAL cameras in ONE gsplat call.

    Measured on this data, ~85% of a refinement iteration is fixed overhead: cutting
    the render resolution 4x or the gaussian count 2.8x each changed the per-iteration
    time by barely 10%. Batching the hypotheses therefore pays the overhead once
    instead of once per hypothesis, which is the actual win.

    `viewmats_cv` is [C,4,4] (OpenCV world-to-camera). Each camera's image depends
    only on its own viewmat, so gradients stay independent and this is equivalent to
    optimising the hypotheses separately.
    Returns (rgb[C,H,W,3], alpha[C,H,W,1], depth[C,H,W,1]).
    """
    from gsplat import rasterization
    C = viewmats_cv.shape[0]
    Ks = K[None].expand(C, 3, 3).contiguous() if K.dim() == 2 else K
    colors, alphas, _ = rasterization(
        means=gauss["means"], quats=gauss["quats"], scales=gauss["scales"],
        opacities=gauss["opacities"], colors=gauss["shs"],
        viewmats=viewmats_cv, Ks=Ks, width=width, height=height,
        sh_degree=sh_degree, render_mode="RGB+ED", packed=False)
    return colors[..., :3], alphas, colors[..., 3:4]


def render_at_pose(gauss, RT_gl, K, width, height, sh_degree):
    """Render at an OpenGL world-to-camera pose. Returns numpy (rgb, alpha, depth)."""
    import torch
    from utils.geometry import gl_w2c_to_cv_viewmat
    V = torch.tensor(gl_w2c_to_cv_viewmat(RT_gl), dtype=torch.float32,
                     device=gauss["means"].device)
    with torch.no_grad():
        rgb, alpha, depth = render(gauss, V, K, width, height, sh_degree)
    return (rgb.clamp(0, 1).cpu().numpy(), alpha[..., 0].cpu().numpy(),
            depth[..., 0].cpu().numpy())


_VIEWMAT_GRAD = {}


def supports_viewmat_grad(device, cache=True):
    """Does the installed gsplat propagate gradients into `viewmats`?

    If it does, the pose can be optimised directly on the camera (16 numbers of
    gradient, and gsplat never has to produce per-gaussian gradients). If not, the
    fallback is to transform the gaussians instead, which is mathematically
    equivalent but pays for a 4xN transform plus a per-gaussian backward.
    """
    key = str(device)
    if cache and key in _VIEWMAT_GRAD:
        return _VIEWMAT_GRAD[key]
    import torch
    ok = False
    try:
        n, res = 64, 32
        g = {"means": torch.randn(n, 3, device=device) * 0.2,
             "quats": torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1),
             "scales": torch.full((n, 3), 0.05, device=device),
             "opacities": torch.full((n,), 0.9, device=device),
             "shs": torch.ones(n, 1, 3, device=device)}
        V = torch.eye(4, device=device)
        V[2, 3] = 2.5
        V = V.clone().requires_grad_(True)
        K = torch.tensor([[50.0, 0, res / 2], [0, 50.0, res / 2], [0, 0, 1]],
                         device=device)
        _, alpha, _ = render(g, V, K, res, res, 0)
        alpha.sum().backward()
        ok = V.grad is not None and torch.isfinite(V.grad).all() and V.grad.abs().sum() > 0
    except Exception:
        ok = False
    if cache:
        _VIEWMAT_GRAD[key] = bool(ok)
    return bool(ok)
