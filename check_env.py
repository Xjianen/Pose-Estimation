#!/usr/bin/env python3
"""Environment check for the pose pipeline.

Run this BEFORE touching data. It verifies each dependency and — importantly —
calls gsplat's `rasterization` with the exact same signature `utils/gaussians.py`
uses, so a version/API mismatch shows up here instead of mid-experiment. It also
reports which pose parameterisation the installed gsplat allows (see
`utils/pose_refine.py`): 'camera' if gradients reach `viewmats`, else 'object'.

    python check_env.py
"""

import argparse
import sys

results = []


def check(name, fn):
    try:
        info = fn()
        results.append((name, True, info))
        print(f"[ OK ] {name}: {info}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def c_python():
    return f"python {sys.version.split()[0]}"


def c_numpy():
    import numpy
    return f"numpy {numpy.__version__}"


def c_cv2():
    import cv2
    return f"opencv {cv2.__version__}"


def c_torch():
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA"
    return (f"torch {torch.__version__}, cuda_available="
            f"{torch.cuda.is_available()}, built_cuda={torch.version.cuda}, dev={dev}")


def c_gsplat_import():
    import gsplat
    return f"gsplat {getattr(gsplat, '__version__', 'unknown')}"


def c_gsplat_render():
    """Render a few random gaussians with the SAME call signature as
    utils.gaussians.render() -> catches API drift across gsplat versions."""
    import torch
    from gsplat import rasterization
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, H, W, sh_degree = 500, 64, 64, 3
    g = torch.Generator(device="cpu").manual_seed(0)
    means = torch.randn(N, 3, generator=g).to(dev) * 0.3
    quats = torch.nn.functional.normalize(torch.randn(N, 4, generator=g).to(dev), dim=-1)
    scales = torch.full((N, 3), 0.02, device=dev)
    opacities = torch.full((N,), 0.8, device=dev)
    shs = torch.zeros(N, (sh_degree + 1) ** 2, 3, device=dev)
    shs[:, 0, :] = 1.0
    viewmat = torch.eye(4, device=dev)
    viewmat[2, 3] = 3.0                                    # camera 3 units back (+z look)
    K = torch.tensor([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]], device=dev)

    colors, alphas, meta = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=shs,
        viewmats=viewmat[None], Ks=K[None], width=W, height=H,
        sh_degree=sh_degree, render_mode="RGB+ED", packed=False)
    assert colors.shape[0] == 1 and colors.shape[1:3] == (H, W), colors.shape
    assert alphas.shape[1:3] == (H, W), alphas.shape
    nch = colors.shape[-1]
    cov = float((alphas > 0.01).float().mean())
    return (f"render OK: colors{tuple(colors.shape)} (ch={nch}, need>=4 for RGB+ED), "
            f"alphas{tuple(alphas.shape)}, alpha_coverage={cov:.2%}")


def c_gsplat_grad():
    """Confirm gradients flow to gaussian means (the pose-optimization path)."""
    import torch
    from gsplat import rasterization
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, H, W = 200, 32, 32
    means = (torch.randn(N, 3, device=dev) * 0.2).requires_grad_(True)
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
    scales = torch.full((N, 3), 0.03, device=dev)
    opacities = torch.full((N,), 0.9, device=dev)
    colors = torch.ones(N, 3, device=dev)
    viewmat = torch.eye(4, device=dev); viewmat[2, 3] = 2.5
    K = torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]], device=dev)
    out, alphas, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmat[None], Ks=K[None], width=W, height=H, packed=False)
    alphas.sum().backward()
    assert means.grad is not None and torch.isfinite(means.grad).all(), "no grad on means"
    return f"grad OK: |d alpha/d means| mean={means.grad.abs().mean().item():.3e}"


def c_gsplat_viewmat_grad():
    """Do gradients reach `viewmats`? Decides which pose parameterisation is used.

    If yes, the pose delta can ride on the camera (16 numbers of gradient, the
    gaussians are never touched). If no, the fallback transforms the gaussians,
    which is equivalent but pays a 4xN transform plus a per-gaussian backward.
    """
    import torch
    from gsplat import rasterization
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, res = 200, 32
    means = torch.randn(n, 3, device=dev) * 0.2
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)
    scales = torch.full((n, 3), 0.03, device=dev)
    opacities = torch.full((n,), 0.9, device=dev)
    colors = torch.ones(n, 3, device=dev)
    V = torch.eye(4, device=dev)
    V[2, 3] = 2.5
    V = V.clone().requires_grad_(True)
    K = torch.tensor([[60.0, 0, res / 2], [0, 60.0, res / 2], [0, 0, 1]], device=dev)
    _, alphas, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=V[None], Ks=K[None], width=res, height=res, packed=False)
    alphas.sum().backward()
    if V.grad is None or not torch.isfinite(V.grad).all() or V.grad.abs().sum() == 0:
        return "NOT available -> pipeline will use param='object' (still correct)"
    return (f"available (|d alpha/d viewmat| mean={V.grad.abs().mean().item():.3e}) "
            f"-> pipeline can use the faster param='camera'")


def c_dino():
    import torch
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev).eval()
    x = torch.zeros(1, 3, 224, 224, device=dev)
    with torch.no_grad():
        o = m.forward_features(x)
    return (f"dinov2_vitl14 OK: cls{tuple(o['x_norm_clstoken'].shape)} "
            f"patch{tuple(o['x_norm_patchtokens'].shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", action="store_true",
                    help="optional: load DINOv2 (~1.1GB). NOT used by the current "
                         "pipeline -- silhouette matching replaced it.")
    args = ap.parse_args()

    print("=" * 70)
    check("python", c_python)
    check("numpy", c_numpy)
    check("opencv", c_cv2)
    check("torch", c_torch)
    check("gsplat import", c_gsplat_import)
    check("gsplat rasterization (same call as utils/gaussians)", c_gsplat_render)
    check("gsplat gradients -> means (object param)", c_gsplat_grad)
    check("gsplat gradients -> viewmats (camera param)", c_gsplat_viewmat_grad)
    if args.dino:
        check("DINOv2 via torch.hub", c_dino)
    else:
        print("[skip] DINOv2 (rerun with --dino to test; needs internet once)")

    print("=" * 70)
    bad = [n for n, ok, _ in results if not ok]
    if bad:
        print(f"FAILED: {bad}\nFix these before running the pipeline.")
        sys.exit(1)
    print("ALL CHECKS PASSED — environment is ready.")


if __name__ == "__main__":
    main()
