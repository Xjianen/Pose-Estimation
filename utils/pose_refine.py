"""Pose refinement: differentiable silhouette alignment against the 3DGS model.

Parameterisation ("camera" vs "object")
---------------------------------------
Both optimise the same 6-DoF SE(3) delta `xi` and produce the SAME effective
world-to-camera matrix `V = V0 @ exp(xi)`, so they are mathematically equivalent
and should agree numerically. They differ only in where the gradient flows:

* **camera** (preferred): pass `V0 @ exp(xi)` straight to gsplat as the viewmat.
  The gaussians are untouched, so there is no per-iteration 4xN transform and
  gsplat never has to produce per-gaussian gradients — only 16 numbers for the
  viewmat. Requires the installed gsplat to backprop into `viewmats`.
* **object**: apply `exp(xi)` to the gaussian means and quats with the viewmat
  pinned at `V0`. Works on every gsplat version (gradients travel through
  means/quats, which is always supported) but costs a transform of every gaussian
  each step plus a per-gaussian backward.

`param="auto"` probes the installed gsplat once and picks camera when possible.

Loss
----
Multi-scale (1, 1/2, 1/4) silhouette agreement:
  * Dice + L1 on the rendered alpha vs the query object mask — overlap term,
  * distance-transform term  mean(alpha * dist_to_object) + mean((1-alpha) * dist_to_bg)
    — this is the important one: it has non-zero gradient over the whole image, so
    refinement can close a large initial viewpoint gap (measured: it closes a 17°
    elevation offset), whereas Dice/IoU alone is flat when silhouettes do not overlap.
  * an optional edge term, only useful if the model has shading (this dataset's
    .ply is pure white, so it is off by default).

Because a symmetric object's pose is only determined up to its symmetry, the
silhouette loss cannot separate a pose from its symmetric counterparts; the
multi-hypothesis search treats those as equivalent solutions.
"""

import numpy as np

from utils.geometry import (cv_viewmat_to_gl_w2c, gl_w2c_to_cv_viewmat,
                            mat_to_quat, quat_mul, se3_exp)
from utils.gaussians import render, render_batch, supports_viewmat_grad
from utils.masks import sobel_magnitude


def resolve_param(param, device):
    """Turn 'auto' into 'camera' or 'object' based on what gsplat supports."""
    if param != "auto":
        return param
    return "camera" if supports_viewmat_grad(device) else "object"


def silhouette_loss(alpha, targets, w_sil=1.0, w_dt=5.0, scales=(1, 2, 4)):
    """Multi-scale Dice + L1 + distance-transform silhouette loss."""
    import torch.nn.functional as F
    total, logs = 0.0, {}
    for s in scales:
        if s == 1:
            A, M, Do, Di = (alpha, targets["mask"], targets["dto"], targets["dti"])
        else:
            def pool(z, s=s):
                return F.avg_pool2d(z[None, None], s)[0, 0]
            A, M = pool(alpha), pool(targets["mask"])
            Do, Di = pool(targets["dto"]), pool(targets["dti"])
        dice = 1 - (2 * (A * M).sum() + 1) / (A.sum() + M.sum() + 1)
        l1 = F.l1_loss(A, M)
        l_dt = (A * Do).mean() + ((1 - A) * Di).mean()
        total = total + (w_sil * (dice + l1) + w_dt * l_dt) / len(scales)
        if s == 1:
            logs = {"dice": dice.item(), "dt": l_dt.item()}
    return total, logs


def refine_pose(gauss, means0, quats0, sh_degree, K, width, height, targets,
                RT_init, iters=80, lr=0.01, w_sil=1.0, w_dt=5.0, w_edge=0.0,
                scales=(1, 2, 4), param="auto", patience=0, min_delta=1e-5,
                tag="", verbose=False):
    """Refine one initial pose. Returns (final_loss, RT_gl_est, iters_used).

    `means0`/`quats0` are only used by the "object" parameterisation; pass the
    live tensors (or None) when using "camera".

    `patience > 0` enables early stopping: stop once `iters` have passed without
    the loss improving by more than `min_delta`. The loss plateaus well before the
    iteration cap on this data, so this is close to free.
    """
    import torch
    import torch.nn.functional as F

    dev = gauss["means"].device
    mode = resolve_param(param, dev)
    V0 = torch.tensor(gl_w2c_to_cv_viewmat(RT_init), dtype=torch.float32, device=dev)
    xi = torch.zeros(6, device=dev, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    last, best, stale, used = float("inf"), float("inf"), 0, 0

    for it in range(iters):
        opt.zero_grad()
        T = se3_exp(xi)
        if mode == "camera":
            # gaussians untouched; the delta rides on the viewmat
            rgb, alpha, _ = render(gauss, V0 @ T, K, width, height, sh_degree)
        else:
            gauss["means"] = (T[:3, :3] @ means0.T).T + T[:3, 3]
            gauss["quats"] = quat_mul(mat_to_quat(T[:3, :3])[None].expand_as(quats0),
                                      quats0)
            rgb, alpha, _ = render(gauss, V0, K, width, height, sh_degree)

        loss, logs = silhouette_loss(alpha[..., 0], targets, w_sil, w_dt, scales)
        if w_edge > 0 and "gray" in targets:
            loss = loss + w_edge * F.l1_loss(
                sobel_magnitude(rgb.mean(-1) * targets["mask"]),
                sobel_magnitude(targets["gray"] * targets["mask"]))

        loss.backward()
        opt.step()
        last = loss.item()
        used = it + 1
        if verbose and (it % 20 == 0 or it == iters - 1):
            print(f"    [{tag}/{mode}] it{it:03d} loss={last:.4f} "
                  f"dice={logs['dice']:.4f} dt={logs['dt']:.5f}")
        if patience > 0:
            if last < best - min_delta:
                best, stale = last, 0
            else:
                stale += 1
                if stale >= patience:
                    if verbose:
                        print(f"    [{tag}/{mode}] early stop at it{it} "
                              f"(no gain > {min_delta} for {patience} iters)")
                    break

    with torch.no_grad():
        T = se3_exp(xi).cpu().numpy()
    return last, cv_viewmat_to_gl_w2c(gl_w2c_to_cv_viewmat(RT_init) @ T), used


def refine_multi_hypothesis(gauss, means0, quats0, sh_degree, K, width, height,
                            targets, hypotheses, **kw):
    """Refine every hypothesis at full cost and keep the lowest final loss.

    Measured on 16 queries: the coarse top-1 was NOT the best hypothesis in 50%
    of cases, so this selection is doing real work rather than being redundant.
    Returns (best_loss, best_RT_gl, best_index, all_losses, total_iters).
    """
    results, total = [], 0
    for k, h in enumerate(hypotheses):
        RT0 = h["RT"] if isinstance(h, dict) else h
        loss, RT, used = refine_pose(gauss, means0, quats0, sh_degree, K, width,
                                     height, targets, RT0, tag=f"h{k}", **kw)
        results.append((loss, RT, k))
        total += used
    results.sort(key=lambda r: r[0])
    best = results[0]
    return (best[0], best[1], best[2],
            [r[0] for r in sorted(results, key=lambda r: r[2])], total)


def refine_multi_hypothesis_batched(gauss, sh_degree, K, width, height, targets,
                                    hypotheses, iters=80, lr=0.01, w_sil=1.0,
                                    w_dt=5.0, w_edge=0.0, scales=(1, 2, 4),
                                    patience=0, min_delta=1e-4, check_every=5,
                                    verbose=False):
    """Refine all hypotheses simultaneously in ONE rasterisation call per step.

    Why: ~85% of an iteration is fixed overhead (measured — cutting resolution 4x
    or gaussian count 2.8x each changed per-iteration time by ~10%). Rendering the
    C hypotheses together pays that overhead once instead of C times, so the wall
    clock approaches that of a single hypothesis.

    This is mathematically the same optimisation as running them separately: image
    i depends only on viewmat i, so d loss_i / d xi_j = 0 for i != j, and Adam keeps
    per-element state. It requires the camera parameterisation (each hypothesis needs
    its own viewmat), hence gsplat viewmat gradients; callers should fall back to
    `refine_multi_hypothesis` when those are unavailable.

    Early stopping applies to the SUM: the loop ends once no hypothesis has improved
    for `patience` checks, so `iters_used` is the max over hypotheses rather than the
    sum — which is exactly where the saving comes from.
    Returns (best_loss, best_RT_gl, best_index, all_losses, iters_used).
    """
    import torch

    dev = gauss["means"].device
    RT0s = [h["RT"] if isinstance(h, dict) else h for h in hypotheses]
    V0 = torch.tensor(np.stack([gl_w2c_to_cv_viewmat(r) for r in RT0s]),
                      dtype=torch.float32, device=dev)                  # [C,4,4]
    C = V0.shape[0]
    xi = torch.zeros(C, 6, device=dev, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    best_sum, stale, used = float("inf"), 0, 0
    per_hyp = [float("inf")] * C

    for it in range(iters):
        opt.zero_grad()
        # scalar se3_exp reused per hypothesis (C is tiny; keeps the verified path)
        T = torch.stack([se3_exp(xi[i]) for i in range(C)])             # [C,4,4]
        _, alpha, _ = render_batch(gauss, torch.bmm(V0, T), K, width, height, sh_degree)
        losses = [silhouette_loss(alpha[i, ..., 0], targets, w_sil, w_dt, scales)[0]
                  for i in range(C)]
        total = sum(losses)
        total.backward()
        opt.step()
        used = it + 1

        if patience > 0 and (it + 1) % check_every == 0:
            cur = float(total.detach())          # one sync per check, not per step
            if cur < best_sum - min_delta * C:
                best_sum, stale = cur, 0
            else:
                stale += 1
                if stale >= max(patience // check_every, 1):
                    if verbose:
                        print(f"    [batched] early stop at it{it}")
                    break
        if verbose and it % 20 == 0:
            print(f"    [batched] it{it:03d} sum_loss={float(total.detach()):.4f}")

    with torch.no_grad():
        per_hyp = [float(l.detach()) for l in losses]
        Tn = torch.stack([se3_exp(xi[i]) for i in range(C)]).cpu().numpy()
    RTs = [cv_viewmat_to_gl_w2c(gl_w2c_to_cv_viewmat(RT0s[i]) @ Tn[i]) for i in range(C)]
    k = int(np.argmin(per_hyp))
    return per_hyp[k], RTs[k], k, per_hyp, used
