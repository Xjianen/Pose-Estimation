"""Silhouette utilities: foreground extraction, scale/position normalisation,
distance fields.

The renders in this dataset put the object on a pure-white background, so the
foreground mask is a simple colour threshold. Templates and queries are put
through the SAME normalisation (square bbox crop) so that matching is invariant
to object scale and image position — which is what lets the camera radius be
recovered analytically instead of being a codebook dimension.
"""

import cv2
import numpy as np


def object_mask(img_bgr, white_thresh=235):
    """Foreground mask from a white-background render. Returns float32 {0,1}."""
    return (img_bgr.min(axis=2) < white_thresh).astype(np.float32)


def read_object_mask(png_path, white_thresh=235):
    """Returns (mask uint8 {0,1}, image width)."""
    img = cv2.imread(png_path)
    if img is None:
        raise FileNotFoundError(png_path)
    return (img.min(axis=2) < white_thresh).astype(np.uint8), img.shape[1]


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def normalise_mask(mask, out, pad=0.12):
    """Square bbox crop (aspect preserved, zero padded) resized to out x out.

    Returns (mask uint8 {0,1} of shape (out,out), pre-crop bbox pixel size).
    The bbox size is what the analytic radius estimate needs.
    """
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


def distance_to_mask(mask_bool):
    """Euclidean distance from every pixel to the nearest True pixel."""
    return cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 3)


def silhouette_targets(mask_np, device, gray_np=None):
    """Torch targets for the refinement loss.

    `dto` is the distance from each pixel to the object (0 inside), `dti` the
    distance from each pixel to the background (0 outside). Weighting the
    rendered alpha by these gives a loss with gradients over the WHOLE image,
    unlike Dice/IoU which is flat when the silhouettes do not overlap — that is
    what lets refinement close a large initial viewpoint gap.

    Rendering/comparing below the query resolution was tried and rejected: it
    barely helped speed (the per-gaussian projection dominates, not the per-pixel
    blending) while translation error more than doubled.
    """
    import torch
    H, W = mask_np.shape
    m_u8 = (mask_np > 0.5).astype(np.uint8)
    diag = float(np.hypot(H, W))
    dto = cv2.distanceTransform(1 - m_u8, cv2.DIST_L2, 5) / diag
    dti = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5) / diag
    out = {"mask": torch.tensor(mask_np.astype(np.float32), device=device),
           "dto": torch.tensor(dto, device=device),
           "dti": torch.tensor(dti, device=device)}
    if gray_np is not None:
        out["gray"] = torch.tensor(gray_np.astype(np.float32), device=device)
    return out


def sobel_magnitude(gray):
    """Edge magnitude of a torch 2-D tensor (used only for shaded models)."""
    import torch
    import torch.nn.functional as F
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=gray.dtype, device=gray.device)[None, None]
    ky = kx.transpose(-1, -2)
    g = gray[None, None]
    return torch.sqrt(F.conv2d(g, kx, padding=1) ** 2
                      + F.conv2d(g, ky, padding=1) ** 2 + 1e-6)[0, 0]


def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0
