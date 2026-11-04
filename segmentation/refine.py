"""Render-buffer-aware mask refinement (no model, no training).

Why this module exists
----------------------
The shipped ``mask/*.png`` is a white-background threshold of the query RGB
(or, equivalently on this dataset, of the constant-background depth/normal
buffer). It has two error sources, and they need completely different fixes:

1. **Structural FP**: the floor plane / soft shadow seen through the chair's
   openings is darker than the white background, so it is fused into the
   silhouette. On a 400-query sample this is ~1.4-2.5k pixels per image and
   it is *far* from the GT boundary.
2. **Boundary quantisation**: the shipped mask and the official
   ``seg_new_*`` label are two different render passes of the same view. The
   query silhouette sits ~1% inside the label silhouette (measured bbox ratio
   old/GT = 0.987/0.991 on 150 queries). Almost all remaining FN and most FP
   therefore live in a 1-2 px ring on the boundary. A stronger segmentation
   network cannot fix this: it is a rasterisation/FOV mismatch, not a semantic
   mistake. A single global affine boundary calibration can.

This module implements both fixes deterministically:

    old mask ──> remove_floor_plane()   (exterior-connected background-like
                                         pixels below the object centroid)
             ──> calibrate_boundary()  (isotropic scale about the image centre)
             ──> optional colour-conditioned 1 px dilation

Measured against official GT (see segmentation/README.md section 9):
    1200-query split (600 calib / 600 held-out), constant fitted on calib:
        shipped mask 0.912 -> floor removal 0.925 -> calibrated 0.945
    2000-query unbiased random sample (seed 42), fixed constant:
        shipped mask 0.900 -> floor removal 0.915 -> calibrated 0.934
        mIoU@trim2 0.955 (the boundary-tolerant metric is above 0.95;
        strict raw mIoU is NOT). Sample-to-sample spread is about +/-0.005.

The floor removal is the same "keep the thin structures" idea as the SAM
pipeline, but it starts from the shipped mask instead of a learned mask, so it
never deletes thin chair legs / frames. That is why it wins on raw mIoU even
though SlimSAM / SAM2 look sharper to the eye.

No weights are downloaded, no training is performed. The single global scale
is a *documented calibration constant* fitted on a disjoint development split
(see calibrate_refine.py); set ``scale=1.0`` for a strictly uncalibrated mask.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    from core import shadow_like
except ImportError:  # package import
    from .core import shadow_like

DEFAULT_SCALE = 1.010        # fitted on a disjoint dev split, see calibrate_refine.py
DEFAULT_ORIGIN = "centre"    # the mismatch is camera-centred (FOV), not object-centred


def remove_floor_plane(rgb, old_mask, guard_above_centroid=8,
                       max_fraction=0.35, close_k=3, dilate_k=3):
    """Strip the floor/shadow band fused into ``old_mask``.

    ``bg_like`` = white background + smooth achromatic shadow. Only the
    connected components that touch the image border are considered
    "exterior"; ``old & exterior`` is therefore the floor plane visible
    through the object's openings, never an enclosed light-grey chair seat.

    Two guards protect real object parts:
      * only pixels below (centroid - ``guard_above_centroid``) are removed;
      * if the candidate would eat more than ``max_fraction`` of the mask,
        nothing is removed.

    Returns ``(clean_mask_uint8, floor_pixels_uint8)``.
    """
    rgb = np.asarray(rgb)
    old = np.asarray(old_mask) > 0
    h, w = old.shape

    sh = shadow_like(rgb).astype(bool)
    white = rgb.min(2) > 200
    bg_like = (sh | white).astype(np.uint8)
    if close_k:
        bg_like = cv2.morphologyEx(bg_like, cv2.MORPH_CLOSE,
                                   np.ones((close_k, close_k), np.uint8))
    _, labels = cv2.connectedComponents(bg_like, 8)
    border = set(np.unique(np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])))
    border.discard(0)
    exterior = (np.isin(labels, list(border)) if border
                else np.zeros((h, w), bool))

    ys, _ = np.nonzero(old)
    cy = int(ys.mean()) if len(ys) else h // 2
    floor = old & exterior
    keep = np.zeros_like(floor)
    keep[max(0, cy - guard_above_centroid):, :] = \
        floor[max(0, cy - guard_above_centroid):, :]
    if keep.sum() > max_fraction * old.sum():
        keep = np.zeros_like(floor)
    if dilate_k:
        keep = cv2.dilate(keep.astype(np.uint8),
                          np.ones((dilate_k, dilate_k), np.uint8)) > 0
    clean = old & ~keep
    if clean.sum() < 0.5 * old.sum():
        clean = old
        keep = np.zeros_like(floor)
    return clean.astype(np.uint8), keep.astype(np.uint8)


def calibrate_boundary(mask, scale=DEFAULT_SCALE, origin=DEFAULT_ORIGIN,
                       dx=0.0, dy=0.0):
    """Global affine compensation of the query/label rasterisation mismatch.

    ``origin='centre'`` scales about the image centre (a FOV difference scales
    radially about the principal point); ``origin='centroid'`` scales about the
    object centroid. ``scale=1.0`` is a no-op (and returns a copy).
    """
    m = np.asarray(mask) > 0
    h, w = m.shape
    if scale == 1.0 and dx == 0 and dy == 0:
        return m.astype(np.uint8)
    if origin == "centroid":
        ys, xs = np.nonzero(m)
        cx, cy = (float(xs.mean()), float(ys.mean())) if len(xs) else (w / 2, h / 2)
    else:
        cx, cy = w / 2.0, h / 2.0
    M = np.array([[scale, 0, cx * (1 - scale) + dx],
                  [0, scale, cy * (1 - scale) + dy]], np.float32)
    return cv2.warpAffine(m.astype(np.uint8), M, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(np.uint8)


def condition_dilate(mask, rgb, threshold=0):
    """Grow 1 px, but only into pixels dark enough to be object, not floor.

    ``threshold<=0`` disables it. This is an optional extra boundary correction;
    the default pipeline uses only the global affine calibration.
    """
    if threshold <= 0:
        return np.asarray(mask, np.uint8)
    m = np.asarray(mask) > 0
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    d = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return (m | (d & (gray < threshold))).astype(np.uint8)


def refine_mask(rgb, old_mask, scale=DEFAULT_SCALE, origin=DEFAULT_ORIGIN,
                dx=0.0, dy=0.0, cond_dilate=0, prior_threshold=200):
    """Full deterministic pipeline.

    ``prior_threshold>200`` slightly loosens the colour prior before floor
    removal (``old | min(rgb)<T``) and recovers very light object parts that
    the shipped threshold missed. T=205/210 adds ~0.001 mIoU; the default
    keeps the shipped mask untouched.

    Returns dict with ``mask`` (uint8 0/255), ``floor`` (removed pixels),
    ``clean`` (mask after floor removal, before calibration).
    """
    rgb = np.asarray(rgb)
    old = np.asarray(old_mask) > 0
    if prior_threshold > 200:
        old = old | (rgb.min(2) < prior_threshold)
    clean, floor = remove_floor_plane(rgb, old)
    cal = calibrate_boundary(clean, scale, origin, dx, dy)
    cal = condition_dilate(cal, rgb, cond_dilate)
    return {"mask": (cal * 255).astype(np.uint8),
            "clean": (clean * 255).astype(np.uint8),
            "floor": (floor * 255).astype(np.uint8)}
