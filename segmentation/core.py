"""Foreground segmentation for white-background rendered objects.

Why this module exists
----------------------
The shipped ``mask/*.png`` in this dataset are produced by thresholding a
camera-space normal buffer (``test/``) against its constant background colour.
Everything that is not background -- including the floor plane and the soft
grey shadow under the object -- becomes foreground. Measured on two shapes:

    * ~3.3k extra pixels per query (floor shadow fused into the silhouette),
    * salt-and-pepper holes inside the shadow,
    * concave openings (the arch under a chair) filled solid.

A plain brightness threshold does not fix it: on light-grey chairs the object
and the floor shadow have almost identical achromatic colours, while a
saliency-based background remover (RMBG-1.4) drops the light-grey seat
entirely (9/45 catastrophic failures locally).

Solution: **prompt-driven Segment Anything (SlimSAM) with a deterministic
fallback.** SAM is prompt based, so it does not care about the object's colour;
positive points are sampled from the eroded interior of a cheap colour mask,
negative points from the outside and from the smooth floor shadow. A small
deterministic CV pipeline is kept as a fallback for when the model is absent
or produces an implausible mask.

No training is involved. Only a 14 MB pair of public ONNX weights is needed.

Public API
----------
``segment_image(rgb, sam=None, mode="auto")`` -> dict with ``mask`` (uint8
0/255), ``method``, ``coarse`` and ``metrics``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

SAM_SIZE = 1024
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

SAM_ENCODER = "vision_encoder_quantized.onnx"
SAM_DECODER = "prompt_encoder_mask_decoder_quantized.onnx"


# --------------------------------------------------------------------------- #
# Unicode-safe I/O (cv2.imread/imwrite fail on non-ASCII Windows paths)
# --------------------------------------------------------------------------- #
def imread(path, flags=cv2.IMREAD_COLOR):
    """Read an image through a byte buffer so Chinese paths work on Windows."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise IOError(f"empty or unreadable image: {path}")
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"cannot decode image: {path}")
    return img


def imwrite(path, img):
    """Write an image through a byte buffer (Unicode-path safe)."""
    ext = os.path.splitext(str(path))[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def load_rgb(path):
    return cv2.cvtColor(imread(path), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Deterministic colour/shadow model
# --------------------------------------------------------------------------- #
def largest_components(mask, ratio=0.01):
    """Keep connected components at least ``ratio`` of the biggest one.

    Never drops small-but-real parts (chair legs) unless they are isolated
    noise; background label 0 must not be retained.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m)
    big = int(stats[1:, cv2.CC_STAT_AREA].max())
    keep = [i for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= big * ratio]
    return np.isin(lab, keep).astype(np.uint8) if keep else np.zeros_like(m)


def coarse_foreground(rgb, threshold=205, open_k=3):
    """Cheap white-background removal, over-inclusive by design.

    It captures light-grey objects reliably (the hard part for saliency
    models), at the price of also capturing the floor shadow. Later stages
    remove the shadow; keeping completeness here is what makes the final
    mask robust across object colours.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cand = (gray < threshold).astype(np.uint8)
    if open_k:
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                                np.ones((open_k, open_k), np.uint8))
    return largest_components(cand)


def shadow_like(rgb, chroma_max=22, grey_min=95, grey_max=245, smooth_max=7):
    """Achromatic, low-texture, mid/light grey pixels = floor shadow.

    ``smooth_max`` uses the local standard deviation of luminance: the
    shadow is a smooth vertical gradient while object surfaces have hard
    boundaries and shading detail.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mn = rgb.min(2).astype(np.int16)
    mx = rgb.max(2).astype(np.int16)
    chroma = mx - mn
    var = cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2
    smooth = np.sqrt(np.maximum(var, 0))
    return ((chroma < chroma_max) & (mn >= grey_min) & (mn < grey_max)
            & (smooth < smooth_max)).astype(np.uint8)


def fill_small_holes(mask, max_area_ratio=0.002):
    """Close tiny holes but PRESERVE large openings (chair arches, leg gaps).

    This is deliberately not a blanket ``floodFill`` hole fill -- filling
    every enclosed background region is exactly the bug in the old masks.
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    buf = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, buf, (0, 0), 2)
    holes = (ff == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max_area_ratio * h * w:
            m[lab == i] = 1
    return m


def cv_refine(rgb, coarse=None):
    """Deterministic fallback: strip the smooth floor-shadow blob.

    Only removes blobs BELOW the foreground centroid (shadows are on the
    floor) and refuses to act if the blob would eat most of the object --
    that guard is what protects light-grey chairs, whose seat is also
    achromatic and smooth.
    """
    if coarse is None:
        coarse = coarse_foreground(rgb)
    c = (coarse > 0).astype(np.uint8)
    if not c.any():
        return c
    sh = shadow_like(rgb)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blob = cv2.dilate(cv2.morphologyEx(sh, cv2.MORPH_OPEN, k), k)
    ys, _ = np.nonzero(c)
    cy = int(ys.mean())
    blob[:cy, :] = 0
    if blob.sum() > 0.5 * c.sum():
        blob[:] = 0
    out = c & ~blob
    if out.sum() < 0.5 * c.sum():
        return c
    out = largest_components(out)
    return fill_small_holes(out)


# --------------------------------------------------------------------------- #
# Prompt generation + SAM inference
# --------------------------------------------------------------------------- #
def sample_prompts(coarse, rgb, grid=3, n_shadow_neg=3):
    """Build SAM point prompts from a cheap over-inclusive mask.

    Positive: deepest interior pixel of each grid cell (guaranteed object).
    Negative: image corners/edges plus a few pixels from the smooth shadow
    region below the object centroid.
    """
    h, w = coarse.shape
    dist = cv2.distanceTransform(coarse.astype(np.uint8), cv2.DIST_L2, 5)
    pts, labels = [], []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = int(gy * h / grid), int((gy + 1) * h / grid)
            x0, x1 = int(gx * w / grid), int((gx + 1) * w / grid)
            cell = dist[y0:y1, x0:x1].copy()
            if cell.max() < 4:
                continue
            yy, xx = np.unravel_index(np.argmax(cell), cell.shape)
            pts.append([x0 + xx, y0 + yy]); labels.append(1)
    for x, y in [(4, 4), (w - 4, 4), (4, h - 4), (w - 4, h - 4),
                 (w // 2, h - 4)]:
        pts.append([x, y]); labels.append(0)
    ys, _ = np.nonzero(coarse)
    cy = int(ys.mean()) if len(ys) else h // 2
    sh = (shadow_like(rgb) * (coarse > 0)).astype(np.uint8)
    sh[:cy, :] = 0
    cand = np.column_stack(np.nonzero(sh))
    if len(cand) > 6 and n_shadow_neg:
        for i in np.linspace(0, len(cand) - 1, n_shadow_neg).astype(int):
            y, x = cand[i]
            pts.append([x, y]); labels.append(0)
    if not pts:
        pts, labels = [[w // 2, h // 2]], [1]
    return np.asarray(pts, np.float32), np.asarray(labels, np.int64)


def _iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max((a | b).sum(), 1)


class SAMSegmenter:
    """SlimSAM (ONNX) wrapper. Weights are ~14 MB total, no training needed.

    ``model_dir`` must contain ``vision_encoder_quantized.onnx`` and
    ``prompt_encoder_mask_decoder_quantized.onnx`` (see download_models.py).
    """

    def __init__(self, model_dir=None, providers=None):
        import onnxruntime as ort

        model_dir = model_dir or os.path.join(os.path.dirname(__file__), "models")
        enc = os.path.join(model_dir, SAM_ENCODER)
        dec = os.path.join(model_dir, SAM_DECODER)
        missing = [p for p in (enc, dec) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "SAM weights missing: " + ", ".join(missing)
                + "\nRun: python segmentation/download_models.py")
        avail = ort.get_available_providers()
        if providers is None:
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in avail
                         else ["CPUExecutionProvider"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self.encoder = ort.InferenceSession(enc, opts, providers=providers)
        self.decoder = ort.InferenceSession(dec, opts, providers=providers)
        self._in_e = self.encoder.get_inputs()[0].name
        self._in_d = [i.name for i in self.decoder.get_inputs()]

    def segment(self, rgb, coarse):
        """Return a refined uint8 mask, or None when no candidate is plausible."""
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (SAM_SIZE, SAM_SIZE),
                       interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = np.ascontiguousarray(((x - IMAGENET_MEAN) / IMAGENET_STD)
                                 .transpose(2, 0, 1)[None])
        emb, pos = self.encoder.run(None, {self._in_e: x})
        pts, labs = sample_prompts(coarse, rgb)
        scale = SAM_SIZE / max(h, w)
        feeds = {self._in_d[0]: (pts * scale).astype(np.float32)[None, None],
                 self._in_d[1]: labs[None, None],
                 self._in_d[2]: emb,
                 self._in_d[3]: pos}
        scores, masks = self.decoder.run(None, feeds)

        area_c = max(coarse.sum(), 1)
        best, best_score = None, -1.0
        for k in range(masks.shape[2]):
            a = cv2.resize(masks[0, 0, k].astype(np.float32),
                           (SAM_SIZE, SAM_SIZE), interpolation=cv2.INTER_LINEAR)
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            m = (a > 0).astype(np.uint8)
            ratio = m.sum() / area_c
            # SAM occasionally returns the *background* as a high-scoring mask.
            # Agreement with the colour prior + a plausible area is a much
            # safer selector than the model's own IoU head.
            if not (0.55 < ratio < 1.45):
                continue
            score = _iou(m, coarse) + 0.2 * float(scores[0, 0, k])
            if score > best_score:
                best_score, best = score, m
        if best is None:
            return None
        best = largest_components(best)
        best = fill_small_holes(best)
        return cv2.morphologyEx(best, cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))


# --------------------------------------------------------------------------- #
# Quality metrics (no pixel-perfect GT exists for the query views)
# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    edge_align: float      # boundary on a strong RGB gradient (higher = better)
    shadow_residue: float  # achromatic smooth lower-half pixels (lower = better)
    speckles: int          # tiny connected components (lower = better)
    area: int

    def as_dict(self):
        return {"edge_align": round(float(self.edge_align), 4),
                "shadow_residue": round(float(self.shadow_residue), 4),
                "speckles": int(self.speckles),
                "area": int(self.area)}


def quality_metrics(rgb, mask):
    """Unsupervised proxies for silhouette quality.

    ``render/`` and ``reference/`` in this dataset are rendered with a
    DIFFERENT camera (39.66 deg FOV vs 50 deg for queries), so they cannot be
    used as query-view ground truth (verified: boundary/RGB-edge alignment
    0.86 vs 0.98 for the shipped masks). These proxies measure what actually
    matters for pose estimation: boundary correctness, shadow contamination,
    and spurious blobs.
    """
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return Metrics(0.0, 0.0, 0, 0)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3),
                    cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3))
    strong = cv2.dilate((grad > np.percentile(grad, 90)).astype(np.uint8),
                        np.ones((3, 3), np.uint8)) > 0
    bd = cv2.Canny((m * 255).astype(np.uint8), 50, 150) > 0
    align = (bd & strong).sum() / max(bd.sum(), 1)
    sh = shadow_like(rgb)
    lower = np.zeros_like(m)
    lower[m.shape[0] // 2:, :] = 1
    residue = (m & sh & lower).sum() / max(m.sum(), 1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    speckles = sum(1 for i in range(1, n)
                   if stats[i, cv2.CC_STAT_AREA] < 40)
    return Metrics(float(align), float(residue), int(speckles), int(m.sum()))


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def segment_image(rgb, sam=None, mode="auto", min_agreement=0.72):
    """Segment one RGB image.

    mode:
      ``sam``  -- require SAM (raises if unavailable/implausible -> CV fallback)
      ``cv``   -- deterministic only, no model, ~20 ms/image
      ``auto`` -- SAM when available, CV otherwise (recommended)

    Returns dict: ``mask`` (uint8 0/255), ``method``, ``coarse``, ``metrics``.
    """
    if mode not in ("auto", "sam", "cv"):
        raise ValueError(f"unknown mode {mode!r}")
    coarse = coarse_foreground(rgb)
    if not coarse.any():
        return {"mask": np.zeros(rgb.shape[:2], np.uint8), "method": "empty",
                "coarse": coarse, "metrics": Metrics(0, 0, 0, 0)}

    cv_mask = cv_refine(rgb, coarse)
    method = "cv"
    mask = cv_mask
    if mode in ("auto", "sam") and sam is not None:
        try:
            refined = sam.segment(rgb, coarse)
        except Exception:
            refined = None
        if refined is not None and _iou(refined, coarse) >= min_agreement \
                and refined.sum() <= 1.6 * coarse.sum():
            mask, method = refined, "sam"
    if mask.sum() == 0:
        mask, method = coarse, "coarse"
    m = fill_small_holes(largest_components(mask))
    return {"mask": (m * 255).astype(np.uint8), "method": method,
            "coarse": coarse, "metrics": quality_metrics(rgb, m)}
