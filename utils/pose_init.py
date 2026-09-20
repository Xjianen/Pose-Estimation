"""Pose initialisation: a silhouette template codebook plus coarse matching.

Two ways to build the codebook:
  * `Codebook.from_references(...)` — uses ONLY the reference views shipped with
    the dataset (the fair setting: PTAD uses those 20 views, so any accuracy gain
    must come from the matcher/refiner rather than from extra templates).
  * `Codebook.render(...)` — renders a dense (azimuth x elevation) grid from the
    3DGS model. Costs ~2 ms/view, so a 1920-template book takes ~4 s.

Matching cost is one matmul over binary masks (~15 ms for 20 templates, still
milliseconds for 2000) with no network forward.

Radius is deliberately NOT a codebook dimension: masks are bbox-crop normalised,
which makes scale non-discriminative, so radius is recovered analytically from
the mask size,  r = r_cb * (bbox_cb / bbox_query) * (f_query / f_cb).
"""

import glob
import json
import os

import numpy as np

from utils.geometry import S4, pose_from_azel
from utils.masks import (distance_to_mask, normalise_mask, read_object_mask)


class Codebook:
    """Silhouette templates + the single shared coarse-matching implementation."""

    def __init__(self, masks, bbox_px, azim, elev, radius, focal, res,
                 mask_res, target, obj_size=1.0):
        self.masks = np.asarray(masks, np.uint8)
        self.N = self.masks.shape[0]
        self.mask_res = int(mask_res)
        self.flat = self.masks.reshape(self.N, -1)
        self.flatf = self.flat.astype(np.float32)
        self.area = self.flat.sum(1).astype(np.float32)
        self.bbox_px = np.asarray(bbox_px, np.float32)
        self.azim = np.asarray(azim, np.float32)
        self.elev = np.asarray(elev, np.float32)
        self.radius = float(radius)
        self.focal = float(focal)
        self.res = int(res)
        self.target = np.asarray(target, np.float64)
        self.obj_size = float(obj_size)
        self._dist = None

    # ---------------------------------------------------------------- build --
    @classmethod
    def from_references(cls, references, mask_res=160, white_thresh=235,
                        target=None, obj_size=1.0):
        """Pack the dataset's own reference views (fair setting, no rendering)."""
        masks, bbox, az, el = [], [], [], []
        radius, focal, res = None, None, None
        for ref in references:
            raw, W = read_object_mask(ref["png"], white_thresh)
            m, sz = normalise_mask(raw, mask_res)
            masks.append(m); bbox.append(sz)
            az.append(ref["azim"] % 360.0); el.append(ref["elev"])
            radius, focal, res = ref["radius"], ref["K"][0][0], W
        return cls(masks, bbox, az, el, radius, focal, res, mask_res,
                   target if target is not None else np.zeros(3), obj_size)

    @classmethod
    def render(cls, gauss, sh_degree, target, azim_step=3.0, elev_range=(15.0, 45.0),
               elev_step=2.0, radius=3.5, res=256, focal=355.0, mask_res=160,
               batch=32, obj_size=1.0, device="cuda", verbose=True):
        """Render a dense (azimuth x elevation) grid of silhouettes with gsplat."""
        import torch
        from gsplat import rasterization
        azims = np.arange(0.0, 360.0, azim_step)
        elevs = np.arange(elev_range[0], elev_range[1] + 1e-9, elev_step)
        grid = [(a, e) for e in elevs for a in azims]
        K = torch.tensor([[focal, 0, res / 2], [0, focal, res / 2], [0, 0, 1.0]],
                         dtype=torch.float32, device=device)
        masks = np.zeros((len(grid), mask_res, mask_res), np.uint8)
        bbox = np.zeros(len(grid), np.float32)
        for i0 in range(0, len(grid), batch):
            chunk = grid[i0:i0 + batch]
            vm = np.stack([S4 @ pose_from_azel(radius, a, e, target) for a, e in chunk])
            viewmats = torch.tensor(vm, dtype=torch.float32, device=device)
            Ks = K[None].expand(len(chunk), 3, 3).contiguous()
            with torch.no_grad():
                _, alphas, _ = rasterization(
                    means=gauss["means"], quats=gauss["quats"], scales=gauss["scales"],
                    opacities=gauss["opacities"], colors=gauss["shs"],
                    viewmats=viewmats, Ks=Ks, width=res, height=res,
                    sh_degree=sh_degree, packed=False)
            a = (alphas[..., 0] > 0.5).to(torch.uint8).cpu().numpy()
            for j in range(len(chunk)):
                masks[i0 + j], bbox[i0 + j] = normalise_mask(a[j], mask_res)
            if verbose and i0 % (batch * 10) == 0:
                print(f"    rendered {i0 + len(chunk)}/{len(grid)}")
        return cls(masks, bbox, [a for a, _ in grid], [e for _, e in grid],
                   radius, focal, res, mask_res, target, obj_size)

    # ------------------------------------------------------------- (de)ser --
    def save(self, path):
        np.savez_compressed(path, masks=self.masks, bbox_px=self.bbox_px,
                            azim=self.azim, elev=self.elev,
                            radius=np.float32(self.radius),
                            focal=np.float32(self.focal), res=np.int32(self.res),
                            mask_res=np.int32(self.mask_res),
                            target=self.target.astype(np.float32),
                            obj_size=np.float32(self.obj_size))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["masks"], d["bbox_px"], d["azim"], d["elev"], float(d["radius"]),
                   float(d["focal"]), int(d["res"]), int(d["mask_res"]),
                   d["target"], float(d["obj_size"]))

    # ---------------------------------------------------------------- match --
    def _template_dist(self):
        if self._dist is None:
            self._dist = np.stack([distance_to_mask(m.astype(bool))
                                   for m in self.masks]).reshape(self.N, -1)
        return self._dist

    def match(self, query_png=None, query_mask=None, metric="chamfer",
              white_thresh=235, query_focal=-1.0, topk=3):
        """Coarse match. Returns a list of hypotheses, best first.

        metric='chamfer' is PTAD's symmetric region chamfer distance;
        metric='iou' is mask IoU. Both reduce to matmuls over the whole codebook.
        Each hypothesis: {idx, azim, elev, radius, score, RT}.
        """
        if query_mask is None:
            raw, W = read_object_mask(query_png, white_thresh)
        else:
            raw, W = query_mask, query_mask.shape[1]
        qm, qsize = normalise_mask(raw, self.mask_res)
        q = qm.reshape(-1).astype(np.float32)
        qa = float(q.sum())

        if metric == "iou":
            inter = self.flatf @ q
            score = -(inter / (self.area + qa - inter + 1e-6))
        else:
            qd = distance_to_mask(qm.astype(bool)).reshape(-1)
            score = ((self._template_dist() @ q) / max(qa, 1e-6)
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
        return hyps, f_q, W

    def intrinsics(self, focal, width, height=None):
        h = height if height is not None else width
        return np.array([[focal, 0.0, width / 2.0],
                         [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]])
