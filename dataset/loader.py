"""Load pose inputs from images/, mask/, mv_images/ and train/."""
from pathlib import Path

import numpy as np
from PIL import Image

from .shad import (QuerySample as BaseQuery, ShadObject as BaseObject,
                   ShadDataset as BaseDataset, mesh_bbox_centre_and_size)
from utils.geometry import pose_from_azel


def read_mask(path):
    """Black background, white foreground; return float32 HxW in {0,1}."""
    with Image.open(path) as image:
        mask = (np.asarray(image.convert("L")) > 127).astype(np.float32)
    if not mask.any():
        raise ValueError(f"Empty object mask: {path}")
    return mask


def remove_floor_shadow(mask, image, grey_min=120, grey_max=170, chroma_max=34,
                        open_k=9, max_fraction=0.5):
    """Drop the floor shadow that some shipped query masks contain.

    Why: the dataset's mask/*.png are thresholded renders, and for several
    lighting variants the soft grey floor shadow under the chair passes the
    white threshold and is fused into the object mask (measured: ~14k px at
    RGB ~145 grey, sitting in the arch between the legs). That extra blob
    breaks the coarse silhouette match (the true-pose template then ranks
    below a wrong-basin one, initial rotation errors of 120-150 deg) and
    biases the refinement.

    Rule: achromatic mid-grey pixels (min-channel in [grey_min, grey_max),
    channel spread < chroma_max) form candidate shadow pixels; morphological
    opening keeps only BLOBS (the thin anti-aliased boundary of dark objects
    also matches the grey test and would erode the outline, but does not
    survive the opening). A blob is accepted as shadow only if it is a
    minority of the mask (< max_fraction) AND sits below the mask centroid
    (shadows are on the floor). Measured on shape_1095: the four corrupted
    queries recover (mask-vs-model IoU 0.80 -> 0.87-0.90) and every clean
    query is left untouched.
    """
    import cv2
    m = (np.asarray(mask) > 0.5).astype(np.uint8)
    rgb = np.asarray(image).astype(np.int16)
    mn, mx = rgb.min(2), rgb.max(2)
    grey = ((mn >= grey_min) & (mn < grey_max)
            & ((mx - mn) < chroma_max)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    blobs = cv2.dilate(cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel), kernel)
    area = int(blobs.sum())
    if area == 0 or area > max_fraction * m.sum():
        return mask, 0
    if np.nonzero(blobs)[0].mean() <= np.nonzero(m)[0].mean():
        return mask, 0
    cleaned = m & ~blobs
    if not cleaned.any():
        return mask, 0
    return cleaned.astype(np.float32), area


# --------------------------------------------------------------------------- #
# Pure-numpy silhouette helpers for the intrinsics calibration below.
# Deliberately free of cv2/torch so the calibration also runs in minimal envs.
# --------------------------------------------------------------------------- #
def _mask_bbox(mask):
    """(w_px, h_px) of the foreground bbox, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return None
    return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def _project_bbox(means, RT_gl, focal, res):
    """Foreground bbox (w_px, h_px) of a point cloud under a pinhole camera."""
    from utils.geometry import gl_w2c_to_cv_viewmat
    V = gl_w2c_to_cv_viewmat(RT_gl)
    P = np.asarray(means) @ V[:3, :3].T + V[:3, 3]
    P = P[P[:, 2] > 1e-6]
    if len(P) == 0:
        return None
    u = focal * P[:, 0] / P[:, 2] + res / 2.0
    v = focal * P[:, 1] / P[:, 2] + res / 2.0
    return float(u.max() - u.min()), float(v.max() - v.min())


def _splat_silhouette(means, RT_gl, focal, res):
    """Point-splat silhouette of the cloud (bool res x res, 1px dilated)."""
    from utils.geometry import gl_w2c_to_cv_viewmat
    V = gl_w2c_to_cv_viewmat(RT_gl)
    P = np.asarray(means) @ V[:3, :3].T + V[:3, 3]
    P = P[P[:, 2] > 1e-6]
    u = (focal * P[:, 0] / P[:, 2] + res / 2.0).astype(np.int64)
    v = (focal * P[:, 1] / P[:, 2] + res / 2.0).astype(np.int64)
    inside = (u >= 0) & (u < res) & (v >= 0) & (v < res)
    img = np.zeros((res, res), bool)
    img[v[inside], u[inside]] = True
    acc = img.copy()
    for d in (-1, 1):                       # thin splats: dilate by 1px
        acc |= np.roll(img, d, 0) | np.roll(img, d, 1)
    return acc


def _normalised_silhouette(mask, res=128):
    """Bbox-crop + resize a boolean mask (scale/position-invariant shape)."""
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = Image.fromarray(crop.astype(np.uint8) * 255).resize((res, res),
                                                               Image.BILINEAR)
    return np.asarray(img) > 127


class QuerySample(BaseQuery):
    def __init__(self, path, target, azim_sign=-1.0, white_thresh=235,
                 mask_path=None, K=None, radius_scale=1.0):
        super().__init__(str(path), target, azim_sign, white_thresh)
        self.mask_path = str(mask_path)
        self.K = K
        self.radius_scale = float(radius_scale)
        self._mask = None
        self.shadow_px = 0

    @property
    def image(self):
        if self._img is None:
            with Image.open(self.path) as image:
                self._img = np.asarray(image.convert("RGB"))[..., ::-1].copy()
        return self._img

    def mask(self):
        if self._mask is None:
            mask = read_mask(self.mask_path)
            if mask.shape != self.image.shape[:2]:
                raise ValueError(f"Image/mask size mismatch: {self.path}")
            mask, removed = remove_floor_shadow(mask, self.image)
            self._mask, self.shadow_px = mask, removed
        return self._mask

    @property
    def gt_radius(self):
        """Effective GT radius: the filename label rescaled by the calibration
        (`radius_scale` != 1 only when the labels were proven inconsistent with
        the reference FOV; see ShadObject._calibrate_query_intrinsics)."""
        if self.params is None:
            return None
        return self.params[0] * self.radius_scale

    def gt_pose(self):
        """Label pose with the calibrated radius scale applied."""
        if self.params is None:
            return None
        r, az, el = self.params
        return pose_from_azel(r * self.radius_scale, self.azim_sign * az, el,
                              self._target)


class ShadObject(BaseObject):
    """Compatible with train.py; calibration RT is OpenGL world-to-camera."""

    def __init__(self, root, azim_sign=-1.0, white_thresh=235,
                 ref_subdir="mv_images", query_subdir="images", sym_azim=0.0,
                 max_refs=0, ref_seed=0, verbose=True):
        root = Path(root).resolve()
        missing = self.missing_assets(root, ref_subdir, query_subdir)
        if missing:
            raise FileNotFoundError(f"{root}: missing {', '.join(missing)}")
        self.root, self.name = str(root), root.name
        self.azim_sign, self.white_thresh, self.sym_azim = azim_sign, white_thresh, sym_azim
        self.max_refs, self.ref_seed = int(max_refs), int(ref_seed)
        self.verbose = bool(verbose)
        self.ref_dir, self.query_dir = str(root / ref_subdir), str(root / query_subdir)
        self.ply_path = str(sorted((root / "point_cloud").glob("*.ply"))[0])
        meshes = sorted(root.glob("*.obj"))
        self.mesh_path = str(meshes[0]) if meshes else None
        refs = self.references()
        if not refs:
            raise ValueError(f"No calibrated references in {self.ref_dir}")
        if meshes:
            self.target, self.obj_size = mesh_bbox_centre_and_size(self.mesh_path)
        else:
            # Intersect the viewing rays; no mesh or query GT is needed.
            matrices, vectors = [], []
            for ref in refs:
                rt = ref["RT"]
                centre = -np.linalg.solve(rt[:3, :3], rt[:3, 3])
                direction = rt[2, :3] / np.linalg.norm(rt[2, :3])
                projection = np.eye(3) - np.outer(direction, direction)
                matrices.append(projection)
                vectors.append(projection @ centre)
            self.target, _, rank, _ = np.linalg.lstsq(
                np.concatenate(matrices), np.concatenate(vectors), rcond=None)
            if rank != 3:
                raise ValueError("Reference cameras cannot determine look-at centre")
            # Only descriptive metadata in the existing pipeline.
            from utils.gaussians import load_gaussians_ply
            xyz = load_gaussians_ply(self.ply_path)["means"]
            self.obj_size = float(np.ptp(xyz, axis=0).max())

    @staticmethod
    def missing_assets(root, ref_subdir="mv_images", query_subdir="images"):
        root = Path(root)
        missing = []
        for subdir, pattern in [("point_cloud", "*.ply"), (ref_subdir, "*.json"),
                                (query_subdir, "*.png"), ("train", "*.png"),
                                ("mask", "*.png")]:
            if not list((root / subdir).glob(pattern)):
                missing.append(f"{subdir}/{pattern}")
        return missing

    def references(self, drop_duplicates=True):
        refs = super().references(drop_duplicates)
        for ref in refs:
            mask_path = Path(self.root) / "train" / Path(ref["png"]).name
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            with Image.open(ref["png"]) as image:
                old_w, old_h = image.size
            with Image.open(mask_path) as image:
                w, h = image.size
            ref["K"] = np.diag([w / old_w, h / old_h, 1.0]) @ np.asarray(ref["K"])
            ref["mask_path"] = str(mask_path)
            ref["mask"] = read_mask(mask_path)
        return refs

    # ------------------------------------------------------------------ #
    # Query intrinsics calibration
    # ------------------------------------------------------------------ #
    def _calibrate_query_intrinsics(self, ref, entries, max_points=250_000):
        """Check whether the query renders share the reference camera FOV.

        Returns (focal_scale, radius_scale, note).

        Why this exists: on shape_1095 the reference renders use fov_x 39.66
        deg (focal 355 @ 256) but the query renders use fov_x 50 deg. Assuming
        the reference FOV for queries makes the silhouette still match with the
        camera 1/0.774 farther away, so every mask LOOKS aligned while the pose
        (and the exported depth) is systematically wrong: a constant ~0.75
        translation "error" at true radius 2.5, with the estimated radius at
        ~3.23.

        Method — the filename (radius, azim, elev) labels are trusted (they
        reproduce the reference RTs to 1e-7 and give < 2 deg rotation error):
        1. project the gaussian means at each query's label pose with the
           reference focal; compare silhouette bboxes with the query masks;
        2. a consistent scale s != 1 has two explanations a bbox cannot
           separate — NEAR: true radius = label and query focal = ref focal * s
           (different FOV), or FAR: true radius = label / s with the reference
           FOV (label units off). Both reproduce the bbox exactly, but their
           PERSPECTIVE differs, so
        3. point-splat silhouettes under both hypotheses decide the winner.
        """
        from utils.gaussians import load_gaussians_ply

        labelled = [(s, wh) for s, wh in entries if s.params is not None]
        if len(labelled) < 3:
            return 1.0, 1.0, "too few labelled queries; assuming reference FOV"
        try:
            means = load_gaussians_ply(self.ply_path)["means"]
        except Exception as exc:                       # unreadable ply -> bail
            return 1.0, 1.0, f"ply unreadable ({exc}); assuming reference FOV"
        if len(means) > max_points:
            means = means[::len(means) // max_points + 1]

        f0 = float(ref["K"][0, 0])                     # ref focal @ mask res
        res0 = ref["mask"].shape[1]
        scales = []
        for sample, (w, h) in labelled:
            r, az, el = sample.params
            RT = pose_from_azel(r, sample.azim_sign * az, el, self.target)
            pred = _project_bbox(means, RT, f0 * w / res0, w)
            actual = _mask_bbox(sample.mask())
            if pred is None or actual is None:
                continue
            scales.append((actual[0] / pred[0] + actual[1] / pred[1]) / 2.0)
        if len(scales) < 3:
            return 1.0, 1.0, "too few measurable queries; assuming reference FOV"
        s, spread = float(np.median(scales)), float(np.std(scales))
        if abs(s - 1.0) <= 0.02:
            return 1.0, 1.0, (f"bbox scale {s:.3f} ~ 1: query FOV matches "
                              "the references")
        if spread > 0.05:
            return 1.0, 1.0, (f"inconsistent bbox scale {s:.3f}+-{spread:.3f}; "
                              "keeping the reference FOV")

        iou_near = iou_far = 0.0
        n = 0
        for sample, (w, h) in labelled[:5]:
            r, az, el = sample.params
            az_true = sample.azim_sign * az
            truth = _normalised_silhouette(sample.mask())
            near = _normalised_silhouette(_splat_silhouette(
                means, pose_from_azel(r, az_true, el, self.target),
                f0 * s * w / res0, w))
            far = _normalised_silhouette(_splat_silhouette(
                means, pose_from_azel(r / s, az_true, el, self.target),
                f0 * w / res0, w))
            iou_near += np.logical_and(truth, near).sum() / max(np.logical_or(truth, near).sum(), 1)
            iou_far += np.logical_and(truth, far).sum() / max(np.logical_or(truth, far).sum(), 1)
            n += 1
        iou_near, iou_far = iou_near / n, iou_far / n
        if iou_near >= iou_far:
            return s, 1.0, (
                f"query FOV DIFFERS from references: bbox scale {s:.3f}, "
                f"perspective favours the label radius (IoU {iou_near:.3f} vs "
                f"{iou_far:.3f}) -> query focal x{s:.3f}")
        return 1.0, 1.0 / s, (
            f"query bbox scale {s:.3f}, perspective favours the reference FOV "
            f"(IoU {iou_far:.3f} vs {iou_near:.3f}) -> GT radius x{1.0 / s:.3f}")

    def queries(self):
        ref = self.references()[0]
        h0, w0 = ref["mask"].shape
        entries = []
        for path in sorted(Path(self.query_dir).glob("*.png")):
            mask_path = Path(self.root) / "mask" / path.name
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            sample = QuerySample(path, self.target, self.azim_sign,
                                 self.white_thresh, mask_path)
            sample.mask()  # Fail early on invalid pairs.
            with Image.open(path) as image:
                entries.append((sample, image.size))

        # Intrinsics: resolution-scaled reference K by default; the calibration
        # detects (and corrects) a query FOV that differs from the references.
        focal_scale, radius_scale, note = self._calibrate_query_intrinsics(
            ref, entries)
        if self.verbose:
            print(f"  query intrinsics: {note}")
            shadowed = [(s.name, s.shadow_px) for s, _ in entries if s.shadow_px > 0]
            if shadowed:
                biggest = max(px for _, px in shadowed)
                print(f"  floor shadow removed from {len(shadowed)}/{len(entries)} "
                      f"query masks (largest blob {biggest} px)")
        for sample, (w, h) in entries:
            K = np.diag([w / w0, h / h0, 1.0]) @ np.asarray(ref["K"], float)
            K[0, 0] *= focal_scale
            K[1, 1] *= focal_scale
            sample.K = K
            sample.radius_scale = radius_scale
        return [sample for sample, _ in entries]

    def __repr__(self):
        return f"ShadObject({self.name}, target={self.target.tolist()})"


class ShadDataset(BaseDataset):
    def __init__(self, root, pattern="shape_*", azim_sign=-1.0, white_thresh=235,
                 ref_subdir="mv_images", query_subdir="images", require_complete=True):
        self.root = str(Path(root).resolve())
        if not Path(self.root).is_dir():
            raise FileNotFoundError(self.root)
        self.pattern, self.azim_sign, self.white_thresh = pattern, azim_sign, white_thresh
        self.ref_subdir, self.query_subdir = ref_subdir, query_subdir
        self.incomplete, self.shape_dirs = {}, []
        for path in sorted(Path(self.root).glob(pattern)):
            if not path.is_dir():
                continue
            missing = ShadObject.missing_assets(path, ref_subdir, query_subdir)
            if missing and require_complete:
                self.incomplete[path.name] = missing
            else:
                self.shape_dirs.append(str(path))

    def object(self, name):
        path = Path(name) if Path(name).is_dir() else Path(self.root) / name
        return ShadObject(path, self.azim_sign, self.white_thresh,
                          self.ref_subdir, self.query_subdir)

    def selected_paths(self, spec=None, limit=0):
        """Select paths without loading any model or images."""
        from fnmatch import fnmatch
        dirs = self.shape_dirs
        if spec not in (None, "", "all"):
            selected = []
            for pattern in spec.split(","):
                hits = [p for p in dirs if fnmatch(Path(p).name, pattern.strip())]
                if not hits:
                    raise ValueError(f"No shape matching {pattern!r}")
                selected.extend(p for p in hits if p not in selected)
            dirs = selected
        return dirs[:limit] if limit else dirs

    def select(self, spec=None, limit=0):
        return [self.object(p) for p in self.selected_paths(spec, limit)]

    def __iter__(self):
        for path in self.shape_dirs:
            yield self.object(path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to one shape directory")
    args = parser.parse_args()
    obj = ShadObject(args.root)
    refs, queries = obj.references(), obj.queries()
    print(obj)
    print(f"References: {len(refs)} (0/360 duplicate removed); queries: {len(queries)}")
    print(f"Gaussian model: {obj.ply_path}")
    print(f"Reference mask: {refs[0]['mask'].shape}; K:\n{refs[0]['K']}")
    print(f"Query: {queries[0].name}; mask: {queries[0].mask().shape}")
    print(f"Query K:\n{queries[0].K}")
    print(f"Query filename pose parameters: {queries[0].params}")
