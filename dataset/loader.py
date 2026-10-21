"""Load pose inputs from images/, mask/, mv_images/ and train/."""
from pathlib import Path

import numpy as np
from PIL import Image

from .shad import (QuerySample as BaseQuery, ShadObject as BaseObject,
                   ShadDataset as BaseDataset, mesh_bbox_centre_and_size)


def read_mask(path):
    """Black background, white foreground; return float32 HxW in {0,1}."""
    with Image.open(path) as image:
        mask = (np.asarray(image.convert("L")) > 127).astype(np.float32)
    if not mask.any():
        raise ValueError(f"Empty object mask: {path}")
    return mask


class QuerySample(BaseQuery):
    def __init__(self, path, target, azim_sign=-1.0, white_thresh=235,
                 mask_path=None, K=None):
        super().__init__(str(path), target, azim_sign, white_thresh)
        self.mask_path = str(mask_path)
        self.K = K

    @property
    def image(self):
        if self._img is None:
            with Image.open(self.path) as image:
                self._img = np.asarray(image.convert("RGB"))[..., ::-1].copy()
        return self._img

    def mask(self):
        mask = read_mask(self.mask_path)
        if mask.shape != self.image.shape[:2]:
            raise ValueError(f"Image/mask size mismatch: {self.path}")
        return mask


class ShadObject(BaseObject):
    """Compatible with train.py; calibration RT is OpenGL world-to-camera."""
    def __init__(self, root, azim_sign=-1.0, white_thresh=235,
                 ref_subdir="mv_images", query_subdir="images", sym_azim=0.0):
        root = Path(root).resolve()
        missing = self.missing_assets(root, ref_subdir, query_subdir)
        if missing:
            raise FileNotFoundError(f"{root}: missing {', '.join(missing)}")
        self.root, self.name = str(root), root.name
        self.azim_sign, self.white_thresh, self.sym_azim = azim_sign, white_thresh, sym_azim
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

    def queries(self):
        # Same camera FOV as the calibrated reference renders.
        ref = self.references()[0]
        h0, w0 = ref["mask"].shape
        samples = []
        for path in sorted(Path(self.query_dir).glob("*.png")):
            mask_path = Path(self.root) / "mask" / path.name
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            with Image.open(path) as image:
                w, h = image.size
            K = np.diag([w / w0, h / h0, 1.0]) @ ref["K"]
            sample = QuerySample(path, self.target, self.azim_sign,
                                 self.white_thresh, mask_path, K)
            sample.mask()  # Fail early on invalid pairs.
            samples.append(sample)
        return samples

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
    print(f"Query filename pose parameters: {queries[0].params}")
