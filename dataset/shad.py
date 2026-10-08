"""SHAD / BrokenChairs180K dataset loading.

Server layout
-------------
    /data/xje/datasets/brokenchairs180k/shapes/          <- dataset root
        shape_1095/
            1095.obj                  mesh (look-at target + object size)
            point_cloud/1095.ply      3DGS model
            mv_images/                reference views: <id>_<r>_<azim>_<elev>.{png,json,npy}
            images/                   query renders: render_<id>_<r>_<azim>_<elev>_<type>.png
            camera/                   query camera json (only for some shapes)
            annotations/              anomaly masks / object masks
        shape_8430/
        ...

`ShadDataset` iterates the `shape_*` folders; `ShadObject` handles one of them.
The object id inside the filenames is ignored — view parameters are read as the
last three numeric fields of the stem, which works for both naming styles.

Filename azimuth sign
---------------------
`mv_images/` names use the same azimuth direction as the stored `RT` matrices
(verified to ~1e-7). `render_*` query names run in the OPPOSITE direction, so a
query labelled azim 245 is physically at azim 115. `query_azim_sign=-1` (default)
applies that correction when building ground truth from a filename. Getting this
wrong makes a working pipeline look ~45° off instead of ~1°.

Object symmetry
---------------
Symmetry is per-object, not a dataset constant. `shape_8430` (a bag) is ~180°
azimuth-symmetric in appearance, so its pose is only determined up to that
symmetry; most chairs are not symmetric. Set the symmetry period per object when
scoring (0 = none, the default).
"""

import glob
import json
import os
from fnmatch import fnmatch

import numpy as np

from utils.geometry import pose_from_azel
from utils.masks import object_mask


def _numbers_in(stem):
    out = []
    for part in stem.split("_"):
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def parse_view_params(path):
    """Last three numeric fields of the stem -> (radius, azim, elev), or None.

    Handles both `8430_3.0_108_20` and `render_8430_3.8_31_29_anomaly`.
    """
    nums = _numbers_in(os.path.splitext(os.path.basename(path))[0])
    return tuple(nums[-3:]) if len(nums) >= 3 else None


def mesh_bbox_centre_and_size(obj_path):
    v = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(v)
    return (v.min(0) + v.max(0)) / 2.0, float((v.max(0) - v.min(0)).max())


class QuerySample:
    """One query image plus its ground-truth pose derived from the filename."""

    def __init__(self, path, target, azim_sign=-1.0, white_thresh=235):
        self.path = path
        self.name = os.path.basename(path)
        self.params = parse_view_params(path)          # (radius, azim_label, elev)
        self.azim_sign = azim_sign
        self.white_thresh = white_thresh
        self._target = target
        self._img = None

    @property
    def image(self):
        if self._img is None:
            import cv2
            self._img = cv2.imread(self.path)
            if self._img is None:
                raise FileNotFoundError(self.path)
        return self._img

    @property
    def shape(self):
        h, w = self.image.shape[:2]
        return w, h

    def mask(self):
        return object_mask(self.image, self.white_thresh)

    def gray(self):
        import cv2
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def gt_pose(self):
        """OpenGL world-to-camera ground truth, or None if the name has no params.

        NOTE the filename values are rounded (azim/elev to 1°, radius to 0.1), so
        this ground truth is only accurate to ~0.5° / ~0.05 — errors below that
        are at the label-quantisation floor, not a real measurement.
        """
        if self.params is None:
            return None
        r, az, el = self.params
        return pose_from_azel(r, self.azim_sign * az, el, self._target)


class ShadObject:
    """A single object's assets and its query set."""

    def __init__(self, root, azim_sign=-1.0, white_thresh=235,
                 ref_subdir="mv_images", query_subdir="images", sym_azim=0.0):
        self.root = os.path.normpath(root)
        self.name = os.path.basename(self.root)
        self.white_thresh = white_thresh
        self.azim_sign = azim_sign
        self.sym_azim = sym_azim

        meshes = sorted(glob.glob(os.path.join(root, "*.obj")))
        if not meshes:
            raise FileNotFoundError(f"no .obj mesh in {root}")
        self.mesh_path = meshes[0]
        plys = sorted(glob.glob(os.path.join(root, "point_cloud", "*.ply"))) \
            or sorted(glob.glob(os.path.join(root, "*.ply")))
        if not plys:
            raise FileNotFoundError(f"no 3DGS .ply under {root}")
        self.ply_path = plys[0]
        self.ref_dir = os.path.join(root, ref_subdir)
        self.query_dir = os.path.join(root, query_subdir)
        self.target, self.obj_size = mesh_bbox_centre_and_size(self.mesh_path)

    @staticmethod
    def missing_assets(root, ref_subdir="mv_images", query_subdir="images"):
        """List of what a shape folder lacks (empty list == usable)."""
        missing = []
        if not glob.glob(os.path.join(root, "*.obj")):
            missing.append("mesh(.obj)")
        if not (glob.glob(os.path.join(root, "point_cloud", "*.ply"))
                or glob.glob(os.path.join(root, "*.ply"))):
            missing.append("gaussians(.ply)")
        if not glob.glob(os.path.join(root, ref_subdir, "*.png")):
            missing.append(f"{ref_subdir}/*.png")
        if not glob.glob(os.path.join(root, query_subdir, "*.png")):
            missing.append(f"{query_subdir}/*.png")
        return missing

    # ------------------------------------------------------------------------
    def references(self, drop_duplicates=True):
        """Reference views with their stored camera parameters.

        The shipped set has 21 files but only 20 distinct views (azimuth 0 and
        360 are the same camera); PTAD uses 20, so duplicates are dropped.
        """
        out, seen = [], set()
        for png in sorted(glob.glob(os.path.join(self.ref_dir, "*.png"))):
            js = os.path.splitext(png)[0] + ".json"
            if not os.path.exists(js):
                continue
            p = parse_view_params(png)
            if p is None:
                continue
            r, az, el = p
            key = (round(az % 360, 3), round(el, 3))
            if drop_duplicates and key in seen:
                continue
            seen.add(key)
            d = json.load(open(js))
            out.append({"png": png, "json": js, "radius": r, "azim": az % 360.0,
                        "elev": el, "K": d["K"],
                        "RT": np.asarray(d["RT"], dtype=np.float64)})
        return out

    def queries(self):
        return [QuerySample(p, self.target, self.azim_sign, self.white_thresh)
                for p in sorted(glob.glob(os.path.join(self.query_dir, "*.png")))]

    def __repr__(self):
        return (f"ShadObject({self.name}: "
                f"mesh={os.path.basename(self.mesh_path)}, "
                f"ply={os.path.basename(self.ply_path)}, "
                f"target={np.round(self.target, 4).tolist()})")


class ShadDataset:
    """A dataset root holding many `shape_*` folders.

        ds = ShadDataset("/data/xje/datasets/brokenchairs180k/shapes")
        for obj in ds.select("shape_1095,shape_8430"):   # or ds (all shapes)
            ...
    """

    def __init__(self, root, pattern="shape_*", azim_sign=-1.0, white_thresh=235,
                 ref_subdir="mv_images", query_subdir="images", require_complete=True):
        self.root = os.path.normpath(root)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"dataset root not found: {self.root}")
        self.pattern = pattern
        self.azim_sign = azim_sign
        self.white_thresh = white_thresh
        self.ref_subdir = ref_subdir
        self.query_subdir = query_subdir

        found = sorted(d for d in glob.glob(os.path.join(self.root, pattern))
                       if os.path.isdir(d))
        self.incomplete = {}
        self.shape_dirs = []
        for d in found:
            miss = ShadObject.missing_assets(d, ref_subdir, query_subdir)
            if miss and require_complete:
                self.incomplete[os.path.basename(d)] = miss
            else:
                self.shape_dirs.append(d)

    # ------------------------------------------------------------------------
    @property
    def names(self):
        return [os.path.basename(d) for d in self.shape_dirs]

    def object(self, name):
        path = name if os.path.isdir(name) else os.path.join(self.root, name)
        return ShadObject(path, self.azim_sign, self.white_thresh,
                          self.ref_subdir, self.query_subdir)

    def select(self, spec=None, limit=0):
        """Pick shapes by comma-separated names/globs; None or 'all' = everything."""
        if spec in (None, "", "all"):
            dirs = self.shape_dirs
        else:
            wanted, dirs = [s.strip() for s in spec.split(",") if s.strip()], []
            for w in wanted:
                hits = [d for d in self.shape_dirs
                        if fnmatch(os.path.basename(d), w) or os.path.basename(d) == w]
                if not hits:
                    raise SystemExit(f"no shape matching {w!r} under {self.root}")
                dirs += [h for h in hits if h not in dirs]
        if limit:
            dirs = dirs[:limit]
        return [ShadObject(d, self.azim_sign, self.white_thresh,
                           self.ref_subdir, self.query_subdir) for d in dirs]

    def __len__(self):
        return len(self.shape_dirs)

    def __iter__(self):
        return iter(self.select())

    def __repr__(self):
        s = f"ShadDataset({self.root}: {len(self.shape_dirs)} usable shapes"
        if self.incomplete:
            s += f", {len(self.incomplete)} skipped as incomplete"
        return s + ")"
