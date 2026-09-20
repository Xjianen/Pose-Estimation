#!/usr/bin/env python3
"""Check whether a shape's rendering/calibration convention matches this pipeline.

Use this before running `train.py` on a new dataset. It is read-only and needs
neither the mesh nor any trust in the query filenames, because every shape ships
its reference views with BOTH the filename parameters and the ground-truth `RT`
matrices — so the convention can be verified against the shape's own data.

What it checks
--------------
1. **Camera-centre convention** — does `C = r*(cos(el)cos(az), sin(el), cos(el)sin(az))`
   (world y-up, sphere about the origin) reproduce the centre implied by each stored
   `RT`?  This tests the world axes, the azimuth/elevation definition and the radius
   units in one shot, and needs no mesh and no look-at target.
2. **Look-at target** — least-squares fit of the single point all reference cameras
   look at, from the rotations alone: z_cam = normalise(C - target), so
   (I - z zᵀ)(C - target) = 0 for every view. Compared against the 3DGS bbox centre.
   A small residual also proves the cameras really do converge on one point.
3. **Full RT rebuild** — reconstruct each reference `RT` from (r, az, el) using the
   fitted target and report the worst element error.
4. **Filename parsing** — the pipeline reads view parameters out of filenames. Two
   rules are compared: "last three numeric fields" and "first field containing a
   decimal point, then the next two". Layouts with extra id fields
   (`render_1334_16105_2.5_20_25_1_normal`) break the first rule silently, which
   would make every reported error meaningless.
5. **Assets** — what `train.py` would find or miss for this shape.

Usage
-----
    python check_convention.py --root shape_1334
    python check_convention.py --data-root /data/xje/datasets/brokenchairs180k/shapes \
        --shapes "shape_*" --max-shapes 20
"""

import argparse
import glob
import json
import os

import numpy as np

from utils.geometry import UP, camera_centre_from_azel, pose_from_azel

TOL = 1e-5


# --------------------------------------------------------------------------- #
def numeric_fields(stem):
    out = []
    for p in stem.split("_"):
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


def parse_last3(stem):
    n = numeric_fields(stem)
    return tuple(n[-3:]) if len(n) >= 3 else None


def parse_first_decimal(stem):
    """radius = first field containing '.', then the next two fields."""
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if "." in p:
            try:
                float(p)
            except ValueError:
                continue
            if i + 2 < len(parts):
                try:
                    return float(parts[i]), float(parts[i + 1]), float(parts[i + 2])
                except ValueError:
                    return None
            return None
    return None


def plausible(v):
    return v is not None and 1.0 <= v[0] <= 8.0 and 0 <= v[1] <= 360 and 0 <= v[2] <= 90


# --------------------------------------------------------------------------- #
def ply_bbox_centre(path):
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
        xyz = np.frombuffer(f.read(n * len(props) * 4), dtype=np.float32
                            ).reshape(n, len(props))[:, :3].astype(np.float64)
    return (xyz.min(0) + xyz.max(0)) / 2.0, n


def fit_lookat_target(centres, rotations):
    """Solve for the point every camera looks at, using only the rotations.

    R_w2c rows are [x y z]; z = normalise(C - target), so (I - z zᵀ)(C - target) = 0.
    Stack those over views and solve in the least-squares sense.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for C, R in zip(centres, rotations):
        z = R[2, :]
        P = np.eye(3) - np.outer(z, z)
        A += P
        b += P @ C
    T = np.linalg.solve(A, b)
    res = [float(np.linalg.norm((np.eye(3) - np.outer(R[2, :], R[2, :])) @ (C - T)))
           for C, R in zip(centres, rotations)]
    return T, max(res)


# --------------------------------------------------------------------------- #
def check_shape(root, ref_subdir="mv_images", query_subdirs=("images",)):
    name = os.path.basename(os.path.normpath(root))
    print(f"\n{'=' * 72}\n{name}")

    js = sorted(glob.glob(os.path.join(root, ref_subdir, "*.json")))
    if not js:
        print(f"  no reference json under {ref_subdir}/ -> cannot verify convention")
        return False
    print(f"  {len(js)} reference views in {ref_subdir}/")

    centres, rotations, dC = [], [], []
    for p in js:
        stem = os.path.splitext(os.path.basename(p))[0]
        v = parse_first_decimal(stem) or parse_last3(stem)
        if v is None:
            print(f"  cannot parse view params from {stem}")
            return False
        r, az, el = v
        RT = np.asarray(json.load(open(p))["RT"], dtype=np.float64)
        R, t = RT[:3, :3], RT[:3, 3]
        C = -R.T @ t
        centres.append(C); rotations.append(R)
        dC.append(float(np.abs(C - camera_centre_from_azel(r, az, el)).max()))

    same_centre = max(dC) < TOL
    print(f"  [1] centre convention      max|dC| = {max(dC):.3e}  -> "
          f"{'SAME' if same_centre else 'DIFFERENT (stop here)'}")
    if not same_centre:
        print("      the world axes, azimuth/elevation definition or radius units "
              "differ from this pipeline")
        return False

    T, res = fit_lookat_target(centres, rotations)
    print(f"  [2] look-at target fitted  {np.round(T, 6)}  max residual {res:.3e}")
    plys = (sorted(glob.glob(os.path.join(root, "point_cloud", "*.ply")))
            or sorted(glob.glob(os.path.join(root, "*.ply"))))
    if plys:
        c, n = ply_bbox_centre(plys[0])
        print(f"      .ply bbox centre       {np.round(c, 6)}  "
              f"|diff| = {np.abs(T - c).max():.3e}  ({n} gaussians)")

    dR = []
    for (C, R), p in zip(zip(centres, rotations), js):
        stem = os.path.splitext(os.path.basename(p))[0]
        r, az, el = parse_first_decimal(stem) or parse_last3(stem)
        RTp = pose_from_azel(r, az, el, T)
        dR.append(float(np.abs(RTp[:3, :3] - R).max()))
    ok = max(dR) < TOL
    print(f"  [3] full RT rebuilt        max|dR| = {max(dR):.3e}  -> "
          f"{'CONVENTION IDENTICAL' if ok else 'rotation convention differs'}")

    print("  [4] filename parsing")
    for sub in (ref_subdir,) + tuple(query_subdirs):
        pngs = sorted(glob.glob(os.path.join(root, sub, "*.png")))
        if not pngs:
            continue
        agree = bad_last3 = 0
        for p in pngs:
            stem = os.path.splitext(os.path.basename(p))[0]
            a, b = parse_last3(stem), parse_first_decimal(stem)
            if a == b:
                agree += 1
            if not plausible(a):
                bad_last3 += 1
        flag = "" if bad_last3 == 0 else "   <-- 'last three' rule is WRONG here"
        print(f"      {sub + '/':14s} {len(pngs):3d} files | rules agree "
              f"{agree}/{len(pngs)} | implausible under 'last three': "
              f"{bad_last3}/{len(pngs)}{flag}")
        if pngs:
            stem = os.path.splitext(os.path.basename(pngs[0]))[0]
            print(f"        e.g. {stem}")
            print(f"             last-three     -> {parse_last3(stem)}")
            print(f"             first-decimal  -> {parse_first_decimal(stem)}")

    print("  [5] assets train.py needs")
    for label, pat in (("mesh (.obj)", "*.obj"), ("mesh (.glb)", "glb/*.glb"),
                       ("3DGS (.ply)", "point_cloud/*.ply")):
        hits = glob.glob(os.path.join(root, pat))
        print(f"      {label:14s} {'yes: ' + os.path.basename(hits[0]) if hits else 'MISSING'}")
    subs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    print(f"      subdirectories  {subs}")
    return ok


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="a single shape folder")
    ap.add_argument("--data-root", default=None, help="folder containing shape_* dirs")
    ap.add_argument("--shapes", default="shape_*")
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--ref-subdir", default="mv_images")
    ap.add_argument("--query-subdirs", default="images",
                    help="comma-separated query folders to check names in")
    args = ap.parse_args()

    if args.root:
        roots = [args.root]
    elif args.data_root:
        roots = sorted(d for d in glob.glob(os.path.join(args.data_root, args.shapes))
                       if os.path.isdir(d))
        if args.max_shapes:
            roots = roots[:args.max_shapes]
    else:
        raise SystemExit("give --root or --data-root")

    qsubs = tuple(s.strip() for s in args.query_subdirs.split(",") if s.strip())
    results = {os.path.basename(os.path.normpath(r)):
               check_shape(r, args.ref_subdir, qsubs) for r in roots}

    print(f"\n{'=' * 72}\nSUMMARY")
    good = [k for k, v in results.items() if v]
    bad = [k for k, v in results.items() if not v]
    print(f"  convention identical : {len(good)}/{len(results)}")
    if bad:
        print(f"  needs attention      : {bad}")


if __name__ == "__main__":
    main()
