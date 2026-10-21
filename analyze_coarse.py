"""Offline coarse-retrieval diagnosis (numpy only, no torch/gsplat needed).

Answers two questions for one shape:

1. Trans-scale bug: with the *reference* focal assumed for queries, how far off
   does the analytic radius recovery `match()` uses land?
2. Bad initial angles: for each query, what initial rotation error does each of
   the top-K templates give, and does the CORRECT template even sit in the top-K?

The dense codebook is emulated by point-splatting the gaussian means (numpy),
which is a good proxy for what gsplat would render.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.loader import ShadObject, read_mask, _splat_silhouette, \
    _normalised_silhouette, _project_bbox, _mask_bbox
from dataset.shad import parse_view_params
from utils.geometry import pose_from_azel, rotation_error_deg, gl_w2c_to_cv_viewmat
from utils.masks import normalise_mask, distance_to_mask


def load_means(ply, max_points=300_000):
    from utils.gaussians import load_gaussians_ply
    m = load_gaussians_ply(ply)["means"]
    return m[::max(1, len(m) // max_points)]


def build_ref_codebook(obj, mask_res=160):
    masks, bbox, az, el = [], [], [], []
    for ref in obj.references():
        m, sz = normalise_mask((ref["mask"] > 0.5).astype(np.uint8) * 255, mask_res)
        masks.append(m)
        bbox.append(sz)
        az.append(ref["azim"] % 360.0)
        el.append(ref["elev"])
    return dict(masks=np.stack(masks), bbox=np.asarray(bbox, float),
                az=np.asarray(az, float), el=np.asarray(el, float),
                radius=float(obj.references()[0]["radius"]),
                focal=float(obj.references()[0]["K"][0, 0]),
                res=int(obj.references()[0]["mask"].shape[1]),
                target=obj.target)


def build_splat_codebook(obj, means, az_step=5.0, elev_range=(15.0, 40.0),
                         elev_step=5.0, radius=None, focal=None, res=256,
                         mask_res=160):
    radius = radius if radius is not None else obj.queries()[0].params[0]
    focal = focal if focal is not None else obj.queries()[0].K[0, 0] * res / 512
    azims = np.arange(0.0, 360.0, az_step)
    elevs = np.arange(elev_range[0], elev_range[1] + 1e-9, elev_step)
    masks, bbox, az, el = [], [], [], []
    for e in elevs:
        for a in azims:
            sil = _splat_silhouette(means, pose_from_azel(radius, a, e, obj.target),
                                    focal, res)
            m, sz = normalise_mask(sil.astype(np.uint8) * 255, mask_res)
            masks.append(m)
            bbox.append(sz)
            az.append(a)
            el.append(e)
    return dict(masks=np.stack(masks), bbox=np.asarray(bbox, float),
                az=np.asarray(az, float), el=np.asarray(el, float),
                radius=float(radius), focal=float(focal), res=res,
                target=obj.target)


def cumulative(cb):
    flat = cb["masks"].reshape(len(cb["azim"] if "azim" in cb else cb["az"]), -1)
    return None


def score_all(cb, qmask, query_focal, W, mask_res=160, topk=3):
    qm, qsize = normalise_mask((qmask > 0.5).astype(np.uint8) * 255, mask_res)
    q = qm.reshape(-1).astype(np.float32)
    qa = float(q.sum())
    qd = distance_to_mask(qm.astype(bool)).reshape(-1)
    flat = cb["masks"].reshape(cb["masks"].shape[0], -1).astype(np.float32)
    area = flat.sum(1)
    dist = np.stack([distance_to_mask(m.astype(bool)).reshape(-1)
                     for m in cb["masks"]])
    score = (dist @ q) / max(qa, 1e-6) + (flat @ qd) / np.maximum(area, 1e-6)
    order = np.argsort(score)
    hyps = []
    for i in order[:topk]:
        r = cb["radius"] * (cb["bbox"][i] / max(qsize, 1e-6)) * (query_focal / cb["focal"])
        hyps.append((int(i), r, float(score[i])))
    return hyps, order


def main(root=r"C:\Users\xiang\Desktop\第三个工作\异常检测\shape_1095"):
    obj = ShadObject(root)
    queries = obj.queries()
    means = load_means(obj.ply_path)
    ref_cb = build_ref_codebook(obj)
    splat_cb = build_splat_codebook(obj, means)

    f_true = queries[0].K[0, 0]
    f_ref = ref_cb["focal"]
    print(f"query focal (calibrated) = {f_true:.1f}   reference focal = {f_ref:.1f}")
    print(f"ref codebook: {ref_cb['az'].size} templates, elev={ref_cb['el'][0]:.0f}, "
          f"radius={ref_cb['radius']:.1f}")
    print(f"splat codebook: {splat_cb['az'].size} templates, "
          f"elev={sorted(set(splat_cb['el']))}, radius={splat_cb['radius']:.1f}")

    def azim_diff(a, b):
        d = abs((a - b) % 360.0)
        return min(d, 360.0 - d)

    hdr = (f"{'query':40s} {'true_az/el':>10s} | "
           f"{'init_rot(WRONG f)':>17s} {'r_init':>6s} | "
           f"{'init_rot(REF cb)':>15s} {'r_init':>6s} | "
           f"{'init_rot(DENSE cb)':>17s} {'r_init':>6s}")
    print("\n" + hdr)
    rows = []
    for qs in queries:
        r_lab, az_lab, el_lab = qs.params
        true_az = (qs.azim_sign * az_lab) % 360.0
        RT_gt = pose_from_azel(r_lab, qs.azim_sign * az_lab, el_lab, obj.target)
        qmask = qs.mask()

        # (a) what the old, uncalibrated run did: assume the reference focal
        ha, _ = score_all(ref_cb, qmask, f_ref * 512.0 / float(512), 512)
        besta = min(rotation_error_deg(
            pose_from_azel(r, ref_cb["az"][i], ref_cb["el"][i], obj.target)[:3, :3],
            RT_gt[:3, :3]) for i, r, _ in ha)
        # (b) correct focal, reference codebook
        hb, _ = score_all(ref_cb, qmask, f_true, 512, topk=3)
        bestb = min(rotation_error_deg(
            pose_from_azel(r, ref_cb["az"][i], ref_cb["el"][i], obj.target)[:3, :3],
            RT_gt[:3, :3]) for i, r, _ in hb)
        rb = [r for _, r, _ in hb]
        # (c) correct focal, dense splat codebook
        hc, orderc = score_all(splat_cb, qmask, f_true, 512, topk=3)
        bestc = min(rotation_error_deg(
            pose_from_azel(r, splat_cb["az"][i], splat_cb["el"][i], obj.target)[:3, :3],
            RT_gt[:3, :3]) for i, r, _ in hc)
        rc = [r for _, r, _ in hc]
        rows.append((qs.name, besta, ha[0][1], bestb, rb[0], bestc, rc[0]))
        print(f"{qs.name[:40]:40s} {true_az:5.0f}/{el_lab:<4.0f} | "
              f"{besta:17.1f} {ha[0][1]:6.2f} | "
              f"{bestb:15.1f} {rb[0]:6.2f} | "
              f"{bestc:17.1f} {rc[0]:6.2f}")

    col = lambda i: np.array([r[i] for r in rows])
    print(f"\ninitial rotation error of the BEST top-3 hypothesis (degrees):")
    print(f"  uncalibrated focal + ref cb : median={np.median(col(1)):6.2f}  "
          f"r_init median={np.median(col(2)):.2f}")
    print(f"  calibrated focal  + ref cb  : median={np.median(col(3)):6.2f}  "
          f"r_init median={np.median(col(4)):.2f}  "
          f">20deg:{int((col(3) > 20).sum())}/{len(rows)}")
    print(f"  calibrated focal  + dense cb: median={np.median(col(5)):6.2f}  "
          f"r_init median={np.median(col(6)):.2f}  "
          f">20deg:{int((col(5) > 20).sum())}/{len(rows)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
