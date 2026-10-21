"""Diagnose: which filename parse rule produced the GT used in the server run?

Re-evaluates the downloaded RT_est poses from results.json against GT built
under two candidate parse rules, using the exact pose math from utils/geometry.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\xiang\Desktop\第三个工作\异常检测\PTAD_plus")
from utils.geometry import pose_from_azel, pose_error  # noqa: E402

RUNS = r"C:\Users\xiang\Desktop\第三个工作\异常检测\runs\shape_1095\results.json"
DATA = r"C:\Users\xiang\Desktop\第三个工作\异常检测\shape_1095"
TARGET = np.array([0.003087502426925734, 0.11657597799105449, -0.001777495765155146])
AZIM_SIGN = -1.0


def numbers_in(stem):
    out = []
    for part in stem.split("_"):
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def parse_new(path):
    """Current local rule: first decimal field + its two successors."""
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    for i, part in enumerate(parts[:-2]):
        if "." in part:
            try:
                return tuple(float(v) for v in parts[i:i + 3])
            except ValueError:
                continue
    nums = numbers_in(os.path.splitext(os.path.basename(path))[0])
    return tuple(nums[-3:]) if len(nums) >= 3 else None


def parse_old(path):
    """Old rule: last three numeric fields."""
    nums = numbers_in(os.path.splitext(os.path.basename(path))[0])
    return tuple(nums[-3:]) if len(nums) >= 3 else None


res = json.load(open(RUNS))
rows = res["per_query"]
print(f"{len(rows)} queries in results.json\n")

print(f"{'query':46s} {'newR':>7s} {'newT':>8s}   {'oldR':>7s} {'oldT':>8s}   {'loggedR':>8s} {'loggedT':>9s}")
print("-" * 105)
sum_new = [0.0, 0.0]
sum_old = [0.0, 0.0]
for rec in rows:
    name = rec["query"]
    RT_est = np.asarray(rec["RT_est"], dtype=np.float64)
    p_new, p_old = parse_new(name), parse_old(name)
    if p_new is None or p_old is None:
        print(f"{name:46s}  parse failed: new={p_new} old={p_old}")
        continue
    gt_new = pose_from_azel(p_new[0], AZIM_SIGN * p_new[1], p_new[2], TARGET)
    gt_old = pose_from_azel(p_old[0], AZIM_SIGN * p_old[1], p_old[2], TARGET)
    rn, tn = pose_error(RT_est, gt_new)
    ro, to = pose_error(RT_est, gt_old)
    sum_new[0] += rn; sum_new[1] += tn
    sum_old[0] += ro; sum_old[1] += to
    print(f"{name[:46]:46s} {rn:7.2f} {tn:8.3f}   {ro:7.2f} {to:8.3f}   {rec.get('rot_deg', float('nan')):8.2f} {rec.get('trans', float('nan')):9.3f}")

n = len(rows)
print("-" * 105)
print(f"mean under NEW parse (first-decimal rule): rot={sum_new[0]/n:6.2f}  trans={sum_new[1]/n:7.3f}")
print(f"mean under OLD parse (last-three rule)   : rot={sum_old[0]/n:6.2f}  trans={sum_old[1]/n:7.3f}")

# Also verify reference convention locally: rebuild ref RT from (r, az, el) vs json RT
print("\nreference RT check (local mv_images jsons):")
worst = 0.0
import glob
for js in sorted(glob.glob(os.path.join(DATA, "mv_images", "*.json"))):
    stem = os.path.splitext(os.path.basename(js))[0]
    p = parse_new(stem + ".png")          # parse expects a path with extension
    d = json.load(open(js))
    RT = pose_from_azel(p[0], p[1], p[2], TARGET)  # refs: filename azim already matches RT
    gt = np.asarray(d["RT"])
    worst = max(worst, np.abs(RT - gt).max())
print(f"  {len(glob.glob(os.path.join(DATA, 'mv_images', '*.json')))} refs, worst |RT rebuild - json RT| = {worst:.2e}")

# --- azim sign test for queries: which sign fits the estimated poses better? ---
print("\nazim-sign test on queries (GT from new parse, sign +1 vs -1):")
for sign in (+1.0, -1.0):
    rots, trans = [], []
    for rec in rows:
        name = rec["query"]
        RT_est = np.asarray(rec["RT_est"], dtype=np.float64)
        r, az, el = parse_new(name)
        gt = pose_from_azel(r, sign * az, el, TARGET)
        rr, tt = pose_error(RT_est, gt)
        rots.append(rr); trans.append(tt)
    rots, trans = np.asarray(rots), np.asarray(trans)
    print(f"  sign={sign:+.0f}: rot median={np.median(rots):7.2f} mean={rots.mean():7.2f} | "
          f"trans median={np.median(trans):6.3f} mean={trans.mean():6.3f} | "
          f"pass(<10deg,<0.1)={int(((rots<10)&(trans<0.1)).sum())}/{len(rots)}")

# --- elev sign test too (maybe elevation counts the other way here) ---
print("\nelev-sign / azim-sign grid on queries:")
best = None
for s_az in (+1.0, -1.0):
    for s_el in (+1.0, -1.0):
        rots, trans = [], []
        for rec in rows:
            name = rec["query"]
            RT_est = np.asarray(rec["RT_est"], dtype=np.float64)
            r, az, el = parse_new(name)
            gt = pose_from_azel(r, s_az * az, s_el * el, TARGET)
            rr, tt = pose_error(RT_est, gt)
            rots.append(rr); trans.append(tt)
        rots, trans = np.asarray(rots), np.asarray(trans)
        n_pass = int(((rots < 10) & (trans < 0.1)).sum())
        line = (f"  az={s_az:+.0f} el={s_el:+.0f}: rot med={np.median(rots):7.2f} "
                f"trans med={np.median(trans):6.3f} pass={n_pass}/{len(rots)}")
        print(line)
        if best is None or np.median(rots) + np.median(trans) < best[0]:
            best = (np.median(rots) + np.median(trans), line)
print(f"\n  best: {best[1]}")
