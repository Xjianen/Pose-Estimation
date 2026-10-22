#!/usr/bin/env python3
"""Aggregate accuracy across all per-shape results.json produced by run_batch.sh.

Each shape ran as its own `python train.py --root ...` process, so its metrics
live in OUTDIR/<shape>/results.json. This walks them all and reports the dataset-
wide rotation/translation pass rates, medians, and the shapes that failed or
missed the threshold.

Usage:
    python aggregate_results.py /data/xje/datasets/brokenchairs180k/runs
    python aggregate_results.py /data/xje/datasets/brokenchairs180k/runs --top-errors 20
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", help="directory containing <shape>/results.json")
    ap.add_argument("--rot-target", type=float, default=10.0)
    ap.add_argument("--trans-target", type=float, default=0.1)
    ap.add_argument("--top-errors", type=int, default=10,
                    help="print this many worst queries by rotation error")
    ap.add_argument("--save", default=None,
                    help="write a merged all_results.json with every per-query "
                         "rot/trans error + dataset summary to this path")
    args = ap.parse_args()

    rows = []
    all_records = []
    failed_shapes = []
    n_json = 0
    for js in sorted(glob.glob(os.path.join(args.runs_dir, "*", "results.json"))):
        n_json += 1
        try:
            d = json.load(open(js))
        except Exception as e:
            failed_shapes.append((os.path.basename(os.path.dirname(js)),
                                  f"results.json unreadable: {e}"))
            continue
        for rec in d.get("per_query", []):
            if "rot_deg" in rec and "trans" in rec:
                rows.append({"shape": rec["shape"], "query": rec["query"],
                             "rot": rec["rot_deg"], "trans": rec["trans"]})
                all_records.append({
                    "shape": rec["shape"], "query": rec["query"],
                    "rot_deg": rec["rot_deg"], "trans": rec["trans"],
                    "sel": rec.get("sel"), "coarse_ms": rec.get("coarse_ms"),
                    "refine_s": rec.get("refine_s"),
                    "iters": rec.get("iters_used"),
                    "query_focal": rec.get("query_focal"),
                    "radius_scale": rec.get("radius_scale"),
                    "ok": bool(rec["rot_deg"] < args.rot_target
                               and rec["trans"] < args.trans_target)})
        for f in d.get("failures", []):
            failed_shapes.append((f.get("shape", "?"),
                                  f"{f.get('stage','?')}: {f.get('error','?')}"))

    if not rows:
        print(f"no per-query rows found under {args.runs_dir} "
              f"({n_json} results.json seen)", file=sys.stderr)
        sys.exit(1)

    rots = np.array([r["rot"] for r in rows])
    trans = np.array([r["trans"] for r in rows])
    rot_ok = rots < args.rot_target
    trans_ok = trans < args.trans_target
    both = rot_ok & trans_ok

    print(f"shapes with results.json : {n_json}")
    print(f"per-query rows           : {len(rows)}")
    print(f"failed/incomplete shapes : {len(failed_shapes)}")
    if failed_shapes:
        for name, err in failed_shapes[:20]:
            print(f"  {name}: {err}")
        if len(failed_shapes) > 20:
            print(f"  ... and {len(failed_shapes)-20} more")
    print()
    print(f"rotation  (target < {args.rot_target} deg): "
          f"pass {int(rot_ok.sum())}/{len(rows)} "
          f"({100*rot_ok.mean():.1f}%)  "
          f"median={np.median(rots):.2f} mean={rots.mean():.2f} max={rots.max():.2f}")
    print(f"translation (target < {args.trans_target}): "
          f"pass {int(trans_ok.sum())}/{len(rows)} "
          f"({100*trans_ok.mean():.1f}%)  "
          f"median={np.median(trans):.3f} mean={trans.mean():.3f} max={trans.max():.3f}")
    print(f"BOTH pass                : "
          f"{int(both.sum())}/{len(rows)} ({100*both.mean():.1f}%)")

    # per-shape BOTH pass rate (quick health view)
    from collections import defaultdict
    per_shape = defaultdict(lambda: [0, 0])
    for r, ok in zip(rows, both):
        per_shape[r["shape"]][1] += 1
        per_shape[r["shape"]][0] += int(ok)
    shape_rates = sorted(per_shape.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
    print(f"\nworst 10 shapes by BOTH pass rate:")
    for name, (ok, tot) in shape_rates[:10]:
        print(f"  {name}: {ok}/{tot} ({100*ok/max(tot,1):.0f}%)")

    print(f"\ntop {args.top_errors} queries by rotation error:")
    worst = sorted(rows, key=lambda r: r["rot"], reverse=True)[:args.top_errors]
    for r in worst:
        print(f"  {r['shape']}/{r['query'][:40]:40s} rot={r['rot']:7.2f} trans={r['trans']:.3f}")

    if args.save:
        out = {
            "runs_dir": args.runs_dir,
            "config": {"rot_target": args.rot_target,
                       "trans_target": args.trans_target},
            "summary": {
                "n_shapes": n_json, "n_queries": len(rows),
                "n_failed_shapes": len(failed_shapes),
                "rot_pass": int(rot_ok.sum()), "rot_pass_rate": float(rot_ok.mean()),
                "trans_pass": int(trans_ok.sum()), "trans_pass_rate": float(trans_ok.mean()),
                "both_pass": int(both.sum()), "both_pass_rate": float(both.mean()),
                "rot_median": float(np.median(rots)), "rot_mean": float(rots.mean()),
                "rot_max": float(rots.max()),
                "trans_median": float(np.median(trans)), "trans_mean": float(trans.mean()),
                "trans_max": float(trans.max())},
            "per_query": all_records,
            "failed_shapes": [{"shape": n, "error": e} for n, e in failed_shapes],
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nmerged results written to: {args.save}")


if __name__ == "__main__":
    main()
