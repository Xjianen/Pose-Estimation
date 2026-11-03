#!/usr/bin/env python3
"""Bin per-query rotation and translation errors into a 2D histogram.

Walks every ``<runs_dir>/<shape>/results.json`` (same layout as
``aggregate_results.py``), extracts ``(rot_deg, trans)`` for each query,
and bins them into a 4x4 grid:

  X (rotation, deg) : <=10 | 10-20 | 20-50 | >50
  Y (translation, cm): <=10 | 10-20 | 20-50 | >50

Outputs a CSV table and a heatmap PNG.  All figure labels are ASCII so the
plot renders correctly on headless servers without CJK fonts.

Usage
-----
    python error_histogram.py /data/xje/datasets/brokenchairs180k/runs
    python error_histogram.py /path/to/runs --csv hist.csv --fig hist.png
    python error_histogram.py /path/to/runs --trans-unit m   # keep metres
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Bin definitions
# --------------------------------------------------------------------------- #
ROT_EDGES = [0, 10, 20, 50, np.inf]
ROT_LABELS = ["<=10", "10-20", "20-50", ">50"]

# Translation edges in *metres* (the unit stored in results.json).
#   10 cm = 0.1 m, 20 cm = 0.2 m, 50 cm = 0.5 m
TRANS_EDGES_M = [0, 0.1, 0.2, 0.5, np.inf]
TRANS_LABELS_CM = ["<=10", "10-20", "20-50", ">50"]


def collect_rows(runs_dir):
    """Return list of (rot_deg, trans_m) from every results.json under runs_dir."""
    rows = []
    n_json = 0
    for js in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        n_json += 1
        try:
            d = json.load(open(js))
        except Exception:
            continue
        for rec in d.get("per_query", []):
            if "rot_deg" in rec and "trans" in rec:
                rows.append((float(rec["rot_deg"]), float(rec["trans"])))
    if not rows:
        print(f"no per-query rows found under {runs_dir} "
              f"({n_json} results.json seen)", file=sys.stderr)
        sys.exit(1)
    return rows, n_json


def build_histogram(rots, trans_m):
    """4x4 count matrix, row=trans bin, col=rot bin."""
    rot_idx = np.digitize(rots, ROT_EDGES[1:-1])      # 0..3
    trans_idx = np.digitize(trans_m, TRANS_EDGES_M[1:-1])
    grid = np.zeros((4, 4), dtype=int)
    for r, t in zip(rot_idx, trans_idx):
        grid[t, r] += 1
    return grid


def write_csv(grid, path, trans_unit="cm"):
    """Write the 4x4 table as CSV with row/col headers."""
    factor = 100 if trans_unit == "cm" else 1
    unit_label = "cm" if trans_unit == "cm" else "m"
    col_hdr = "trans_" + unit_label + "\\rot_deg," + ",".join(ROT_LABELS)
    lines = [col_hdr]
    for i, tlab in enumerate(TRANS_LABELS_CM):
        vals = ",".join(str(grid[i, j]) for j in range(4))
        lines.append(f"{tlab},{vals}")
    # append a total row / col for convenience
    lines.append("")
    lines.append("row_totals," + ",".join(str(grid[i].sum()) for i in range(4)))
    lines.append("col_totals," + ",".join(str(grid[:, j].sum()) for j in range(4)))
    lines.append(f"grand_total,{grid.sum()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"CSV  -> {path}")


def plot_heatmap(grid, path, total):
    """Save a colour-coded 4x4 heatmap."""
    import matplotlib
    matplotlib.use("Agg")                       # headless-safe
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    # imshow expects row 0 at top; we want smallest translation at bottom,
    # so flip the grid vertically.
    disp = grid[::-1]
    ylabels = TRANS_LABELS_CM[::-1]

    im = ax.imshow(disp, cmap="YlOrRd", aspect="auto")

    # cell text (count + percentage)
    for i in range(4):
        for j in range(4):
            cnt = disp[i, j]
            pct = 100 * cnt / max(total, 1)
            colour = "white" if cnt > total * 0.15 else "black"
            ax.text(j, i, f"{cnt}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11, color=colour)

    ax.set_xticks(range(4))
    ax.set_xticklabels(ROT_LABELS, fontsize=11)
    ax.set_yticks(range(4))
    ax.set_yticklabels(ylabels, fontsize=11)
    ax.set_xlabel("Rotation Error (deg)", fontsize=12)
    ax.set_ylabel("Translation Error (cm)", fontsize=12)
    ax.set_title(f"Pose Error Distribution (N={total})", fontsize=13)

    # colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Query Count", fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.clear()
    print(f"FIG  -> {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir",
                    help="directory containing <shape>/results.json files")
    ap.add_argument("--csv", default="error_histogram.csv",
                    help="output CSV path (default: error_histogram.csv)")
    ap.add_argument("--fig", default="error_histogram.png",
                    help="output figure path (default: error_histogram.png)")
    args = ap.parse_args()

    rows, n_json = collect_rows(args.runs_dir)
    rots = np.array([r[0] for r in rows])
    trans_m = np.array([r[1] for r in rows])

    total = len(rows)
    grid = build_histogram(rots, trans_m)

    # ---- console summary --------------------------------------------------
    print(f"shapes with results.json : {n_json}")
    print(f"per-query rows           : {total}")
    print()
    header = f"{'trans\\rot':>10s}  " + "  ".join(f"{l:>6s}" for l in ROT_LABELS) + "  |  total"
    print(header)
    print("-" * len(header))
    for i, tlab in enumerate(TRANS_LABELS_CM):
        vals = "  ".join(f"{grid[i, j]:>6d}" for j in range(4))
        print(f"{tlab:>10s}  {vals}  |  {grid[i].sum():>6d}")
    print("-" * len(header))
    col_totals = [str(grid[:, j].sum()) for j in range(4)]
    print(f"{'total':>10s}  " + "  ".join(f"{v:>6s}" for v in col_totals)
          + f"  |  {grid.sum():>6d}")
    print()

    # rotation-only distribution
    print("Rotation distribution:")
    for j, lab in enumerate(ROT_LABELS):
        c = grid[:, j].sum()
        print(f"  {lab:>6s} deg : {c:>6d}  ({100*c/total:.1f}%)")
    print()
    # translation-only distribution
    print("Translation distribution:")
    for i, lab in enumerate(TRANS_LABELS_CM):
        c = grid[i].sum()
        print(f"  {lab:>6s} cm  : {c:>6d}  ({100*c/total:.1f}%)")
    print()

    # ---- files ------------------------------------------------------------
    write_csv(grid, args.csv)
    plot_heatmap(grid, args.fig, total)


if __name__ == "__main__":
    main()
