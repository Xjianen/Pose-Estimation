#!/bin/bash
# Batch-render reference_camera (colored depth + depth.npy) and masks for every
# shape under brokenchairs180k/shapes. N workers share one GPU; each worker runs
# one shape at a time as an independent python process (--root single shape).
#
# Usage:
#   ./run_batch.sh              # default 8 workers
#   ./run_batch.sh 12           # 12 workers (raise only if nvidia-smi shows headroom)
#
# Resumable: a shape whose reference_camera/<shape>/.done sentinel exists is
# skipped, so re-running picks up where it stopped. To force-rerun one shape,
# delete its .done file first:
#   rm /data/xje/datasets/brokenchairs180k/reference_camera/shape_XXXX/.done
#
# Monitor in another terminal while this runs:
#   watch -n5 nvidia-smi
# If GPU memory stays below ~70%, stop this and rerun with a higher worker count.

set -u

SHAPES=/data/xje/datasets/brokenchairs180k/shapes
REFOUT=/data/xje/datasets/brokenchairs180k/reference_camera
OUTDIR=/data/xje/datasets/brokenchairs180k/runs
WORKERS=${1:-8}

mkdir -p "$REFOUT" "$OUTDIR"
cd "$(dirname "$0")"

# One shape per line; xargs -P runs up to $WORKERS in parallel. Each line is a
# single --root invocation, so a crash on one shape never affects the others and
# the whole batch keeps going.
ls -d "$SHAPES"/shape_* | xargs -P "$WORKERS" -I {} bash -c '
  shape="{}"
  name="$(basename "$shape")"
  ref="'"$REFOUT"'/$name"
  if [ -f "$ref/.done" ]; then
    echo "[skip] $name (done)"; exit 0
  fi
  echo "[$(date +%H:%M:%S)] [start] $name"
  python train.py --root "$shape" \
    --out-dir "'"$OUTDIR"'/$name" \
    --reference-out-dir "'"$REFOUT"'" \
    --lean --skip-if-done
  echo "[$(date +%H:%M:%S)] [done] $name (rc=$?)"
'

echo
echo "============================================================"
echo "batch finished. sentinel-based progress:"
done_n=$(find "$REFOUT" -maxdepth 2 -name .done 2>/dev/null | wc -l)
total_n=$(ls -d "$SHAPES"/shape_* 2>/dev/null | wc -l)
echo "  $done_n / $total_n shapes done"
echo "  reference_camera under: $REFOUT/<shape>/{<query>_depth.png, <query>_depth.npy}"
echo "  per-shape results.json + mask under: $OUTDIR/<shape>/{results.json, mask/}"
