#!/bin/bash
# Batch-render reference_camera (colored depth + depth.npy) and masks for every
# shape under brokenchairs180k/shapes. N workers share one GPU; each worker runs
# one shape at a time as an independent python process (--root single shape).
#
# Uses --codebook-mode refs (PTAD-fair: only the 20 shipped reference views, no
# dense gsplat codebook). Uses GPU 1 (CUDA_VISIBLE_DEVICES=1).
#
# Usage:
#   ./run_batch.sh              # default 8 workers, GPU 1
#   CUDA_VISIBLE_DEVICES=0 ./run_batch.sh   # override to GPU 0
#   ./run_batch.sh 12           # 12 workers (raise only if nvidia-smi shows headroom)
#
# Runs in FOREGROUND: Ctrl+C stops the whole batch cleanly.
# To run detached instead:  nohup ./run_batch.sh > batch.log 2>&1 &
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

# ---- GPU selection ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TORCH_CUDA_ARCH_LIST="8.0"        # suppress "no CUDA arch list" warning

SHAPES=/data/xje/datasets/brokenchairs180k/shapes
REFOUT=/data/xje/datasets/brokenchairs180k/reference_camera
OUTDIR=/data/xje/datasets/brokenchairs180k/runs
WORKERS=${1:-8}
COUNT=/tmp/ptad_progress.count
OKC=/tmp/ptad_ok.count
FAILC=/tmp/ptad_fail.count
LOCK=/tmp/ptad_progress.lock

mkdir -p "$REFOUT" "$OUTDIR"
cd "$(dirname "$0")"
TOTAL=$(ls -d "$SHAPES"/shape_* 2>/dev/null | wc -l)
echo 0 > "$COUNT"; echo 0 > "$OKC"; echo 0 > "$FAILC"
echo "total shapes: $TOTAL, workers: $WORKERS (each ~1.5-3GB; GPU $CUDA_VISIBLE_DEVICES, codebook=refs)"

# One shape per line -> xargs -P runs up to $WORKERS in parallel. Each line is a
# single --root --quiet invocation, so a crash on one shape never affects the
# others and the whole batch keeps going. flock keeps the progress counter
# atomic across parallel workers.
worker() {
  local shape="$1" name ref status rc
  name="$(basename "$shape")"
  ref="$REFOUT/$name"
  if [ -f "$ref/.done" ]; then
    status="(skip)"
    rc=0
  else
    if python train.py --root "$shape" --out-dir "$OUTDIR/$name" \
        --reference-out-dir "$REFOUT" --codebook-mode refs \
        --lean --skip-if-done --quiet; then
      status="ok"
      rc=0
    else
      status="FAILED"
      rc=1
    fi
  fi
  local n
  (
    flock 9
    n=$(cat "$COUNT"); echo $((n+1)) > "$COUNT"
    if [ "$rc" -eq 0 ]; then
      ok=$(cat "$OKC"); echo $((ok+1)) > "$OKC"
    else
      fail=$(cat "$FAILC"); echo $((fail+1)) > "$FAILC"
    fi
  ) 9>"$LOCK"
  n=$(cat "$COUNT")
  echo "[$n/$TOTAL] $name $status"
}
export -f worker
export REFOUT OUTDIR COUNT OKC FAILC LOCK TOTAL

ls -d "$SHAPES"/shape_* | xargs -P "$WORKERS" -I {} bash -c 'worker "$@"' _ {}

echo
echo "============================================================"
ok=$(cat "$OKC"); fail=$(cat "$FAILC")
echo "shapes: $ok ok / $fail failed / $TOTAL total"
echo "aggregating per-query accuracy across all shapes..."
python aggregate_results.py "$OUTDIR" --save "$OUTDIR/all_results.json" --top-errors 20
echo
done_n=$(find "$REFOUT" -maxdepth 2 -name .done 2>/dev/null | wc -l)
echo "  $done_n / $TOTAL reference_camera dirs complete"
echo "  merged rot/trans errors: $OUTDIR/all_results.json"
echo "  reference_camera under: $REFOUT/<shape>/{<query>_depth.png, <query>_depth.npy}"
echo "  per-shape results.json + mask under: $OUTDIR/<shape>/{results.json, mask/}"
