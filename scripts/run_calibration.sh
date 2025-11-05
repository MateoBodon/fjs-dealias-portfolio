#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="src"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

CPUS=${1:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}

python3 experiments/synthetic/calibrate_thresholds.py \
  --p-assets 64 80 96 \
  --n-groups 36 \
  --replicates 14 20 \
  --alpha 0.015 \
  --delta-abs-grid 0.35 0.5 \
  --trials-null 40 \
  --trials-alt 40 \
  --edge-modes scm tyler \
  --replicate-bins 12-16 17-22 \
  --asset-bins 64-96 \
  --q-max 2 \
  --workers "$CPUS" \
  --verbose \
  --out calibration/edge_delta_thresholds.json
