#!/usr/bin/env bash
# Helper script for Step 2 synthetic calibration (p=200 slice).
# Usage:
#   ./scripts/manual/run_calibration_p200.sh
# Optional: set WORKERS to override default worker count (defaults to 10).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKERS="${WORKERS:-10}"

cd "${REPO_ROOT}"

PYTHONPATH=src:. python -u experiments/synthetic/calibrate_thresholds.py \
  --workers "${WORKERS}" \
  --p-assets 200 \
  --n-groups 252 \
  --trials-null 60 \
  --trials-alt 60 \
  --delta-abs-grid 0.5 \
  --delta-frac-grid 0.015 0.02 \
  --stability-grid 0.35 \
  --out reports/rc-20251104/calibration/thresholds_p200.json

echo ""
echo "Calibration complete. Monitor progress during execution with:"
echo "  tail -f reports/rc-20251104/calibration/thresholds_p200.json"
