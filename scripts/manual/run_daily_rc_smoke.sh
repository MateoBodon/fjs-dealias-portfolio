#!/usr/bin/env bash
# Launch DoW/vol-state daily RC smoke runs with calibrated gating telemetry.
#
# Usage:
#   bash scripts/manual/run_daily_rc_smoke.sh
#   PARALLEL=1 bash scripts/manual/run_daily_rc_smoke.sh   # run both in parallel
#
# Progress tips:
#   tail -f reports/rc-20251104/dow-rc-smoke/diagnostics.csv
#   tail -f reports/rc-20251104/dow-rc-smoke/regime.csv
#   tail -f reports/rc-20251104/vol-rc-smoke/diagnostics.csv
#   tail -f reports/rc-20251104/vol-rc-smoke/regime.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH=src:.

RC_DATE="20251104"
CALIB_PATH="reports/rc-${RC_DATE}/calibration/thresholds.json"
if [[ ! -f "${CALIB_PATH}" ]]; then
  echo "Calibration file ${CALIB_PATH} not found. Run calibration first." >&2
  exit 1
fi

COMMON=(
  python -u experiments/daily/run.py
  --returns-csv data/returns_daily.csv
  --window 126
  --horizon 21
  --start 2015-01-01
  --end 2022-12-31
  --gate-delta-calibration "${CALIB_PATH}"
  --gate-delta-frac-min 0.02
)

run_cmd() {
  local label=$1
  shift
  echo "[$(date '+%H:%M:%S')] launching ${label} ..."
  "$@" && echo "[$(date '+%H:%M:%S')] ${label} complete."
}

DOW_CMD=(
  "${COMMON[@]}"
  --group dow
  --shrinker rie
  --prewhiten ff5mom
  --gate-mode strict
  --gate-stability-min 0.3
  --out "reports/rc-${RC_DATE}/dow-rc-smoke/"
)

VOL_CMD=(
  "${COMMON[@]}"
  --group vol
  --shrinker oas
  --prewhiten off
  --gate-mode soft
  --gate-soft-max 2
  --gate-accept-nonisolated
  --out "reports/rc-${RC_DATE}/vol-rc-smoke/"
)

if [[ "${PARALLEL:-0}" == "1" ]]; then
  run_cmd "dow-rc-smoke" "${DOW_CMD[@]}" &
  PID1=$!
  run_cmd "vol-rc-smoke" "${VOL_CMD[@]}" &
  PID2=$!
  wait "${PID1}"
  wait "${PID2}"
else
  run_cmd "dow-rc-smoke" "${DOW_CMD[@]}"
  run_cmd "vol-rc-smoke" "${VOL_CMD[@]}"
fi
