# RUNBOOK — Next RC Reproduction
_Last updated: 2025-11-04_

## 0. Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export PYTHONPATH=src
```

## 1. Daily Panels with Replicates
Launch the Day-of-Week (`dow`) and volatility-state (`vol`) smoke slices with calibrated gating and regime telemetry.
```bash
# sequential (recommended)
bash scripts/manual/run_daily_rc_smoke.sh

# optional: run both slices in parallel
PARALLEL=1 bash scripts/manual/run_daily_rc_smoke.sh

# monitor progress
tail -f reports/rc-$(date +%Y%m%d)/dow-rc-smoke/regime.csv
tail -f reports/rc-$(date +%Y%m%d)/vol-rc-smoke/regime.csv
```
Outputs land in `reports/rc-$(date +%Y%m%d)/{dow,vol}-rc-smoke/` with `full/`, `calm/`, `crisis/` folders (`metrics.csv`, `risk.csv`, `dm.csv`, `diagnostics.csv`, `diagnostics_detail.csv`, `delta_mse.png`) plus `overlay_toggle.md`, `regime.csv`, `prewhiten_*`, and enriched gating telemetry.

## 2. Overlay Baseline Comparisons
Repeat `scripts/manual/run_daily_rc_smoke.sh` with desired overrides:
- `SHRINKER=quest`, `PREWHITEN=ff5mom` for RIE/QuEST comparisons.
- `SHRINKER=ewma`, `PREWHITEN=off`, `GATE_MODE=strict|soft` to sweep gating knobs.
Document the exact command variants that ship in the bundle.

## 3. Calibration Sweep
Generate calibrated δ_frac lookups for p∈{100,200}, n=252, then merge.
```bash
# ~4 min on M3 Pro
PYTHONPATH=src:. python experiments/synthetic/calibrate_thresholds.py \
  --workers 8 \
  --p-assets 100 \
  --n-groups 252 \
  --trials-null 60 --trials-alt 60 \
  --delta-abs-grid 0.5 \
  --delta-frac-grid 0.015 0.02 \
  --stability-grid 0.35 \
  --out reports/rc-$(date +%Y%m%d)/calibration/thresholds_p100.json

# second slice (p=200)
PYTHONPATH=src:. python experiments/synthetic/calibrate_thresholds.py \
  --workers 10 \
  --p-assets 200 \
  --n-groups 252 \
  --trials-null 60 --trials-alt 60 \
  --delta-abs-grid 0.5 \
  --delta-frac-grid 0.015 0.02 \
  --stability-grid 0.35 \
  --out reports/rc-$(date +%Y%m%d)/calibration/thresholds_p200.json


# merge + ROC
python scripts/manual/merge_calibration_thresholds.py \
  --inputs \
  reports/rc-$(date +%Y%m%d)/calibration/thresholds_p100.json \
  reports/rc-$(date +%Y%m%d)/calibration/thresholds_p200.json \
  --out reports/rc-$(date +%Y%m%d)/calibration/thresholds.json
```
Artifacts: combined `thresholds.json` + `roc.png` with FPR/Power scatter.

## 4. Crisis/Calm Evaluation Bundle
```bash
python tools/collect_rc.py \
  --rc-date $(date +%Y%m%d) \
  --runs \
    reports/rc-$(date +%Y%m%d)/dow-rc-smoke \
    reports/rc-$(date +%Y%m%d)/vol-rc-smoke \
  --overlay reports/rc-$(date +%Y%m%d)/calibration/thresholds.json
```
Outputs: `memo.md`, `overlay_toggle.md`, gallery assets, `regime.csv`, and telemetry CSVs in `reports/rc-$(date +%Y%m%d)/`.

## 5. ETF Alt-Panel Demo
```bash
python experiments/etf_panel/run.py \
  --returns-csv data/etf_returns.csv \
  --window 126 --horizon 21 \
  --shrinker rie \
  --out reports/etf-rc/
```

## 6. Testing & Smoke Targets
```bash
pytest \
  tests/baselines/test_covariance.py \
  tests/baselines/test_load_factors.py \
  tests/fjs/test_overlay.py \
  tests/experiments/test_eval_run.py \
  tests/experiments/test_daily_grouping.py \
  tests/synthetic/test_calibration.py

make smoke-daily
```

## 7. Memo & Gallery Refresh
```bash
make rc RC_DATE=$(date +%Y%m%d)
make gallery RC_DATE=$(date +%Y%m%d)
```
Review deliverables:
```bash
less reports/rc-$(date +%Y%m%d)/memo.md
open figures/rc-$(date +%Y%m%d)/index.html
```

## 8. Packaging
```bash
git status
pytest -q
make smoke-daily
```
If clean and green, tag the RC:
```bash
VERSION=$(date +%Y%m%d)
git tag -a rc-${VERSION} -m "RC ${VERSION}"
git push origin rc-${VERSION}
```
