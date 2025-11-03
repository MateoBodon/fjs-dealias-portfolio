# RUNBOOK — Next RC Reproduction
_Last updated: 2025-11-03_

## 0. Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export PYTHONPATH=src
```

## 1. Daily Panels with Replicates
Run the Day-of-Week (`dow`) and volatility-state (`vol`) replicates using the daily harness.
```bash
python experiments/daily/run.py \
    --returns-csv data/daily_returns.csv \
    --design dow \
    --window 126 --horizon 21 \
    --shrinker rie \
    --edge-mode tyler \
    --out reports/rc-$(date +%Y%m%d)/dow

python experiments/daily/run.py \
    --returns-csv data/daily_returns.csv \
    --design vol \
    --window 126 --horizon 21 \
    --shrinker rie \
    --edge-mode huber \
    --out reports/rc-$(date +%Y%m%d)/vol
```
Both commands write `metrics.csv`, `risk.csv`, `dm.csv`, `diagnostics.csv`, and `delta_mse.png` for full/calm/crisis windows plus `overlay_toggle.md`.

## 2. Overlay Baseline Comparisons
Re-run with EWMA and QuEST shrinkage while toggling observed-factor prewhitening.
```bash
python experiments/daily/run.py \
    --returns-csv data/daily_returns.csv \
    --design dow \
    --shrinker quest \
    --prewhiten ff5_mom \
    --out reports/rc-$(date +%Y%m%d)/dow_quest

python experiments/daily/run.py \
    --returns-csv data/daily_returns.csv \
    --design vol \
    --shrinker ewma \
    --ewma-halflife 30 \
    --overlay-off \
    --out reports/rc-$(date +%Y%m%d)/vol_ewma
```

## 3. Calibration Sweep
Synthetic null/power calibration at matched dimensionality.
```bash
python experiments/synthetic/calibrate_thresholds.py \
    --p 120 --n 180 \
    --fpr-target 0.015 \
    --sims 2000 \
    --out calibration/thresholds.json
```
Inspect the report: `python tools/show_thresholds.py calibration/thresholds.json`.

## 4. Crisis/Calm Evaluation Bundle
Aggregate replicates and baselines into a single RC drop.
```bash
python tools/collect_rc.py \
    --rc-date $(date +%Y%m%d) \
    --runs reports/rc-$(date +%Y%m%d)/dow reports/rc-$(date +%Y%m%d)/vol \
    --include reports/rc-$(date +%Y%m%d)/dow_quest reports/rc-$(date +%Y%m%d)/vol_ewma \
    --overlay calibration/thresholds.json
```
Outputs live under `reports/rc-$(date +%Y%m%d)/` with `memo.md`, `gallery/`, `overlay_toggle.md`, and telemetry CSVs.

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
pytest tests/baselines/test_prewhiten.py \
       tests/baselines/test_rie.py \
       tests/baselines/test_ewma.py \
       tests/fjs/test_overlay.py \
       tests/experiments/test_eval_run.py \
       tests/experiments/test_daily_run.py \
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
