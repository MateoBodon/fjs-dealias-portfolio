# AGENTS.md — fjs-dealias-portfolio
_Last updated: 2025-11-02_

## Sprint Focus
- Daily evaluation with replicated groups, observed-factor prewhitening, and overlay diagnostics.
- ΔMSE/VaR/ES reporting split by calm/crisis regimes plus detection telemetry (edge margin, isolation share, stability).
- ETF alt-panel demo reusing the daily harness; refreshed RC artefacts under `reports/rc-20251103/` including an overlay toggle note.

## Core Commands
- Prewhiten + evaluate daily panel:
  ```bash
  python experiments/eval/run.py \
      --returns-csv <returns.csv> \
      --window 126 --horizon 21 \
      --shrinker rie \
      --out reports/rc-YYYYMMDD/
  ```
  Adds `metrics.csv`, `risk.csv`, `dm.csv`, `diagnostics.csv`, and `delta_mse.png` per regime (full/calm/crisis).
- ETF demo (countries/sectors):
  ```bash
  python experiments/etf_panel/run.py \
      --returns-csv data/etf_returns.csv \
      --out reports/etf-rc/
  ```
- Unit tests:
  ```bash
  pytest tests/baselines/test_prewhiten.py tests/fjs/test_overlay.py tests/experiments/test_eval_run.py
  ```

## Implementation Notes
- `src/baselines/factors.py` supplies `load_observed_factors` with FF5+MOM lookup and MKT proxy fallback plus `prewhiten_returns` returning residuals, betas, intercepts, and R².
- `src/fjs/overlay.py` defines `OverlayConfig`, `detect_spikes`, and `apply_overlay`; only accepted detections swap eigenvalues, every other direction shrinks via `shrinker` (default RIE). Deterministic gating uses `seed` and fixed `a_grid`.
- `experiments/eval/run.py` pulls daily returns via `load_daily_panel` (with fallback for wide CSVs), runs prewhitening, rolls calm/crisis splits from an EWMA vol proxy, and writes diagnostics/plots. Overlay toggle commentary lives in `reports/<run>/overlay_toggle.md`.
- `experiments/etf_panel/run.py` is a thin CLI wrapper around the daily evaluation defaults.

## RC Refresh
- Sample RC artefacts generated with the new harness live in `reports/rc-20251103/` (full/calm/crisis CSVs + `overlay_toggle.md`).
- Memo/gallery targets (`make rc`, `make gallery`) remain under `experiments/equity_panel`; run after updating detection tables to propagate overlay notes into the memo.
