# Runbook

## Smoke vs. full equity runs

- Smoke run (≈5 min, 6+1 weeks, 120 assets):
  ```bash
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py \
      --config experiments/equity_panel/config.smoke.yaml \
      --no-progress --workers 4 --assets-top 120 --stride-windows 3 \
      --resume --cache-dir .cache --precompute-panel --drop-partial-weeks
  ```
- Full run (≈90 min, 156+4 weeks): remove the stride/top overrides.

## New estimator switches

- `--estimator {aliased,dealias,lw,oas,cc,factor}` tags the run and cache entries.
- `--factor-csv tests/data/factors_tiny.csv` enables the observed-factor covariance; the CSV must be wide, date-indexed.
- Shrinkage benchmarks (LW, OAS, constant-correlation) and the factor estimator all appear in `metrics_summary.csv`; Diebold–Mariano columns (`dm_stat_*`, `dm_p_*`) compare them to the de-aliased baseline.

## Min-variance regularisation and turnover

- `--minvar-ridge 1e-3` adds λ²I to the min-var covariance; tweak with `--minvar-box lo,hi` (defaults 0,0.05).
- Turnover diagnostics (`*_turnover`, `*_turnover_cost`) are written per window.
- Apply transaction costs via `--turnover-cost 25` (basis points).

## Summaries

- `python tools/summarize_run.py <run_dir>` prints estimator MSE breakdowns plus DM p-values.
- `tools/list_runs.py` remains the index of archived runs.
