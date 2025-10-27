# Runbook

## Smoke vs. full equity runs

- Smoke run (≈5 min, 6+1 weeks, 80 assets):
  ```bash
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/equity_panel/run.py \
      --config experiments/equity_panel/config.smoke.yaml \
      --no-progress --workers $(python -c 'import os;print(os.cpu_count() or 4)') \
      --assets-top 80 --stride-windows 4 --resume --cache-dir .cache \
      --precompute-panel --drop-partial-weeks --estimator oas
  ```
  Artifacts land in `experiments/equity_panel/outputs_smoke/<design>_J*_solver-*_est-*_prep-*/`.
- Full run (≈90 min, 156+4 weeks): drop the stride/top overrides and let `--estimator dealias`.

## Test suites

- `make test-fast` → `pytest -m unit` (deterministic micro-tests; keep data to ≤12 assets, ≤6 weeks).
- `make test-integration` → `pytest -m integration` (smoke-sized multi-module flows).
- `make test-slow` → `pytest -m slow` (long synthetic studies, optional locally).
- `make test-all` → union of unit + integration (default CI footprint). Mark multi-minute jobs as `heavy`.

## New estimator switches

- `--estimator {aliased,dealias,lw,oas,cc,factor,tyler_shrink}` tags the run, cache entries, and `run_meta`.
- `--factor-csv tests/data/factors_tiny.csv` enables the observed-factor covariance; the CSV must be wide, date-indexed.
- Shrinkage benchmarks (LW, OAS, constant-correlation, Tyler) and the factor estimator all appear in `metrics_summary.csv`; Diebold–Mariano columns (`dm_stat_*`, `dm_p_*`) compare them to the de-aliased baseline.

## Robust preprocessing toggles

- `--winsorize q` applies column-wise clipping to the empirical `[q, 1-q]` quantiles (mutually exclusive with Huber).
- `--huber c` clips at `median ± c·MAD` per column; use when thin tails help guard ablations.
- Preprocess selections become part of cache keys, panel manifests, artifact directories, and `run_meta.preprocess_flags`.
- Pair with `--estimator tyler_shrink` to use the Tyler–ridge covariance in evaluation and DM tables.

## Min-variance regularisation and turnover

- `--minvar-ridge 1e-3` adds λ²I to the min-var covariance; tweak with `--minvar-box lo,hi` (defaults 0,0.05).
- Turnover diagnostics (`*_turnover`, `*_turnover_cost`) are written per window.
- Apply transaction costs via `--turnover-cost 25` (basis points).

## Summaries

- `python tools/summarize_run.py <run_dir>` prints estimator MSE breakdowns plus DM p-values.
- `tools/list_runs.py` remains the index of archived runs.

## Clean outputs

- Inspect pending moves: `python tools/clean_outputs.py --dry-run`
- Delete legacy artifacts once verified: `python tools/clean_outputs.py --purge`

## Build gallery

- Build tables/plots for configured runs: `make gallery`
- Inspect generated assets: `ls figures/smoke/<run_tag>/{tables,plots}`
