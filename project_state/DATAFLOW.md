# Dataflow

## Directories & artifacts
- `data/returns_daily.csv` — Wide daily returns (WRDS CRSP) validated by `data/registry.json` (sha256 `96ac7d…3197` per README).
- `data/factors/ff5mom_daily.csv` — FF5+MOM factors (sha256 `469d…908ca`), registry in `data/factors/registry.json`.
- `data/wrds/` — raw WRDS parquet exports (ignored by git) including labels for DoW/vol-state.
- `reports/rc-YYYYMMDD/` — RC drops: metrics/risk/dm/diagnostics CSVs, regime.csv, summary.json, memo/brief, manifests.
- `figures/rc/` — Gallery plots (edge hist, detection, DM p-values, ablations).
- `reports/synthetic/` & `reports/figures/` — Null/power score tables, ROC curves, calibration defaults.
- `experiments/equity_panel/outputs*` — Smoke/nested/crisis run outputs (rolling_results, detection_summary, plots).
- `ablations/` — Ablation tables (CSV) produced by ablate runner.
- Caches: `.cache/` for per-window stats and MP edge cache (`MP_EDGE_CACHE_DIR` env override or `.cache/mp_edges`).

## Input expectations
- Returns CSVs: tidy (`date,ticker,ret`) or wide matrices. Loaders validate registry hashes; duplicates dropped.
- Factor CSVs: expected columns `MKT,SMB,HML,RMW,CMA,MOM` when available; fallback to market proxy if absent.
- Balanced panels: Week×Day cubes require equal replicates per week/year; partial-week policy `drop` (default) or `impute`.

## Transformations
1. **Preprocessing**: winsorize/huber clip (optional), drop partial weeks or impute, intersect tickers across kept weeks/days.
2. **Prewhitening**: regress returns on factors (FF5/FF5+MOM/custom) → residuals, betas, fitted, R² telemetry; factors scaled to decimals (not %).
3. **Balancing**: `data.panels.build_balanced_weekday_panel` or `eval.balance/build_balanced_window` enforce equal replicates and fixed universe; manifests record hashes and preprocessing flags.
4. **Mean squares**: compute MS1/MS2 (/MS3 for nested) and covariance components (`Sigma1_hat`, etc.). Cs plug-ins estimated with top-eigen trimming.
5. **Detection**: for each window, derive MP edge (SCM/Tyler/Huber), apply delta/delta_frac buffers, t-vector isolation, stability perturbations, off-component leakage caps. Optional calibrated δ_frac lookup by (edge_mode, p×t).
6. **Overlay**: substitute de-aliased spikes into baseline covariance (RIE/LW/OAS/CC/factor/POET/Tyler) capped by q_max/gate; PSD adjustments ensure numerical stability.
7. **Evaluation**: compute portfolio forecasts (EW, min-var box/long-only), realised risk over hold-out weeks, ΔMSE, VaR/ES violations, DM/sign tests, alignment diagnostics.
8. **Aggregation & reporting**: rolling_results + detection_summary feed metrics_summary, summary.json, gallery tables/plots; memos/briefs pull from figures/rc and run manifests.

## Environment / path assumptions
- `EXEC_MODE` sets thread caps; `MP_CACHE_DIR` or `MP_EDGE_CACHE_DIR` enables MP edge caching.
- Registries must be current; `tools/update_registry.py` recomputes hashes/row counts after data refreshes.
- WRDS credentials live outside repo (`.pgpass`, env vars) and must not be printed; scripts use `io/wrds_connect.py` / `scripts/secrets/setup_wrds_keychain.sh`.
