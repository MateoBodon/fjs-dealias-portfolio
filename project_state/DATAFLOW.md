# Dataflow

## Directories & artifacts
- `data/returns_daily.csv` — WRDS CRSP daily returns (registry hash `96ac7d…3197`).
- `data/factors/ff5mom_daily.csv` — FF5+MOM factors (registry hash `469d…908ca`); registry `data/factors/registry.json`.
- `data/prices_daily.csv`, `data/prices_sample.csv` — price inputs; `data/returns_balanced_weekly.parquet` cached balanced panel; `data/meta/universe_2016_2024.json` universe metadata.
- `data/wrds/` — raw WRDS parquet exports (ignored) incl. DoW/vol labels.
- `experiments/equity_panel/outputs*` — Weekly smoke/crisis/nested/ablation/rc-lite-sanity artifacts (rolling_results, detection_summary, metrics_summary, E1–E4 plots, manifests, run_meta).
- `reports/rc-YYYYMMDD*/` — RC drops and rc-lite-sanity: metrics/risk/dm/diagnostics CSVs, resolved_config, regime.csv, summary/kill_criteria, memo/brief copies.
- `figures/rc/`, `reports/figures/` — Gallery plots, ROC curves, ablation heatmaps, inject-spike/sensitivity figures.
- `reports/synthetic/` — Null/power score tables and calibration outputs.
- `ablations/` — Ablation tables (CSV) from ablate runner.
- Caches: `.cache/mp_edges` (MP cache), `.cache/rc-lite` (rc-lite-sanity weekly cache), general `.cache/` for window payloads.

## Inputs & expectations
- Returns CSVs: tidy (`date,ticker,ret`) or wide; loaders validate registries and drop duplicates.
- Factor CSVs: columns `MKT,SMB,HML,RMW,CMA,MOM`; fall back to market proxy if absent.
- Balanced panels: Week×Day cubes with equal replicates; partial-week policy `drop` (default) or `impute`.
- Grouping metadata: DoW/vol labels computed in `experiments.daily.grouping` or derived from CRSP label files.

## Transformations
1) **Preprocess**: optional winsorize/huber clip; drop/impute partial weeks; intersect tickers; enforce minimum history.
2) **Prewhiten**: regress returns on factors (FF5/MOM/custom) → residuals, betas, R² telemetry; factor columns normalized to decimals.
3) **Balance**: `data.panels.build_balanced_weekday_panel` (weekly) or `eval.balance.build_balanced_window` (daily) enforce replicates/universe; manifests store hashes and flags.
4) **Mean squares**: compute MS1/MS2 (/MS3 nested) and Σ̂ components; Cs plug-ins estimated with top-eigen trimming.
5) **Detection**: per window, derive MP edge (SCM/Tyler/Huber); apply δ/δ_frac buffer, t-vector isolation, stability perturbations, off-component leak caps; optional calibrated δ_frac lookup (edge_mode × p×t).
6) **Overlay**: substitute de-aliased spikes into baseline covariance (RIE/LW/OAS/CC/factor/POET/Tyler) with q_max/gate; PSD safeguards.
7) **Evaluation**: portfolio forecasts (EW, min-var box/LO), realised risk, ΔMSE/QLIKE, VaR/ES coverage, DM/sign tests, alignment diagnostics, flip-set DM.
8) **Aggregation**: rolling_results + detection_summary → metrics_summary/summary.json; daily eval writes regime CSVs + plots; rc-lite-sanity adds kill_criteria/limitations; gallery/memo/brief pull from these artifacts.

## Environment / path assumptions
- `EXEC_MODE` (deterministic/throughput) sets BLAS/OpenMP caps via `meta.runtime`.
- MP cache path via `MP_CACHE_DIR`/`MP_EDGE_CACHE_DIR`; unset uses in-memory cache only.
- Registries must be current; refresh with `tools/update_registry.py` / `verify_dataset.py` after data changes.
- WRDS credentials external (`.pgpass`, env vars); `scripts/secrets/setup_wrds_keychain.sh` assists. Avoid logging SQL/creds.
