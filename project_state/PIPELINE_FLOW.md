# Pipeline Flow

## Weekly Equity Panel (experiments/equity_panel/run.py)
- **Entry**: `python experiments/equity_panel/run.py --config <yaml> [overrides]` (also via `make rc`, `make rc-lite`, `make rc-lite-sanity`).
- **Steps**:
  1) Load config YAML (smoke/nested/crisis/rc/ablation/gallery) and CLI overrides.
  2) Load daily returns (`finance.io.load_returns_csv` or prices→returns), optional winsorize/huber clip.
  3) Build balanced Week×Day panel (`data.panels.build_balanced_weekday_panel`) with partial-week policy; optional precompute/cache manifest.
  4) Prewhiten (FF5+MOM or off) using `experiments.prewhiten.apply_prewhitening`; record telemetry.
  5) For each rolling window (window_weeks × horizon_weeks), optionally resume cached stats (`meta.cache` keyed by code signature/design/nested replicates). Compute mean squares (`fjs.balanced` or `balanced_nested`), estimate Cs, MP edge(s).
  6) Detect spikes via `fjs.dealias.dealias_search` (oneway/nested/DoW/vol designs) with robust edge modes (SCM/Tyler/Huber), delta/eps/eta guardrails, off-component leak caps. Optionally apply calibrated δ_frac from `calibration/edge_delta_thresholds.json`.
  7) Gate detections (isolation, stability/alignment/energy thresholds, q_max/q2 alignment), optionally soft top-k.
  8) Build overlay covariance by substituting detections into shrinkage/factor baselines (LW/OAS/CC/RIE/factor/poet/tyler) and compare to aliased baseline via EW + min-var (box/long-only) portfolios; compute ΔMSE, VaR/ES, DM tests; plot spectra/edges (E1–E4) and spike time-series.
  9) Persist per-window CSVs (`rolling_results`, `detection_summary`, `diagnostics`, `diagnostics_detail`), summary.json, figures, panel manifest, run_meta.json. Crisis slices optionally restricted by date window.
  10) Optional ablation grid (`--ablations` / config.ablation.smoke.yaml) writes ablation_summary.csv and plots.
- **Outputs**: run directory under `experiments/equity_panel/outputs*` or `reports/rc-YYYYMMDD/*`; figures under `figures/rc/...`; memo/brief/gallery via tools.

## Daily Overlay Evaluation (experiments/eval/run.py)
- **Entry**: `python experiments/eval/run.py --returns-csv data/returns_daily.csv --config experiments/eval/config.yaml [overrides]` (wrapped by `make rc-lite`, `scripts/manual/run_daily_rc_smoke.sh`).
- **Steps**:
  1) Resolve EvalConfig (defaults + thresholds.json + YAML + CLI) and load daily panel (winsor + min history); optional factor prewhitening (FF5+MOM/custom/off) using registry or fallback proxy.
  2) Group dates by design (`week`, `dow`, `dow_vol`, `dow_month`, pure vol state) using `experiments.daily.grouping` with min replicate constraints.
  3) Apply NaN policies (asset/row drop), balance per group (`eval.clean`, `eval.balance`), enforce asset caps (`assets_top`), compute EW volatility proxy.
  4) For each rolling window: build sample covariance, detect spikes via `fjs.overlay.detect_spikes` (with calibrated/strict gating), build overlay covariance (baseline shrinker), compute EW and min-var forecasts, ΔMSE, VaR/ES backtests, DM/sign tests; collect diagnostics (edge margins, isolation, alignment) and flip-set DM.
  5) Split metrics by regime (full/calm/crisis based on volatility quantiles), write CSVs (metrics/risk/dm/diagnostics/diagnostics_detail), plots (delta_mse, flip_dm, histograms), resolved_config.json, prewhiten_diagnostics.
- **Outputs**: `reports/rc-YYYYMMDD/{dow-*,vol-*}/` or custom out_dir with regime CSVs + plots.

## Synthetic Calibration (experiments/synthetic)
- **Null/Power harness**: `experiments/synthetic/null.py` and `power.py` simulate MANOVA scores across edge modes; write `null_scores.parquet`, `power_scores.parquet`, ROC curves (`roc_null.png`, `roc_power.png`).
- **Threshold sweep**: `experiments/synthetic/calibrate_thresholds.py` (or `make sweep:acceptance`, `make calibrate-thresholds`) runs grids over delta/energy/stability, shards via `tools/shard_grid.py`, consolidates with `tools/reduce_calibration.py` into `calibration_defaults.json` and `calibration/edge_delta_thresholds.json` (+ figures under `reports/figures/`).
- **Threshold evaluation**: `synthetic/threshold_eval.py` compares calibrated thresholds against stored score tables.

## Synthetic Benchmarks (synthetic_oneway)
- `python experiments/synthetic_oneway/run.py` (or `make run-synth`) executes S1/S3/S4/S5 synthetic scenarios, writing figures (`figures/synthetic/`) and `summary.json`.

## Reporting / RC packaging
- `make gallery` / `make rc` / `make rc-lite` invoke `tools/build_gallery.py`, `build_memo.py`, `build_brief.py`, `tools/make_summary.py` with gallery/rc YAML configs to produce tables/plots in `figures/rc/`, memos/briefs in `reports/`, and merged regime/manifest CSVs.
- `tools/summarize_run.py` and `tools/prewhiten_effect.py` provide quick textual/CSV diagnostics; `tools/clean_outputs.py` archives/purges legacy outputs.

## Data ingestion
- `scripts/data/fetch_wrds_crsp.py` / `fetch_sharadar.py` → raw exports under `data/wrds/`.
- `scripts/data/make_weekly.py` / `make_balanced_weekly.py` build weekly panels.
- `tools/update_registry.py` / `tools/verify_dataset.py` refresh and validate dataset hashes in registries.

## AWS/Hetzner execution wrappers
- `scripts/aws_run.sh <target>` syncs code/data to EC2, runs Make target with telemetry to `reports/aws/<run_id>/`.
- Exec mode: `EXEC_MODE={deterministic,throughput}` controls thread caps via `meta.runtime` in all heavy runners.
