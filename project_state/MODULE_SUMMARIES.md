# Module Summaries

## src/fjs (MANOVA core)
- `balanced.py` — Balanced one-way MANOVA utilities: input validation, group means, mean square estimators (`mean_squares`), `BalancedConfig`, placeholder `compute_balanced_weights`.
- `balanced_nested.py` — Nested Year⊃Week mean squares with `NestedDesignMetadata` (d, c, N, order, replicates) and label validation.
- `mp.py` — Marčenko–Pastur machinery: edge/root finders (`mp_edge`, `m_edge`, `marchenko_pastur_edges`), admissible root (`admissible_m_from_lambda`), t-vector (`t_vec`), Cs plug-in estimation (`estimate_Cs_from_MS`), cache controls (`configure_mp_cache`, `clear_mp_cache`), z′/z″ utilities, PDF stub.
- `dealias.py` — Algorithm 1 spike search & substitution (`dealias_search`, `dealias_covariance`); helper design defaults, input validation, admissible root checks, angular grid/rotation, merge duplicate detections, design params TypedDicts, result dataclass.
- `gating.py` — Post-detection gating: outlier counting, top-k selection, calibrated δ_frac lookup (`lookup_calibrated_delta`), scoring helpers.
- `overlay.py` — Overlay wrapper: `OverlayConfig` dataclass, coarse candidates, baseline covariance chooser, `detect_spikes`, gating (strict/soft), calibrated/relative delta handling, `apply_overlay` (PSD-guarded substitution).
- `robust.py` — Tyler/Huber scatter estimators and MP edge scaling helper (`edge_from_scatter`).
- `spectra.py` — Eigen diagnostics: `topk_eigh`, `project_alignment`, spectrum plots (`plot_spectrum_with_edges`), spike time-series plots, spectrum estimation.
- `theta_solver.py` — `ThetaSolverParams` + `solve_theta_for_t2_zero` root finder for k=2 designs with stability probes.

## src/finance (covariance, portfolios, IO)
- `design.py` — Build combined return+factor design matrices; week grouping labels.
- `eval.py` — Rolling windows; risk metrics; `oos_variance_forecast`, `weekly_cov_from_components`, `variance_forecast_from_components`, `evaluate_portfolio`.
- `factors.py` — Observed-factor covariance via cross-sectional OLS; design alignment; industry factor handling.
- `ledoit.py` — Ledoit–Wolf shrinkage wrapper with PSD checks.
- `portfolio.py` — PGD min-var with ridge/box (`minvar_ridge_box`), turnover and cost utilities, memoised penalised covariances.
- `portfolios.py` — cvxpy-based min-var solvers (box/long-only); fail-loud on missing cvxpy, optional skip flag (no EW fallback).
- `returns.py` — Log-return computation; weekly aggregation; balanced Week×Day construction (`balance_weeks`, `weekly_panel`).
- `robust.py` — Winsorize/huberize returns; Tyler shrinkage covariance.
- `shrinkage.py` — OAS and constant-correlation shrinkage with PSD/finite guards.
- `io.py` — Price/return CSV loaders with registry validation, long→wide conversion.
- `loader.py` — Balanced weekly panel loader from daily prices; rolling windows with fixed universe enforcement.

## src/baselines (shrinkers & prewhitening)
- `covariance.py` — Sample covariance; LW/OAS/CC shrinkage; RIE, QuEST clipping, EWMA covariance with PSD symmetrisation.
- `factors.py` — Load observed factors (FF5/MOM, market proxy); `PrewhitenResult`; prewhiten returns; percent-scale detection.

## src/data (registries/panels)
- `panels.py` — `PanelManifest`, `BalancedPanel`, hash utilities, balanced Week×Day builder, save/load manifest/panel pickles.
- `registry.py` — Dataset registry loader/validator with SHA256 checking (`assert_registered_dataset`).
- `factors.py` — Factor registry loader/validator (`load_registered_factors`), SHA checks, timestamp coercion.

## src/eval (helpers for daily eval)
- `clean.py` — NaN filtering policy with `NaNPolicyTelemetry/Result`.
- `balance.py` — Balance per-group replicates (`BalanceResult/Telemetry`), asset intersection controls.

## src/evaluation (metrics & diagnostics)
- `dm.py` — Diebold–Mariano test with Newey–West variance.
- `evaluate.py` — ΔMSE/QLIKE/VaR/ES metrics, sign test, bootstrap CI, coverage tests, alignment diagnostics, plotting helpers, `DeltaSummary`, metrics summary aggregation.
- `factor.py` — Observed-factor covariance, POET-lite estimator with IC selection (`POETResult`).

## src/meta (runtime/cache/metadata)
- `cache.py` — `window_key`, save/load window payloads (JSON + NPZ).
- `run_meta.py` — `RunMeta` dataclass; `code_signature` hashing; Git SHA helper; PDF hash collector; detection counting; `write_run_meta`.
- `runtime.py` — `ExecModeSettings`, thread-cap application, exec-mode resolution, worker count helpers, metadata snapshot.

## src/report & plotting
- `report.gather` — Run loader, run discovery, detection/edge extraction, estimator panel aggregation.
- `report.tables` — Estimator/rejection/ablation table writers (CSV/MD/TeX).
- `report.plots` — Detection rates, edge histograms, alignment angles, DM bars, ablation heatmaps.
- `plotting.utils` — Figure roots, E1–E4 plot builders, guardrail plots.

## src/synthetic
- `calibration.py` — Calibration sweep config/result dataclasses, panel simulation, seed batching, threshold selection/writing.
- `threshold_eval.py` — Evaluate calibrated thresholds vs score tables; `DetectionArrays`.

## src/io
- `crsp_daily.py` — CRSP snapshot fetch with rowcount probe, cleaning, DoW/vol labels, parquet writer.
- `wrds_connect.py` — WRDS connection helper (credential-aware).

## src/utils
- `credentials.py` — WRDS credential stubs; keychain lookup placeholder.

## experiments (pipelines & helpers)
- `equity_panel/run.py` — Weekly MANOVA runner: config parsing/defaults, data loading/balancing, prewhitening, caching (`meta.cache`), detection (`dealias_search`), gating (calibrated/fixed), overlay baseline selection, portfolio evaluation, plots (E1–E4), ablation support, crisis slices, run metadata.
- `equity_panel/sweep_acceptance.py` — Small synthetic sweep near acceptance thresholds.
- Configs (`config*.yaml`) — smoke, crisis, nested, rc, rc-lite/gallery, ablation, acceptance, calibration.
- `eval/run.py` — Daily overlay runner: config resolution, grouping (week/DoW/vol/dow_month/dowxvol), NaN/balance policies, prewhitening, overlay detection, risk metrics, diagnostics/plots, resolved config echo.
- `eval/config.py` / `diagnostics.py` / `inject_spike.py` / `sensitivity.py` — Config deep-merge & thresholds loader, diagnostic enums, spike injection harness, gating sensitivity sweeps.
- `daily/grouping.py` / `config.py` / `run.py` — Grouping utilities (week, DoW, vol-state, DoW×vol, DoW×month) with error reporting and CLI wrapper.
- `synthetic/null.py`, `power.py`, `power_null.py`, `calibrate_thresholds.py`, `harness_utils.py` — Synthetic ROC sweeps, score simulation, calibration CLI, cache/meta helpers, plotting and default writer.
- `synthetic_oneway/run.py` — Synthetic S1/S3/S4/S5 benchmarks, bias/recall plots, summary JSON, multi-spike support.
- `ablate/run.py` — YAML-driven parameter sweep; E5-style summary extraction; supports calm/crisis sampling.
- `etf_panel/run.py` — ETF demo atop daily eval defaults, writes overlay_toggle markdown.
- `prewhiten.py` — Prewhitening selection, telemetry computation, diagnostics writers (`PrewhitenTelemetry`).

## tools (reporting, maintenance, monitoring)
- `build_gallery.py`, `build_memo.py`, `build_brief.py` — Gallery/ memo/brief generation from YAML manifests and run dirs.
- `make_summary.py` — Cross-run summary (detection/perf/kill criteria) with `SummaryArtifacts`.
- `summarize_run.py`, `summarize_rc_sanity.py` — Text/CSV summaries for run dirs and rc-lite-sanity batch.
- `aggregate_runs.py`, `list_runs.py` — Run discovery and aggregation.
- `prewhiten_effect.py` — Compare prewhiten on/off runs.
- `reduce_calibration.py`, `shard_grid.py` — Calibration sharding/reduction.
- `run_monitor.py` — Progress/metrics tailer with resource telemetry.
- `clean_outputs.py` — Targeted clean/archival of legacy outputs.
- `update_registry.py`, `verify_dataset.py` — Dataset/factor registry refresh and validation.
- `plot_rc_hist.py` — RC histogram plot helper.

## scripts
- `scripts/data/*.py` — WRDS/Sharadar ingestion, balanced weekly builders, return summaries.
- `scripts/manual/*.sh|*.py` — Calibration and RC smoke wrappers, merge calibration shards.
- `scripts/aws_run.sh` / `scripts/aws_provision.sh` — AWS dispatch/provision helpers (micromamba + telemetry).
- `scripts/bench_linalg.py` — BLAS sizing probe.
- `scripts/secrets/setup_wrds_keychain.sh` — WRDS credential setup (no secrets committed).
