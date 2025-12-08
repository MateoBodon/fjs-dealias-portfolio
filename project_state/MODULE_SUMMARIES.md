# Module Summaries

## src/fjs
- `balanced.py` — Balanced one-way MANOVA utilities: input validation, group means, mean square estimators (`mean_squares`), balanced config dataclass, placeholder balanced weight solver.
- `balanced_nested.py` — Nested Year⊃Week MANOVA mean squares, metadata capture (d,c,N,I,J,replicates), validation of balanced hierarchical labels.
- `mp.py` — Closed-form Marčenko–Pastur transforms: edge/root finders (`mp_edge`, `m_edge`, `admissible_m_from_lambda`), t-vector evaluation (`t_vec`), Cs plug-in estimation, in-memory/on-disk MP edge cache, derivatives z′/z″ utilities.
- `dealias.py` — Algorithm 1 spike search and covariance substitution: grid of a-vectors, MP edge buffering, t-vector filtering, stability checks, off-component leakage caps, optional Cs sensitivity bands, theta root finder integration, merging of near-duplicate detections, `dealias_covariance` replacer.
- `gating.py` — Post-detection gates: isolate outlier count, rank top-k detections, lookup calibrated δ_frac from JSON tables.
- `overlay.py` — Overlay wrapper: config dataclass, optional coarse candidates, shrinkage/factor baseline selection, spike detection (`detect_spikes`), gating (strict/soft), delta calibration resolution, and substitution (`apply_overlay`).
- `robust.py` — Tyler and Huber scatter estimators plus MP edge scaling helper `edge_from_scatter`.
- `spectra.py` — Eigen diagnostics: top-k eigensolver, projection alignment, spectrum/edge plots, spike time-series plots, spectrum sorter.
- `theta_solver.py` — For k=2 designs, root-find θ such that off-component t2=0 with stability checks.

## src/finance
- `design.py` — Build combined return+factor design matrices; week grouping labels.
- `eval.py` — Rolling window generator, risk metrics, variance forecast wrappers for multiple estimators (dealias, LW/OAS/CC, factor, Tyler, factor_obs, POET), weekly covariance reconstruction from MANOVA components, portfolio evaluation, alignment of detections.
- `factors.py` — Observed-factor covariance via cross-sectional OLS; design alignment; handles industry factors.
- `ledoit.py` — Ledoit–Wolf shrinkage wrapper with PSD checks.
- `portfolio.py` — Projected-gradient min-var with ridge/box constraints (`minvar_ridge_box`), turnover utilities, memoised penalised covariances.
- `portfolios.py` — cvxpy-based min-var solvers (long-only/box) with equal-weight fallback.
- `returns.py` — Log-return computation; weekly aggregation; balanced Week×Day construction with `balance_weeks`.
- `robust.py` — Winsorize/huberize returns; Tyler M-estimator with ridge.
- `shrinkage.py` — OAS and constant-correlation shrinkage implementations plus helpers for PSD/finite handling.
- `io.py` — CSV loaders for prices/returns with registry validation; conversion to wide matrices.
- `loader.py` — Balanced weekly panel loader from daily prices, rolling windows with fixed universe enforcement.

## src/baselines
- `covariance.py` — Convenience wrappers: sample covariance, LW/OAS/CC shrinkage, RIE, QuEST clipping, EWMA covariance.
- `factors.py` — Load observed factors (FF5/MOM), construct market proxy, prewhiten returns; data classes for prewhiten result.

## src/data
- `panels.py` — Build balanced Week×Day panels, compute/manipulate manifests, hashing utilities, save/load panel pickles.
- `registry.py` — Dataset registry loader/validator with SHA256 checking and exceptions.
- `factors.py` — Factor registry validation/loading with optional env override; returns entry metadata.

## src/eval (clean/balance helpers used by experiments.eval)
- `clean.py` — NaN filtering policy for per-window panels, telemetry capture.
- `balance.py` — Enforce balanced replicates per group with optional asset intersection; reasons/telemetry included.

## src/evaluation
- `dm.py` — Diebold–Mariano test with Newey–West long-run variance.
- `evaluate.py` — ΔMSE/QLIKE/VaR/ES metrics, sign test, block-bootstrap CI, coverage tests, alignment diagnostics, plotting helpers, metrics summary aggregation.
- `factor.py` — Observed-factor covariance and POET-lite estimator with IC selection.

## src/meta
- `cache.py` — Hash key generator for per-window cache, save/load split between JSON scalars and NPZ arrays.
- `run_meta.py` — Run metadata dataclass; code signature hash; helper to write `run_meta.json` including detection counts and figure hashes.
- `runtime.py` — Execution mode resolution (deterministic/throughput), thread caps application, worker scaling.

## src/report / plotting
- `report.gather` — Load run artifacts, discover tagged runs, aggregate estimator panels, DM stats extraction.
- `report.tables` — Table builders for estimators, rejections, ablations (CSV/MD/TeX outputs).
- `report.plots` — Plot generators (detection, DM p-values, isolation bars, stability scatter, histograms, ablation heatmaps).
- `plotting.utils` — Shared plot styling and save helpers.

## src/synthetic
- `calibration.py` — Threshold calibration helpers for MP edge delta, ROC construction.
- `threshold_eval.py` — Evaluate calibrated thresholds against null/power score tables.

## src/io
- `crsp_daily.py` — CRSP-specific loader utilities (returns panel creation).
- `wrds_connect.py` — WRDS connection helper (credential-aware).

## src/utils
- `credentials.py` — Stub for retrieving secrets (kept minimal to avoid leakage).

## experiments
- `equity_panel/run.py` — Weekly MANOVA experiment driver: config parsing, data loading/balancing, prewhitening, per-window detection (with caching, nested/oneway/DoW/vol designs), overlay gating, portfolio evaluation (EW, min-var), plotting (E1–E4), ablation grid support, crisis slices, run metadata writing.
- `equity_panel/sweep_acceptance.py` — Small synthetic sweep around acceptance thresholds.
- Configs (`config*.yaml`) — smoke, crisis, nested, ablation, gallery, rc presets.
- `eval/run.py` — Daily overlay runner: load daily panel, prewhiten, group by design (week, DoW, vol-state, month), apply overlay per window with risk metrics, per-regime outputs, optional plots, writes resolved_config and diagnostics.
- `eval/config.py`/`diagnostics.py`/`inject_spike.py`/`sensitivity.py` — Eval config resolver, diagnostic reason codes, spike injection for tests, sensitivity utilities.
- `daily/*.py` — Grouping utilities for daily panels (DoW, vol-state), config data classes.
- `synthetic/null.py`, `power.py`, `calibrate_thresholds.py`, `power_null.py`, `harness_utils.py` — Synthetic ROC sweeps, score simulation, threshold calibration CLI, run metadata writing.
- `synthetic_oneway/run.py` — Lightweight synthetic benchmarks (S1/S3/S4/S5 grid) driven by YAML config.
- `ablate/run.py` — Parameter ablation runner reading YAML matrices.
- `etf_panel/run.py` — ETF demo wrapper atop daily evaluation harness.
- `prewhiten.py` — Shared prewhitening selection/telemetry writer for experiments.

## tools
- `build_gallery.py` — Generate gallery tables/plots from run directories (YAML-driven).
- `build_memo.py` — Assemble memo markdown from runs + summary artifacts.
- `build_brief.py` — One-page advisor brief using gallery inputs.
- `make_summary.py` — Cross-run summary tables (detection, performance, kill criteria).
- `summarize_run.py` — Print/summarise run directories (detection, ΔMSE, DM, gallery links).
- `clean_outputs.py` — Move/clean legacy experiment outputs.
- `aggregate_runs.py` — Combine multiple run directories into aggregated CSVs.
- `prewhiten_effect.py` — Compare prewhiten on/off runs for deltas.
- `reduce_calibration.py` — Consolidate calibration shards into threshold JSONs/plots.
- `shard_grid.py` — Shard calibration grids for distributed runs.
- `run_monitor.py` — Tail metrics/progress JSONL during long runs.
- `update_registry.py` / `verify_dataset.py` — Refresh/check dataset registry digests.
- `list_runs.py` — Enumerate available runs with metadata.

## scripts
- `scripts/data/*.py` — Fetch WRDS/Sharadar data, build balanced weekly panels, summarise returns.
- `scripts/run_calibration.sh`, `scripts/manual/run_daily_rc_smoke.sh`, `scripts/manual/merge_calibration_thresholds.py`, `scripts/manual/run_calibration_p200.sh` — Convenience wrappers for sweeps/RC smoke.
- `scripts/aws_run.sh`, `scripts/aws_provision.sh` — AWS dispatch/provision helpers with micromamba + telemetry.
- `scripts/bench_linalg.py` — Quick linalg perf test for BLAS sizing.
- `scripts/secrets/setup_wrds_keychain.sh` — WRDS credential setup (no secrets in repo).
