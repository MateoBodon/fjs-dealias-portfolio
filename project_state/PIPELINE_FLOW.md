# Pipeline Flow

## rc-lite-sanity (Make target)
- **Entry**: `make rc-lite-sanity` (uses `RC_*` env defaults in Makefile).
- **Steps**:
  1) Verify dataset/factor registries.
  2) Run *daily* eval twice on 2023-01-01→2023-06-30, assets_top=50: DoW (tyler, rie shrinker) and vol-state (tyler, oas). Gate soft with calibrated δ_frac (min 0.02 DoW / 0.015 vol), q_max=2, prewhiten FF5+MOM.
  3) Run *weekly* smoke: DoW (6-week window, 1-week horizon, 2023Q1 slice) and nested (52×1 window, 2022–2023H1) with calibrated gating on tyler edges, prewhiten FF5+MOM, cached panel.
  4) Build summary (`tools/make_summary.py`) and rc-lite-sanity digest (`tools/summarize_rc_sanity.py`).
- **Outputs**: `reports/rc-<date>-sanity-<stamp>/` (daily metrics/dm/diagnostics + kill_criteria/limitations), `experiments/equity_panel/outputs_rc-lite-<date>_<stamp>/` (weekly smoke artifacts), cached stats in `.cache/rc-lite`.

## Weekly Equity Panel (experiments/equity_panel/run.py)
- **Entry**: `python experiments/equity_panel/run.py --config <yaml> [overrides]` (via `make rc`, `make rc-lite`, or rc-lite-sanity step 3).
- **Flow**:
  1) Parse YAML (smoke/nested/crisis/rc/ablation/gallery) + CLI overrides; apply exec-mode thread caps.
  2) Load returns (prices→returns or CSV), optional winsor/huber; build balanced Week×Day panel (`data.panels`, `finance.loader`), optionally cached/manifested.
  3) Prewhiten via `experiments.prewhiten.apply_prewhitening` (FF5+MOM/custom/off) and log telemetry.
  4) For each rolling window (window_weeks × horizon_weeks, stride): optionally resume cached stats; compute mean squares (`fjs.balanced`/`balanced_nested`), Cs, MP edge(s) (SCM/Tyler/Huber).
  5) Detect spikes (`fjs.dealias`) with guardrails (δ/δ_frac/ε/η, off-component leak, energy floor, optional θ root find); optionally calibrated δ_frac from `calibration/edge_delta_thresholds.json`.
  6) Gate detections (isolation, stability/alignment, q_max/q2, soft top-k) and substitute into baseline covariance (LW/OAS/CC/RIE/factor/POET/Tyler). Compute portfolio metrics (EW, min-var box/LO), ΔMSE, VaR/ES, DM/sign tests; plot spectra/edges (E1–E4), spike time-series.
  7) Persist rolling_results, detection_summary, diagnostics(+_detail), metrics_summary, summary.json, figures, panel_manifest, run_meta.json. Ablations (`--ablations`) emit ablation_summary.csv.
- **Outputs**: `experiments/equity_panel/outputs*` (smoke/crisis/ablation/rc-lite-sanity) or `reports/rc-YYYYMMDD/`; figures under `figures/rc/...`.

## Daily Overlay Evaluation (experiments/eval/run.py)
- **Entry**: `python experiments/eval/run.py --returns-csv data/returns_daily.csv --config experiments/eval/config.yaml [overrides]` (used in rc-lite-sanity, rc-lite, rc-dow/rc-vol).
- **Flow**:
  1) Resolve EvalConfig (defaults + thresholds.json + YAML + CLI); load daily panel (winsor bounds, min_history). Prewhiten (FF5+MOM/custom/off) via registry/fallback; telemetry recorded.
  2) Group dates (`week`, `dow`, `dow_vol`, `dow_month`, `vol`, `dowxvol`) using `experiments.daily.grouping` with min-count/min-replicate guards.
  3) Apply NaN policy (`eval.clean`), balance per group (`eval.balance`), cap assets (`assets_top`), compute EW vol proxy.
  4) Per rolling window: sample covariance → `fjs.overlay.detect_spikes` (calibrated/strict gating, coarse candidate) → baseline shrinker → `apply_overlay`. Compute EW + min-var forecasts, ΔMSE/QLIKE, VaR/ES coverage, DM/sign tests, flip-set DM; collect diagnostics (edge margins, isolation, alignment, reason codes).
  5) Split by regimes (full/calm/crisis); write metrics/risk/dm/diagnostics/diagnostics_detail CSVs, plots (delta_mse, flip_dm, histograms), resolved_config.json, prewhiten diagnostics.
- **Outputs**: `reports/rc-YYYYMMDD/<design-edge>/` or custom out_dir; rc-lite-sanity outputs include kill_criteria/limitations under `summary/`.

## Synthetic Calibration (experiments/synthetic)
- **Null/Power harness**: `experiments/synthetic/null.py`, `power.py` simulate score tables; `power_null.py` caches calibration shards/meta.
- **Threshold sweep**: `experiments/synthetic/calibrate_thresholds.py` (or `make sweep:acceptance`) sweeps δ_abs/δ_frac/η/energy over edge modes and p×t grids; shard via `tools/shard_grid.py`; consolidate with `tools/reduce_calibration.py` into `calibration_defaults.json` and `calibration/edge_delta_thresholds.json`; plots under `reports/figures/`.
- **Threshold evaluation**: `synthetic/threshold_eval.py` benchmarks calibrated thresholds against stored null/power scores.

## Synthetic Benchmarks (synthetic_oneway)
- **Entry**: `python experiments/synthetic_oneway/run.py` (or `make run-synth`).
- **Flow**: simulate S1/S3/S4/S5 scenarios (bias/recall/guardrail analysis), plot histograms/heatmaps, write `summary.json` and figures under `figures/synthetic/`.

## Ablation & Sensitivity
- `experiments/ablate/run.py` — YAML-driven sweeps (ablation_matrix*.yaml) over overlay hyperparams; optional calm/crisis sampling; emits `ablation_summary.csv`.
- `experiments/eval/sensitivity.py` — grid over gate mode, δ_frac, alignment cos, stability η; runs eval CLI per combo; writes heatmaps/tables under `reports/rc-sensitivity/`.
- `experiments/eval/inject_spike.py` — inject synthetic spikes into eval windows to measure recall/FP; writes plots under `reports/figures/`.

## Reporting / Packaging
- `make gallery` / `make memo` / `make brief` — call `tools/build_gallery.py`, `build_memo.py`, `build_brief.py` with `experiments/equity_panel/config.rc.yaml`; produce tables/plots in `figures/rc/`, memos/briefs in `reports/`.
- `tools/make_summary.py` — aggregate RC directories into summary/kill_criteria CSVs.
- `tools/summarize_run.py`, `tools/prewhiten_effect.py`, `tools/summarize_rc_sanity.py` — lightweight textual/CSV digests for individual runs and rc-lite-sanity batches.

## Data ingestion
- `scripts/data/fetch_wrds_crsp.py` / `fetch_sharadar.py` → raw exports under `data/wrds/`.
- `scripts/data/make_weekly.py` / `make_balanced_weekly.py` → balanced weekly panels (`returns_balanced_weekly.parquet`).
- Registries refreshed via `tools/update_registry.py` / `tools/verify_dataset.py`.

## AWS/Hetzner execution wrappers
- `scripts/aws_run.sh <target>` syncs code/data to EC2, runs Make target, writes telemetry under `reports/aws/<run_id>/`.
- `EXEC_MODE={deterministic,throughput}` (via `meta.runtime`) sets BLAS/thread caps across runners; rc-lite-sanity defaults to deterministic caps unless overridden.
