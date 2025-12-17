# Experiments

## rc-lite-sanity (daily + weekly)
- **Script/Target**: `make rc-lite-sanity` (wraps `experiments/eval/run.py` twice + two weekly smoke runs).
- **Configs**: Makefile env knobs (RC_DOW_GROUP_MIN/REPS, RC_VOL_GROUP_MIN/REPS, RC_GATE_DELTA_FRAC_MIN{_VOL}, RC_Q_MAX, RC_VOL alignment, assets_top=50, window=60, horizon=10).
- **Purpose**: Fast health check combining daily DoW + vol-state eval (2023H1 slice) and weekly DoW/nested smoke windows.
- **Outputs**: `reports/rc-<date>-sanity-<stamp>/` (metrics/risk/dm/diagnostics, delta_mse.png, flip_dm.png, kill_criteria.json, limitations.md, regime.csv) and `experiments/equity_panel/outputs_rc-lite-<date>_<stamp>/` (weekly smoke artifacts).

## Daily Overlay Evaluation
- **Script**: `experiments/eval/run.py` (Make targets: `rc-dow`, `rc-vol`, `rc-week`, `rc-dowxvol`, `rc-lite`, `rc-lite-sanity`).
- **Configs**: `experiments/eval/config.yaml`, `experiments/eval/thresholds.json`, CLI flags (`--group-design`, `--window`, `--horizon`, `--assets-top`, `--edge-mode`, `--shrinker`, `--gate-mode`, `--gate-delta-frac-min/max`, `--q-max`, `--q2-alignment-min-cos`, `--overlay-delta`, `--coarse-candidate`, `--prewhiten`, `--use-factor-prewhiten`, `--factors-csv`).
- **Purpose**: Rolling daily overlay eval by design (week/DoW/vol/dow_month/dowxvol), producing ΔMSE/QLIKE, VaR/ES, DM/sign tests, flip-set diagnostics.
- **Outputs**: Per-regime CSVs (metrics, risk, dm, diagnostics, diagnostics_detail), plots (delta_mse, flip_dm, histograms), resolved_config.json, prewhiten_diagnostics under `reports/rc-YYYYMMDD/<design-edge>/` or custom out_dir.

## Weekly Equity Panel (MANOVA)
- **Script**: `experiments/equity_panel/run.py` (Make targets: `rc-lite`, `rc`, `rc-lite-sanity` weekly component, `rc-data`, `rc-ablations`).
- **Configs**: `config.smoke.yaml`, `config.nested.smoke.yaml`, `config.crisis.*.yaml`, `config.rc.yaml`, `config.gallery.yaml`, `config.ablation.smoke.yaml`.
- **Purpose**: Weekly MANOVA detection on balanced panels (oneway/DoW/vol/nested) with overlay on shrinkage/factor baselines; produces E1–E4 plots, metrics_summary, detection diagnostics.
- **Outputs**: Run directories under `experiments/equity_panel/outputs*` or `reports/rc-YYYYMMDD/` containing rolling_results.csv, detection_summary.csv, diagnostics*.csv, metrics_summary.csv, summary.json, run_meta.json, E1–E4 figures, panel manifest.

## Synthetic Calibration / ROC
- **Scripts**: `experiments/synthetic/null.py`, `experiments/synthetic/power.py`, `experiments/synthetic/calibrate_thresholds.py`, `experiments/synthetic/power_null.py`.
- **Configs/CLI**: grids over `--delta-abs-grid`, `--delta-frac-grid`, `--stability-grid`, `--energy-floor-grid`, `--edge-modes`, `--p-assets`, `--n-groups`, `--replicates`, `--trials-null/alt`; sharding via `tools/shard_grid.py`; reduction via `tools/reduce_calibration.py`.
- **Purpose**: Calibrate δ/η/energy thresholds for MP edge gating; produce ROC curves and defaults JSON.
- **Outputs**: Score tables (`reports/synthetic/null_harness`, `power_harness`), ROC figures under `reports/figures/`, `calibration_defaults.json`, `calibration/edge_delta_thresholds.json`.

## Nested Synthetic Kill-test
- **Script**: `experiments/synthetic/nested_killtest.py`.
- **Config**: `experiments/synthetic/config.nested.killtest.yaml` (p≈200, years=2, weeks 6–8, reps=5, delta=0.35, delta_frac_min configurable, tyler/huber edge, calibrated lookup).
- **Purpose**: Stress-test nested gating (FPR/Power + skip reasons) on year⊃week structure matched to weekly smoke windows.
- **Outputs**: `reports/synthetic_nested_killtest/{nested_killtest_trials.csv,summary.csv,summary.md,run.json}`.

## Synthetic Benchmarks (one-way)
- **Script**: `experiments/synthetic_oneway/run.py` (Make: `run-synth`).
- **Configs**: YAML at `experiments/synthetic_oneway/config.yaml` (S1/S3/S4/S5 knobs).
- **Purpose**: Bias/recall/guardrail analysis for S1/S3/S4/S5 scenarios; optional multi-spike simulation.
- **Outputs**: `figures/synthetic/`, `experiments/synthetic_oneway/summary.json`.

## Ablation Grid
- **Script**: `experiments/ablate/run.py` (Make: `rc-ablations`).
- **Configs**: `experiments/ablate/ablation_matrix_tiny.yaml` (tiny grid), `ablation_matrix.yaml`.
- **Purpose**: Sweep overlay hyperparams (delta, eps, eta, q_max, gate mode) on selected slices; supports calm/crisis sampling.
- **Outputs**: `ablations/ablation_matrix.csv`, `experiments/equity_panel/outputs_ablation_smoke/ablation_summary.csv`, heatmaps in gallery.

## Sensitivity / Spike injection
- **Scripts**: `experiments/eval/sensitivity.py` (gate/δ/η grids); `experiments/eval/inject_spike.py` (μ-injection recall/FP).
- **Outputs**: `reports/rc-sensitivity/*` heatmaps/tables; `reports/figures/inject_*` plots + manifest.

## ETF Demo
- **Script**: `experiments/etf_panel/run.py`.
- **Purpose**: ETF country/sector demo using daily eval harness; writes overlay_toggle markdown.
- **Outputs**: Similar diagnostics/plots as daily eval under specified `out` dir.
