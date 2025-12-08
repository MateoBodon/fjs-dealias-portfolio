# Experiments

## Equity Panel (weekly MANOVA)
- **Script**: `experiments/equity_panel/run.py`
- **Configs**: `config.smoke.yaml` (short 2023 slice), `config.crisis.2020.yaml`, `config.crisis.2022.yaml`, `config.nested.*.yaml`, `config.rc.yaml`, `config.gallery.yaml`, `config.ablation.smoke.yaml`.
- **Purpose**: Detect MANOVA spikes on weekly aggregated returns; compare de-aliased overlay vs shrinkage/factor baselines; generate E1–E4 plots, metrics_summary, detection diagnostics.
- **Key args**: `--design {oneway,nested,dow,vol}`, `--edge-mode {scm,tyler,huber}`, `--dealias-delta{,_frac,_eps}`, `--a-grid`, `--q-max`, `--gating-mode {fixed,calibrated}`, `--gating-calibration calibration/edge_delta_thresholds.json`, `--prewhiten {off,ff5,ff5mom,custom}`, `--factor-csv`, `--cache-dir`, `--resume`, `--ablations`.
- **Outputs**: Run dir containing rolling_results.csv, detection_summary.csv, diagnostics.csv, diagnostics_detail.csv, metrics_summary.csv, summary.json, plots (spectrum, spike series, DM pvals), panel_manifest.json, run_meta.json.

## Daily Overlay Evaluation
- **Script**: `experiments/eval/run.py` (defaults resolved via `experiments/eval/config.py` + `thresholds.json`).
- **Purpose**: Faster daily pipeline by group design (week/DoW/vol-state/month) to gauge overlay behaviour, VaR/ES, DM tests by regime.
- **Key args**: `--returns-csv`, `--factors-csv`, `--group-design {week,dow,dow_vol,dow_month,vol}`, `--window`, `--horizon`, `--shrinker`, `--q-max`, `--edge-mode`, `--gate-mode {strict,soft}`, `--gate-delta-calibration`, `--assets-top`, `--prewhiten`, `--use-factor-prewhiten`.
- **Outputs**: per-regime CSVs (metrics, risk, dm, diagnostics, diagnostics_detail), delta_mse.png, flip_dm.png, resolved_config.json, prewhiten_* files.

## Synthetic Calibration / ROC
- **Null**: `experiments/synthetic/null.py --trials ... --edge-modes scm tyler --out reports/synthetic/null_harness`
- **Power**: `experiments/synthetic/power.py --mu-values 4 6 8 --null-scores <path>`
- **Calibration sweep**: `experiments/synthetic/calibrate_thresholds.py` (or `make sweep:acceptance`, `make calibrate-thresholds`) sweeps δ_frac / energy / stability to populate `calibration_defaults.json` and `calibration/edge_delta_thresholds.json`; supports sharding + reduction.
- **Threshold evaluation**: `synthetic/threshold_eval.py` compares calibrated thresholds vs score tables.

## Synthetic Benchmarks (one-way)
- **Script**: `experiments/synthetic_oneway/run.py` (invoked via `make run-synth`).
- **Purpose**: S1/S3/S4/S5 synthetic scenarios measuring detection and MSE behaviour; writes `figures/synthetic/` and `summary.json`.

## Ablation Grid
- **Script**: `experiments/ablate/run.py` with matrices in `ablate/ablation_matrix.yaml` or `.tiny.yaml`.
- **Purpose**: Sweep overlay hyperparameters (delta, eps, eta, q_max, etc.) on small slices; outputs `ablation_summary.csv` for gallery embedding.

## ETF Demo
- **Script**: `experiments/etf_panel/run.py`.
- **Purpose**: Apply daily evaluation harness to ETF country/sector panel; writes overlay_toggle.md and same diagnostics as daily eval.

## Support / Utility Experiments
- `experiments/eval/inject_spike.py` injects synthetic spikes for testing detection sensitivity.
- `experiments/eval/sensitivity.py` explores gating sensitivity to Cs perturbations.
- `experiments/equity_panel/sweep_acceptance.py` runs small acceptance sweeps on equity data.
