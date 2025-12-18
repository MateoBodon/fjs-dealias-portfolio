# Config Reference

## Equity Panel (`experiments/equity_panel/*.yaml` + CLI)
Common keys (defaults in `run.py:DEFAULT_CONFIG` unless overridden):
- Data/time: `data_path`, `start_date`, `end_date`, `window_weeks`, `horizon_weeks`, `stride_windows`, `assets_top`.
- Detection: `dealias_delta`, `dealias_delta_frac`, `dealias_eps`, `stability_eta_deg`, `signed_a`, `nonnegative_a`, `a_grid`, `cs_drop_top_frac`, `cs_sensitivity_frac`, `energy_min_abs`, `off_component_leak_cap`, `oneway_a_solver {auto,rootfind,grid}`, `use_tvector` (bool), `target_component`.
- Design: `design {oneway,nested,dow,vol}`, `nested_replicates`, `partial_week_policy {drop,impute}`, optional `design_override` block.
- Edge/gating: `edge_mode {scm,tyler,huber}`, `edge_huber_c`, `gating.enable`, `gating.q_max`, `gating.require_isolated`, `gating.mode {fixed,calibrated}`, `gating.calibration_path` (default `calibration/edge_delta_thresholds.json`), `alignment_top_p`, `require_isolated` CLI flag, `gate_mode`/`gate_soft_max` overrides via CLI.
- Overlay/baselines: `estimator {dealias,lw,oas,cc,factor,tyler_shrink,poet}`, `minvar_ridge`, `minvar_box [lo,hi]`, `minvar_condition_cap`, `turnover_cost_bps`, `use_factor_prewhiten`, `prewhiten {off,ff5,ff5mom,custom}`, `factor_csv`.
- Paths: `output_dir`, `cache_dir`, `resume`, `precompute_panel`, `label`, `crisis_label`, `ablations`.

CLI notable flags: `--design`, `--edge-mode`, `--gating-mode {fixed,calibrated}`, `--gating-calibration <json>`, `--oneway-a-solver`, `--allow-non-isolated/--require-isolated`, `--coarse-candidate`, `--use-tvector`, `--target-component`, `--cs-drop-top-frac`, `--energy-min-abs`, `--off-component-leak-cap`, `--output-dir`, `--cache-dir`.

## Daily Evaluation (`experiments/eval/config.py`, `config.yaml`, `thresholds.json`, CLI)
- Data: `returns_csv` (required), `factors_csv` optional, `assets_top`, `min_history`, winsor bounds (`winsor_lower/upper`), `use_factor_prewhiten`.
- Grouping: `group_design {week,dow,dow_vol,dow_month,vol,dowxvol}`, `group_min_count`, `group_min_replicates`, `min_reps_dow`, `min_reps_vol`, `calm_quantile`, `crisis_quantile`, `vol_ewma_span`.
- Overlay: `overlay_delta`, `overlay_delta_frac`, `overlay_a_grid`, `edge_mode`, `require_isolated`, `q_max`, `q2_alignment_min_cos`, `angle_min_cos`, `alignment_top_p`, `cs_drop_top_frac`, `coarse_candidate`.
- Gating: `gate_mode {strict,soft}`, `gate_soft_max`, `gate_delta_calibration`, `gate_delta_frac_min/max`, `gate_stability_min`, `gate_alignment_min`, `gate_accept_nonisolated`.
- Portfolios: `mv_gamma`, `mv_tau`, `mv_box_lo/hi`, `mv_turnover_bps`, `mv_condition_cap`, `mv_seed`.
- Outputs: `out_dir`, `echo_config`, `reason_codes`, `bootstrap_samples`, `seed`, `max_windows`, `workers`.

## Synthetic calibration (`experiments/synthetic/calibrate_thresholds.py`)
- Grids: `--delta-abs-grid`, `--delta-frac-grid`, `--stability-grid`, `--energy-floor-grid`, `--edge-modes`, `--p-assets`, `--n-groups`, `--replicates`, `--trials-null/alt`.
- Execution: `--workers`, `--run-id`, `--exec-mode`, `--mp-cache-dir`, sharding `--shard-manifest`, `--shard-id`, `--batch-size`.
- Outputs: `--out` (cell JSONs), `--figures-out`; reduction via `tools/reduce_calibration.py` produces `calibration_defaults.json` + `calibration/edge_delta_thresholds.json`.

## Calibration defaults
- `calibration_defaults.json` — top-level `parameters` (delta, delta_frac, eps, stability_eta_deg, energy_floor, edge_mode), `selection` stats, `config` metadata, `generated_at`.
- `calibration/edge_delta_thresholds.json` — `thresholds.{edge_mode}.{p}x{t}.delta_frac` entries used by `fjs.gating.lookup_calibrated_delta`.

## Nested synthetic kill-test (`experiments/synthetic/config.nested.killtest.yaml`)
- Keys: `n_assets`, `years`, `weeks_options`, `replicates`, `trials_per_scenario`, `spikes` map (null/moderate/strong μ), `edge_modes {tyler,huber}`, `delta`, `delta_frac_min`, `eps`, `stability_eta_deg`, `a_grid`, `cs_drop_top_frac`, `off_component_leak_cap`, `energy_min_abs`, `allow_nonisolated`, `nonisolated_*` gates, `require_isolated`, `use_tvector`, `q_max`, `calibration_path`, `seed`, `out_dir`.
- Script: `experiments/synthetic/nested_killtest.py` reads this YAML, simulates nested year⊃week panels, logs skip reasons/detection coverage, and writes `reports/synthetic_nested_killtest/`.

## Make targets / env vars (excerpt)
- **Shared**: `EXEC_MODE`, `RC_RETURNS`, `RC_FACTORS`, `RC_GATE_DELTA_FRAC_MIN{_VOL}`, `RC_Q_MAX`, `VOL_Q2_ALIGNMENT_MIN_COS`, `RC_OVERLAY_DELTA`, `RC_COARSE_CANDIDATE`, `RC_GATE_MODE`, `RC_GATE_ACCEPT_NONISOLATED`, `RC_GATE_STABILITY_MIN`, `RC_MV_GAMMA`, `RC_MV_BOX`, `RC_MV_TURNOVER_BPS`, `RC_MV_CONDITION_CAP`, `RC_PREWHITEN`, `RC_USE_FACTOR_PREWHITEN`, `RC_WORKERS`.
- **rc-lite-sanity specific**: `RC_DOW_GROUP_MIN`, `RC_DOW_GROUP_REPS`, `RC_VOL_GROUP_MIN`, `RC_VOL_GROUP_REPS`, `RC_DOW_MIN_REPS`, `RC_VOL_MIN_REPS`, `RC_DOW_SHRINKER`, `RC_VOL_SHRINKER`, `RC_LITE_BASE`, `RC_LITE_CACHE`, `RC_OUT_SANITY`.
- **Synthetic**: `HARNESS_TRIALS`, `CALIB_TRIALS_NULL/ALT`, `CALIB_P_ASSETS`, `CALIB_REPLICATES`, `MP_CACHE_DIR`.

## Registries
- `data/registry.json` and `data/factors/registry.json` store sha256/rows/date spans. Update via `tools/update_registry.py --dataset <file> --wrds-source <table> --note ...`.

## AWS/runner scripts
- `scripts/aws_run.sh <target>` expects Make target and optional `EXEC_MODE`; writes telemetry under `reports/aws/<run_id>/`.
- Manual scripts under `scripts/manual/` call rc-lite/rc/calibration targets with explicit env overrides.
