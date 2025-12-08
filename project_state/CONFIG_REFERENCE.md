# Config Reference

## Equity Panel (`experiments/equity_panel/*.yaml`)
Common keys (with defaults in `run.py:DEFAULT_CONFIG`):
- `data_path` (returns CSV), `start_date`, `end_date`, `frequency`.
- Windowing: `window_weeks` (156 smoke=6), `horizon_weeks` (4 smoke=1), `stride_windows` (CLI placeholder).
- Detection: `dealias_delta`, `dealias_delta_frac`, `dealias_eps`, `stability_eta_deg`, `signed_a`/`nonnegative_a`, `a_grid`, `cs_drop_top_frac`, `cs_sensitivity_frac`, `energy_min_abs`, `off_component_leak_cap`, `oneway_a_solver {auto,rootfind,grid}`, `target_component`.
- Design: `design {oneway,nested,dow,vol}`, `nested_replicates`, `partial_week_policy {drop,impute}`.
- Edge/gating: `edge_mode {scm,tyler,huber}`, `edge_huber_c`, `gating.enable`, `gating.q_max`, `gating.require_isolated`, `gating.mode {fixed,calibrated}`, `gating.calibration_path` (default `calibration/edge_delta_thresholds.json`), `gate_mode`/`gate_soft_max` overrides via CLI, `alignment_top_p`.
- Overlay/baselines: `estimator {dealias,lw,oas,cc,factor,tyler_shrink}`, `minvar_ridge`, `minvar_box [lo,hi]`, `minvar_condition_cap`, `turnover_cost_bps`, `use_factor_prewhiten`, `prewhiten {off,ff5,ff5mom,custom}`, `factor_csv`.
- Paths: `output_dir`, `cache_dir`, `resume` flag, `precompute_panel`, `label`, `crisis_label`, `ablations`.

## Daily Evaluation (`experiments/eval/config.py` defaults + `config.yaml` + `thresholds.json`)
- Data: `returns_csv` (required), `factors_csv` optional, `assets_top` cap, `min_history` (via DailyLoader), winsor bounds from loader (winsor_lower/upper), `use_factor_prewhiten`.
- Grouping: `group_design {week,dow,dow_vol,dow_month,vol}`, `group_min_count`, `group_min_replicates`, `min_reps_dow`, `min_reps_vol`, `calm_quantile`, `crisis_quantile`, `vol_ewma_span`.
- Overlay: `overlay_delta`, `overlay_delta_frac`, `overlay_a_grid`, `edge_mode`, `require_isolated`, `q_max`, `q2_alignment_min_cos`, `angle_min_cos`, `alignment_top_p`, `cs_drop_top_frac`, `coarse_candidate`.
- Gating: `gate_mode {strict,soft}`, `gate_soft_max`, `gate_delta_calibration`, `gate_delta_frac_min/max`, `gate_stability_min`, `gate_alignment_min`, `gate_accept_nonisolated`.
- Portfolios: `mv_gamma`, `mv_tau`, `mv_box_lo/hi`, `mv_turnover_bps`, `mv_condition_cap`, `mv_seed`.
- Outputs: `out_dir`, `echo_config`, `reason_codes`, `bootstrap_samples`, `seed`.

## Synthetic calibration (`experiments/synthetic/calibrate_thresholds.py`)
- Grid: `--delta-abs-grid`, `--delta-frac-grid`, `--stability-grid`, `--energy-floor-grid`, `--edge-modes`, `--p-assets`, `--n-groups`, `--replicates`, `--trials-null/alt`.
- Execution: `--workers`, `--run-id`, `--exec-mode`, `--mp-cache-dir`, sharding options `--shard-manifest`, `--shard-id`.
- Outputs: `--out` (cell JSONs under run-id), `--figures-out`, reduced thresholds JSON via `tools/reduce_calibration.py`.

## Calibration defaults
- `calibration_defaults.json` — top-level `parameters` (delta, delta_frac, eps, stability_eta_deg, energy_floor, edge_mode), `selection` stats, `config` metadata.
- `calibration/edge_delta_thresholds.json` — `thresholds.{edge_mode}.{p}x{t}.delta_frac` entries for lookup by `fjs.gating.lookup_calibrated_delta`.

## Make targets / env vars
- `EXEC_MODE={deterministic,throughput}` controls thread caps (`meta.runtime`).
- `MP_CACHE_DIR` / `MP_EDGE_CACHE_DIR` set on-disk MP edge cache; `.cache/mp_edges` used if set.
- `HARNESS_TRIALS`, `RUN_ID`, `SHARD_MANIFEST`, `SHARD_ID` configure synthetic sweeps.
- `RC_DATE`, `RC_GATE_DELTA_FRAC_MIN_VOL`, `RC_VOL_*` environment knobs influence RC make targets.

## Registries
- `data/registry.json` and `data/factors/registry.json` store sha256/rows/date spans. Update via `tools/update_registry.py --dataset <file> --wrds-source <table> --note ...`.

## AWS/runner scripts
- `scripts/aws_run.sh <target>` expects Make target and optional `EXEC_MODE`; writes telemetry under `reports/aws/<run_id>/`.
- `scripts/manual/run_daily_rc_smoke.sh` uses env `PARALLEL=1` to run DoW/vol slices concurrently.
