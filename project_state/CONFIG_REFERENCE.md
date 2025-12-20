---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Config Reference

## Data & registries
- Returns: `data/returns_daily.csv` (sha256 `96ac7dd3…3197`) — verified via `tools/verify_dataset.py ... --registry data/registry.json`.
- Factors: `data/factors/ff5mom_daily.csv` (sha256 `469d44ad…908ca`) — verified via `data/factors/registry.json`.
- Never commit raw WRDS exports (`data/wrds/`).

## Make targets (from `project_state/_generated/make_targets.txt`)
- `make rc`, `rc-lite`, `rc-lite-sanity`, `rc-dow`, `rc-vol`, `rc-week`, `rc-dowxvol`, `rc-sensitivity`, `inject-spike`, `calibrate-thresholds`, `sweep:acceptance`, `gpt-bundle`. Remote variants: `make aws:<target> AWS_ARGS="..."`.
- Key env knobs: `EXEC_MODE={deterministic,throughput}`, `RC_GATE_DELTA_FRAC_MIN` / `_VOL`, `RC_Q_MAX`, `Q_MAX_VOL`, `RC_DOW_MIN_REPS`, `RC_VOL_MIN_REPS`, `RC_VOL_GROUP_REPS`, `RC_OVERLAY_DELTA`, `RC_GATE_MODE`, `RC_PREWHITEN`, `RC_USE_FACTOR_PREWHITEN`, `RC_REQUIRE_ISOLATED` / `RC_VOL_REQUIRE_ISOLATED`, `RC_FACTORS`, `RC_RETURNS`, `HARNESS_TRIALS`.
- Smoke shortcut: `make run:equity_nested_smoke_tiny` (deterministic nested weekly smoke capped at three windows; emits gating_diagnostics).

## Equity weekly runner (`experiments/equity_panel/run.py`)
- Config files: `experiments/equity_panel/config*.yaml` (see EXPERIMENTS.md for list). CLI overrides include:
  - `--design {oneway,dow,vol,nested}`; `--nested-replicates`; `--estimator {aliased,dealias,lw,oas,cc,factor,tyler_shrink}`.
  - Rolling window controls: `--window-weeks`, `--horizon-weeks`, `--max-windows` (caps evaluated windows; persisted in `config_resolved.yaml`).
  - Edge/gating: `--edge-mode {scm,tyler,huber}` (+ `--edge-huber-c`), `--gating-mode {fixed,calibrated}`, `--gating-calibration path`, `--gating-diagnostics`.
  - Overlay params: `--delta-frac`, `--eps`, `--a-grid`, `--eta`, `--off-leak`, `--energy-min-abs`, `--target-component`, `--signed-a/--nonnegative-a`, `--cs-drop-top-frac`.
  - Portfolio: `--minvar-ridge`, `--minvar-box lo,hi`, `--minvar-condition-cap`, `--turnover-cost`.
  - Preprocessing/cache: `--prewhiten {off,ff5,ff5mom,custom}`, `--use-factor-prewhiten {0,1}`, `--factor-csv path`, `--precompute-panel`, `--cache-dir`, `--resume`, `--winsorize q`, `--huber c`, `--drop-partial-weeks/--impute-partial-weeks`.
  - Crisis/ablations: `--crisis start:end`, `--ablations`, `--sigma-ablation`.

## Daily eval runner (`experiments/eval/run.py`)
- Config layering: defaults (`experiments/eval/config.yaml`), optional thresholds (`thresholds.json`), YAML config (`--config`), then CLI overrides.
- Common CLI flags: `--returns-csv`, `--factors-csv`, `--window`, `--horizon`, `--group-design {dow,vol,week,dowxvol}`, `--assets-top`, `--require-isolated`, `--q-max`, `--q2-alignment-min-cos`, `--edge-mode {tyler,scm,huber}`.
- Overlay/gating: `--overlay-delta`, `--overlay-delta-frac`, `--gate-mode {strict,soft}`, `--gate-delta-frac-min/max`, `--gate-stability-min`, `--gate-accept-nonisolated`, `--gate-delta-calibration`, `--coarse-candidate`.
- MV controls: `--mv-gamma`, `--mv-tau`, `--mv-box-lo/--mv-box-hi`, `--mv-turnover-bps`, `--mv-condition-cap`, `--mv-solver {projgrad,cvxpy}`, `--mv-skip-on-missing-solver`, `--mv-solver-name` (cvxpy backend string).
- Prewhitening: `--prewhiten {off,ff5,ff5mom,custom}`, `--use-factor-prewhiten {0,1}`.
- Outputs: `--out` (default `reports/eval-latest`), writes `resolved_config.json`, `run.json`, metrics/risk/dm CSVs, diagnostics, plots (when matplotlib available).

## Synthetic / calibration
- `experiments/synthetic/null.py` / `power.py`: `--trials`, `--edge-modes`, `--defaults-path`, `--out`, `--figures-out`.
- `experiments/synthetic/calibrate_thresholds.py`: grids for delta_abs/delta_frac/stability, assets/groups/replicates; supports `--run-id`, `--shard-manifest`, `--shard-id`, `--exec-mode`, `--mp-cache-dir`.
- Sharding helpers: `tools/shard_grid.py`, reduction via `tools/reduce_calibration.py --run-id <id>`.

## Environment variables
- `EXEC_MODE` (used by `meta/runtime.py` and Makefile) toggles deterministic thread caps.
- `FJS_FORCE_MISSING_CVXPY=1` forces `finance.portfolios` to behave as if cvxpy is absent; pair with `mv_skip_on_missing_solver` when exercising skip paths.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` respected by `meta/runtime.configure_exec_mode`.
