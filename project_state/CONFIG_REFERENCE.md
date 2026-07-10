---
generated: 2026-07-10T17:32:11-04:00
git_sha: 193a325dc681ebc4da67b44715a92e4f63113019
git_branch: portfolio/fjs-recenter-m1-20260710
commands:
  - Ticket 37 live config and data-contract audit
---
# Config Reference

## Data & registries
- Returns: `data/returns_daily.csv` (sha256 `96ac7dd318245cf1a8b434bb358a9344bf282992fc9fe66f0282023696563197`) — verified via `tools/verify_dataset.py ... --registry data/registry.json`.
- Factors: `data/factors/ff5mom_daily.csv` (sha256 `469d44ad0c5cac556c60c1f258e14245acfcc9f2901ad443f41b64309bf908ca`) — verified via `data/factors/registry.json`.
- Never commit raw WRDS exports (`data/wrds/`).
- `data/returns_daily.csv` is legacy development input only: it lacks PERMNO and
  point-in-time security/universe provenance, so it is ineligible for the
  Ticket 37 flagship result.
- Shared CRSP/public authorities, exact manifests, and hashes are frozen in
  `docs/strategy/FJS_SCIENTIFIC_RECENTER_PREDECLARATION.md`. Restricted raw data
  remain outside Git.

## Make targets (subset; see `project_state/_generated/make_targets.txt` for full list)
- `setup`, `fmt`, `lint`, `test`, `test-fast`, `test-integration`, `test-slow`, `test-all`, `rc`, `rc-lite`, `rc-lite-sanity`, `rc-data`, `rc-eval`, `rc-summary`, `rc-ablations`, `rc-dow`, `rc-vol`, `rc-week`, `rc-dowxvol`, `rc-sensitivity`, `run-synth`, `run-equity`, `run:equity_smoke`, `run:equity_nested_smoke_tiny`, `calibrate-thresholds`, `sweep:acceptance`, `gpt-bundle`
- Remote variants: `make aws:<target> AWS_ARGS="..."`.
- Key env knobs: `EXEC_MODE`, `RC_GATE_DELTA_FRAC_MIN`/`_VOL`, `RC_Q_MAX`, `Q_MAX_VOL`, `RC_DOW_MIN_REPS`, `RC_VOL_MIN_REPS`, `RC_VOL_GROUP_REPS`, `RC_OVERLAY_DELTA`, `RC_GATE_MODE`, `RC_PREWHITEN`, `RC_USE_FACTOR_PREWHITEN`, `RC_REQUIRE_ISOLATED`/`RC_VOL_REQUIRE_ISOLATED`, `RC_FACTORS`, `RC_RETURNS`, `HARNESS_TRIALS`.

## Equity weekly runner (`experiments/equity_panel/run.py`)
- Config files: `experiments/equity_panel/config*.yaml` (see EXPERIMENTS.md).
- CLI overrides (selected):
  - Design: `--design {oneway,dow,vol,nested}`, `--nested-replicates`.
  - Estimator: `--estimator {aliased,dealias,lw,oas,cc,factor,tyler_shrink}`.
  - Edge/gating: `--edge-mode {scm,tyler,huber}`, `--edge-huber-c`, `--gating-mode {fixed,calibrated}`, `--gating-calibration`, `--gating-diagnostics`.
  - Overlay params: `--delta-frac`, `--eps`, `--a-grid`, `--eta`, `--off-leak`, `--energy-min-abs`, `--target-component`, `--signed-a/--nonnegative-a`, `--cs-drop-top-frac`.
  - Windowing: `--window-weeks`, `--horizon-weeks`, `--max-windows`.
  - Portfolio: `--minvar-ridge`, `--minvar-box`, `--minvar-condition-cap`, `--turnover-cost`.
  - Preprocessing/cache: `--prewhiten {off,ff5,ff5mom,custom}`, `--use-factor-prewhiten`, `--factor-csv`, `--winsorize`, `--huber`, `--precompute-panel`, `--cache-dir`, `--resume`.

## Daily eval runner (`experiments/eval/run.py`)
- Config layering: defaults (`experiments/eval/config.yaml`), thresholds (`experiments/eval/thresholds.json`), optional YAML `--config`, then CLI overrides.
- Common CLI flags: `--returns-csv`, `--factors-csv`, `--window`, `--horizon`, `--group-design {dow,vol,week,dowxvol}`, `--assets-top`, `--universe-csv`, `--universe-as-of`, `--require-isolated`, `--q-max`, `--q2-alignment-min-cos`, `--edge-mode {tyler,scm,huber}`.
- `--assets-top` is fail-closed: both universe arguments are required. The CSV
  schema is exactly `as_of_date,ticker,rank`; the requested date must not be
  after evaluation start, ranks/tickers must be unique, selected tickers must
  exist, and the runner records source and selected-universe SHA-256 values.
  This static snapshot is for safe bounded runs; the flagship needs rolling
  point-in-time membership.
- Overlay/gating: `--overlay-delta`, `--overlay-delta-frac`, `--gate-mode {strict,soft}`, `--gate-delta-frac-min/max`, `--gate-stability-min`, `--gate-accept-nonisolated`, `--gate-delta-calibration`, `--coarse-candidate`.
- Every overlay candidate must carry one explicit source label from
  `{fjs,coarse,oracle,sham}`. Missing, unknown, or mixed sources fail. Pre-gate
  and accepted diagnostics preserve source counts.
- Unknown baseline shrinker names and a repeated overlay eigendecomposition
  failure raise errors rather than silently falling back.
- MV controls: `--mv-gamma`, `--mv-tau`, `--mv-box-lo/--mv-box-hi`, `--mv-turnover-bps`, `--mv-condition-cap`, `--mv-solver {projgrad,cvxpy}`, `--mv-skip-on-missing-solver`, `--mv-solver-name`.
- Comparison validity: `--min-comparison-windows` enforces minimum aligned windows for Δ metrics and DM tests, writing `n_effective_*` and `comparison_valid_*` flags.
- Outputs: `--out` directory with `resolved_config.json`, `run.json`, metrics/risk/dm CSVs, diagnostics, optional plots.

## Inject-spike evaluation (`experiments/eval/inject_spike.py`)
- Purpose: weak-spike injection sweeps on real residual windows to test detection/gating response by design.
- Key flags: `--inject-mode {total,between,within}`, `--mu-grid`, `--inject-frac-min`, `--inject-frac-max`, `--max-windows`, `--window-sampling {first,random}`, `--window-sampling-seed`, `--seed`, `--run-id`, `--out`.
- Data/config flags mirror daily eval runner: `--returns-csv`, `--factors-csv`, `--config`, `--thresholds`, `--group-design`, `--use-factor-prewhiten`.
- Detector promotion is governed by the frozen Ticket 37 stop-line and the
  persisted reducer in `src/fjs/detector_contract.py`; a flat/underpowered curve
  cannot unlock broad empirical work.

## Synthetic / calibration
- `experiments/synthetic/null.py` / `power.py`: `--trials`, `--edge-modes`, `--defaults-path`, `--out`, `--figures-out`.
- `experiments/synthetic/calibrate_thresholds.py`: grids for delta/stability/asset/group/replicate bins; supports `--run-id`, `--shard-manifest`, `--shard-id`, `--exec-mode`, `--mp-cache-dir`.
- Sharding helpers: `tools/shard_grid.py`, reduction via `tools/reduce_calibration.py --run-id <id>`.

## Environment variables
- `EXEC_MODE` toggles deterministic thread caps (`meta/runtime.py`).
- `FJS_FORCE_MISSING_CVXPY=1` forces the missing-solver path in `finance.portfolios` (pair with `--mv-skip-on-missing-solver`).
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` respected by `meta/runtime.configure_exec_mode`.
