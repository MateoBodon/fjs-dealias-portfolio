# Dependency Graph (textual)

## Core library imports
- `fjs.mp` → numpy; consumed by `fjs.dealias`, `fjs.theta_solver`, `fjs.overlay`, `experiments.synthetic`, `evaluation.evaluate` (alignment diagnostics).
- `fjs.balanced` / `fjs.balanced_nested` → numpy; consumed by `fjs.dealias`, `finance.eval`, `experiments.equity_panel`, `experiments.synthetic`.
- `fjs.dealias` → (`fjs.balanced*`, `fjs.mp`, `fjs.theta_solver`); consumed by `fjs.overlay`, `finance.eval` (dealias estimator), `experiments.equity_panel`, `experiments.eval`.
- `fjs.overlay` → (`fjs.dealias`, `fjs.gating`, shrinkers from `baselines.covariance`/`finance.ledoit`/`finance.shrinkage`); consumed by `experiments.eval`, optionally by `equity_panel`.
- `fjs.gating` → json/numpy; consumed by `fjs.overlay`, `experiments.equity_panel` (calibrated gate), `experiments.eval`.
- `fjs.robust` → numpy; used for robust MP edges (Tyler/Huber) in overlay/equity_panel.
- `fjs.spectra` → numpy/matplotlib; plotting helpers used by equity_panel.

- `finance.*` → numpy/pandas; depend on `fjs` (dealias_covariance) and `evaluation.factor`/`baselines` for factor/POET; consumed by experiment runners.
- `baselines.*` → numpy/pandas; shrinkers and prewhitening used by experiments/eval/equity_panel.
- `data.*` → pandas; registries and balanced panels used by finance loaders, experiments, registry tools.
- `evaluation.*` → numpy/pandas/scipy; metrics/DM/coverage used by equity_panel + eval runners and reporting.
- `eval.clean/balance` → numpy/pandas; used by `experiments.eval.run` to enforce balanced regimes.
- `meta.cache/run_meta/runtime` → used by equity_panel, eval, synthetic calibration for cache keys, metadata, thread caps.
- `report.*` → pandas/matplotlib; used by gallery/memo/brief/summary tools.
- `plotting.utils` → matplotlib; used by equity_panel for E1–E4 figures.

## Experiment runners and helpers
- `experiments/equity_panel/run.py` imports: finance (eval/portfolio/returns/io/robust), fjs (balanced/nested/mp/dealias/gating/spectra/robust), baselines (factors, covariance), meta (cache/runtime/run_meta), data.panels, evaluation (alignment/metrics), plotting utils, experiments.prewhiten. Outputs → figures, run_meta, memo/brief via tools.
- `experiments/eval/run.py` imports: experiments.daily grouping, experiments.prewhiten, baselines (covariance/factors), data.factors, eval.clean/balance, evaluation.dm/evaluate/factor, finance (minvar/turnover), fjs.overlay, meta.runtime. Outputs feed rc-lite/rc-lite-sanity + rc targets.
- `experiments/daily/run.py` (wrapper) imports grouping utilities and calls eval runner; minimal external deps.
- `experiments/ablate/run.py` imports equity_panel helpers to sweep configs and read ablation matrices.
- `experiments/synthetic/*.py` import `fjs.mp/dealias/balanced`, `synthetic.calibration/threshold_eval`, harness_utils; use pandas/matplotlib.
- `experiments/synthetic_oneway/run.py` imports `fjs.mp` utilities directly for analytic edges.

## Tools / scripts
- `tools/build_*`, `make_summary`, `summarize_rc_sanity`, `aggregate_runs`, `list_runs` → depend on `report.*`, pandas/matplotlib, sometimes `synthetic.calibration`/`data.registry`.
- `tools/reduce_calibration.py` → reads JSON shards produced by synthetic sweeps.
- `tools/update_registry.py` / `verify_dataset.py` → depend on `data.registry` / `data.factors`.
- `scripts/data/*.py` → pandas/numpy + finance/data loaders; `scripts/aws_run.sh` wraps Make targets.

## High fan-out modules
- `fjs.dealias` / `fjs.mp` — numerical core touched by most detection/overlay flows.
- `experiments/equity_panel/run.py` — orchestrates data, fjs, finance, meta, plotting; writes run_meta.
- `experiments/eval/run.py` — daily counterpart with heavy dependencies on overlay + finance + eval.clean/balance.
- `experiments/prewhiten.py` — centralises factor selection for both daily and weekly pipelines.

## Coupling / gaps
- `experiments/eval/run.py` imports `data.loader` but falls back to an inline loader when missing (no `src/data/loader.py` exists); keep fallback in mind when refactoring.
- No circular imports observed; plotting and reporting modules are leaf dependents.
- MP cache path is set via environment (`MP_CACHE_DIR`/`MP_EDGE_CACHE_DIR`); callers must configure before first edge query to avoid mixed cache content across runs.
