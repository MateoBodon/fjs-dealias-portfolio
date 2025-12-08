# Dependency Graph (textual)

## Core library
- `fjs.mp` ← numpy; used by `fjs.dealias`, `fjs.theta_solver`, `fjs.overlay`, `evaluation.evaluate` (alignment), synthetic harness.
- `fjs.balanced` / `fjs.balanced_nested` ← numpy; used by `fjs.dealias`, `finance.eval`, `experiments.equity_panel`, synthetic harness.
- `fjs.dealias` ← `fjs.balanced`, `fjs.mp`, `fjs.theta_solver`; consumed by `fjs.overlay`, `finance.eval` (dealias estimator), `experiments.equity_panel`, `experiments.eval`.
- `fjs.overlay` ← `fjs.dealias`, `fjs.gating`, shrinkage from `baselines.covariance`/`finance.ledoit`/`finance.shrinkage`; used by `experiments.eval` and optionally `equity_panel` overlays.
- `fjs.gating` ← json/numpy; used by `fjs.overlay`, `experiments.equity_panel` (calibrated gating), `experiments.eval`.
- `fjs.robust` ← numpy; used by `experiments.equity_panel` (Tyler/Huber edges), `fjs.overlay` edge scaling.
- `fjs.spectra` ← numpy, matplotlib; used by `experiments.equity_panel` plotting.

- `finance.*` ← numpy/pandas; depend on `fjs` (dealias_covariance) and `evaluation.factor`/`baselines` for factor/POET; used by experiment runners.
- `baselines.*` ← numpy/pandas; used by experiments and evaluation as shrinker baselines + prewhitening.
- `data.*` ← pandas; used by finance loaders, experiments, registry tools.
- `evaluation.*` ← numpy/pandas/scipy; used by experiments/equity_panel and experiments/eval for metrics/DM tests.
- `eval.*` (clean/balance) ← numpy/pandas; used exclusively by `experiments.eval.run`.
- `meta.cache`/`meta.run_meta`/`meta.runtime` used by `experiments.equity_panel`, `experiments.eval`, calibration scripts.
- `report.*` ← pandas/matplotlib; used by tools (`build_gallery`, `build_memo`, `build_brief`, `make_summary`).

## Experiments / pipelines
- `experiments/equity_panel/run.py` imports: finance (eval/portfolio/returns/io/robust), fjs (balanced/nested/mp/dealias/gating/spectra/robust), baselines (factors, covariance), meta (cache/runtime/run_meta), data.panels, evaluation (metrics, check), plotting utils. Outputs consumed by reporting tools.
- `experiments/eval/run.py` imports: experiments.daily grouping, experiments.prewhiten, baselines (covariance/factors), data.factors, eval.clean/balance, evaluation.dm/evaluate/factor, finance (minvar, turnover), fjs.overlay, meta.runtime. Outputs feed gallery/memo via run directories.
- `experiments/synthetic/*.py` import fjs.mp/dealias/balanced, synthetic.calibration/threshold_eval, harness_utils; use matplotlib/pandas.
- `experiments/ablate/run.py` imports equity_panel runner helpers to sweep configs.

## Tools / scripts
- `tools/*` rely on `report.*` loaders, pandas/matplotlib; some (reduce_calibration, update_registry) call `data.registry`/`synthetic.calibration`.
- `scripts/data/*.py` depend on pandas/numpy plus finance/data loaders; `scripts/aws_run.sh` invokes Make targets and tools.

## High fan-out modules
- `fjs.dealias` and `fjs.mp` are core numerical primitives for detection; many modules depend on them.
- `experiments/equity_panel/run.py` orchestrates most other modules (data, fjs, finance, meta, plotting).
- `experiments/eval/run.py` is the daily counterpart with heavy dependencies on overlay + finance.

## Notable absence / isolation
- `evaluation.check_dealiased_applied` (small helper in evaluation/__init__.py) is used by equity_panel to verify overlay application.
- `synthetic.calibration` / `threshold_eval` are mostly used by synthetic runners and tools/reduce_calibration.
- `io/wrds_connect.py` and `utils/credentials.py` are isolated, only referenced by scripts or external consumers.

No circular import was observed; caching and plotting modules are leaf dependents.
