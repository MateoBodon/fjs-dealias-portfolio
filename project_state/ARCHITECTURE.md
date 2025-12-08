# Architecture

## Project purpose
Robust equity-return risk forecasting via FJS-style MANOVA de-aliasing. The codebase builds balanced weekly/daily panels, detects MANOVA spikes above Marčenko–Pastur edges (SCM/Tyler/Huber), substitutes de-aliased directions into covariance estimates, and benchmarks against shrinkage and factor baselines. Pipelines produce reproducible RC drops (figures + memos) and synthetic calibration sweeps.

## Top-level layout
- `src/` — core library: MANOVA math (`fjs/`), covariance/shrinkage (`finance/`, `baselines/`), evaluation metrics (`evaluation/`, `eval/`), data/registry helpers (`data/`, `io/`, `meta/`), plotting/report glue (`plotting/`, `report/`), synthetic utils (`synthetic/`), utilities (`utils/`).
- `experiments/` — runnable pipelines and configs: equity_panel (weekly MANOVA), eval (daily overlay), synthetic (null/power sweeps), ablate grids, ETF demo, synthetic_oneway toy, prewhitening helpers.
- `tools/` — CLI/report helpers (gallery, memo/brief, summarization, cleaning, calibration reduction, run monitor, registry updater).
- `scripts/` — data ingestion (WRDS/Sharadar), calibration wrappers, AWS runner, manual RC scripts.
- `reports/`, `figures/`, `results/`, `ablations/` — generated artifacts: RC drops, ROC figures, memo/briefs, ablation tables.
- `data/` — committed derived panels and registries (`returns_daily.csv`, factor CSVs, registries). Raw WRDS exports stay under `data/wrds/` (ignored).
- `docs/`, `README.md`, `METHODS.md`, `RUNBOOK.md`, `PLAN.md` — narrative documentation and runbooks.
- `tests/` — unit/integration/regression tests covering core math, overlay gating, pipelines, reporting, and datasets.
- `Makefile` — convenience targets (`make setup`, `make test-fast`, `make rc`, `make sweep:acceptance`, etc.).

## Core logical components
- **Data loading & balancing**: `data.panels`, `finance.io`, `finance.returns`, `experiments.daily` loaders build balanced Week×Day cubes, enforce fixed universes, handle winsor/huber clipping and partial weeks.
- **Prewhitening**: `baselines.factors` + `experiments.prewhiten` regress returns on FF5+MOM (or fallbacks) and emit residuals/telemetry.
- **MANOVA stats**: `fjs.balanced`, `fjs.balanced_nested` compute mean squares / covariance components for one-way or nested designs; `fjs.mp` provides MP edges, admissible roots, and t-vectors; `fjs.theta_solver` refines a-grid angles.
- **Detection & overlay**: `fjs.dealias` (Algorithm 1 search), `fjs.gating` (calibrated δ_frac lookups), `fjs.overlay` (spike detection + substitution onto shrinkage/factor baselines), `fjs.robust` (Tyler/Huber scatter edges), `fjs.spectra` (diagnostic plots).
- **Evaluation/metrics**: `evaluation.evaluate` (ΔMSE, VaR/ES, DM/sign tests, CI), `evaluation.dm`, `finance.eval` (rolling forecasts, weekly covariance reconstruction), `experiments.eval` runner (daily regimes, overlay gating), `experiments.equity_panel` runner (weekly design, crisis/nested/smoke batches).
- **Calibration**: `experiments.synthetic` harnesses + `synthetic.calibration`/`threshold_eval` generate ROC tables and `calibration_defaults.json`, `calibration/edge_delta_thresholds.json`.
- **Reporting**: `report.gather/tables/plots`, `tools/build_gallery.py`, `build_memo.py`, `build_brief.py`, `tools/make_summary.py` assemble advisor-facing artifacts.
- **CLI surfaces**: `experiments/equity_panel/run.py`, `experiments/eval/run.py`, `experiments/synthetic/*.py`, `tools/*.py`, `scripts/*.sh|*.py`.

## High-level data flow
Raw WRDS daily returns → (`finance.io`, `baselines.factors`) optional prewhiten → balanced Week×Day panel (`data.panels`) → MANOVA mean squares (`fjs.balanced*`) → MP edge & spike search (`fjs.dealias`, `fjs.mp`, robust edges) → gating (`fjs.overlay/gating`) → overlay-adjusted covariance → portfolio forecasts (`finance.eval`, `evaluation.evaluate`) → metrics/diagnostics CSVs → gallery/memo/brief (`report.*`, `tools/*`) stored under `reports/` and `figures/`.

## High-level control flow
- **Weekly equity panel**: `experiments/equity_panel/run.py` orchestrates data prep, per-window cache, detection/overlay, portfolio evaluation, plots, summaries, run metadata.
- **Daily overlay eval (DoW/vol-state)**: `experiments/eval/run.py` groups daily residuals, applies overlay per regime, computes ΔMSE/DM/VaR/ES, writes regime CSVs/plots.
- **Synthetic calibration**: `experiments/synthetic/null.py` & `power.py` simulate scores, `calibrate_thresholds.py` sweeps thresholds and writes calibration defaults; `synthetic/calibration.py` provides MP edge delta utilities.
- **Reporting**: `tools/build_gallery.py` + `build_memo.py` + `build_brief.py` consume run directories/configs and emit figures/memos; `tools/make_summary.py` creates cross-run summary tables.
- **Auxiliary**: `scripts/data/*.py` fetch/preprocess data; `tools/update_registry.py` maintains dataset digests; `scripts/aws_run.sh` dispatches remote runs with telemetry.
