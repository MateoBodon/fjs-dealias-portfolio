# Architecture

## Project purpose
Robust equity-return risk forecasting via FJS-style MANOVA de-aliasing. The code builds balanced weekly/daily panels, detects MANOVA spikes above Marčenko–Pastur edges (SCM/Tyler/Huber), substitutes de-aliased directions into covariance estimates, and benchmarks against shrinkage and factor baselines. Pipelines ship reproducible RC drops (figures + memos) and synthetic calibration sweeps. Latest operational drop is the **rc-lite-sanity** batch (2025-12-09 stamp) combining daily DoW/vol-state eval and weekly smoke runs.

## Top-level layout
- `src/` — core library: MANOVA math (`fjs/`), covariance/shrinkage (`finance/`, `baselines/`), evaluation metrics (`evaluation/`, `eval/`), data/registry helpers (`data/`, `io/`, `meta/`), plotting/report glue (`plotting/`, `report/`), synthetic utils (`synthetic/`), utilities (`utils/`).
- `experiments/` — runnable pipelines and configs: equity_panel (weekly MANOVA), eval (daily overlay), daily grouping helpers, synthetic harness, ablation runner, ETF demo, synthetic_oneway toy, prewhitening helpers, rc-lite-sanity outputs (`outputs_rc-lite-*`).
- `tools/` — CLI/report helpers (gallery, memo/brief, summary, calibration reduction, run monitor, registry updater, rc-sanity summariser).
- `scripts/` — data ingestion (WRDS/Sharadar), calibration wrappers, AWS dispatch/provision, manual RC scripts.
- `reports/`, `figures/`, `results/`, `ablations/` — generated artifacts: RC drops, ROC figures, memos/briefs, ablation tables, rc-lite-sanity summaries.
- `data/` — committed derived panels and registries (`returns_daily.csv`, factor CSVs, registries, balanced_weekly parquet); raw WRDS exports live in `data/wrds/` (ignored).
- `docs/`, `Long_Term_Plan.md`, `METHODS.md`, `RUNBOOK.md`, `REPORT.md`, `AGENTS.md`, `Paper and Project Context/` — narrative documentation, research context, and plans.
- `tests/` — unit/integration/smoke coverage for core math, gating, pipelines, reporting, datasets.
- `Makefile` — canonical targets (`make rc-lite-sanity`, `make rc-lite`, `make rc`, `make sweep:acceptance`, etc.).

## Core logical components
- **Data loading & balancing**: `data.panels`, `finance.io`, `finance.returns`, `experiments.daily` loaders build balanced Week×Day cubes; `finance.loader` builds weekly panels with fixed universes; partial-week policies `drop`/`impute`.
- **Prewhitening**: `baselines.factors` + `experiments.prewhiten` regress on FF5+MOM (or fallbacks) and emit residuals + telemetry.
- **MANOVA stats**: `fjs.balanced`, `fjs.balanced_nested` compute mean squares / covariance components for one-way or nested designs; `fjs.mp` supplies MP edges, admissible roots, t-vectors; `fjs.theta_solver` refines a-grid angles.
- **Detection & overlay**: `fjs.dealias` (Algorithm 1 search), `fjs.gating` (calibrated δ_frac lookups), `fjs.overlay` (spike detection + substitution onto shrinkage/factor baselines), `fjs.robust` (Tyler/Huber scatter edges), `fjs.spectra` (diagnostics).
- **Evaluation/metrics**: `evaluation.evaluate` (ΔMSE, VaR/ES, DM/sign tests, CI, alignment diagnostics), `evaluation.dm`, `finance.eval` (rolling forecasts, weekly covariance reconstruction), `experiments.eval` runner (daily regimes), `experiments.equity_panel` runner (weekly designs, nested/DoW/vol).
- **Calibration**: `experiments.synthetic` harnesses + `synthetic.calibration`/`threshold_eval` generate ROC tables and populate `calibration_defaults.json`, `calibration/edge_delta_thresholds.json`.
- **Reporting**: `report.gather/tables/plots`, `tools/build_gallery.py`, `build_memo.py`, `build_brief.py`, `tools/make_summary.py`, `tools/summarize_rc_sanity.py` assemble advisor-facing artifacts.
- **CLI surfaces**: `experiments/equity_panel/run.py`, `experiments/eval/run.py`, `experiments/synthetic/*.py`, `experiments/ablate/run.py`, `tools/*.py`, `scripts/*.sh|*.py`.

## High-level data flow
WRDS daily returns → optional winsor/huber → (optional) factor prewhitening → balanced Week×Day panel (`data.panels`/`finance.loader`) → MANOVA mean squares (`fjs.balanced*`) → MP edge & spike search (`fjs.dealias`, `fjs.mp`, robust edges) → gating (`fjs.gating`/`fjs.overlay`) → overlay-adjusted covariance on shrinkage/factor baseline → portfolio forecasts (`finance.eval`, `evaluation.evaluate`) → metrics/diagnostics CSVs → gallery/memo/brief (`report.*`, `tools/*`) under `reports/` and `figures/`. Synthetic pipelines feed calibration JSONs; rc-lite-sanity additionally writes kill-criteria + summary tables.

## High-level control flow
- **Weekly equity panel**: `experiments/equity_panel/run.py` handles data prep, per-window cache, detection/overlay, portfolio evaluation, plots, run metadata. Used in rc-lite, rc, rc-lite-sanity weekly smoke (DoW + nested).
- **Daily overlay eval (DoW/vol-state/week/month/dow×vol)**: `experiments/eval/run.py` groups daily residuals, applies overlay per regime, computes ΔMSE/DM/VaR/ES, writes regime CSVs/plots; used by rc-lite-sanity and rc/rc-lite targets.
- **Synthetic calibration**: `experiments/synthetic/null.py` & `power.py` simulate scores; `calibrate_thresholds.py` + `tools/reduce_calibration.py` sweep δ/energy/stability into defaults JSONs; `synthetic.calibration` provides MP edge delta utilities.
- **Ablations & sensitivity**: `experiments/ablate/run.py` sweeps overlay hyperparams; `experiments/eval/sensitivity.py` grids gate/δ/η on daily panels.
- **Reporting**: `tools/build_gallery.py` / `build_memo.py` / `build_brief.py` use run directories + summary tables; `tools/make_summary.py` builds kill-criteria/summary CSVs; `tools/summarize_rc_sanity.py` merges rc-lite-sanity outputs.
- **Auxiliary**: `scripts/data/*.py` fetch/preprocess data; `tools/update_registry.py`/`verify_dataset.py` maintain dataset digests; `scripts/aws_run.sh` dispatches remote runs; `tools/run_monitor.py` tails progress/metrics JSONL.
