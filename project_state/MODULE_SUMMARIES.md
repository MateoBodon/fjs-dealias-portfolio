---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Module Summaries

## Core packages
- **src/fjs** — spectral de-alias overlay. `dealias.py` (core transform + admissibility), `overlay.py` (detect_spikes/apply_overlay), `mp.py` (MP edge & stability), `gating.py` (acceptance), `balanced.py` / `balanced_nested.py` (balanced MANOVA sums), `theta_solver.py`, `spectra.py`, `robust.py`.
- **src/finance** — base covariance + portfolio layer. `shrinkage.py`/`ledoit.py` (LW/OAS/RIE helpers), `robust.py` (Tyler/Huber SCM), `factors.py` (factor covariance), `returns.py`/`io.py`/`loader.py` (data alignment), `design.py` (group labels), `portfolio.py` + `portfolios.py` (EW/min-var/box/turnover with solver skip/fail-loud), `eval.py` (covariance evaluation).
- **src/baselines** — evaluation baselines (`covariance.py`/`factors.py`) used by `experiments/eval/run.py`.
- **src/evaluation** — `evaluate.py` (rolling metrics + alignment), `dm.py` (DM tests), `factor.py` (observed-factor + POET-lite).
- **src/eval** — `clean.py` (NaN policy / winsor/Huber), `balance.py` (balanced window construction) reused in daily eval.
- **src/data** — registry + panel utilities (`registry.py`, `panels.py`, `factors.py`).
- **src/meta** — cache/provenance (`cache.py`, `run_meta.py`, `runtime.py`, `completeness.py`).
- **src/report** — `gather.py`, `tables.py`, `plots.py` for summaries and memo/galleries.
- **src/plotting** — shared plotting utilities.
- **src/io** — WRDS connectors (`wrds_connect.py`, `crsp_daily.py`).
- **src/synthetic** — calibration helpers (`calibration.py`, `threshold_eval.py`).
- **src/utils** — credentials helper (`utils/credentials.py`).

## Experiment & tool layers
- **experiments/equity_panel** — weekly group runner (`run.py`) with configs `config*.yaml`; emits `outputs_*` dirs (detection summaries, gating diagnostics, plots).
- **experiments/eval** — daily evaluation CLI (`run.py`), config resolver (`config.py`), diagnostics/injection/sensitivity helpers.
- **experiments/synthetic** — null/power harness, nested kill-test, calibration sweeps.
- **experiments/ablate** — ablation grid runner (`run.py`).
- **experiments/daily** — quick daily smoke runner (`run.py`) + groupings.
- **experiments/prewhiten.py** — factor prewhitening CLI.
- **experiments/etf_panel** — ETF panel demo.
- **experiments/synthetic_oneway** — S1/S3/S4/S5 synthetic figures.
- **tools/** — ops/reporting utilities (verify/update registries, summaries, gallery/memo/brief builders, calibration reducers, run monitors, project_state generator).

## Module inventory (AST-derived)
- **baselines**: `baselines`, `baselines.covariance`, `baselines.factors`
- **eval**: `eval`, `eval.balance`, `eval.clean`
- **evaluation**: `evaluation`, `evaluation.dm`, `evaluation.evaluate`, `evaluation.factor`
- **experiments**: `experiments`, `experiments.ablate`, `experiments.ablate.run`, `experiments.daily`, `experiments.daily.config`, `experiments.daily.grouping`, `experiments.daily.run`, `experiments.equity_panel`, `experiments.equity_panel.reasons`, `experiments.equity_panel.run`, `experiments.equity_panel.sweep_acceptance`, `experiments.etf_panel.run`, `experiments.eval.config`, `experiments.eval.diagnostics`, `experiments.eval.inject_spike`, `experiments.eval.run`, `experiments.eval.sensitivity`, `experiments.prewhiten`, `experiments.synthetic`, `experiments.synthetic.calibrate_thresholds`, `experiments.synthetic.harness_utils`, `experiments.synthetic.nested_killtest`, `experiments.synthetic.null`, `experiments.synthetic.power`, `experiments.synthetic.power_null`, `experiments.synthetic_oneway`, `experiments.synthetic_oneway.run`
- **finance**: `finance`, `finance.design`, `finance.eval`, `finance.factors`, `finance.io`, `finance.ledoit`, `finance.loader`, `finance.portfolio`, `finance.portfolios`, `finance.returns`, `finance.robust`, `finance.shrinkage`
- **fjs**: `fjs`, `fjs.balanced`, `fjs.balanced_nested`, `fjs.dealias`, `fjs.gating`, `fjs.mp`, `fjs.overlay`, `fjs.robust`, `fjs.spectra`, `fjs.theta_solver`
- **io**: `io.crsp_daily`, `io.wrds_connect`
- **meta**: `meta.cache`, `meta.completeness`, `meta.run_meta`, `meta.runtime`
- **plotting**: `plotting`, `plotting.utils`
- **report**: `report`, `report.gather`, `report.plots`, `report.tables`
- **synthetic**: `synthetic`, `synthetic.calibration`, `synthetic.threshold_eval`
- **tools**: `tools.aggregate_runs`, `tools.build_brief`, `tools.build_gallery`, `tools.build_memo`, `tools.clean_outputs`, `tools.generate_project_state`, `tools.list_runs`, `tools.make_summary`, `tools.paper_v1_ablation`, `tools.plot_rc_hist`, `tools.prewhiten_effect`, `tools.reduce_calibration`, `tools.run_monitor`, `tools.shard_grid`, `tools.summarize_rc_sanity`, `tools.summarize_run`, `tools.summarize_weekly_diagnostics`, `tools.update_registry`, `tools.verify_dataset`
- **utils**: `utils.credentials`
