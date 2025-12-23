---
generated: 2025-12-22T21:06:25Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
---


# Dependency Graph


Source: `project_state/_generated/import_graph.json` (internal imports among src/, experiments/, tools/).

## Summary
- modules: 93
- internal edges: 121
- isolated modules (no internal in/out): 19

## Top fan-out (internal imports)
- `experiments.equity_panel.run` -> 22
- `experiments.eval.run` -> 16
- `finance.eval` -> 7
- `experiments.eval.inject_spike` -> 6
- `experiments.synthetic.nested_killtest` -> 5
- `experiments.synthetic.power_null` -> 5
- `experiments.synthetic_oneway.run` -> 5
- `fjs.overlay` -> 5
- `experiments.synthetic.calibrate_thresholds` -> 4
- `fjs.dealias` -> 3
- `tools.build_gallery` -> 3
- `baselines.covariance` -> 2
- `experiments.ablate.run` -> 2
- `experiments.daily` -> 2
- `experiments.daily.run` -> 2

## Top fan-in (imported by others)
- `fjs.dealias` <- 7
- `fjs.balanced` <- 6
- `experiments.eval.run` <- 5
- `evaluation.evaluate` <- 4
- `experiments.synthetic.harness_utils` <- 4
- `finance.shrinkage` <- 4
- `fjs.robust` <- 4
- `baselines` <- 3
- `evaluation.dm` <- 3
- `finance.ledoit` <- 3
- `fjs.gating` <- 3
- `fjs.mp` <- 3
- `fjs.spectra` <- 3
- `report.gather` <- 3
- `baselines.covariance` <- 2

## Adjacency list (grouped by top-level package)
### `baselines`
- `baselines` -> (none)
- `baselines.covariance` -> `finance.ledoit`, `finance.shrinkage`
- `baselines.factors` -> (none)

### `eval`
- `eval` -> (none)
- `eval.balance` -> (none)
- `eval.clean` -> (none)

### `evaluation`
- `evaluation` -> (none)
- `evaluation.dm` -> (none)
- `evaluation.evaluate` -> `evaluation.dm`
- `evaluation.factor` -> `finance.factors`

### `experiments`
- `experiments` -> (none)
- `experiments.ablate` -> (none)
- `experiments.ablate.run` -> `experiments.eval.run`, `tools.make_summary`
- `experiments.daily` -> `experiments.config`, `experiments.grouping`
- `experiments.daily.config` -> (none)
- `experiments.daily.grouping` -> (none)
- `experiments.daily.run` -> `experiments.daily.config`, `experiments.eval.run`
- `experiments.equity_panel` -> (none)
- `experiments.equity_panel.reasons` -> (none)
- `experiments.equity_panel.run` -> `baselines`, `evaluation`, `evaluation.evaluate`, `experiments.equity_panel.reasons`, `experiments.prewhiten`, `finance.eval`, `finance.io`, `finance.portfolio`, `finance.portfolios`, `finance.returns`, `finance.robust`, `fjs.balanced`, `fjs.balanced_nested`, `fjs.dealias`, `fjs.gating`, `fjs.mp`, `fjs.robust`, `fjs.spectra`, `meta`, `meta.cache`, `meta.run_meta`, `plotting`
- `experiments.equity_panel.sweep_acceptance` -> `experiments.equity_panel.run`
- `experiments.etf_panel.run` -> `experiments.eval.run`
- `experiments.eval.config` -> `experiments.eval.run`
- `experiments.eval.diagnostics` -> (none)
- `experiments.eval.inject_spike` -> `eval.balance`, `eval.clean`, `experiments.daily.grouping`, `experiments.eval.config`, `experiments.eval.run`, `fjs.overlay`
- `experiments.eval.run` -> `baselines`, `baselines.covariance`, `baselines.factors`, `eval.balance`, `eval.clean`, `evaluation.dm`, `evaluation.evaluate`, `evaluation.factor`, `experiments.daily.grouping`, `experiments.eval.config`, `experiments.eval.diagnostics`, `experiments.prewhiten`, `finance`, `finance.portfolio`, `fjs.overlay`, `meta`
- `experiments.eval.sensitivity` -> `evaluation.dm`, `tools.verify_dataset`
- `experiments.prewhiten` -> `baselines`
- `experiments.synthetic` -> `experiments.synthetic.harness_utils`
- `experiments.synthetic.calibrate_thresholds` -> `experiments.synthetic.harness_utils`, `fjs`, `meta`, `synthetic.calibration`
- `experiments.synthetic.harness_utils` -> `experiments.synthetic_oneway.run`, `fjs.robust`
- `experiments.synthetic.nested_killtest` -> `experiments.equity_panel.run`, `fjs.balanced_nested`, `fjs.dealias`, `fjs.gating`, `fjs.robust`
- `experiments.synthetic.null` -> `experiments.synthetic.harness_utils`
- `experiments.synthetic.power` -> `experiments.synthetic.harness_utils`
- `experiments.synthetic.power_null` -> `evaluation.evaluate`, `experiments.synthetic_oneway.run`, `finance.eval`, `fjs.dealias`, `fjs.robust`
- `experiments.synthetic_oneway` -> (none)
- `experiments.synthetic_oneway.run` -> `fjs.balanced`, `fjs.dealias`, `fjs.spectra`, `meta.run_meta`, `plotting`

### `finance`
- `finance` -> `eval`, `io`
- `finance.design` -> (none)
- `finance.eval` -> `evaluation.factor`, `finance.factors`, `finance.ledoit`, `finance.robust`, `finance.shrinkage`, `fjs.balanced`, `fjs.dealias`
- `finance.factors` -> (none)
- `finance.io` -> (none)
- `finance.ledoit` -> `finance.shrinkage`
- `finance.loader` -> `finance.io`
- `finance.portfolio` -> (none)
- `finance.portfolios` -> (none)
- `finance.returns` -> (none)
- `finance.robust` -> (none)
- `finance.shrinkage` -> (none)

### `fjs`
- `fjs` -> (none)
- `fjs.balanced` -> (none)
- `fjs.balanced_nested` -> (none)
- `fjs.dealias` -> `fjs.balanced`, `fjs.mp`, `fjs.theta_solver`
- `fjs.gating` -> (none)
- `fjs.mp` -> (none)
- `fjs.overlay` -> `baselines.covariance`, `finance.ledoit`, `finance.shrinkage`, `fjs.dealias`, `fjs.gating`
- `fjs.robust` -> (none)
- `fjs.spectra` -> (none)
- `fjs.theta_solver` -> `fjs.mp`

### `io`
- `io.crsp_daily` -> (none)
- `io.wrds_connect` -> `utils.credentials`

### `meta`
- `meta.cache` -> (none)
- `meta.completeness` -> (none)
- `meta.run_meta` -> (none)
- `meta.runtime` -> (none)

### `plotting`
- `plotting` -> `utils`
- `plotting.utils` -> `evaluation.evaluate`, `fjs.spectra`

### `report`
- `report` -> (none)
- `report.gather` -> (none)
- `report.plots` -> (none)
- `report.tables` -> (none)

### `synthetic`
- `synthetic` -> (none)
- `synthetic.calibration` -> `fjs.balanced`, `synthetic.threshold_eval`
- `synthetic.threshold_eval` -> `fjs.balanced`, `fjs.dealias`

### `tools`
- `tools.aggregate_runs` -> (none)
- `tools.build_brief` -> `report.gather`
- `tools.build_gallery` -> `report.gather`, `report.plots`, `report.tables`
- `tools.build_memo` -> `report.gather`
- `tools.clean_outputs` -> (none)
- `tools.generate_project_state` -> (none)
- `tools.list_runs` -> (none)
- `tools.make_summary` -> `meta.completeness`
- `tools.plot_rc_hist` -> (none)
- `tools.prewhiten_effect` -> (none)
- `tools.reduce_calibration` -> `experiments.synthetic`
- `tools.run_monitor` -> (none)
- `tools.shard_grid` -> `experiments.synthetic.calibrate_thresholds`
- `tools.summarize_rc_sanity` -> `meta.completeness`
- `tools.summarize_run` -> (none)
- `tools.summarize_weekly_diagnostics` -> (none)
- `tools.update_registry` -> (none)
- `tools.verify_dataset` -> (none)

### `utils`
- `utils.credentials` -> (none)
