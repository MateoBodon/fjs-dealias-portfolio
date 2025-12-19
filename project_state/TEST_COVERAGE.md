---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Test Coverage

- **Commands** — `make test-fast` (pytest -m unit), `make test-integration`, `make test-slow`, `make test` (full). Latest recorded run (2025-12-19 ticket-08) shows 68 passed, 151 deselected on `make test-fast` (per PROGRESS.md). No tests were executed during this project_state rebuild.
- **Areas covered** (representative):
  - FJS math/gating: `tests/test_dealias.py`, `test_mp_edge_and_root.py`, `test_gating.py`, `test_theta_solver.py`, `test_balanced*`, `test_nested_balanced.py`, `test_nested_smoke.py`, `tests/fjs/test_overlay.py`.
  - Synthetic harness: `tests/test_power_null.py`, `tests/synthetic/test_calibration.py`, `tests/synthetic/test_harness_utils.py`, `tests/test_calibrate_defaults.py`, `tests/test_threshold_eval.py`.
  - Evaluation runners: `tests/experiments/test_eval_run.py`, `test_gating_diagnostics.py`, `test_skip_reasons.py`, `tests/test_pipeline_smoke.py`.
  - Finance/portfolio: `tests/test_portfolios_missing_solver.py`, `tests/test_eval_missing_solver.py`, `tests/test_minvar_regularized.py`, `tests/test_shrinkage.py`, `tests/test_factor_cov.py`, `tests/test_cache_switch_estimator.py`.
  - Reporting: `tests/test_report_gather.py`, `tests/test_report_tables.py`, `tests/test_report_plots.py`, `tests/tools/test_make_summary.py`, `tests/tools/test_summarize_rc_sanity.py`, `tests/test_gpt_bundle.py`.
  - Data/registry: `tests/data/test_factors_registry.py`, `tests/io/test_wrds_snapshot.py`, `tests/test_data_registry.py`.
- **Gaps / heavy tests**
  - Full RC / RC-lite / AWS paths are not in fast suite; rely on smoke configs and manual runs.
  - Crisis configs, vol-state acceptance, and nested kill-test FPR are not yet regression-tested.
  - Plotting relies on matplotlib availability; skipped when library missing.
