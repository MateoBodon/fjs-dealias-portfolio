---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Test Coverage

- **Commands** — `make test-fast` (pytest -m unit), `make test-integration`, `make test-slow`, `make test` (full).
- **Latest recorded runs** — per PROGRESS.md: ticket-09 ran `make test-fast` (2025-12-20); ticket-15 ran targeted eval tests + `make test-fast`.
- **Areas covered** (representative):
  - FJS math/gating: `tests/test_dealias.py`, `tests/test_mp_edge_and_root.py`, `tests/test_gating.py`, `tests/test_theta_solver.py`, `tests/test_balanced*`, `tests/test_nested_balanced.py`, `tests/test_nested_smoke.py`, `tests/fjs/test_overlay.py`.
  - Synthetic harness: `tests/test_power_null.py`, `tests/synthetic/test_calibration.py`, `tests/synthetic/test_harness_utils.py`, `tests/test_calibrate_defaults.py`, `tests/test_threshold_eval.py`.
  - Evaluation runners: `tests/experiments/test_eval_run.py`, `tests/experiments/test_gating_diagnostics.py`, `tests/experiments/test_skip_reasons.py`, `tests/test_pipeline_smoke.py`.
  - Finance/portfolio: `tests/test_portfolios_missing_solver.py`, `tests/test_eval_missing_solver.py`, `tests/test_minvar_regularized.py`, `tests/test_shrinkage.py`, `tests/test_factor_cov.py`, `tests/test_cache_switch_estimator.py`.
  - Reporting: `tests/test_report_gather.py`, `tests/test_report_tables.py`, `tests/test_report_plots.py`, `tests/tools/test_make_summary.py`, `tests/tools/test_summarize_rc_sanity.py`, `tests/test_gpt_bundle.py`.
  - Data/registry: `tests/data/test_factors_registry.py`, `tests/io/test_wrds_snapshot.py`, `tests/test_data_registry.py`.
- **Gaps / heavy tests**
  - Full RC/RC-lite/AWS paths are not part of the fast suite; rely on smokes + manual runs.
  - Crisis configs, vol-state acceptance, and nested kill-test FPR remain mostly smoke-tested.
  - Plotting tests skip when matplotlib is unavailable.
- **This rebuild** — no tests executed (documentation-only changes).
