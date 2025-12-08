# Test Coverage

## Test suite structure (pytest, markers)
- Markers: `unit` (default), `integration`, `slow`, `heavy` (see `pytest.ini`; default excludes slow/heavy).
- Commands: `make test-fast` (unit), `make test-integration`, `make test-slow`, `make test` (full).

## Covered areas
- **Core FJS math**: `tests/test_dealias.py`, `test_dealias_search.py`, `test_mp.py`, `test_mp_edge_and_root.py`, `test_theta_solver.py`, `test_balanced.py`, `test_balanced_sigma_a.py`, `test_nested_balanced.py`, `test_robust_edge.py`, `test_s5_pairing_alignment.py` (alignment/energy diagnostics), `test_dealias_guardrails.py`.
- **Gating/overlay**: `tests/test_gating.py`, `test_overlay.py`, `test_cache_switch_estimator.py` (cache interactions), `test_alignment_diag.py`.
- **Finance/shrinkage**: `tests/test_shrinkage.py`, `baselines/test_covariance.py`, `test_minvar_regularized.py`, `test_var_backtests.py`, `test_weekly_cov_identity.py`, `finance` min-var turnover etc.
- **Data/registry**: `tests/test_data_registry.py`, `data/test_factors_registry.py`, `io/test_wrds_snapshot.py`.
- **Prewhitening/factors**: `baselines/test_prewhiten.py`, `baselines/test_load_factors.py`, `experiments/test_equity_prewhiten.py`, `experiments/test_prewhiten_utils.py`.
- **Experiments**: `tests/test_pipeline_smoke.py`, `tests/experiments/test_eval_run.py`, `test_equity_ablations.py`, `test_sweep_cli.py`, `test_equity_ablation_emits_sum0` etc. (many pytest-generated directories), `test_sharadar_pipeline_smoke.py`.
- **Synthetic**: `tests/synthetic/test_calibration.py`, `test_harness_utils.py`, `test_power_null.py`, `test_threshold_eval.py`.
- **Reporting**: `tests/test_report_gather.py`, `test_report_tables.py`, `test_report_plots.py`, `tools/test_make_summary.py`, `test_diagnostics.py`.
- **Utilities**: `test_cache.py`, `test_cache_switch_estimator.py`, `test_pairing.py`, `test_metrics_qlike.py`.

## Gaps / risks
- Balanced weight computation and MP PDF stubs untested (not implemented).
- Crisis/nested full-length RC flows not covered in tests (only smoke slices). Heavy performance tests marked `heavy`/`slow` and often skipped.
- AWS/HPC scripts and thread-cap behaviour rely on manual testing; limited automated coverage.
- PSD clipping paths in `apply_overlay`/`variance_forecast_from_components` rely on numerical tolerance; no dedicated regression tests.
