# RESULTS

- Code: `finance/portfolios.py` now exposes `skip_reason/solver_used`, honors `FJS_FORCE_MISSING_CVXPY`, and never drops to EW when cvxpy is absent; `experiments/eval/run.py` adds `mv_solver=cvxpy` + `mv_skip_on_missing_solver`, propagates `skipped/skip_reason/solver_status` into metrics/diagnostics, and new tests cover the missing-solver path.
- Tests: `source .venv/bin/activate && make test-fast` (pass; 68 passed, 151 deselected).
- Smoke (cvxpy present): `reports/eval-smoke-ticket08-proof/normal/` from `--mv-solver cvxpy` (max_windows=2) shows `solver_status=optimal` and `skipped=False` for all MV rows (`metrics_detail.csv`).
- Forced-missing (env flag + skip): `reports/eval-smoke-ticket08-proof/missing-skip/` from `FJS_FORCE_MISSING_CVXPY=1 --mv-skip-on-missing-solver` records MV rows with `skipped=True`, `skip_reason=missing_solver`, `solver_status=missing_solver`; diagnostics `full/diagnostics.csv` reports `mv_skipped_share=1.0`.
- Default fail-loud verified via unit tests: `optimize_portfolio` and `_min_variance_weights` raise `MissingSolverError` when cvxpy is missing and skip flag is false.
- Bundle: `docs/gpt_bundles/20251219_204908_ticket-08_20251219_202301_ticket-08_solver-missing-proof.zip`.
