---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Research Notes

- **Window coverage fix pending in this branch** — The ticket-05 daily DoW run is capped by `window_coverage`; later branches mention a fix, but this branch still records the capped attempt (see `reports/rc-ticket-05-20251221_221902/summary/limitations.md`).
- **Weekly gating diagnostics improved** — Ticket-09 smoke adds `guard_unknown` and explicit skip_reason columns; `guard_other` is absent in the latest smoke (see `experiments/equity_panel/outputs_smoke/.../weekly_diagnostics.md`).
- **Nested calibration gap remains** — Nested weekly smokes report `calibration_missing_p_T` for p≈188 and T∈{70,80}; calibrated grid does not cover these regimes.
- **MV solver handling is fail-loud** — `finance.portfolios` raises on missing cvxpy unless explicitly skipped (`mv_skip_on_missing_solver`); no equal-weight fallback.
- **RC-lite sanity shows harmful deltas** — Deterministic sanity run remains valid but overlay ΔMSE > 0 in summary, so delta comparisons are marked invalid in summary tables (see `reports/rc-20251220-sanity-20251220_233700/summary/summary_perf.csv`).
