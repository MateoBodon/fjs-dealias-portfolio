---
generated: 2025-12-23T22:59:22Z
git_sha: def4b2a1a49ad0c045896c8d4e9d9a3433e160a0
git_branch: codex/ticket-18-inject-spike-sensitivity
commands:
  - manual edit (ticket-18 injection notes)
---
# Research Notes

- **Injection sensitivity (ticket-18)** — Real-data injection smoke (2024-01-01→2024-03-31, window=40, horizon=5, assets_top=30) reports baseline detection/acceptance = 0 and μ=3/6/9 detection/acceptance = 0 (acceptance drought; `n_changed=0`, skip histogram empty). Curve + plot: `reports/inject_spike/20251223_225141/curve.csv` and `reports/inject_spike/20251223_225141/curve.png`.
- **Window coverage fix pending in this branch** — The ticket-05 daily DoW run is capped by `window_coverage`; later branches mention a fix, but this branch still records the capped attempt (see `reports/rc-ticket-05-20251221_221902/summary/limitations.md`).
- **Weekly gating diagnostics improved** — Ticket-09 smoke adds `guard_unknown` and explicit skip_reason columns; `guard_other` is absent in the latest smoke (see `experiments/equity_panel/outputs_smoke/.../weekly_diagnostics.md`).
- **Nested calibration gap remains** — Nested weekly smokes report `calibration_missing_p_T` for p≈188 and T∈{70,80}; calibrated grid does not cover these regimes.
- **MV solver handling is fail-loud** — `finance.portfolios` raises on missing cvxpy unless explicitly skipped (`mv_skip_on_missing_solver`); no equal-weight fallback.
- **RC-lite sanity shows harmful deltas** — Deterministic sanity run remains valid but overlay ΔMSE > 0 in summary, so delta comparisons are marked invalid in summary tables (see `reports/rc-20251220-sanity-20251220_233700/summary/summary_perf.csv`).
