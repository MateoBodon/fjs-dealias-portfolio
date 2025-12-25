---
generated: 2025-12-25T21:49:13Z
git_sha: 22523fc301aa7228193bf135ae9615974cb631c0
git_branch: codex/ticket-18-inject-spike-sensitivity
commands:
  - manual edit (ticket-18 injection notes update)
---
# Research Notes

- **Injection diagnostics (ticket-23)** — DoW injection run (2022-01-01→2022-12-31, assets_top=80, max_windows=25) remains flat-zero; new `gating_reasons.csv` shows pre-gate `tvec_compute_error` + `tvec_off_component` dominating (no candidates admitted). Week runs on the same regime were too slow to complete locally (t-vector search dominates runtime), so week viability remains unverified pending a faster run path.
- **Injection sensitivity (ticket-18)** — Make target run (2024-01-01→2024-06-30, window=40, horizon=5, assets_top=25) reports baseline detection/acceptance = 0 and μ=3/4/5 detection/acceptance = 0 with `n_detected=0` (pre-gate drought). Curve + plot: `reports/inject_spike/20251225_213525/curve.csv` and `reports/inject_spike/20251225_213525/curve.png`. Larger slice with μ=3/6/9/12/15 (`reports/inject_spike/20251224_051700/`) and the smaller 2024-01-01→2024-03-31 slice matched the zero-response pattern.
- **Window coverage fix pending in this branch** — The ticket-05 daily DoW run is capped by `window_coverage`; later branches mention a fix, but this branch still records the capped attempt (see `reports/rc-ticket-05-20251221_221902/summary/limitations.md`).
- **Weekly gating diagnostics improved** — Ticket-09 smoke adds `guard_unknown` and explicit skip_reason columns; `guard_other` is absent in the latest smoke (see `experiments/equity_panel/outputs_smoke/.../weekly_diagnostics.md`).
- **Nested calibration gap remains** — Nested weekly smokes report `calibration_missing_p_T` for p≈188 and T∈{70,80}; calibrated grid does not cover these regimes.
- **MV solver handling is fail-loud** — `finance.portfolios` raises on missing cvxpy unless explicitly skipped (`mv_skip_on_missing_solver`); no equal-weight fallback.
- **RC-lite sanity shows harmful deltas** — Deterministic sanity run remains valid but overlay ΔMSE > 0 in summary, so delta comparisons are marked invalid in summary tables (see `reports/rc-20251220-sanity-20251220_233700/summary/summary_perf.csv`).
