---
generated: 2025-12-20T20:36:15+00:00
git_sha: 39e2889e353f54cbd502162f8aca5e9b3cc212b0
git_branch: codex/ticket-09-gating-diagnostics-attribution
commands:
  - manual update during ticket-09 (gating diagnostics attribution)
---

# Known Issues / Limitations

- **Evaluation contamination (aligned windows)** — RESOLVED in tickets 11/15: Δ/DM now use aligned window intersections with `n_effective_*` and `comparison_valid*` surfaced; caps/truncation recorded in run metadata and excluded from headline aggregates. Residual risk: tiny caps can still invalidate comparisons when aligned sample < `min_comparison_windows` (default 30); monitor n_effective in smoke outputs before citing results.

- **Nested calibration refreshed (2025-12-20)** — Synthetic kill-test (p=200, weeks 6–8, reps=5, tyler) shows 0/220 null detections (Wilson hi=0.017) with power 1.0 on moderate/strong spikes at delta_frac=0.05; calibration saved to `calibration/nested_edge_delta_thresholds.json` (RUN_NAME `20251220_011519_ticket-10_nested-null-fpr`) with embedded metadata/config hash. Real-data nested smoke rechecked (deterministic, max_windows=3) and still blocked: all 3 windows skipped with `calibration_missing_p_T` (p≈188, T=70/80 outside calibrated grid) and stability guards; delta_frac fell back to config 0.008. Need calibrated coverage for smaller p to unlock nested acceptance.
- **Guardrail attribution unclear** — RESOLVED in ticket-09: gating diagnostics emit explicit skip_reason_primary/detail plus exception_type/stage/message_short; `guard_other` is unreachable in smokes (0 counts) and unknown guard keys now surface as `guard_unknown` with key detail. Monitor future smokes for regressions.
- **Crisis degradation risk** — Crisis 2020/2022 configs have shown harmful overlay in past runs; keep gates strict and validate completeness before using crisis outputs.
- **Vol-state acceptance low** — Vol-state design remains below target detection/acceptance (≈0–0.7% in recent runs) despite prewhitening tweaks.
- **Cache staleness** — `.cache/` keys do not encode reporting/evaluation code; invalidate caches when changing summary logic or gating defaults.
- **Optional dependencies** — `cvxpy` is required for MV optimisation; absence now raises `MissingSolverError` unless `mv_skip_on_missing_solver` is set (skip produces flagged empty weights). Matplotlib is optional; plots are skipped when missing.
- **Large outputs** — Historical outputs in `experiments/equity_panel/outputs_*` and `reports/` can be heavy; avoid deleting or overwriting, create new timestamped dirs instead.
