---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Known Issues / Limitations

- **Nested calibration refreshed (2025-12-20)** — Synthetic kill-test (p=200, weeks 6–8, reps=5, tyler) now shows 0/220 null detections (Wilson hi=0.017) with power 1.0 on moderate/strong spikes at delta_frac=0.05; calibration saved to `calibration/nested_edge_delta_thresholds.json` (RUN_NAME `20251220_011519_ticket-10_nested-null-fpr`). Real-data nested acceptance still needs re-check post-calibration.
- **Guardrail attribution unclear** — `guard_other` and `diagnostic_failure` dominate recent weekly diagnostics (ticket-07); root causes not yet traced, so gating decisions may be hiding actionable failures.
- **Crisis degradation risk** — Crisis 2020/2022 configs have shown harmful overlay in past runs; keep gates strict and validate completeness before using crisis outputs.
- **Vol-state acceptance low** — Vol-state design remains below target detection/acceptance (≈0–0.7% in recent runs) despite prewhitening tweaks.
- **Cache staleness** — `.cache/` keys do not encode reporting/evaluation code; invalidate caches when changing summary logic or gating defaults.
- **Optional dependencies** — `cvxpy` is required for MV optimisation; absence now raises `MissingSolverError` unless `mv_skip_on_missing_solver` is set (skip produces flagged empty weights). Matplotlib is optional; plots are skipped when missing.
- **Large outputs** — Historical outputs in `experiments/equity_panel/outputs_*` and `reports/` can be heavy; avoid deleting or overwriting, create new timestamped dirs instead.
