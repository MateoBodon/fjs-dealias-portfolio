---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Known Issues / Limitations

- **Nested coverage still near zero** — Even with relaxed guardrails, nested smoke/rc-lite-sanity runs accept nothing; nested kill-test shows high FPR. Needs re-tuning before relying on nested designs.
- **Guardrail attribution unclear** — `guard_other` and `diagnostic_failure` dominate recent weekly diagnostics (ticket-07); root causes not yet traced, so gating decisions may be hiding actionable failures.
- **Crisis degradation risk** — Crisis 2020/2022 configs have shown harmful overlay in past runs; keep gates strict and validate completeness before using crisis outputs.
- **Vol-state acceptance low** — Vol-state design remains below target detection/acceptance (≈0–0.7% in recent runs) despite prewhitening tweaks.
- **Cache staleness** — `.cache/` keys do not encode reporting/evaluation code; invalidate caches when changing summary logic or gating defaults.
- **Optional dependencies** — `cvxpy` is required for MV optimisation; absence now raises `MissingSolverError` unless `mv_skip_on_missing_solver` is set (skip produces flagged empty weights). Matplotlib is optional; plots are skipped when missing.
- **Large outputs** — Historical outputs in `experiments/equity_panel/outputs_*` and `reports/` can be heavy; avoid deleting or overwriting, create new timestamped dirs instead.
