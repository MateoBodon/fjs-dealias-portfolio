---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Roadmap (near-term)

- **Guardrail diagnostics** — Add explicit counters/labels for `guard_other` / `diagnostic_failure` paths in weekly runner; rerun ticket-07 smokes to confirm attribution.
- **Nested/vol acceptance lift** — Experiment with softer isolation, eta, and delta_frac for nested and vol-state designs on smoke panels; log power/FPR on nested kill-test + synthetic harness.
- **Crisis safety check** — Rerun crisis configs (2020/2022) with completeness reporting and compare overlay vs baseline ΔMSE, VaR/ES; consider tighter gate defaults for crisis regimes.
- **Calibration refresh** — After gating changes, regenerate `calibration/edge_delta_thresholds.json` and ROC figures; update `calibration_defaults.json` with before/after hashes.
- **Reporting polish** — Ensure `tools/make_summary.py` / `summarize_rc_sanity.py` consume new diagnostics; keep memo/gallery templates aligned with latest metrics fields.
- **AWS pipeline decision** — Decide whether to restore AWS rc targets (configure `INSTANCE_DNS`/SSH) or document deprecation.
