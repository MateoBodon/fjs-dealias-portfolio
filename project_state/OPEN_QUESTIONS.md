---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Open Questions

- What is driving `guard_other` / `diagnostic_failure` in weekly gating diagnostics (ticket-07)? Need per-guard counters and repro using `experiments/equity_panel/outputs_smoke_ticket07_20251219_173231`.
- How to raise nested design acceptance without blowing FPR? Explore isolation/eta/delta_frac relaxations with `config.nested.smoke.yaml` and nested kill-test configs.
- Can vol-state design hit 2–6% detection/acceptance on balanced panels? Try group_min_replicates/rep counts plus q2 alignment tweaks.
- Crisis safety: Do current gates prevent harmful overlay in 2020/2022 crisis configs? Require targeted reruns with `config.crisis.*.yaml` and completeness checks.
- Factor vs overlay: Is observed-factor covariance + POET-lite sufficient once prewhitening is applied? Compare against dealias overlay on the same runs (use eval runner outputs).
- AWS/Hetzner parity: Are AWS rc targets still blocked by missing `INSTANCE_DNS`/SSH config? (see `reports/aws/` manifests); decide whether to deprecate or restore.
