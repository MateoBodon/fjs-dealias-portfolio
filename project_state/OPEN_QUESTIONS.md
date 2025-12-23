---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Open Questions

- How to eliminate daily DoW window coverage caps for paper runs? (See ticket-05 capped run in `reports/rc-ticket-05-20251221_221902/`.)
- How to raise nested acceptance without inflating FPR? Extend calibration grid beyond p≈188, T≈70/80 and retest `config.nested.smoke.tiny.yaml`.
- Do weekly designs still emit `guard_other` or `diagnostic_failure` in larger smokes after ticket-09 attribution changes? Monitor `gating_diagnostics.csv` in new weekly runs.
- Can vol-state design hit the 2–6% acceptance band on balanced panels? Requires focused `config.smoke.yaml` sweeps.
- Crisis safety: need uncapped crisis runs (2020/2022 configs) with completeness checks to assess harm in stress regimes.
