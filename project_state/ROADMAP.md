---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Roadmap

## Near-term (next runs)
- Add/restore `experiments/eval/config.paper_v1.yaml` (or update Makefile) so paper runs are reproducible without silent fallback.
- Rerun daily DoW paper config with uncapped windows and validate headline metrics (`cap_active=false`, `comparison_valid_*` = 1).
- Expand nested calibration grid to cover p≈188, T≈70/80; re-run `config.nested.smoke.tiny.yaml` to confirm detection coverage.
- Run weekly smokes with `--gating-diagnostics` to confirm `guard_unknown=0` and absence of `guard_other` after ticket-09 changes.

## Mid-term
- Vol-state acceptance tuning to reach 2–6% detection/acceptance band on balanced panels.
- Crisis-window evaluations (2020/2022 configs) with completeness checks to validate safety.
- Compare overlay vs factor-only (observed-factor + POET-lite) on aligned window sets.

## Longer-term (from `Long_Term_Plan.md`)
- Maintain synthetic null/power calibration suite and documented operating points.
- Establish publishable RC-lite grid with advisor-ready memo/brief outputs.
