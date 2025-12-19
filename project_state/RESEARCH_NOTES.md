---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Research Notes

- **Overlay rarely fires on balanced panels** — Despite calibrated delta_frac grids, many designs still show zero detections (nested, vol-state) or heavy `guard_other` counts (ticket-07 diagnostics). Synthetic micro smokes failing with `diagnostic_failure` suggest gate diagnostics need finer attribution.
- **MV solver handling clarified** — `finance.portfolios` now fails loud by default when `cvxpy` is missing; skip path is explicit (`skip_on_missing_solver`, propagated through metrics). No more EW fallback when solver absent.
- **Crisis performance still fragile** — Prior RC-lite runs (2025-11-21) show small acceptance with mixed ΔMSE signs; crisis slices (2020/2022) remain riskier and are gated via `config.crisis.*.yaml`.
- **Calibration defaults** — `calibration/defaults.json` and `calibration/edge_delta_thresholds.json` (2025-11-21) remain in force; energy_floor≈0.108 for SCM edge at 2% FPR. Any re-tune must log before/after thresholds.
- **Prewhitening impact** — FF5+MOM prewhitening is the default in eval/equity flows; factor registries are enforced. Vol-state acceptance remains low even with prewhitening, indicating gating issues beyond factor alignment.
- **Completeness tooling** — `meta/completeness.py`, `tools/make_summary.py`, and `tools/summarize_rc_sanity.py` now surface missing sections and harmful overlay effects; helps avoid silently treating incomplete RC drops as valid.
