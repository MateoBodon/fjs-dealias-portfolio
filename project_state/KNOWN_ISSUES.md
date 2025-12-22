---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# Known Issues

- **Missing paper-v1 config file** — Makefile references `experiments/eval/config.paper_v1.yaml`, but the file is absent in this branch; `experiments/eval/config.py` silently falls back to defaults when the file is missing.
- **Nested calibration coverage gap** — Weekly nested smokes show `calibration_missing_p_T` for p≈188, T∈{70,80} (see `experiments/equity_panel/outputs_nested_smoke_tiny/`).
- **Weekly detection scarcity** — Many weekly/nested runs still report zero detections; overlay effectively off in those windows (see `experiments/equity_panel/outputs_*/*/summary.json`).
- **Capped runs not headline** — `tools/make_summary.py` excludes `cap_active=true` runs from headline tables and lists cap sources in limitations; ticket-05 daily DoW run is capped by window coverage.
- **Heavy ablation runtime** — `make rc-ablations` can time out locally; use smaller grids or remote hosts.
