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

- **Weekly detection scarcity**: Many weekly/nested runs still report zero detections; overlay is effectively off in those windows (see `experiments/equity_panel/outputs_nested_smoke*/summary.json`).
- **Capped runs not headline (resolved 2025-12-21)**: `tools/make_summary.py` now excludes `cap_active=true` runs from headline tables and lists cap sources in limitations (ticket-02).
- **Holdout-empty windows counted as caps (resolved 2025-12-22)**: daily eval window planning now drops holdout-empty windows from `windows_requested` and logs `windows_dropped_holdout_empty`, preventing false `window_coverage` caps on uncapped runs (ticket-06).
- **Heavy ablation runtime**: `make rc-ablations` can time out on local hosts; use smaller grids or remote hosts.
