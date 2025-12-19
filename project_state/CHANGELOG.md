---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# project_state Change Log

- **2025-12-19** — Rebuilt project_state folder; added `tools/generate_project_state.py` + refreshed `_generated/{repo_inventory,symbol_index,import_graph,make_targets}`; regenerated FUNCTION_INDEX/DEPENDENCY_GRAPH; updated configs/results/issues to reflect ticket-05/07/08 runs and MV solver skip/fail-loud behaviour.
