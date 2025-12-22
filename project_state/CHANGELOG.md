---
generated: 2025-12-22T21:04:17Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - python3 - <<'PY' (write project_state docs)
---
# project_state Change Log

- **2025-12-22** — Rebuilt `project_state/` and `_generated` artifacts; refreshed module inventory, dependency graph, config references, and run summaries for ticket-05/09/14/15 state. Updated `tools/generate_project_state.py` to scope AST/indexing to src/experiments/tools and to include signatures/bases.
- **2025-12-19** — Rebuilt project_state folder; added `tools/generate_project_state.py` + refreshed `_generated/{repo_inventory,symbol_index,import_graph,make_targets}`; regenerated FUNCTION_INDEX/DEPENDENCY_GRAPH; updated configs/results/issues to reflect ticket-05/07/08 runs and MV solver skip/fail-loud behaviour.
