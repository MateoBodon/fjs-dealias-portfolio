---
generated: 2025-12-22T20:55:48Z
git_sha: a7d76d8cf7f5fe4c9765c335530064170a0ca87a
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py
  - python3 - <<'PY' (emit FUNCTION_INDEX.md + DEPENDENCY_GRAPH.md)
  - manual edit (INDEX.md)
---

# Project State Index

How to read this folder without repo access:
- **ARCHITECTURE.md** — system overview, layers, and main entrypoints.
- **MODULE_SUMMARIES.md** — package/module inventory with role notes.
- **FUNCTION_INDEX.md** — AST-derived list of top-level classes/functions (line numbers + signatures).
- **DEPENDENCY_GRAPH.md** — internal import fan-in/out summary (see `_generated/import_graph.json`).
- **PIPELINE_FLOW.md** — execution paths for RC/RC-lite/synthetic/reporting.
- **DATAFLOW.md** — datasets, registries, caches, and IO contracts.
- **EXPERIMENTS.md** — configs/targets and expected outputs.
- **CURRENT_RESULTS.md** — latest validated runs and metrics (capped runs flagged).
- **RESEARCH_NOTES.md** — recent findings/interpretations tied to artifacts.
- **OPEN_QUESTIONS.md** — decisions/analyses still pending.
- **KNOWN_ISSUES.md** — bugs, limitations, and guardrails.
- **ROADMAP.md** — near- and mid-term milestones (aligned to `Long_Term_Plan.md`).
- **CONFIG_REFERENCE.md** — CLI flags, env vars, YAML keys.
- **SERVER_ENVIRONMENT.md** — runtime/dependency expectations.
- **TEST_COVERAGE.md** — test surface + gaps.
- **STYLE_GUIDE.md** — coding/doc conventions.
- **CHANGELOG.md** — changes to `project_state/` itself.
- **_generated/** — machine outputs (do not edit): `repo_inventory.json`, `symbol_index.json`, `import_graph.json`, `make_targets.txt`.
