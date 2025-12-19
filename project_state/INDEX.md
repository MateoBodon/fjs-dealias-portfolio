---
generated: 2025-12-19T21:04:10+01:00
git_sha: ce4c1b224c43028bb5388efdebbe0e8eb52e6c61
git_branch: chore/project_state_refresh
commands:
  - python3 tools/generate_project_state.py (latest run excludes heavy caches/outputs)
  - python3 - <<'PY' (emit project_state docs and indexes)
---

# Project State Index

How to read this folder without repo access:
- **ARCHITECTURE.md** — system overview, layers, and main entrypoints.
- **MODULE_SUMMARIES.md** — what each top-level package does.
- **FUNCTION_INDEX.md** — AST-derived list of public classes/functions (with line numbers).
- **DEPENDENCY_GRAPH.md** — internal import fan-in/fan-out snapshot.
- **PIPELINE_FLOW.md** — execution paths for RC/RC-lite/synthetic/reporting.
- **DATAFLOW.md** — datasets, registries, caches, and IO contracts.
- **EXPERIMENTS.md** — configs/targets and expected outputs.
- **CURRENT_RESULTS.md** — latest validated runs and metrics.
- **RESEARCH_NOTES.md** — interpretations + context from recent work.
- **OPEN_QUESTIONS.md** — decisions/analyses still pending.
- **KNOWN_ISSUES.md** — bugs, limitations, and safety guardrails.
- **ROADMAP.md** — near- and mid-term milestones tied to LONG_TERM_PLAN.
- **CONFIG_REFERENCE.md** — CLI flags, env vars, YAML keys.
- **SERVER_ENVIRONMENT.md** — runtime/dependency expectations.
- **TEST_COVERAGE.md** — test surface + gaps.
- **STYLE_GUIDE.md** — coding/doc conventions.
- **CHANGELOG.md** — doc-only changes to `project_state/`.
- `_generated/` — machine outputs (do not edit): `repo_inventory.json`, `symbol_index.json`, `import_graph.json`, `make_targets.txt`.
