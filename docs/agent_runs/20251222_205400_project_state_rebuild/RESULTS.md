# Results

## Changes
- Updated `tools/generate_project_state.py` to:
  - limit AST indexing to `src/`, `experiments/`, and `tools/`.
  - include function signatures and class bases in `symbol_index.json`.
  - preserve `experiments.*` and `tools.*` module prefixes for import graph clarity.
- Regenerated machine artifacts: `project_state/_generated/{repo_inventory.json,symbol_index.json,import_graph.json,make_targets.txt}`.
- Rewrote all `project_state/*.md` with fresh metadata headers, updated module inventory, config reference, and current results summaries.
- Added new Known Issue noting missing `experiments/eval/config.paper_v1.yaml` and silent fallback behavior.
- Created run log directory: `docs/agent_runs/20251222_205400_project_state_rebuild/`.
- Updated `PROGRESS.md` with this rebuild entry.

## Artifacts
- Project state bundle: `docs/gpt_bundles/project_state_20251222_205400_a7d76d8.zip`.
- Updated docs: `project_state/` (all required files), `_generated/` indices, `PROGRESS.md`.
