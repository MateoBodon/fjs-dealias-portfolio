# FJS-TKT-024

## Goal
Make agentic/Codex-worker generated artifacts not dirty git status by replacing `.gitignore.append` with real `.gitignore` rules and verifying tools/agentic outputs land in ignored paths.

## Scope
- Ensure root `.gitignore` contains ignores for `docs/_generated/`, `docs/_bundles/`, `docs/agent_runs/`, `project_state/_generated/`.
- Remove `.gitignore.append` after integrating its rules.
- Confirm `tools/agentic/project_state_refresh.py` and `tools/agentic/repo_snapshot.py` outputs are covered by ignores.

## Acceptance Criteria
- `.gitignore` contains the intended ignore patterns.
- `.gitignore.append` is removed.
- Running `python3 tools/agentic/project_state_refresh.py --zip` and `python3 tools/agentic/repo_snapshot.py` does not introduce untracked files (git status clean aside from expected tracked diffs).
- `make test-fast` passes.

## Plan
1. Inspect `.gitignore` and `.gitignore.append` for intended ignore rules.
2. Update `.gitignore` if needed and delete `.gitignore.append`.
3. Verify tool outputs match ignored paths (`docs/_generated/`, `docs/_bundles/`, `project_state/_generated/`).
4. Run `make test-fast` and the agentic tool commands; confirm `git status --porcelain` is clean of new untracked files.
5. Update `PROGRESS.md`, run log files, and generate the GPT bundle.

## Notes
- No experiment logic changes required.
