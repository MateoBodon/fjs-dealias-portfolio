# FJS-TKT-030

## Goal
Normalize tracking policy enforcement by migrating local ignores, moving run dumps to canonical scratch zones, and ensuring ticket/run-log scaffolding is tracked.

## Scope
- Clear policy rules from `.git/info/exclude`.
- Move untracked run dumps under `reports/_runs/`.
- Ensure docs scaffolding README placeholders are tracked.
- Track existing tickets and agent run logs.

## Acceptance Criteria
- `.git/info/exclude` contains no shared policy rules.
- Untracked run dumps under `reports/` are moved to `reports/_runs/`.
- `docs/tickets/README.md`, `docs/agent_runs/README.md`, and `docs/artifacts/README.md` are tracked.
- `docs/tickets/*.md` and `docs/agent_runs/**` are tracked (small files only).
- `git status --porcelain` shows no run-dump directories under `reports/`.
