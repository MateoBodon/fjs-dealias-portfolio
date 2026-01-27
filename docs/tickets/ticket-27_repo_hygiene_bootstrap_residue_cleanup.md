# Ticket-27: Repo hygiene - remove bootstrap residue + enforce ignore rules

## Goal
Restore a clean, tracking-policy-compliant tree by removing bootstrap residue (`*.bak.*`, `*.append*`), ensuring scratch ignores (esp. `reports/_runs/`) stay enforced, and adding a guardrail test.

## Context
- Diagnosis flagged dirty worktree with backup residue + untracked scratch (`project_state/_generated/git_status.txt` in `docs/gpt_outputs/fjs01-26-26prompt-1diagnosis.md`).
- Tracking policy requires scratch output to live under ignored zones (`reports/_runs/`, `artifacts/_local/`).
- A guardrail is needed so backup residue cannot silently return.

## Constraints
- No surprise top-level directories.
- Bulky outputs go only to `artifacts/_local/` or `reports/_runs/` (ignored).
- If a cleanup/validation run is performed, create a run log under `docs/agent_runs/<RUN_NAME>/`.
- Align with `TRACKING_POLICY.md` and guardrails (`make check-data-policy`, `make validate-runlogs`).

## Plan
1. Create a run log for the cleanup and capture baseline `git status -sb` and `git clean -ndx`.
2. Remove `*.bak.*` / `*.append*` residue and resolve any untracked bootstrap leftovers.
3. Verify `.gitignore` compliance for scratch zones and add guardrail test.
4. Update docs (`PROGRESS.md`, ticket file, test coverage) and run checks.

## Acceptance criteria
- [ ] `git status -sb` is clean (no modified tracked files, no residue).
- [ ] No `*.bak.*` or `*.append*` files exist under the repo root.
- [ ] `reports/_runs/` is ignored per `TRACKING_POLICY.md` and does not appear in `git status`.
- [ ] New guardrail test exists: `tests/test_repo_hygiene.py`.
- [ ] `make test-fast`, `make validate-runlogs`, and `make check-data-policy` pass.

## Test plan
- [ ] `git status -sb`
- [ ] `git clean -ndx`
- [ ] `pytest -q tests/test_repo_hygiene.py`
- [ ] `make test-fast`
- [ ] `make validate-runlogs`
- [ ] `make check-data-policy`

## Artifacts / Outputs
- `docs/agent_runs/<RUN_NAME>/` (run log)
- `tests/test_repo_hygiene.py`
- `.gitignore` (if drift fix required)
- `docs/tickets/ticket-27_repo_hygiene_bootstrap_residue_cleanup.md`

## Notes / Risks
- Risk: deleting something valuable from untracked residue. Mitigation: dry-run `git clean -ndx` before removals.
- Rollback: restore tracked files with `git restore <path>` if needed.
