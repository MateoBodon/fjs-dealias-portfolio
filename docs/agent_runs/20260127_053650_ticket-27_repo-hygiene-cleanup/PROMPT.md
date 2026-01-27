# Ticket-27: Repo hygiene - remove bootstrap residue + enforce ignore rules

## Goal

Restore a clean, tracking-policy-compliant `main` by removing bootstrap residue (`*.bak.*`, `*.append*`), fixing `.gitignore` drift (esp. `reports/_runs/`), and adding a guardrail test so it can’t regress.

## Context

The current snapshot shows a dirty worktree with many untracked backup/residue files and at least one scratch/run directory that should not be showing up in `git status`. This is explicitly called out as tracking-policy drift and a reproducibility risk in `fjs01-26-26prompt-1diagnosis.md` (Ticket 1). 

Key paths referenced in the diagnosis (inspect in-repo):

* `project_state/_generated/git_status.txt` (evidence of modified files + untracked `*.bak.*`/`*.append*`)
* `.gitignore` (likely drift causing `reports/_runs/` to appear)
* `TRACKING_POLICY.md` (states `reports/_runs/` should be ignored)
* `docs/agent_runs/20260127_024404_ticket-00_agentic-bootstrap-refresh/` (untracked run log folder to decide on)
* `reports/_runs/` (untracked scratch dir that should be ignored)

## Constraints

* **Tracking/logging constraints**

  * No surprise top-level directories.
  * Bulky outputs must go to `artifacts/_local/` or `reports/_runs/` (and must be ignored appropriately).
  * If you perform a “run” (cleanup script, refresh, validation sweep), create a run log under `docs/agent_runs/<RUN_NAME>/` and ensure `make validate-runlogs` passes.
* **Repo-specific constraints**

  * Align with `TRACKING_POLICY.md` and existing guardrails (`make check-data-policy`, `make validate-runlogs`).
  * Do **not** “fix” hygiene by weakening policy checks—fix the root cause (ignore rules + residue generation).

## Plan

1. **Create a run log for this cleanup**

   * Create `docs/agent_runs/<RUN_NAME>/` (suggest: `<YYYYMMDD_HHMMSS>_ticket-27_repo-hygiene-cleanup/`).
   * Record: baseline `git status -sb`, baseline `git clean -ndx`, and a short narrative of what you changed and why.
2. **Audit + classify residue**

   * List all `*.bak.*` / `*.append*` files and any unexpected untracked directories.
   * For each “modified tracked file” reported by `git status`, decide: revert (if accidental) vs keep (if intentional) and commit as part of this ticket.
3. **Fix ignore drift (policy compliance)**

   * Ensure `.gitignore` matches `TRACKING_POLICY.md` for ignored scratch zones, especially `reports/_runs/`.
   * Confirm `artifacts/_local/` and `reports/_runs/` are not accidentally being tracked or causing noise.
4. **Remove residue safely**

   * Use `git clean -ndx` (dry run) first; then remove residue (prefer `git clean -fdx` only after verifying patterns are correct).
   * Ensure no `*.bak.*` / `*.append*` remains anywhere (repo root, `docs/`, `tools/agentic/`, etc.).
   * Decide the fate of `docs/agent_runs/20260127_024404_ticket-00_agentic-bootstrap-refresh/`:

     * If it’s a legitimate run log, bring it into compliance and commit it.
     * If it’s not a legitimate run, move/delete it (do **not** leave untracked).
5. **Add guardrail test**

   * Add `tests/test_repo_hygiene.py` that fails if any `*.bak.*` or `*.append*` exists anywhere under repo root.
   * Optional (if cheap + stable): also assert `.gitignore` contains an ignore entry for `reports/_runs/` to prevent recurrence of that exact drift.

## Acceptance criteria

* [ ] `git status -sb` is clean on `main` (no modified tracked files; no untracked residue).
* [ ] No `*.bak.*` or `*.append*` files exist anywhere in the repo (including `docs/` and `tools/agentic/`).
* [ ] `reports/_runs/` is ignored as required by `TRACKING_POLICY.md` (and does **not** appear in `git status`).
* [ ] `docs/agent_runs/20260127_024404_ticket-00_agentic-bootstrap-refresh/` is either:

  * [ ] committed as a valid run log that passes validation, **or**
  * [ ] removed/moved so it does not appear as untracked clutter.
* [ ] New guardrail test exists and passes: `tests/test_repo_hygiene.py`.
* [ ] `make test-fast`, `make validate-runlogs`, and `make check-data-policy` all pass.

## Test plan

* [ ] `git status -sb`
* [ ] `git clean -ndx` (dry run, verify only junk would be removed)
* [ ] `pytest -q tests/test_repo_hygiene.py`
* [ ] `make test-fast`
* [ ] `make validate-runlogs`
* [ ] `make check-data-policy`

## Artifacts / Outputs

Expected changes:

* `docs/tickets/ticket-27_repo_hygiene_bootstrap_residue_cleanup.md` (this ticket)
* `.gitignore` (to fix ignore drift, especially `reports/_runs/`)
* `tests/test_repo_hygiene.py` (new)
* `docs/agent_runs/<RUN_NAME>/` (new run log for this cleanup)

No new top-level directories. No bulky outputs expected; if any temporary outputs are needed, they must be placed under `artifacts/_local/` or `reports/_runs/` and kept out of Git.

## Notes / Risks

* **Risk: accidental deletion of something valuable.** Mitigation: use `git clean -ndx` first; rely on Git for rollback of tracked files; for untracked items, inspect before deletion.
* **Rollback plan:** If a tracked file was unintentionally modified, restore with `git checkout -- <path>` (or `git restore <path>`). If `.gitignore` changes cause unintended ignores, revert the `.gitignore` edit and re-run `make check-data-policy`.
* **Common failure mode:** “Fixing” this by weakening validations (ignore this temptation). The goal is to tighten hygiene, not loosen policy enforcement.
