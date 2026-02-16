# 2026-02-16 Full Analysis (Analysis.md)

Source note: Full analysis text captured from the ticket-32 review thread and preserved as an immutable GPT output artifact.

## Verdict

FAIL

## Evidence

Run log folder exists + has required files:
`docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/` contains `PROMPT.md`, `COMMANDS.md`, `RESULTS.md`, `TESTS.md`, `META.json` (and also `META.md`).

Tests were actually run and recorded:
`docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/TESTS.md` shows:

- `. .venv/bin/activate && make validate-runlogs` -> PASS
- `. .venv/bin/activate && make test-fast` -> PASS (`83 passed, 171 deselected`)

`PROGRESS.md` has new entries referencing the run:

- `2026-02-16T02:34Z - ticket-31 docs recenter + snapshot refresh` (references `RUN_NAME`)
- `2026-02-16T02:49Z - ticket-31 bundle generation` (references bundle path + verification)

Ticket-31 doc deliverables are present in the diff:
`DIFF.patch` includes updates to `PROJECT.md`, `README.md`, `docs/PLAN_OF_RECORD.md`, `docs/CODEX_SPRINT_TICKETS.md`, `project_state/CURRENT_RESULTS.md`, `project_state/KNOWN_ISSUES.md`, `project_state/OPEN_QUESTIONS.md`, plus a new file `docs/gpt_outputs/20260216_analysis.md` (content visible in the diff).

Critical audit inconsistency (this breaks the logging contract):

- `BUNDLE_META.md` says `head_sha: 8bd1282541112293a3e6c823b7e32bbeaa8ef5c2`.
- `LAST_COMMIT.txt` also shows the last commit is `8bd1282541112293a3e6c823b7e32bbeaa8ef5c2`.
- But `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.json` records:
  - `git_sha_before = 1371b3c2e7...`
  - `git_sha_after = 1371b3c2e7...` (same as before)
- And the new `PROGRESS.md` ticket-31 entry also cites git sha `1371b3c2e7...`

This makes the run log not audit-grade, which violates the contract in `docs/DOCS_AND_LOGGING_SYSTEM.md` (`META.json` must carry correct before/after SHAs).

Review noise / workflow drift:

- `BUNDLE_META.md` sets `base_ref: origin/main`, so `DIFF.patch` is a stacked range diff that includes changes across `Makefile/`, `tools/`, and `tests/` in addition to ticket-31 doc changes. That is not inherently illegal, but it makes "is ticket-31 done?" harder to verify cleanly.

## Required fixes (if FAIL)

Fix the ticket-31 run log metadata to be truthful and consistent:

- Update `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.json` (and its mirror `META.md`) so:
  - `git_sha_after` equals `8bd1282541112293a3e6c823b7e32bbeaa8ef5c2` (per `LAST_COMMIT.txt` / `BUNDLE_META.md`)
  - `git_sha_before` equals the actual starting SHA for the run (likely the parent of `8bd128...`, but confirm via git)
  - `dirty_at_start` / `dirty_at_end` reflect reality (and if dirty, include a `git status --porcelain` snapshot in the run log directory)

Fix `PROGRESS` without breaking append-only discipline:

- Do not edit the existing ticket-31 PROGRESS entry in place.
- Append a new errata entry that:
  - references the same `RUN_NAME`
  - states the corrected git sha (and optionally the corrected branch)
  - states that the earlier entry's SHA was incorrect and is superseded

(Optional but strongly recommended) Make the bundle reviewable as "ticket-31 only":

- Either rebase/merge so `origin/main` includes the earlier tickets, or
- regenerate the bundle with a base that isolates ticket-31 (for example, `BUNDLE_BASE=<parent-of-8bd128>`), so reviewers are not forced to reason through unrelated diffs.

## Suggested follow-up tickets

### docs/tickets/ticket-32_promote_full_analysis_and_patch_ticket31_meta.md

Goal: Promote the full "extensive analysis" into canonical repo docs and patch ticket-31's audit metadata (`META.json` + `PROGRESS` errata) so the repo truth is consistent and reviewer-safe.

Acceptance criteria:

- Full analysis is committed under `docs/gpt_outputs/` and linked from `docs/PLAN_OF_RECORD.md`
- Ticket-31 run log `META.json` has correct before/after SHAs (`after == 8bd128...`)
- `PROGRESS` has an append-only errata correcting ticket-31 SHA/branch
- `. .venv/bin/activate && make validate-runlogs` and `. .venv/bin/activate && make test-fast` pass

### docs/tickets/ticket-33_injection_flat_zero_root_cause_mini_diagnostics.md

Goal: Run a minimal injection diagnostic that conclusively explains why week-design inject-spike is flat-zero (injection mismatch vs gating/admissible-root vs numerical).

Acceptance criteria:

- A single small deterministic inject run produces a per-stage debug table (pre-gate vs post-gate failures)
- Either a non-flat detection curve appears on at least one controlled config, or artifact-backed explanation why not
- Results written to `project_state/RESEARCH_NOTES.md` with exact artifact paths

### docs/tickets/ticket-34_one_advisor_ready_uncapped_week_run.md

Goal: Produce one uncapped, comparison-valid week-design run with clean summary tables safe to show an advisor/recruiter.

Acceptance criteria:

- `cap_active=false` and `comparison_valid_*` true with meaningful `n_effective_*`
- Summary tables exist and include acceptance/detection/skip accounting
- Run log + `PROGRESS` updated with exact commands and artifact paths

## Ticket-32: Promote full analysis + patch ticket-31 audit metadata

### Goal

Promote the full external project analysis into canonical repo docs and fix ticket-31's audit metadata (run log META + PROGRESS errata) so the repository truth is consistent and reviewer-safe.

### Context

Ticket-31 successfully recentered several docs, but the audit trail is currently internally inconsistent:

- Bundle metadata says `head_sha = 8bd1282541112293a3e6c823b7e32bbeaa8ef5c2` (`BUNDLE_META.md`, `LAST_COMMIT.txt`).
- Ticket-31 run log `META.json` incorrectly has `git_sha_before == git_sha_after == 1371b3c2e7...` at `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.json`.
- `PROGRESS.md`'s ticket-31 entry also cites the wrong SHA.
- The "extensive deep analysis" (uploaded as `Analysis.md` in chat) is not fully captured in the repo; ticket-31 added only a short snapshot at `docs/gpt_outputs/20260216_analysis.md`.

Relevant files to touch:

- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.json`
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.md`
- `PROGRESS.md` (append-only errata entry; do not rewrite history)
- `docs/PLAN_OF_RECORD.md` (add link to full analysis doc under `docs/gpt_outputs/`)
- `docs/gpt_outputs/20260216_analysis.md` (do not edit; treat as immutable snapshot)
- NEW: `docs/gpt_outputs/20260216_analysis_full.md` (commit the full analysis content here)

### Constraints

- No surprise top-level directories.
- Bulky outputs go to `artifacts/_local/` or `reports/_runs/` (do not commit bulky outputs).
- Run log must be created under `docs/agent_runs/<RUN_NAME>/` for this ticket.
- `PROGRESS.md` updates must be append-only (add an errata entry; do not rewrite prior entries).
- Run and record:
  - `. .venv/bin/activate && make validate-runlogs`
  - `. .venv/bin/activate && make test-fast`

### Plan

1. Create a new run log for ticket-32 (`tools/agentic/runlog_init.py ...`) and capture baseline `git status --porcelain`.
2. Add `docs/gpt_outputs/20260216_analysis_full.md` containing the full analysis text from the uploaded `Analysis.md` (verbatim, with minimal formatting fixes only).
3. Update `docs/PLAN_OF_RECORD.md` "Ground-truth status references" to include a link to `docs/gpt_outputs/20260216_analysis_full.md` (keep the short snapshot link too).
4. Patch ticket-31 run log metadata:
   - Update `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.json` and `META.md` so `git_sha_after` matches `8bd1282541112293a3e6c823b7e32bbeaa8ef5c2` and `git_sha_before` is the true starting SHA.
   - If either `dirty_at_start` or `dirty_at_end` is true, add `git_status_start.txt` / `git_status_end.txt` snapshots in the run log folder for audit clarity.
5. Append a `PROGRESS.md` errata entry that:
   - references the ticket-31 `RUN_NAME`,
   - states the corrected SHAs and that prior entry's SHA was incorrect,
   - points to the patched ticket-31 run log `META.json` for the canonical fix.
6. Run `. .venv/bin/activate && make validate-runlogs` and `. .venv/bin/activate && make test-fast`, record both in the new ticket-32 run log.
7. Produce a new bundle:
   - `make gpt-bundle TICKET=32 RUN_NAME=<RUN_NAME>`
   - ensure it includes the new analysis doc + the errata/meta fixes.

### Acceptance criteria

- `docs/gpt_outputs/20260216_analysis_full.md` exists and contains the full external analysis (from uploaded `Analysis.md`) without truncation.
- `docs/PLAN_OF_RECORD.md` links to both:
  - `docs/gpt_outputs/20260216_analysis.md` (short snapshot)
  - `docs/gpt_outputs/20260216_analysis_full.md` (full analysis)
- Ticket-31 run log `META.json` is corrected so `git_sha_after == 8bd1282541112293a3e6c823b7e32bbeaa8ef5c2`, and `git_sha_before` is accurate.
- `PROGRESS.md` has an append-only errata entry correcting ticket-31 SHA/branch info (no rewriting old entries).
- `. .venv/bin/activate && make validate-runlogs` passes.
- `. .venv/bin/activate && make test-fast` passes.

### Test plan

- `git status --porcelain`
- `. .venv/bin/activate && make validate-runlogs`
- `. .venv/bin/activate && make test-fast`
- `. .venv/bin/activate && make gpt-bundle TICKET=32 RUN_NAME=<RUN_NAME>` and verify the bundle contains the new doc and the patched META/errata via `unzip -l ...`

### Artifacts / Outputs

- `docs/agent_runs/<RUN_NAME>/PROMPT.md`
- `docs/agent_runs/<RUN_NAME>/COMMANDS.md`
- `docs/agent_runs/<RUN_NAME>/RESULTS.md`
- `docs/agent_runs/<RUN_NAME>/TESTS.md`
- `docs/agent_runs/<RUN_NAME>/META.json`
- `docs/gpt_outputs/20260216_analysis_full.md`
- `artifacts/_local/gpt_bundles/<STAMP>_32_<RUN_NAME>.zip`

### Notes / Risks

Risk: Editing historical run logs can be seen as "rewriting history."
Mitigation: Keep edits minimal (only to correct objective SHAs), record the change in a new ticket-32 run log, and append a PROGRESS errata entry.

Risk: `docs/gpt_outputs/` is intended to be immutable; modifying existing files violates that norm.
Mitigation: Do not edit `docs/gpt_outputs/20260216_analysis.md`; add `20260216_analysis_full.md` as a new immutable artifact.

Rollback plan: If anything goes sideways, `git restore` the touched files and revert the commit; no data artifacts should be committed outside allowed directories.
