# Docs recenter + snapshot refresh (advisor-credible repo truth)

## Goal
Update the repo’s canonical documentation so that “what this project is” + “what is currently true” matches the actual pipeline state and recent runs (and remove known inconsistencies in reported results).

## Context
This repo has strong engineering/audit infrastructure (tickets → runs → logs → bundles), but the *canonical docs are stale and internally inconsistent*, which undermines advisor/paper credibility.

Key inputs to reconcile:
- `PROGRESS.md` (source of truth for what actually ran + what changed)
- `docs/PLAN_OF_RECORD.md` (research framing / acceptance gates)
- `PROJECT.md` (currently template-ish; should summarize purpose/goals/current state)
- `README.md` (Current Status section is dated 2025-12-20; likely stale)
- `project_state/CURRENT_RESULTS.md` (contains at least one arithmetic inconsistency: “detection_rate_mean ≈ 4.16% (1751/1774 windows)”)
- `project_state/KNOWN_ISSUES.md`, `project_state/RESEARCH_NOTES.md`, `project_state/OPEN_QUESTIONS.md` (should reflect today’s blockers and priorities)

External analysis to incorporate (uploaded in this chat):
- `Analysis.md` (recentered critique: pipeline is real + impressive, but publishability hinges on non-flat injection response + meaningful real-window overlay impact)

## Constraints
- Follow TRACKING_POLICY.md:
  - no surprise top-level directories
  - bulky outputs go to `artifacts/_local/` or `reports/_runs/` (and remain gitignored)
- Create a run log under `docs/agent_runs/<RUN_NAME>/` with:
  - PROMPT.md, COMMANDS.md, RESULTS.md, TESTS.md, META.json
- This ticket should be **docs/snapshot-focused** (do not change core research logic unless required to fix factual reporting errors).

## Plan
1. Create a run log:
   - `python3 tools/agentic/runlog_init.py --ticket "31" --summary "Docs recenter + snapshot refresh" --run-name "<RUN_NAME>"`
2. Add the uploaded analysis into the repo as an immutable GPT output:
   - Create `docs/gpt_outputs/20260216_analysis.md` containing the content of `Analysis.md`.
   - Reference it from `docs/PLAN_OF_RECORD.md` as “current external audit”.
3. Recenter `PROJECT.md`:
   - Fill in: one-liner purpose, goals/non-goals, current state, biggest risks, quickstart.
   - Keep it consistent with `docs/PLAN_OF_RECORD.md` and `PROGRESS.md`.
4. Update `README.md` “Current Status” to reflect current reality:
   - Replace the 2025-12-20 status bullets with a minimal, accurate snapshot:
     - tests green baseline
     - main blockers (injection flat-zero, doc staleness, etc.)
     - the next “research-validity gate” runs
5. Refresh `project_state/*` content for correctness:
   - Fix any obvious arithmetic/logical inconsistencies in `project_state/CURRENT_RESULTS.md`
     - For each referenced run, compute or cite correct numerator/denominator from the actual output artifacts (e.g., summary CSVs) instead of guessing.
     - If a number cannot be verified from artifacts in-repo, remove it or mark it explicitly “unverified”.
   - Ensure `KNOWN_ISSUES.md` matches the actual top blockers in `PROGRESS.md` and the recenter analysis.
6. Update `docs/CODEX_SPRINT_TICKETS.md`:
   - Add Ticket #31 (this ticket) and ensure the top priorities match reality (injection flat-zero and one advisor-ready run).
7. Append a new entry to `PROGRESS.md` summarizing exactly what changed and what was validated.
8. Run required checks and record them in TESTS.md:
   - `make validate-runlogs`
   - `make test-fast`

## Acceptance criteria
- [ ] A complete run log exists at `docs/agent_runs/<RUN_NAME>/` including `META.json`.
- [ ] `docs/gpt_outputs/20260216_analysis.md` exists and is referenced from `docs/PLAN_OF_RECORD.md`.
- [ ] `PROJECT.md` is fully filled out (no placeholder sections) and accurately describes the repo.
- [ ] `README.md` “Current Status” is updated to a current snapshot consistent with `PROGRESS.md`.
- [ ] `project_state/CURRENT_RESULTS.md` contains **no arithmetic contradictions** and every stated metric is either:
  - verified from an artifact path in-repo, or
  - explicitly labeled unverified/removed.
- [ ] `project_state/KNOWN_ISSUES.md` reflects the real current blockers (not stale ones).
- [ ] `docs/CODEX_SPRINT_TICKETS.md` includes Ticket #31 and the ordering matches reality.
- [ ] `PROGRESS.md` has a new append-only entry referencing this run and listing tests + modified docs.
- [ ] `make validate-runlogs` and `make test-fast` pass.

## Test plan
- [ ] `make validate-runlogs`
- [ ] `make test-fast`
- [ ] Quick sanity grep for the known inconsistency:
  - `rg -n "4\\.16% \\(1751/1774" project_state/CURRENT_RESULTS.md` returns nothing

## Artifacts / Outputs
- Run log: `docs/agent_runs/<RUN_NAME>/`
- New doc: `docs/gpt_outputs/20260216_analysis.md`
- No new bulky outputs expected; if anything transient is produced, keep it in `artifacts/_local/` or `reports/_runs/`.

## Notes / Risks
- Risk: Regenerating `project_state/*` mechanically could create a large diff. Prefer **surgical edits** focused on correctness unless you intentionally run the project_state generator.
- Risk: Some metrics in CURRENT_RESULTS may not be reproducible if the referenced output directories are missing. In that case, remove the metric or mark as unverified rather than leaving inconsistent numbers.
- Rollback: If doc edits become confusing, revert to the previous versions and re-apply only the minimal “truth fixes” (PROJECT.md fill + CURRENT_RESULTS arithmetic fix + PROGRESS entry).