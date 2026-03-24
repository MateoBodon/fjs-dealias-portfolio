# Prompt

Ticket: **31**
Run: **20260216_032804_ticket-31_docs-recenter-snapshot-refresh**
Summary: Docs recenter + snapshot refresh

## Goal
- Recenter canonical docs so advisor/recruiter-facing truth is current: what the project is, what works, core blockers, and next research-validity gates.

## Constraints
- No new top-level directories.
- Bulky outputs only in `artifacts/_local/` or `reports/_runs/`.
- Create complete run log in `docs/agent_runs/<RUN_NAME>/`.
- Update `PROGRESS.md` append-only.
- Run and record `make validate-runlogs` and `make test-fast`.
- Do not modify core research logic (`src/`, `experiments/`) unless needed for factual corrections.

## Requested doc scope
- Add `docs/gpt_outputs/20260216_analysis.md`.
- Update `docs/PLAN_OF_RECORD.md` priorities to focus on injection flat-zero and one advisor-ready uncapped run.
- Fill `PROJECT.md` (no placeholders).
- Refresh README Current Status.
- Correct arithmetic/verification issues in `project_state/CURRENT_RESULTS.md`.
- Add Ticket #31 in `docs/CODEX_SPRINT_TICKETS.md` and align ordering.
- Append `PROGRESS.md` entry.

## Definition of Done
- [x] Canonical docs updated and internally consistent.
- [x] `CURRENT_RESULTS.md` contradiction removed and metrics artifact-verified or removed.
- [x] Required run-log files present.
- [x] Required validation/test commands executed and logged.
