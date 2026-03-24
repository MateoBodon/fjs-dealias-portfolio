# Prompt

Ticket: **32**
Run: **20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta**
Summary: Promote full analysis + patch ticket-31 audit metadata

## Goal
- Promote the full analysis artifact into `docs/gpt_outputs/`, correct ticket-31 audit metadata (`META.json`/`META.md` + dirty snapshots), and append a PROGRESS errata entry without rewriting history.

## Constraints
- No new top-level directories.
- Keep `docs/gpt_outputs/20260216_analysis.md` immutable and add `docs/gpt_outputs/20260216_analysis_full.md` as a new artifact.
- Preserve append-only discipline in `PROGRESS.md`.
- Run and record `. .venv/bin/activate && make validate-runlogs` and `. .venv/bin/activate && make test-fast`.

## Plan
1. Initialize ticket-32 run log and capture baseline git status.
2. Patch ticket-31 metadata + add dirty status snapshots.
3. Add full analysis doc and link it from `docs/PLAN_OF_RECORD.md`.
4. Append ticket-31 SHA errata in `PROGRESS.md`.
5. Run required validations/tests and produce ticket-32 bundle.

## Files touched
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.json`
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/META.md`
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/git_status_start.txt`
- `docs/agent_runs/20260216_032804_ticket-31_docs-recenter-snapshot-refresh/git_status_end.txt`
- `docs/gpt_outputs/20260216_analysis_full.md`
- `docs/PLAN_OF_RECORD.md`
- `PROGRESS.md`
- `docs/agent_runs/20260216_050358_ticket-32_promote-full-analysis-patch-ticket31-meta/*`

## Definition of Done
- [x] Ticket-31 before/after SHAs corrected and dirty snapshots added.
- [x] Full analysis doc added and linked from PLAN_OF_RECORD.
- [x] PROGRESS append-only errata added.
- [x] `make validate-runlogs` passes.
- [x] `make test-fast` passes.
