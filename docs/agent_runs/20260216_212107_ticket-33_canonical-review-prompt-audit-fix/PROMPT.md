# Prompt

Ticket: **33**
Run: **20260216_212107_ticket-33_canonical-review-prompt-audit-fix**
Summary: Canonical project review + Codex continuation prompt + ticket-32 audit drift fix

## Goal
- Fix ticket-32 audit inconsistencies (bundle SHA/path drift), add the canonical continuation prompt, and update canonical docs so the full-audit reference is unambiguous.

## Constraints
- Follow `AGENTS.md` and `docs/DOCS_AND_LOGGING_SYSTEM.md` stop-the-line rules.
- Keep `PROGRESS.md` append-only (errata entries only; no in-place rewrite of historical claims).
- Treat `docs/gpt_outputs/20260216_analysis.md` and `docs/gpt_outputs/20260216_analysis_full.md` as immutable.
- Run and record:
  - `. .venv/bin/activate && make validate-runlogs`
  - `. .venv/bin/activate && make test-fast`
- Produce a ticket-33 review bundle with `make gpt-bundle`.

## Plan
1. Inspect ticket-32 run log and bundle metadata, then align `META.json` + `RESULTS.md` + `PROGRESS.md` errata to one canonical bundle story.
2. Add canonical doc assets:
   - `docs/prompts/codex_continuation.md`
   - `docs/gpt_outputs/20260216_project_review_full.md` (canonical full-review path)
   - update `docs/PLAN_OF_RECORD.md` and sprint/ticket docs.
3. Resolve workflow drift by tracking `docs/tickets/ticket-31_docs_recenter_snapshot_refresh.md` (ghost untracked ticket file).
4. Run validations/tests and generate the ticket-33 bundle.

## Notes
- Exact uploaded `Analysis.md` text is not present in workspace artifacts in this run; do not fabricate a verbatim replacement from memory.

## Definition of Done
- [ ] Ticket-32 runlog/bundle references are consistent.
- [ ] Canonical continuation prompt exists and is linked in results/progress.
- [ ] PLAN_OF_RECORD full-audit link and analysis-full disambiguation are updated.
- [ ] Ghost ticket drift is resolved (committed or removed; not left untracked).
- [ ] Required validations/tests pass and are logged.
- [ ] Ticket-33 bundle artifact is generated and recorded.
