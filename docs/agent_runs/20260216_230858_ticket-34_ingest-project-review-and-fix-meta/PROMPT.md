# Prompt

Ticket: **34**
Run: **20260216_230858_ticket-34_ingest-project-review-and-fix-meta**
Summary: Ingest canonical full review provenance and close ticket-33 metadata/audit drift

## Goal
- Ensure the canonical full review path is meaningful and provenance-labeled.
- Close ticket-33 audit drift via append-only PROGRESS errata and metadata verification.
- Produce validation/test evidence and a ticket-34 review bundle.

## Constraints
- Keep `PROGRESS.md` append-only.
- Do not fabricate review text; use uploaded `docs/Analysis.md` verbatim content.
- Keep ticket-33 metadata correction truthful (`git_sha_after=7003d53fc31cf00e1a7b2032a620abd0e39a7d53`).

## Plan
1. Initialize run log and capture start status.
2. Add provenance header to canonical full review file.
3. Verify ticket-33 metadata SHA against git history.
4. Append ticket-34 PROGRESS errata with exact commands and artifact paths.
5. Run `make validate-runlogs`, `make test-fast`, and produce ticket-34 bundle.

## Files touched (actual)
- `docs/gpt_outputs/20260216_project_review_full.md`
- `docs/prompts/codex_continuation.md`
- `docs/tickets/ticket-34_ingest_full_project_review_and_fix_ticket33_meta.md`
- `PROGRESS.md`
- `docs/agent_runs/20260216_230858_ticket-34_ingest-project-review-and-fix-meta/*`

## Definition of Done
- [x] Canonical full review path is non-placeholder and provenance-labeled.
- [x] Ticket-33 SHA drift is explicitly corrected in append-only PROGRESS errata.
- [x] Validation and unit tests pass.
- [x] Ticket-34 run log is complete.
- [ ] GPT bundle generated for ticket-34.
