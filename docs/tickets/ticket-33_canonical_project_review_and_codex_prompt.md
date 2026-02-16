# Ticket-33: Canonical project review + Codex continuation prompt

## Goal
Make canonical project-review docs and ticket-32 audit metadata internally consistent and reusable.

## Scope
- Promote canonical full-review artifact path under `docs/gpt_outputs/`.
- Add reusable continuation prompt at `docs/prompts/codex_continuation.md`.
- Fix ticket-32 runlog/bundle SHA/path drift with append-only errata.

## Acceptance criteria
- `docs/gpt_outputs/20260216_project_review_full.md` is the canonical full-review path.
- `docs/PLAN_OF_RECORD.md` points full audit to that path and labels `20260216_analysis_full.md` as a ticket/bundle review capture.
- `docs/prompts/codex_continuation.md` exists and encodes read order, non-negotiables, blockers, and priorities.
- Ticket-32 `META.json`, `RESULTS.md`, and `PROGRESS.md` errata agree on canonical bundle path + `head_sha`.
- `. .venv/bin/activate && make validate-runlogs` passes.
- `. .venv/bin/activate && make test-fast` passes.

## Notes
- The exact uploaded `Analysis.md` text must be available in-repo (or pasted into run prompt) before claiming verbatim full-review ingestion complete.
