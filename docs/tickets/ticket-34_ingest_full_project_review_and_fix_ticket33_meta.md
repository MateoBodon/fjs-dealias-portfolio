# Ticket-34: Ingest full project review + fix ticket-33 runlog metadata drift

## Goal
Ingest the verbatim full project analysis into the repo's canonical review path and repair ticket-33 audit metadata drift so the documentation and runlog chain is referee/advisor-credible.

## Context
Ticket-33 created the canonical path for the full review but initially left it as a placeholder when the exact uploaded analysis text was not yet available.

Concrete issues addressed by this ticket:
- `docs/gpt_outputs/20260216_project_review_full.md` must contain the real analysis text (not placeholder content).
- `docs/PLAN_OF_RECORD.md` points to that path as the canonical full audit reference; the target file must be meaningful.
- Ticket-33 metadata/provenance must consistently reference commit `7003d53fc31cf00e1a7b2032a620abd0e39a7d53` as the post-run SHA for run `20260216_212107_ticket-33_canonical-review-prompt-audit-fix`.
- `PROGRESS.md` must include append-only errata for any previously logged ticket-33 SHA mismatch.

## Key Files
- Canonical docs:
  - `docs/PLAN_OF_RECORD.md`
  - `docs/CODEX_SPRINT_TICKETS.md`
  - `docs/prompts/codex_continuation.md`
- Canonical full review output:
  - `docs/gpt_outputs/20260216_project_review_full.md`
- Ticket-33 run log:
  - `docs/agent_runs/20260216_212107_ticket-33_canonical-review-prompt-audit-fix/`
- Provenance log:
  - `PROGRESS.md`

## Constraints
- No surprise top-level directories.
- Keep run logs under `docs/agent_runs/<RUN_NAME>/` with required files (`PROMPT.md`, `COMMANDS.md`, `RESULTS.md`, `TESTS.md`, `META.json`, plus `META.md` compatibility).
- `PROGRESS.md` is append-only; corrections must be appended as errata.
- Do not fabricate review text; use uploaded source text verbatim.

## Plan
1. Initialize a ticket-34 run log and capture start status.
2. Ensure canonical full-review file is non-placeholder and carries provenance header.
3. Verify ticket-33 `META.json`/`META.md` SHA correctness against git history.
4. Append a ticket-34 errata entry to `PROGRESS.md` documenting correction chain.
5. Optionally polish continuation prompt read order to include canonical full review.
6. Run validation/tests and generate a ticket-34 bundle.

## Acceptance Criteria
- `docs/gpt_outputs/20260216_project_review_full.md` contains the full analysis text and a provenance header indicating verbatim ingest from uploaded `Analysis.md`.
- `docs/PLAN_OF_RECORD.md` continues to point to `docs/gpt_outputs/20260216_project_review_full.md` and that file is meaningful.
- Ticket-33 runlog metadata is truthful: `git_sha_after` equals `7003d53fc31cf00e1a7b2032a620abd0e39a7d53`.
- `PROGRESS.md` includes an append-only ticket-34 errata note correcting ticket-33 SHA reporting drift.
- Ticket-34 run log exists under `docs/agent_runs/<RUN_NAME>/` with required files filled.
- `. .venv/bin/activate && make validate-runlogs` passes.
- `. .venv/bin/activate && make test-fast` passes.

## Test Plan
- `. .venv/bin/activate && make validate-runlogs`
- `. .venv/bin/activate && make test-fast`
- `rg -n "BLOCKED: the exact \\Analysis\\.md source text" docs/gpt_outputs/20260216_project_review_full.md` returns no matches.
- `git check-ignore -v docs/gpt_outputs/20260216_project_review_full.md` returns no output.
- `python3 -c 'import json; print(json.load(open("docs/agent_runs/20260216_212107_ticket-33_canonical-review-prompt-audit-fix/META.json"))["git_sha_after"])'` prints `7003d53fc31cf00e1a7b2032a620abd0e39a7d53`.

## Outputs
- Ticket-34 run log: `docs/agent_runs/<RUN_NAME>/`
- Canonical full review: `docs/gpt_outputs/20260216_project_review_full.md`
- Append-only errata update: `PROGRESS.md`
- Bundle (local-only): `artifacts/_local/gpt_bundles/<STAMP>_34_<RUN_NAME>_*.zip`
