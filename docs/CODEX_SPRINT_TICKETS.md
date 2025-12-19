# Codex Sprint Tickets (FJS)

Rules (binding; see `AGENTS.md` + `docs/DOCS_AND_LOGGING_SYSTEM.md`):
- A ticket is **DONE** only if it has:
  - `RUN_NAME`, git SHA, **bundle path** under `docs/gpt_bundles/`, and tests recorded.
  - A complete run log in `docs/agent_runs/<RUN_NAME>/` with PROMPT/COMMANDS/RESULTS/TESTS/META.
- No silent fallbacks (especially portfolio solver fallbacks). If a dependency is missing, either **fail loud** or **explicitly skip** with run marked incomplete and summaries excluding it.

Status legend:
- NEXT, IN-PROGRESS, NEEDS-FOLLOW-UP, BLOCKED, DONE

## Ticket table (ordered by current priority)

| Ticket | Goal | Status | Latest RUN_NAME | Bundle (required for DONE) | Notes / Evidence |
|---|---|---:|---|---|---|
| ticket-08 | **Eliminate silent MV solver fallback** (cvxpy missing → EW) | **NEXT** | (to be created) | (to be created) | Stop-the-line validity issue noted in `project_state/KNOWN_ISSUES.md` (“Optional dependencies: cvxpy required… absence silently falls back”). |
| ticket-07 | Weekly detection “drought” diagnostics (emit gating trace artifact) | **NEEDS-FOLLOW-UP** | `20251219_173231_ticket-07_weekly-drought-diagnostics` | `docs/gpt_bundles/20251219_180641_ticket-07_20251219_173231_ticket-07_weekly-drought-diagnostics.zip` | Feature landed, but doc protocol not met: no PROGRESS entry; RESULTS.md missing bundle pointer; META.json SHA mismatch; synthetic skip_reason=`diagnostic_failure` everywhere; real smoke dominated by `guard_other`. See run log `docs/agent_runs/.../RESULTS.md`. |
| ticket-05 | rc-lite-sanity completeness hardening | **NEEDS-FOLLOW-UP** | `20251219_044404_ticket-05_rc-sanity-summary-hardening` | **MISSING** | PROGRESS has results, but (per prior tickets file) bundling was blocked before ticket-06. After ticket-06, re-run `make gpt-bundle` for ticket-05 so it becomes reviewable. |
| ticket-06 | Restore `make gpt-bundle` fail-loud target + regression test | **DONE** | `20251219_072353_ticket-06_gpt-bundle-restore` | `docs/gpt_bundles/20251219_074334_ticket-06_20251219_072353_ticket-06_gpt-bundle-restore.zip` | Implemented bundling + regression guard; recorded in `PROGRESS.md` top entry. |

## Ticket detail: ticket-07 (follow-ups required before “DONE”)
Follow-ups to close ticket-07 cleanly:
- Backfill documentation protocol:
  - Add PROGRESS entry for ticket-07 (run name, SHA, bundle path, key metrics).
  - Update `docs/agent_runs/<RUN_NAME>/RESULTS.md` to include bundle path; update `COMMANDS.md` to include `make gpt-bundle ...` invocation.
  - Fix `META.json` to reflect final git SHA (should match `LAST_COMMIT.txt`).
- Make diagnostics actionable:
  - Break down `guard_other` into specific sub-reasons (or add a more granular reason code in the gate).
  - Explain and/or fix `skip_reason=diagnostic_failure` on synthetic weekly micro run (is it expected? if so, rename; if not, fix root cause).
  - Update `project_state/CONFIG_REFERENCE.md` to document `diagnostics.gating_trace` / `--gating-diagnostics`.

## Ticket detail: ticket-08 (NEXT)
Goal:
- Remove the “cvxpy missing → silently use equal-weight MV” behavior.

Acceptance criteria:
- When MV is requested and solver is unavailable:
  - default behavior is **fail loud** with a clear error message, OR
  - MV is skipped only when explicitly allowed, and run is marked incomplete + summaries exclude MV.
- Add unit test(s) that simulate missing solver and assert no silent fallback.
- `make test-fast` passes.
- Deterministic smoke run using real data triggers the behavior and is recorded in the run log.
- Update `project_state/KNOWN_ISSUES.md` and `project_state/CONFIG_REFERENCE.md` accordingly.
- Bundle generated and path recorded in `docs/agent_runs/<RUN_NAME>/RESULTS.md`.
