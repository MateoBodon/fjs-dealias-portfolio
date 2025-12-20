TICKET: ticket-09
RUN_NAME: 20251220_<HHMMSS>_ticket-09_weekly-gating-attribution

You are Codex running in the repo `fjs-dealias-portfolio`.

Hard constraints (binding):
- Obey `AGENTS.md` stop-the-line rules. In particular:
  - No `guard_other` blobs masking guardrails.
  - No `diagnostic_failure` without exception type + minimal context.
  - No silent fallbacks anywhere.
- Do NOT write a long upfront plan. Explore → implement → test → smoke → document end-to-end.
- Keep changes small and auditable. Prefer 2–4 logical commits max.
- Every commit MUST include “Tests run:” in the commit body with exact commands.
- You MUST create a run log folder: `docs/agent_runs/<RUN_NAME>/` with PROMPT/COMMANDS/RESULTS/TESTS/META.
- End by producing a bundle: `make gpt-bundle TICKET=ticket-09 RUN_NAME=<RUN_NAME>` and record the bundle path in `docs/agent_runs/<RUN_NAME>/RESULTS.md`.

Goal:
Fix weekly gating diagnostics attribution so every non-acceptance window has an actionable reason code.
Eliminate `guard_other` and make `diagnostic_failure` always carry exception_type + stage + minimal context.

Scope (expected touch points):
- `experiments/equity_panel/run.py` (skip reason inference + diagnostics writer)
- Possibly `src/fjs/overlay.py` / `src/fjs/gating.py` if reason surfaces are missing
- `tools/summarize_weekly_diagnostics.py` (if it produces `weekly_diagnostics.md`)
- Tests: `tests/experiments/test_gating_diagnostics.py` (extend), add new regression tests as needed.

Required output behavior:
1) `gating_diagnostics.csv` must contain:
   - `skip_reason_primary` (string, REQUIRED)
   - `skip_reason_detail` (string, optional but preferred)
   - `exception_type` (string, REQUIRED when failure is exception-driven)
   - `exception_stage` (string, REQUIRED when failure is exception-driven)
   - optional: `exception_msg` (truncated), `missing_inputs`, etc.
2) `weekly_diagnostics.md` must include:
   - counts/shares by `skip_reason_primary`
   - top 5 windows per dominant reason with key fields (window_id/date range + minimal stats)
3) `guard_other` share must be 0 OR the concept must be removed entirely (no fallback bucket).
   - If an unknown guard key appears, it must be surfaced explicitly as e.g. `skip_reason_primary=guard_unknown` with detail=the key (actionable), and tests should make this hard to miss.

Implementation steps you must follow (no long plan, just do it):
A) Setup + branch
- Create feature branch: `codex/ticket-09-weekly-gating-attribution`.
- Create `docs/agent_runs/<RUN_NAME>/` and write:
  - `PROMPT.md` (this text)
  - Start `COMMANDS.md` and append every command you run, in order.

B) Diagnose current behavior
- Search for `guard_other`, `diagnostic_failure`, and where `gating_diagnostics.csv` fields are assembled.
- Identify the precise site(s) where:
  - guardrail counters are aggregated into `guard_other`
  - exceptions are caught and flattened into `diagnostic_failure` without typed context

C) Implement actionable reason attribution
- Centralize reason mapping:
  - Prefer a small enum/constant mapping rather than ad-hoc strings.
  - Map every existing guardrail counter / failure mode to a `skip_reason_primary`.
  - Ensure “exception-driven” failures include `exception_type` and `exception_stage`.
- Update the diagnostics writer to always emit the new columns.
- Update the markdown summary (`weekly_diagnostics.md`) generation to use the new columns and to print “top windows” per reason.

D) Tests (must fail on old behavior)
- Extend `tests/experiments/test_gating_diagnostics.py` (or add a new test module) to assert:
  - `skip_reason_primary` column exists and is non-empty for skipped windows
  - `guard_other` is absent or has 0 count/share
  - any row with `skip_reason_primary=diagnostic_failure` has non-empty `exception_type` and `exception_stage`
- Add at least one regression test that would fail on the current baseline behavior.

E) Run minimal verification
- Run: `make test-fast` (required).
- Run targeted pytest for gating diagnostics tests (required).

F) Real-data smoke (required)
- Run deterministic weekly smoke that produces gating diagnostics:
  - `EXEC_MODE=deterministic make run:equity_smoke`
  - or the explicit runner command used by the repo for equity smoke with `--gating-diagnostics`
- In `docs/agent_runs/<RUN_NAME>/RESULTS.md`, include:
  - output directory path(s)
  - a short excerpt showing the new columns exist (header + 2–3 rows)
  - counts showing `guard_other` is 0 (or absent) and `diagnostic_failure` rows include exception_type/stage if any exist

G) Documentation updates (required)
- Update `PROGRESS.md` with:
  - branch + final git SHA
  - exact commands
  - output paths
  - a blunt statement of what changed and remaining limitations
- Update:
  - `project_state/KNOWN_ISSUES.md` to remove/soften the `guard_other` issue if truly fixed
  - `project_state/CURRENT_RESULTS.md` only if you ran something that changes validated claims (otherwise don’t touch it)
- Update `docs/CODEX_SPRINT_TICKETS.md`:
  - mark ticket-09 DONE
  - select the next ticket (exactly one) based on the file’s ordering principle

H) Commits (required)
- Make small logical commits (e.g., “reason mapping + schema”, “weekly md summary”, “tests”).
- Each commit message body must include:
  - `Tests run: ...`

I) Bundle (required)
- Run: `make gpt-bundle TICKET=ticket-09 RUN_NAME=<RUN_NAME>`
- Record the resulting zip path in `docs/agent_runs/<RUN_NAME>/RESULTS.md`.
- Also record `unzip -l <zip>` output in the run log (either in RESULTS.md or a `bundle_contents.txt` file).

Stop condition:
- If you discover a stop-the-line rule violation unrelated to this ticket, stop and fix it only if it blocks ticket-09; otherwise document it in RESULTS.md and leave it for the next ticket.
