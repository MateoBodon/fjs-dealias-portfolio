# Codex Sprint Tickets (FJS)

Rules (binding; see `AGENTS.md` + `docs/DOCS_AND_LOGGING_SYSTEM.md`):
- A ticket is **DONE** only if it has:
  - `RUN_NAME`, git SHA, **bundle path** under `docs/gpt_bundles/`, and tests recorded.
  - A complete run log in `docs/agent_runs/<RUN_NAME>/` with `PROMPT.md`, `COMMANDS.md`, `RESULTS.md`, `TESTS.md`, `META.json`.
- **Stop-the-line**: no silent fallbacks in evaluation.
  - In particular: if MV optimization is requested and the solver dependency is unavailable, the run must either **fail loud** (default) or **explicitly skip** with the run marked incomplete and summaries excluding MV metrics.

Status legend:
- NEXT, IN-PROGRESS, NEEDS-FOLLOW-UP, BLOCKED, DONE

## Ticket table (ordered by current priority)

| Ticket | Goal | Status | Latest RUN_NAME | Bundle (required for DONE) | Notes / Evidence |
|---|---|---:|---|---|---|
| ticket-08 | **Eliminate silent MV solver fallback** (cvxpy missing must not silently use EW) | **DONE** | `20251219_202301_ticket-08_solver-missing-proof` | `docs/gpt_bundles/20251219_204908_ticket-08_20251219_202301_ticket-08_solver-missing-proof.zip` | CVXPy path exercised: make test-fast, unit tests force `MissingSolverError` by default; forced-missing smoke (`FJS_FORCE_MISSING_CVXPY=1 --mv-skip-on-missing-solver`) flags MV as `skipped`/`missing_solver`, and normal cvxpy smoke logs `solver_status=optimal` for MV. |
| ticket-05 | rc-lite-sanity completeness hardening | **NEEDS-FOLLOW-UP** | `20251219_044404_ticket-05_rc-sanity-summary-hardening` | **MISSING** | Implementation exists (see PROGRESS.md entry), but ticket is not reviewable until a bundle is generated + recorded. |
| ticket-07 | Weekly detection “drought” diagnostics (emit gating diagnostics artifacts) | **DONE** | `20251219_173231_ticket-07_weekly-drought-diagnostics` | `docs/gpt_bundles/20251219_180641_ticket-07_20251219_173231_ticket-07_weekly-drought-diagnostics.zip` | Delivered gating diagnostics artifacts; follow-on work is now “make it actionable” (separate ticket if/when needed). |
| ticket-06 | Restore `make gpt-bundle` fail-loud target + regression test | **DONE** | `20251219_072353_ticket-06_gpt-bundle-restore` | `docs/gpt_bundles/20251219_074334_ticket-06_20251219_072353_ticket-06_gpt-bundle-restore.zip` | Bundle target restored + regression guard `tests/test_gpt_bundle.py`. |

---

## Ticket detail: ticket-08 (DONE)

- Evidence run: `RUN_NAME=20251219_202301_ticket-08_solver-missing-proof` (bundle `docs/gpt_bundles/20251219_204908_ticket-08_20251219_202301_ticket-08_solver-missing-proof.zip`).
- Tests: `make test-fast`; regression tests force cvxpy missing — default raises `MissingSolverError`; skip flag returns `skipped=True`, `skip_reason=missing_solver`, empty weights (no EW fallback).
- Smokes (deterministic, max_windows=2):
  - Normal cvxpy path `reports/eval-smoke-ticket08-proof/normal/metrics_detail.csv` shows MV rows `skipped=False`, `solver_status=optimal`.
  - Forced missing with `FJS_FORCE_MISSING_CVXPY=1 --mv-skip-on-missing-solver` writes `reports/eval-smoke-ticket08-proof/missing-skip/` with MV rows `skipped=True`, `skip_reason=missing_solver`; diagnostics `mv_skipped_share=1.0`.
- Docs updated: `project_state/CONFIG_REFERENCE.md` documents `mv_solver`/`mv_skip_on_missing_solver` + `FJS_FORCE_MISSING_CVXPY`; run log recorded under `docs/agent_runs/20251219_202301_ticket-08_solver-missing-proof/`.

---

## Ticket detail: ticket-05 (NEEDS-FOLLOW-UP)

Goal:
- Ensure rc-lite-sanity summaries **exclude incomplete / capped runs** and emit explicit `completeness.json` so paper tables cannot be “accidentally” built from contaminated runs.

What is missing:
- A reviewable bundle for the existing run `20251219_044404_ticket-05_rc-sanity-summary-hardening`.

To close:
- Run: `make gpt-bundle TICKET=ticket-05 RUN_NAME=20251219_044404_ticket-05_rc-sanity-summary-hardening`
- Record the resulting zip path in:
  - `docs/agent_runs/20251219_044404_ticket-05_rc-sanity-summary-hardening/RESULTS.md`
  - This ticket table row
  - `PROGRESS.md` (append a “Bundle:” line to the ticket-05 entry)
