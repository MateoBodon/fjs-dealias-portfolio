# Ticket-08 — MV solver fallback must fail loud (no silent EW)

RUN_NAME (create new): YYYYMMDD_HHMMSS_ticket-08_solver-fallback-fail-loud

## Acceptance criteria (must be satisfied to mark DONE)
- [ ] No silent fallback: if MV optimization is requested and the solver dependency is unavailable, the run must either:
  - raise a clear error early (default), OR
  - skip MV only if an explicit config flag allows it, and mark outputs unmistakably (e.g., solver_status="missing_dependency", mv_skipped=true, run incomplete so summaries exclude it).
- [ ] Add unit test(s) that simulate missing cvxpy and assert we do NOT compute MV metrics using EW weights.
- [ ] `make test-fast` passes.
- [ ] Deterministic real-data smoke:
  - find the smallest command/config that requests MV weights; run it with `EXEC_MODE=deterministic`.
  - record whether cvxpy is installed; if missing, confirm the failure/skip behavior is explicit and documented.
- [ ] Update docs:
  - `project_state/KNOWN_ISSUES.md`: remove/replace the “silent fallback to EW” issue.
  - `project_state/CONFIG_REFERENCE.md`: document any new config/CLI knobs and the new fail-loud/skip behavior.
  - `PROGRESS.md`: new dated entry with RUN_NAME, git SHA, commands, tests, and bundle path.
  - `docs/CODEX_SPRINT_TICKETS.md`: mark ticket-08 DONE with RUN_NAME + bundle path.
- [ ] Run log complete at `docs/agent_runs/<RUN_NAME>/` with PROMPT/COMMANDS/RESULTS/TESTS/META.
- [ ] Generate and record bundle path:
  - `make gpt-bundle TICKET=ticket-08 RUN_NAME=<RUN_NAME>`
  - record the produced `docs/gpt_bundles/...zip` path in `docs/agent_runs/<RUN_NAME>/RESULTS.md`.

## Constraints (binding)
- Follow `AGENTS.md` stop-the-line rules.
- Do NOT change research semantics or gating thresholds in this ticket.
- Do NOT add “silent fallbacks” anywhere. If you introduce an escape hatch, it must be explicit in config and must poison completeness / summary aggregation.
- Prefer small logical commits. Every commit body must include `Tests: ...`.

## Execution checklist (do, don’t explain upfront)
1) Create branch:
   - `git checkout -b ticket-08-solver-fallback-fail-loud`

2) Create run log dir + initialize files:
   - `mkdir -p docs/agent_runs/<RUN_NAME>/`
   - Write PROMPT.md (this file verbatim), COMMANDS.md, RESULTS.md, TESTS.md, META.json (include start time, hostname, git sha, python version).

3) Repo exploration (fast):
   - Use `rg` to locate MV/min-var code paths (search: cvxpy, minvar, box constraints, mean-variance, qp, converged).
   - Identify where the “fallback to EW” happens and where `converged=False` is currently set but ignored.

4) Implement fix (minimal, correctness-first):
   - Default: MV requested + solver missing => raise a RuntimeError with actionable install/config message.
   - Optional explicit escape hatch: allow skipping MV only if a config flag is set; in that case:
     - output must include solver_status + mv_skipped + converged flag
     - summaries/completeness must treat the run as incomplete or exclude MV metrics.
   - Ensure no downstream code computes MV risk metrics if MV weights are invalid/skipped.

5) Tests:
   - Add unit test(s) simulating missing solver via monkeypatching import to raise ImportError.
   - Ensure tests assert “no silent EW fallback” and that the error/skip state is explicit.
   - Run:
     - `source .venv/bin/activate && make test-fast`
     - plus any targeted pytest commands; record in TESTS.md.

6) Real-data deterministic smoke:
   - Find the smallest existing config/command that requests MV weights.
   - Run it with `EXEC_MODE=deterministic`.
   - If cvxpy is absent: confirm the run fails loudly (or skips only with explicit flag) and record the observed behavior in RESULTS.md.
   - If cvxpy is present: confirm MV runs and solver_status="ok" is recorded.

7) Docs updates:
   - Update `project_state/KNOWN_ISSUES.md` and `project_state/CONFIG_REFERENCE.md`.
   - Add a new entry at the top of `PROGRESS.md`.
   - Update `docs/CODEX_SPRINT_TICKETS.md`.

8) Commit in small steps:
   - Each commit body must include `Tests: ...` and note key artifact paths.
   - Keep formatting clean; avoid drive-by refactors.

9) Generate bundle and record path:
   - `make gpt-bundle TICKET=ticket-08 RUN_NAME=<RUN_NAME>`
   - Put the resulting zip path into RESULTS.md.
