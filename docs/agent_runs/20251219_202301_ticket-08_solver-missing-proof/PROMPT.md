# ticket-08 — MV solver fallback: **prove** fail-loud / explicit-skip (no silent EW)

## Acceptance criteria (must be evidenced by tests + artifacts)
1) **No silent fallback**:
   - If MV optimization requests the cvxpy/exact solver and cvxpy is unavailable, the system must **not** silently switch to equal-weight (or any other “success-shaped” fallback).
2) **Default is fail-loud**:
   - The default behavior when the required solver is missing is an explicit exception (`MissingSolverError` or equivalent) with a clear message.
3) **Optional explicit skip** (only if configured):
   - When `skip_on_missing_solver=True` (or the repo’s equivalent flag), MV is **explicitly skipped** and the result object/output must carry:
     - `skipped=true`
     - a machine-readable `skip_reason` like `missing_solver`
     - `solver_status` indicating missing/unavailable
     - MV metrics are NaN/None and are excluded from summary aggregations (run marked incomplete, if that’s the repo pattern).
4) **Regression tests**:
   - A unit test must simulate “cvxpy missing” without uninstalling packages (e.g., monkeypatch import) and assert (1)-(3).
5) **Real-data deterministic smoke**:
   - Run a deterministic smoke that actually hits the cvxpy-required MV path on a small real dataset.
   - Run it twice:
     - normal env (cvxpy present): MV path executes, `solver_status=ok`.
     - forced-missing solver: behavior matches (2) by default, or (3) when explicitly configured.
6) **Audit trail**:
   - New run log under `docs/agent_runs/<RUN_NAME>/` with PROMPT/COMMANDS/RESULTS/TESTS/META.
   - `make test-fast` must pass and be recorded in TESTS.md.
   - Generate a bundle at the end: `make gpt-bundle TICKET=ticket-08 RUN_NAME=<RUN_NAME>` and record the zip path in RESULTS.md.
   - Update `docs/CODEX_SPRINT_TICKETS.md` + `PROGRESS.md` to reflect DONE/remaining gaps.

---

## Codex task (do not write a long plan; execute end-to-end)

You are running in the Codex CLI inside this repo. Follow `AGENTS.md` and `docs/DOCS_AND_LOGGING_SYSTEM.md` as binding.

### 0) Setup: branch + run log
- Create a feature branch: `codex/ticket-08-solver-missing-proof`.
- Pick a fresh `RUN_NAME` like: `YYYYMMDD_HHMMSS_ticket-08_solver-missing-proof`.
- Create `docs/agent_runs/<RUN_NAME>/` with the required files and write this prompt verbatim into `PROMPT.md`.

### 1) Reproduce / locate the problematic path
- Use `rg` to locate:
  - where cvxpy is imported/loaded (e.g., `_get_cvxpy`, `import cvxpy as cp`, etc.)
  - MV optimization entrypoints (likely in `src/finance/portfolios.py` and/or `src/finance/portfolio.py`)
  - any fallback that returns equal-weight weights on failure.
- Write exact reproduction commands into `COMMANDS.md` as you go.

### 2) Implement (or harden) fail-loud + explicit-skip
- Ensure there is a single, well-defined exception type for missing solver (e.g., `MissingSolverError`).
- Ensure there is exactly one controlled “skip” mechanism, and it is **explicit** and **opt-in** (flag/param/config).
- Remove/forbid any “success-shaped” fallback to EW for MV requests.
- If an evaluation runner consumes `OptimizationResult`, ensure it propagates `skipped/skip_reason/solver_status` into the written metrics JSON/CSV so summaries cannot silently treat EW as MV.

### 3) Add regression tests (required)
- Add/modify tests to simulate missing solver deterministically:
  - Prefer `pytest` + `monkeypatch` to force the cvxpy import helper to raise `ImportError`.
  - Assert:
    - default: raises missing-solver exception
    - opt-in skip: returns result with `skipped=True`, `skip_reason="missing_solver",` and no MV weights/metrics computed.
- Keep tests small and fast; add to `make test-fast` coverage.

### 4) Minimal real-data deterministic smoke (required)
- Find the smallest real-data smoke entrypoint/config already in-repo that can request the cvxpy/exact MV solver (search CLI flags and config schema).
- Run deterministically and save outputs under a ticket-specific directory, e.g. `reports/eval-smoke-ticket08-proof/`.
- Run two modes:
  1) normal (cvxpy present): MV executes; record the key output lines proving the cvxpy path ran.
  2) forced-missing (without uninstall): trigger the missing-solver path and confirm fail-loud default OR explicit skip when configured.

### 5) Verification + documentation
- Run at minimum: `make test-fast` (record in `docs/agent_runs/<RUN_NAME>/TESTS.md`).
- Update:
  - `project_state/CONFIG_REFERENCE.md` to document the exact knob/env var used to force “missing solver” in tests/smoke.
  - `project_state/KNOWN_ISSUES.md` only if a new limitation is discovered.
  - `PROGRESS.md` with the new run + key results + bundle path.
  - `docs/CODEX_SPRINT_TICKETS.md` mark ticket-08 DONE only if all acceptance criteria are met.

### 6) Bundle (required finish)
- Run: `make gpt-bundle TICKET=ticket-08 RUN_NAME=<RUN_NAME>`
- Record the resulting zip path in `docs/agent_runs/<RUN_NAME>/RESULTS.md`.
- Ensure `LAST_COMMIT.txt` matches HEAD and that tests are recorded in commit bodies (format: `Tests: make test-fast`).

### Guardrails
- Do **not** “fix” by disabling MV or always-skipping; default must be fail-loud, and skip must be explicit and surfaced.
- If you must introduce a new config/env var to force the missing-solver path for smoke, document it and keep it scoped (e.g., `FJS_FORCE_MISSING_CVXPY=1`).
- Keep commits small and logical; every commit message body must include `Tests: ...` (or `Tests: not run (reason)`).
