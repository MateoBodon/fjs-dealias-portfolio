ticket-10: FAIL (no commits + empty DIFF.patch → not auditable/mergeable).

Keep ticket-11 / ticket-12 in queue, but do not proceed until we can review and merge ticket-10’s claimed calibration plumbing.

Next ticket to run (exactly one): add and run a fixup ticket

Select: NEW ticket-14 — “Ticket‑10 Fixup: make nested calibration mergeable + auditable”

Rationale: the technical result might be correct, but we can’t merge or trust it without committed diffs + a non-empty patch + a nested-specific smoke.


# Codex Sprint Tickets (NEXT SPRINT ONLY)

Ordering principle: unblock validity first, then calibrate/extend.

---

## Ticket #1 (ticket-09) — Fix weekly gating diagnostics attribution (kill `guard_other`)
**Goal (1 sentence)**  
Make weekly gating diagnostics actionable: every “no acceptance” window must have an attributable reason code (no `guard_other` blob, no opaque `diagnostic_failure`).

**Files/modules likely involved**
- `experiments/equity_panel/run.py` (notably `_infer_skip_reason(...)` and gating diagnostics writer)
- `src/fjs/overlay.py` (gating summaries / reason surfaces if needed)
- `src/fjs/gating.py` (calibrated delta lookup surfaces if needed)
- `tools/summarize_weekly_diagnostics.py` (if it post-processes reason codes)
- Tests:
  - `tests/experiments/test_gating_diagnostics.py`
  - (possibly) `tests/test_skip_reasons.py`

**Acceptance criteria**
- `gating_diagnostics.csv` includes:
  - a primary reason code field (e.g., `skip_reason_primary`)
  - optional structured detail fields (`skip_reason_detail`, `exception_type`)
- `weekly_diagnostics.md` includes:
  - counts by primary reason
  - top 5 windows per dominant reason (with key stats)
- On the smoke config, `guard_other` count/share is **0** OR provably unreachable.
- `diagnostic_failure` only appears with:
  - exception type + minimal context (which stage failed, what inputs missing)
- Tests pass (`make test-fast`).

**Minimal tests/commands**
- `make test-fast`
- `pytest -m unit -k "gating_diagnostics or skip_reason"`
- Real-data smoke:
  - `EXEC_MODE=deterministic make run:equity_smoke`

**Expected artifacts/logs**
- `docs/agent_runs/<RUN_NAME>/` with PROMPT/COMMANDS/RESULTS/TESTS/META
- Updated weekly smoke outputs:
  - `experiments/equity_panel/outputs_*/gating_diagnostics.csv`
  - `experiments/equity_panel/outputs_*/weekly_diagnostics.md`
- `make gpt-bundle TICKET=ticket-09 RUN_NAME=<RUN_NAME>`

---

## Ticket #2 (ticket-10) — Nested null-FPR: reproduce + calibrate (or de-scope nested)
**Goal (1 sentence)**  
Make nested design statistically defensible by demonstrating synthetic null-FPR control (or explicitly de-scope nested from paper v1 with a documented failure analysis).

**Files/modules likely involved**
- `experiments/synthetic/nested_killtest.py`
- `experiments/synthetic/power_null.py` (if we generalize calibration)
- `src/fjs/gating.py` and/or `src/fjs/overlay.py` (design-aware calibration hooks)
- Calibration outputs:
  - `calibration/edge_delta_thresholds.json`
  - `calibration/defaults.json`
- Reporting:
  - `reports/synthetic/...` summaries and figures

**Acceptance criteria**
- Nested synthetic null-FPR ≤ target (pick current target; default 2%) for the declared operating point(s).
- Threshold selection is:
  - produced by a script (not hand-edited)
  - recorded with git SHA + run metadata
- If nested cannot satisfy FPR without killing power:
  - update `project_state/KNOWN_ISSUES.md` + `docs/PLAN_OF_RECORD.md` to explicitly de-scope nested for paper v1
  - add a “why nested fails” summary (parameter sensitivity + failure mode)

**Minimal tests/commands**
- `make test-fast`
- `python -m experiments.synthetic.nested_killtest --config <cfg> --out reports/synthetic/nested_killtest/<RUN_ID>`
- (Optional) `make calibrate-thresholds` if we unify calibration infra.

**Expected artifacts/logs**
- `reports/synthetic/nested_killtest/<RUN_ID>/` with summary markdown + tables
- Updated calibration JSON(s) if calibration succeeds
- Run log + gpt bundle for ticket-10

---

## Ticket #3 (ticket-11) — Evaluation contamination via caps/selection: enforce aligned comparisons
**Goal (1 sentence)**  
Ensure loss comparisons are not biased by hidden caps/skips: DM tests and ΔLoss must be computed on aligned window sets with explicit `n_effective` and skip/cap summaries.

**Files/modules likely involved**
- `experiments/eval/run.py` (alignment, skip policy, DM computations)
- `src/eval/metrics.py` (loss definitions; if needed)
- `src/finance/portfolios.py` (skip semantics already exist; ensure propagated)
- Summary tooling:
  - `tools/make_summary.py`
  - `tools/summarize_rc_sanity.py`
- Tests:
  - `tests/test_skip_reasons.py`
  - (new) regression test for “aligned DM requires same window set”

**Acceptance criteria**
- Every reported ΔLoss includes:
  - `n_effective` (count of aligned windows)
  - skip counts by reason for each arm (baseline vs overlay)
- Any cap/truncation is recorded in run metadata and surfaced in `limitations.md`.
- Summary aggregation never mixes capped with uncapped runs without a clear label/segregation.
- Tests pass.

**Minimal tests/commands**
- `make test-fast`
- Targeted eval smoke (small window count):
  - `EXEC_MODE=deterministic make rc-lite-sanity` (already produces daily eval outputs)

**Expected artifacts/logs**
- Updated `reports/.../summary/limitations.md` and summary CSVs showing `n_effective` + skip stats
- Run log + gpt bundle for ticket-11

---

## Ticket #4 (ticket-12) — Portfolio solver behavior: enforce fail-loud/explicit-skip everywhere
**Goal (1 sentence)**  
Guarantee that *all* pipelines (daily + weekly) share the same “no silent fallback” contract for MV optimization.

**Files/modules likely involved**
- `src/finance/portfolios.py` (contract already exists; ensure weekly uses it)
- `experiments/equity_panel/run.py` (propagate solver status + skip reason)
- Tests:
  - add or extend a test that simulates missing cvxpy / missing solver and asserts explicit skip or raised error

**Acceptance criteria**
- With solver missing:
  - pipeline either fails-loud (default) OR returns `skipped=true` with `MissingSolverError` recorded
  - never produces equal-weight fallback
- Weekly outputs include `solver_status` / `skipped` fields where relevant.
- Tests pass.

**Minimal tests/commands**
- `make test-fast`
- If repo supports it, run missing-solver smoke:
  - `FJS_FORCE_MISSING_CVXPY=1 EXEC_MODE=deterministic make rc-lite-sanity`
  - (or equivalent env knob used in existing tests)

**Expected artifacts/logs**
- Run log + gpt bundle for ticket-12

---

## Ticket #5 (ticket-13) — Advisor-ready “validity RC” rerun + results refresh
**Goal (1 sentence)**  
Produce one clean, reproducible run (daily + weekly smokes) that meets validity criteria and update the advisor-facing summaries.

**Files/modules likely involved**
- No major code changes expected; focus on running + documenting.
- Docs to update:
  - `PROGRESS.md`
  - `project_state/CURRENT_RESULTS.md`
  - `project_state/KNOWN_ISSUES.md` (if any item is resolved)
  - `reports/memo.md` / `reports/brief.md` if those are part of the current workflow

**Acceptance criteria**
- `EXEC_MODE=deterministic make rc-lite-sanity` completes.
- Outputs include complete summaries (`summary_perf.csv`, `summary_detection.csv`, `limitations.md`, `completeness.json`).
- Weekly smokes produce attributable gating diagnostics (post ticket-09).
- Docs updated with run IDs, key metrics, and explicit limitations.

**Minimal tests/commands**
- `make test-fast`
- `EXEC_MODE=deterministic make rc-lite-sanity`
- `make memo` (if memo build is part of the expected artifacts)

**Expected artifacts/logs**
- New `reports/rc-.../` directory + weekly outputs directory
- Run log + gpt bundle for ticket-13
