# CODEX_SPRINT_TICKETS (NEXT SPRINT ONLY)
Date: 2025-12-17  
Source of priorities: `fjsprompt1output.md`, `project_state/KNOWN_ISSUES.md`, `project_state/CURRENT_RESULTS.md`.

**Sprint goal:** restore pipeline validity (no broken FPR, no cap-contaminated metrics, no silent solver fallbacks) and produce a trustworthy `rc-lite-sanity` run.

---

## Ticket 01 — Fix nested synthetic null FPR (currently ≈1.0)
**Goal (1 sentence):** Make the nested gating path non-broken by ensuring the nested synthetic kill-test has controlled null acceptance (no unconditional accept).

**Files / modules likely involved**
- `experiments/synthetic/nested_killtest.py`
- `experiments/synthetic/config.nested.killtest.yaml`
- `src/fjs/gating.py`
- `src/fjs/dealias.py`
- `src/fjs/mp.py`
- (maybe) `calibration/edge_delta_thresholds.json` (only if a lookup bug is found)

**Acceptance criteria**
- Running the nested kill-test (null scenario) yields **null acceptance <= 0.05** (or the explicitly configured target) on a nontrivial number of trials (>=200) in deterministic mode.
- The kill-test report clearly states:
  - p, weeks, reps, edge_mode, delta/delta_frac, seed, git SHA
  - acceptance rate by scenario (null/moderate/strong)
  - top skip/rejection reasons if acceptance is low
- Add a regression test that would have failed under the “accept everything” bug:
  - either a deterministic fixture where the previous code accepted and now rejects, or
  - a bounded Monte Carlo test with fixed seeds and a conservative upper bound.

**Minimal tests / commands**
- `make test-fast`
- `pytest -m unit -k "gating or nested"`
- Run kill-test (inspect `--help` for exact flags):
  - `PYTHONPATH=src python experiments/synthetic/nested_killtest.py --config experiments/synthetic/config.nested.killtest.yaml --exec-mode deterministic --run-id <RUN_ID>`

**Expected artifacts/logs**
- `reports/synthetic_nested_killtest/<RUN_ID>/run.json`
- `reports/synthetic_nested_killtest/<RUN_ID>/summary.md`
- `docs/agent_runs/<ts>_ticket-01_nested-null-fpr/` (see docs protocol)

---

## Ticket 02 — Remove evaluation contamination via caps (cap must be explicit + default OFF)
**Goal (1 sentence):** Prevent window/asset caps from silently driving “percent_changed≈100%” and other misleading top-line metrics.

**Files / modules likely involved**
- `experiments/eval/config.py`
- `experiments/eval/run.py`
- `tools/make_summary.py`
- `tools/summarize_rc_sanity.py`
- `src/meta/run_meta.py`

**Acceptance criteria**
- Any cap (`max_windows`, “cap to first K windows”, etc.) is:
  - default OFF in RC/rc-lite-sanity configs and Make targets
  - recorded in run metadata (`run_manifest.json` or equivalent)
  - clearly indicated in summary tables when enabled
- percent_changed is computed on the **true evaluated window set** and is not trivially 100% due to truncation artifacts.
- Add a unit/integration test: turning cap on changes the “windows evaluated” count and is reported as such.

**Minimal tests / commands**
- `make test-fast`
- `pytest -m integration -k "rc_lite_sanity or eval"`
- `EXEC_MODE=deterministic make rc-lite-sanity` (verify percent_changed is sensible with caps OFF)

**Expected artifacts/logs**
- Updated `reports/rc-*-sanity-*/summary_sanity.json` showing cap metadata
- `docs/agent_runs/.../` log with before/after comparison notes

---

## Ticket 03 — Fail-loud MV solver fallback + record solver backend
**Goal (1 sentence):** If MV portfolio evaluation is requested and the solver backend is unavailable, the run must error loudly and record solver identity when available.

**Files / modules likely involved**
- `src/finance/portfolio.py` (or solver wrapper)
- `experiments/eval/run.py`
- `experiments/equity_panel/run.py`
- `src/meta/run_meta.py`
- `project_state/KNOWN_ISSUES.md` (update when fixed)

**Acceptance criteria**
- When MV is requested and `cvxpy` (or required backend) is missing:
  - the run exits with a clear error message and nonzero exit code
  - no EW fallback occurs silently
- When MV runs successfully:
  - solver backend is recorded in run metadata and metrics outputs

**Minimal tests / commands**
- `pytest -m unit -k "portfolio or solver"`
- `make test-fast`
- Optional: run a smoke MV eval locally/Hetzner and verify metadata shows backend.

**Expected artifacts/logs**
- `docs/agent_runs/.../` includes a “solver missing” test case evidence
- Updated run metadata schema/fields documented somewhere discoverable (e.g., `src/meta/run_meta.py` or docs)

---

## Ticket 04 — Detection drought diagnostics (weekly/nested: explain “0 detections”)
**Goal (1 sentence):** When weekly/nested runs have 0 detections, produce enough per-window diagnostics to tell whether the issue is “no candidates above edge” vs “candidates rejected by gating”.

**Files / modules likely involved**
- `src/fjs/dealias.py`
- `src/fjs/gating.py`
- `src/report/plots.py`
- `tools/summarize_run.py`
- (optional) `experiments/equity_panel/run.py`

**Acceptance criteria**
- Weekly outputs include a per-window diagnostic artifact (CSV/JSON):
  - max generalized eigenvalue, MP edge(s), edge margin
  - number of candidates considered
  - gate rejection reasons (dominant reason code)
- Summary/memo includes a short “why no detections” section for drought runs.

**Minimal tests / commands**
- `make test-fast`
- `pytest -m integration -k "equity_panel"`
- Run weekly smoke (small):
  - command pattern from README §5.1, with `--design dow` or `--design nested`

**Expected artifacts/logs**
- `experiments/equity_panel/outputs_smoke/.../diagnostics_*` contains drought explanation
- `docs/agent_runs/.../` with a before/after drought run comparison

---

## Ticket 05 — rc-lite-sanity summary hardening (partial run detection + missing sections)
**Goal (1 sentence):** Ensure rc-lite-sanity summaries are complete and do not silently treat partial runs as valid.

**Files / modules likely involved**
- `tools/summarize_rc_sanity.py`
- `tools/make_summary.py`
- `src/meta/run_meta.py`
- `project_state/KNOWN_ISSUES.md` (update after fix)

**Acceptance criteria**
- Summary always includes:
  - daily DoW + daily vol-state sections (when present)
  - weekly DoW + weekly nested sections (when present)
- If a run dir is missing expected outputs:
  - summary flags it prominently as incomplete
  - incomplete runs are excluded from aggregates
- Existing “partial RC dir” issue is resolved (`reports/rc-20251208/` style).

**Minimal tests / commands**
- `pytest -m unit -k "summary or run_meta"`
- `EXEC_MODE=deterministic make rc-lite-sanity`
- `python tools/summarize_rc_sanity.py <run_dir>`

**Expected artifacts/logs**
- `reports/rc-*-sanity-*/summary_sanity.json` with all sections
- `docs/agent_runs/.../` including a completeness check snapshot

---
