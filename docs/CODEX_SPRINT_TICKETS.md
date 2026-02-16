# CODEX_SPRINT_TICKETS.md — Next sprint (ordered)

Sprint scope: **unblock publishable validity + get one advisor‑ready uncapped RC run**.  
Rule: do not expand the experiment grid until Ticket #18 and Ticket #20 are closed.

Priority order as of 2026-02-16:
1. Ticket #18 (injection flat-zero debugging)
2. Ticket #20 (advisor-ready uncapped run)
3. Ticket #31 (docs recenter + canonical truth refresh, completed)

---

## Ticket #31 — Docs recenter + snapshot refresh

**Status:** DONE (doc-only, 2026-02-16)

**Goal (1 sentence):** Make canonical docs (`PROJECT.md`, `README.md`, `PLAN_OF_RECORD`, `project_state`) reflect current truth: strong engineering pipeline, unresolved injection flat-zero blocker, and next validity gates.

**Delivered in this ticket:**
- Added external audit snapshot: `docs/gpt_outputs/20260216_analysis.md`.
- Recentered priorities in `docs/PLAN_OF_RECORD.md` around Ticket #18 then Ticket #20.
- Replaced placeholder `PROJECT.md` with concrete purpose/state/risks/done criteria.
- Updated `README.md` current status to remove stale dated claims.
- Corrected `project_state/CURRENT_RESULTS.md` arithmetic inconsistency and left only artifact-verified metrics.
- Added run log and PROGRESS entry with required validation/test commands.

---

## Ticket #21 — gpt-bundle diff auditability (FAILED)

**Status:** FAIL — `DIFF.patch` only captured the last commit, hiding multi-commit changes.

---

## Ticket #22 — gpt-bundle range diff + bundle meta

**Status:** DONE

**Goal (1 sentence):** Make `make gpt-bundle` produce a reviewable full-range diff (merge-base..HEAD) with explicit base/head metadata.

---

## Ticket #16 — Paper config integrity: kill silent fallback (TOP PRIORITY)

**Goal (1 sentence):** Eliminate the silent `paper-v1` config fallback and make missing/invalid configs fail loudly so “paper runs” are reproducible and audit-safe.

**Files/modules likely involved:**
- `experiments/eval/config.py`
- `experiments/eval/config.yaml`
- `Makefile`
- `experiments/eval/config.paper_v1.yaml` (create or rename to match Make target)
- `tests/experiments/test_eval_run.py`
- Docs: `project_state/KNOWN_ISSUES.md`, `docs/PLAN_OF_RECORD.md`, `PROGRESS.md`

**Acceptance criteria:**
- The Make target(s) that run “paper-v1” **load a real config file** (no default fallback).
- If a requested config file path is missing, the run **fails loudly** with a clear error (non-zero exit).
- `run.json` records:
  - resolved config path/name
  - config hash (sha256 of `resolved_config.json`)
  - git SHA + dirty flag
- `project_state/KNOWN_ISSUES.md` no longer lists “Missing paper-v1 config file”.

**Minimal tests/commands:**
- `make test-fast`
- `pytest tests/experiments/test_eval_run.py -k config`
- Real-data smoke (small):
  - `EXEC_MODE=deterministic make rc-lite-sanity`
  - `EXEC_MODE=deterministic make rc-dow` (or the fixed paper target, if light enough)

**Expected artifacts/logs:**
- `docs/agent_runs/<RUN_NAME>/` with PROMPT/COMMANDS/RESULTS/TESTS/META (+ DIFF.patch recommended)
- A smoke run under `reports/` showing `run.json` includes config path/hash and no silent fallback

---

## Ticket #17 — Nested calibration grid coverage: remove `calibration_missing_p_T` skips

**Goal (1 sentence):** Extend nested calibration to cover real-data observed \((p,T)\) (notably p≈188, T∈{70,80}) and unblock nested smokes by eliminating `calibration_missing_p_T`.

**Files/modules likely involved:**
- `experiments/synthetic/nested_killtest.py`
- `experiments/synthetic/config.nested.killtest.yaml`
- `calibration/nested_edge_delta_thresholds.json`
- `src/fjs/gating.py` (lookup logic / strictness)
- `tests/synthetic/test_calibration.py`
- `tests/test_threshold_eval.py`

**Acceptance criteria:**
- `make run:equity_nested_smoke_tiny` produces windows that **do not** skip with `calibration_missing_p_T`.
- Calibration JSON includes audit metadata (run_name, timestamp, git_sha, config_hash) and thresholds for the new grid cells.
- Synthetic nested null-FPR at the operating point remains **≤ target** (e.g., 2%) for newly added \((p,T)\) cells.

**Minimal tests/commands:**
- `make test-fast`
- `python -m experiments.synthetic.nested_killtest --config experiments/synthetic/config.nested.killtest.yaml --out reports/synthetic/nested_killtest/<RUN_ID> --calibration-out calibration/nested_edge_delta_thresholds.json`
- `EXEC_MODE=deterministic make run:equity_nested_smoke_tiny`

**Expected artifacts/logs:**
- `reports/synthetic/nested_killtest/<RUN_ID>/` summary + tables
- Updated `calibration/nested_edge_delta_thresholds.json`
- Weekly tiny smoke outputs under `experiments/equity_panel/outputs_nested_smoke_tiny/...` with skips attributed to something other than missing calibration

---

## Ticket #18 — Injection sensitivity on real windows (detection/acceptance vs μ)

**Status:** FAIL — flat-zero curve (no detections/acceptances across μ).
**Latest note:** Week full run (ticket-24) remains flat-zero; pre-gate dominated by `tvec_off_component` + no-root reasons (see run `reports/inject_spike/20251226_ticket24_week_full_fix/`).

**Goal (1 sentence):** Prove the detection + gating stack responds to known spikes under real-data noise by running injection sensitivity and producing a μ→(detection, acceptance) curve.

**Files/modules likely involved:**
- `experiments/eval/inject_spike.py`
- `experiments/eval/run.py` (plumbing for injected windows / diagnostics)
- `tools/make_summary.py` (or a dedicated injector summarizer)
- `project_state/RESEARCH_NOTES.md` (record conclusions + plots)

**Acceptance criteria:**
- Produces a CSV and plot/table:
  - injected_mu → detection_rate and acceptance_rate
  - baseline false positive rate on non-injected windows is reported
- For at least one design and edge_mode, detection_rate increases monotonically with μ and acceptance_rate is non-zero at moderate μ.
- Artifacts stored under a timestamped `reports/` subdir and referenced in `PROGRESS.md`.

**Minimal tests/commands:**
- `make test-fast`
- `make inject-spike`
- Optional coarse sweep: `make inject-spike-coarse`

**Expected artifacts/logs:**
- `reports/inject_spike/<RUN_ID>/{curve.csv,curve.png(or .pdf),run.json,resolved_config.json}`
- run log in `docs/agent_runs/<RUN_NAME>/`

---

## Ticket #23 — Injection diagnostics + max-windows sampling

**Status:** DONE

**Goal (1 sentence):** Make `inject_spike.py` diagnostic (per-window + gating attribution), add max-windows sampling, and run real-data smokes that explain why `week` is flat.

**Acceptance criteria:**
- CLI supports `--max-windows`, `--window-sampling`, and `--window-sampling-seed` with deterministic sampling post-filtering.
- Outputs include `windows_detail.csv` and `gating_reasons.csv` with required columns and guardrail reason buckets.
- `run.json` captures sampling metadata, baseline vs injected window counts, and reason-bucket summaries.
- Unit tests cover deterministic sampling, output schema, and missing-config hard errors.
- Real-data inject-spike smokes run and are referenced in `PROGRESS.md` (with gating histograms when flat).

---

## Ticket #19 — Conditional effect reporting + weight-change stats (changed windows only)

**Status:** DONE

**Goal (1 sentence):** Add “changed-window” performance reporting so we can tell whether the overlay ever matters when it triggers.

**Files/modules likely involved:**
- `tools/make_summary.py`
- `tools/summarize_rc_sanity.py`
- `experiments/eval/run.py` (ensure “changed-window” flags are emitted consistently)
- `src/evaluation/*` (if summary computation lives there)
- `tests/tools/test_make_summary.py`
- `tests/experiments/test_eval_run.py`

**Acceptance criteria:**
- Summary tables include:
  - ΔMSE/ΔQLIKE conditional on **changed windows**
  - `n_changed` counts and changed fractions
  - weight-change magnitude stats (median ‖Δw‖, turnover delta) for EW and MV
- Unit tests assert:
  - the changed-window set matches the semantics used for `n_effective_*` and aligned comparisons

**Minimal tests/commands:**
- `make test-fast`
- `EXEC_MODE=deterministic make rc-lite-sanity`
- `PYTHONPATH=src:. python tools/make_summary.py --rc-dir <rc-lite-sanity-dir>`

**Expected artifacts/logs:**
- Updated summary CSVs + an updated `limitations.md` template section describing conditional reporting
- run log in `docs/agent_runs/<RUN_NAME>/`

---

## Ticket #20 — Advisor-ready uncapped RC run (primary design = `week`) + memo outputs

**Status:** PARTIAL — uncapped sample_spike run completed to populate changed-window stats; advisor-ready full week run (n_effective threshold + memo outputs) still pending.

**Goal (1 sentence):** Produce one uncapped, advisor-ready RC run with meaningful effective sample size and clean summaries (including conditional effects + skip/acceptance).

**Files/modules likely involved:**
- `experiments/eval/run.py`
- `experiments/eval/config.yaml` (and/or paper YAML if needed)
- `tools/make_summary.py`
- `tools/build_memo.py` (if memo output is part of the standard advisor packet)
- Docs: `project_state/CURRENT_RESULTS.md`, `PROGRESS.md`

**Acceptance criteria:**
- Run has `cap_active=false` and appears in `summary/completeness.json` as complete/uncapped.
- `summary/summary_perf.csv` and `summary/summary_detection.csv` are non-empty.
- Effective sample size threshold (justify if different):
  - `n_effective_mse >= 150` for EW at minimum (or a documented alternative)
- Includes:
  - skip reason histogram table
  - conditional-effect table
  - (if supported) regime slicing / crisis slices

**Minimal tests/commands:**
- `make test-fast`
- Main run:
  - `EXEC_MODE=throughput make rc-week RC_WORKERS=$(nproc)`
- Summarize:
  - `PYTHONPATH=src:. python tools/make_summary.py --rc-dir <reports/rc-week-dir>`
- Optional advisor memo:
  - `make memo` (or `python tools/build_memo.py ...` if configured)

**Expected artifacts/logs:**
- `reports/<rc-week-dir>/summary/*` including perf + detection + limitations + completeness
- `docs/agent_runs/<RUN_NAME>/`
- Updates to `PROGRESS.md` and `project_state/CURRENT_RESULTS.md` referencing the run

---

### Sprint rule-of-thumb (enforced)
If Ticket #18 shows injection sensitivity works but Ticket #20 still shows conditional ΔLoss ~0, we stop expanding the grid and prepare a pivot memo (diagnostic/negative result framing).
