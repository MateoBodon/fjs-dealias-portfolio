# Sprint Log

## 2025-11-02T02:40Z
- **Step**: Initialized GPT-5 Codex session, reviewed repo_plan scope, switched to `feat/codex-sprint-01`.
- **Decisions**: Confirmed instructions emphasise overlay implementation and reproducible RC artifacts; defer existing `AGENTS.md` reconciliation to docs milestone.
- **Checks**: `pytest -q -x` currently fails at `tests/test_dealias.py::test_dealias_search_limits_sigma2_false_positives` (baseline false positives high); will address in overlay overhaul.
- **Next Actions**: Scaffold data/fjs/baseline modules per Sprint 1 milestones, design accompanying tests before implementation.

## 2025-11-02T02:53Z
- **Step**: Scaffolded new data/grouping, edge/overlay, baseline, experiment, script, and tooling modules with placeholder implementations plus stub tests; added detection config.
- **Decisions**: APIs raise `NotImplementedError` to make missing functionality explicit while enabling test scaffolding.
- **Checks**: `pytest -q -x` still halts at `tests/test_dealias.py::test_dealias_search_limits_sigma2_false_positives` (expected pre-overlay fix).
- **Next Actions**: Implement daily loader and grouping logic with balanced universe enforcement and update tests accordingly.

## 2025-11-02T03:00Z
- **Step**: Implemented daily loader with winsorisation/balanced-universe enforcement and DoW/vol-state groupers; replaced stub tests with behavioural checks.
- **Decisions**: Rank-based quantile binning for volatility states keeps bins evenly populated; weekdays enforce Mon–Fri coverage to align with trading calendar.
- **Checks**: `pytest -q -x` continues to fail at `tests/test_dealias.py::test_dealias_search_limits_sigma2_false_positives` (legacy overlay behaviour).
- **Next Actions**: Wire robust edge estimation and overlay detection routines before tightening failing dealias tests.

## 2025-11-02T03:06Z
- **Step**: Added Tyler/Huber edge estimator with buffering and implemented overlay detection/replacement pipeline plus synthetic spike/null tests.
- **Decisions**: Detection keeps full eigen-decomposition context to allow rank-one updates; shrinkage only applied off-spike towards robust edge.
- **Checks**: `pytest -q -x` still halts at `tests/test_dealias.py::test_dealias_search_limits_sigma2_false_positives` (pre-refactor false positives).
- **Next Actions**: Integrate RIE and EWMA baselines and begin calibrating overlay thresholds against synthetic suites.

## 2025-11-02T03:11Z
- **Step**: Implemented RIE eigenvalue clipping and EWMA covariance with weighted means; replaced stub baseline tests with numerical checks.
- **Decisions**: Clipped MP bulk to its mean to stabilise RIE while leaving spikes intact; EWMA debiases weights by default to maintain unit mass.
- **Checks**: `pytest -q -x` still fails at `tests/test_dealias.py::test_dealias_search_limits_sigma2_false_positives` pending overlay refactor.
- **Next Actions**: Build synthetic calibration harness for threshold tuning and generate artefact scaffolding.

## 2025-11-02T03:14Z
- **Step**: Implemented calibration harness producing thresholds.json and ROC CSV/plots; added regression test exercising low-trial sweep.
- **Decisions**: Shared FPR across signal levels to stabilise grid evaluation; fallback selects lowest-FPR margin if FPR≤2% grid empty.
- **Checks**: `pytest -q -x` still red at `tests/test_dealias.py::test_dealias_search_limits_sigma2_false_positives` (pre-threshold tuning baseline).
- **Next Actions**: Expand rolling evaluation pipeline and tie new baselines/overlay into experiments outputs.

## 2025-11-02T03:19Z
- **Step**: Built rolling evaluation harness generating ΔMSE, DM, and VaR coverage summaries plus CLI artifacts.
- **Decisions**: Compared overlay against RIE baseline for DM; min-var portfolios stabilised with ridge 5e-4; produced bar plots when matplotlib present.
- **Checks**: `pytest -q -x` still fails at `tests/test_dealias.py::test_dealias_search_limits_sigma2_false_positives` pending overlay gating refresh.
- **Next Actions**: Finalise docs/tooling, hook AGENTS/README, and ensure memo/gallery targets run green.

## 2025-11-02T03:36Z
- **Step**: Refreshed docs (README, AGENTS), wired memo/gallery builders, simplified Makefile targets, and generated RC artifacts via `make rc` (reports/rc-20251101).
- **Decisions**: Fallback to plain-text tables when optional plotting/tabulate deps missing; default Makefile commands run with `PYTHONPATH=.` to expose packages.
- **Checks**: `pytest -q -x` still red at `tests/test_dealias.py::test_dealias_search_limits_sigma2_false_positives`; other suites pass. `make rc` and `make gallery` completed successfully.
- **Next Actions**: Address legacy `dealias_search` FPR regression in follow-up; prepare final review summary.

## 2025-11-02T03:47Z
- **Step**: Reproduced `tests/test_dealias.py::test_dealias_search_limits_sigma2_false_positives` failure and instrumented `dealias_search` with `FJS_DEBUG` per-candidate logging (edge margins, admissibility, isolation, angle, reason codes).
- **Decisions**: Structured debug output as JSON for downstream parsing; enumerated candidate/a-grid indices including dynamically added root-find vectors.
- **Checks**: Target test still failing pending threshold integration; debug traces verified via `FJS_DEBUG=1` dry run.
- **Next Actions**: Load calibrated thresholds, align gating parameters, and drive null FPR back under 2%.

## 2025-11-02T04:29Z
- **Step**: Added detection settings loader (YAML + thresholds.json), hardened `dealias_search`/overlay defaults (Tyler edge, isolation gating, off-component cap, t-eps), and introduced calibrated slow tests (null FPR ≤2%, power ≥80%).
- **Decisions**: Default `t_eps=0.06`, `off_component_cap=0.3`, `require_isolated=True`, and `q_max=1` keep null false positives suppressed; unit/integration tests override settings when legacy behaviour is required.
- **Checks**: `tests/test_dealias.py`, `tests/fjs/test_config.py`, and `tests/fjs/test_dealias_calibrated.py` now pass locally; failing null FPR test resolved.
- **Next Actions**: Refresh docs, rebuild RC artifacts (`make rc && make gallery`), and add memo postmortem before final regression (`pytest -q`).

## 2025-11-02T04:59Z
- **Step**: Rebuilt RC artifacts (`reports/rc-20251102/`), appended detection postmortem, and updated synthetic/power harnesses plus theta/guardrail tests to use calibrated detection settings.
- **Decisions**: Loosened synthetic overrides (`off_component_cap=None`, `t_eps=0.05`, `require_isolated=False`) inside simulation modules to maintain power while retaining strict defaults elsewhere.
- **Checks**: `pytest -q` green (warnings only from sparse multi-spike means); `make rc` + `make gallery` regenerated memo/gallery with detection diagnostics.
- **Next Actions**: Stage RC artifacts, documentation updates, and push branch after final review.

## 2025-11-02T05:33Z
- **Step**: Added Day-of-Week/Vol-State replicates to the daily loader, VIX-aware volatility bucketing with realised-vol fallback, and updated data unit tests.
- **Decisions**: Store group keys alongside balanced panels; default vol-state bins stay at four levels with deterministic EWMA-based fallback when VIX gaps.
- **Checks**: `pytest tests/data/test_loader.py tests/data/test_groups.py` passes.
- **Next Actions**: Implement factor prewhitening pipeline and extend residual de-aliasing tests.
