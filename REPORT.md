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
