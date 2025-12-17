# 2025-12-17 — Codex Sprint 02 (codex/nested-killtests-sanity)

## Goals
Clarify nested design viability via calibration + synthetic kill-tests, add explicit skip-reason logging, and make rc-lite-sanity summaries flag overlay harm on daily DoW/vol slices.

## Planned tasks
1. Add nested detection skip-reason logging; pipe into detection_summary and summary.json; add test.
2. Extend Tyler/Huber edge calibration coverage for p≈180–220, T≈60–80; update artifacts; add test for nested config thresholds.
3. Create nested-mimicking synthetic kill-test config and run null/power sweeps; summarize results.
4. Fix rc-lite-sanity summaries: vol-state aggregation and overlay harm flags for daily DoW/vol.
5. Housekeeping: mark canonical rc-lite-sanity run, archive/label stale outputs, update CURRENT_RESULTS and KNOWN_ISSUES.

## Commands to run (expected)
- make test-fast
- make sweep:acceptance (nested p≈200 T≈60–80 slice)
- make calibrate-thresholds (after sweeps)
- python -m experiments.synthetic.<cfg> or equivalent for nested kill-tests
- make rc-lite-sanity

## Log
- Setup: checked out branch `codex/nested-killtests-sanity`; read AGENTS.md, Long_Term_Plan.md, project_state/INDEX.md, ROADMAP.md, CURRENT_RESULTS.md.
- Commands: `git status --short --branch` (saw pending edits in project_state), `git checkout -b codex/nested-killtests-sanity`.
- Installed venv + deps: `python3 -m venv .venv`, `pip install -e .[dev]`.
- Tests: `make test-fast` (venv) → pass (65 passed, 142 deselected).
- Initial `make test-fast` failed (pytest missing) before venv setup; resolved after creating venv and installing dev extras.
- Calibration: ran `python -m experiments.synthetic.calibrate_thresholds` (nested slice p={188,200}, T={60,70,80}, replicates=5, tyler+huber, delta_abs=0.35, delta_frac∈{0.015,0.02}, stability=0.3, trials 30/15) with run_id `calib-20251217T223400Z`; resumed to completion; artifacts in `reports/synthetic/calib_nested_20251217*.json`.
- Merged new calibration cells into `calibration/edge_delta_thresholds.json` (backup at `calibration/edge_delta_thresholds.backup_20251217`); added direct keys 188x{60,70,80}, 200x{60,70,80} for tyler+huber and backfilled 64x/96x entries.
- Tests: `make test-fast` (venv) → pass (65 passed, 144 deselected) after calibration merge and new lookup tests.
- Note: earlier heavier calibration grids (p={180,188,200,220}, trials 120/80) timed out; kept run_id dirs under `reports/synthetic/calib/` for traceability but did not write outputs.
- Synthetic nested kill-test: added `experiments/synthetic/nested_killtest.py` + config `experiments/synthetic/config.nested.killtest.yaml`; ran with trials_per_scenario=12 (tyler, p≈200, weeks 6–8, reps=5, delta=0.35, delta_frac_min=0.05). Outputs in `reports/synthetic_nested_killtest/` show detection_rate=1.0 even in null scenario → FPR ~1 (overlay unsafe), logged in summary.md.
- Regenerated rc-lite-sanity summary via `tools/summarize_rc_sanity.py` (daily dow/vol + weekly dow/nested) with new overlay deltas/effect flags; updated `reports/rc-20251208-sanity-20251209_001356/summary_sanity.json` and regime.csv.
- Tests: `make test-fast` (after summary/doc updates) → pass (65 passed, 144 deselected).
