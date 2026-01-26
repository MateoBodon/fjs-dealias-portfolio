# FJS-TKT-028

## Goal
Make inject-spike a decisive diagnostic by adding structured tvec telemetry and standardizing between-mode real-window curves.

## Scope
- Add additive telemetry output to `experiments/eval/inject_spike.py`.
- Update Makefile defaults/targets to prefer `--inject-mode between`.
- Extend `tests/experiments/test_inject_spike.py`.
- Do not change overlay/gating thresholds.

## Acceptance Criteria
- Write `pre_gate_telemetry.csv` (or `tvec_debug.csv`) with per-window fields: `window_id`, `mu`, `inject_mode`, `target_component`, `q`, `delta_frac`, `root_count`, `selected_root`, `failure_reason` + scalar checks.
- Make inject-spike default to `inject-mode=between` or add `inject-spike-between-smoke` target.
- Smoke run yields detection/acceptance = 0 at `mu=0` and >0 for at least one `mu>0`.
- Tests assert telemetry file exists and non-empty.
- `make test-fast` and `pytest -q tests/experiments/test_inject_spike.py` pass.

## Plan
1. Inspect `experiments/eval/inject_spike.py` and current inject-spike make targets to find injection mode defaults and telemetry hooks.
2. Add structured telemetry CSV writing (pre-gate) with per-window fields and scalar checks.
3. Standardize between-mode real-window curves or default target to between mode in Makefile.
4. Update tests in `tests/experiments/test_inject_spike.py` to validate telemetry output.
5. Run required tests and create run log, then update `PROGRESS.md` and decision log if needed.

## Notes
- Keep diffs minimal and avoid refactors.
- Maintain existing overlay/gating thresholds.
