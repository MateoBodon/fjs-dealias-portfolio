# FJS-TKT-027

## Goal
Make week RC eval runs always emit `run.json`, `resolved_config.json`, and a non-empty `run.log` (with START/END lines), even when exiting early after prewhiten.

## Scope
- Update `experiments/eval/run.py` to create the output directory and write a `run.json` stub early.
- Add explicit stage/status updates and ensure early exits map to terminal statuses.
- Add regression coverage in `tests/experiments/test_eval_week_artifacts.py` for early-exit artifacts.
- Update `PROGRESS.md` and the run log under `docs/agent_runs/`.

## Acceptance Criteria
- `run.json` always exists for week runs even on failure.
- `resolved_config.json` is always written.
- `run.json` includes terminal status `ok|no_windows|error` and a stage tag.
- `run.log` is non-empty with START/END lines.
- Regression test asserts artifacts + status schema on early-exit path.
- `make test-fast` passes.

## Plan
1. Add run status/log scaffolding in `experiments/eval/run.py` (early stub + stage updates).
2. Ensure early-return/no-window paths write terminal status and log END line in `experiments/eval/run.py`.
3. Add regression test for week early-exit artifacts in `tests/experiments/test_eval_week_artifacts.py`.
4. Run required tests and capture results in `docs/agent_runs/<RUN_NAME>/`.
5. Update `PROGRESS.md` and generate GPT bundle.

## Notes
- Keep overlay math/gating thresholds/portfolio logic unchanged.
