# FJS-TKT-023

## Goal
Finish FJS-TKT-022 acceptance: clean git status, harden weekly exception handling, and require non-empty detection_summary.csv for tiny weekly smoke.

## Scope
- Add missing scaffold pieces (PROJECT.md, tools/agentic/) without overwriting repo-specific files.
- In `experiments/equity_panel/run.py`, ensure locals used in exception attribution are initialized before the try block.
- In `tests/experiments/test_gating_diagnostics.py`, require detection_summary.csv to exist and be non-empty.
- Keep working tree clean by tracking docs/tickets and scaffold additions.

## Acceptance Criteria
- `git status` clean after changes.
- Weekly runner exception path does not raise UnboundLocalError (locals initialized).
- `make run:equity_nested_smoke_tiny` produces non-empty `gating_diagnostics.csv` and `detection_summary.csv`.
- No `skip_reason_primary == diagnostic_failure` in diagnostics or detection summary.
- Tests assert detection_summary.csv exists and is non-empty.

## Plan
1. Bootstrap missing scaffold pieces and align `tools/agentic/gpt_bundle.py` with repo gpt-bundle requirements.
2. Update weekly runner exception locals in `experiments/equity_panel/run.py`.
3. Strengthen detection_summary assertions in `tests/experiments/test_gating_diagnostics.py`.
4. Run `make test-fast` and `make run:equity_nested_smoke_tiny`.
5. Update run log + `PROGRESS.md`, then generate GPT bundle.

## Notes
- Run name: 20260125_012533_ticket-23_finish-weekly-acceptance
- Tests: `make test-fast && make run:equity_nested_smoke_tiny`
