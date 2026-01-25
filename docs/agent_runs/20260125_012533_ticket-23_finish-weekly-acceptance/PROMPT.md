# Prompt

Ticket: FJS-TKT-023
Goal: Finish FJS-TKT-022 acceptance: make git status clean (no untracked ticket/scaffold dirs), harden the exception path in weekly runner, and enforce non-empty detection_summary.csv for the tiny weekly smoke.

Scope/constraints:
- Clean working tree by either committing or removing docs/tickets/FJS-TKT-022.md and tools/agentic/.
- In experiments/equity_panel/run.py initialize locals used in except (e.g., calibration_missing) before try.
- In tests/experiments/test_gating_diagnostics.py require detection_summary.csv exists + non-empty for the smoke/test harness (or add a tiny integration test that asserts this).

Acceptance criteria:
- git status clean after changes.
- weekly runner exception path cannot raise UnboundLocalError (locals initialized).
- make run:equity_nested_smoke_tiny produces non-empty detection_summary.csv and gating_diagnostics.csv; no skip_reason_primary==DIAGNOSTIC_FAILURE.
- tests assert detection_summary.csv exists+non-empty.

Test command: make test-fast && make run:equity_nested_smoke_tiny
Risk: med

Notes:
- Step 1 requires confirming Agentic System scaffold (AGENTS.md, PROJECT.md, tools/agentic/) and running /prompts:bootstrap if missing.
- Must emit gpt_bundle.zip via gpt-bundle skill after work.
