# Commands

- `git checkout -b codex/ticket-36-meta-sha-guardrail`
  - Created ticket branch from `71a700bb15a7f39b70a705215d5258e2d24549f3`.
- `python3 tools/agentic/runlog_init.py --ticket "36" --summary "Enforce runlog META.json SHA correctness + fix ticket-35 meta drift" --run-name "20260217_020118_ticket-36_enforce-runlog-meta-sha-correctness-fix-ticket35-drift"`
  - Initialized run log scaffold under `docs/agent_runs/`.
- `pytest -q tests/test_validate_runlog.py tests/test_gpt_bundle.py tests/test_gpt_bundle_diff.py`
  - Targeted tests passed after guardrail implementation.
- `. .venv/bin/activate && make validate-runlogs`
  - Passed; ticket-35 metadata fix validated.
- `. .venv/bin/activate && make test-fast`
  - Passed (`90 passed, 171 deselected`).
